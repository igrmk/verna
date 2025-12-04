import time
import json
import sys
import textwrap
import argparse
from verna import db_types
from enum import Enum, auto
import psycopg
from openai import OpenAI
from pydantic import BaseModel, Field
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

from verna.upper_str_enum import UpperStrEnum
from verna.config import get_parser, Sections, print_config, ReasoningLevel

from rich.console import Console
from rich.text import Text

from jinja2 import Environment, StrictUndefined

CON = Console()

JINJA_ENV = Environment(
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,
)


class Mode(UpperStrEnum):
    LEXEME = auto()
    TEXT = auto()


class Language(UpperStrEnum):
    ENGLISH = auto()
    RUSSIAN = auto()
    OTHER = auto()


class Lexeme(BaseModel):
    text: str
    language: Language
    rp: list[str] = Field(default_factory=list)
    past_simple: str | None = None
    past_participle: str | None = None


class Card(BaseModel):
    lexeme: Lexeme
    translations: list[Lexeme]
    example: str | None


class TranslatorResponse(BaseModel):
    mode: Mode | None = None
    language: Language
    typo_note: str | None = None
    rp: str | None = None
    translation: str | None = None
    cards: list[Card] = Field(default_factory=list)


class ExampleResponse(BaseModel):
    example: str | None


TRANSLATION_INSTRUCTIONS = JINJA_ENV.from_string(
    textwrap.dedent("""
        You are a translator and dictionary.
        Swear words are allowed when necessary.
        Informal language is allowed as well.
        Do not explain your actions. Output ONLY JSON matching the schema.
        Between UK and US variants, choose UK.

        Let Q be the user input.

        Input variables:
          - WORD_COUNT(Q) = {{ word_count }}

        Set `language` to Q's language.
        If `language` is `OTHER`, output ONLY {"language": "OTHER"}.

        If you suspect a genuine spelling typo, add a short note to `typo_note`.
        Not a typo:
          - informal usage
          - colloquial expressions
          - contractions such as "wanna" or "gonna"
          - UK-only spelling variants

        Fill `mode` to:
          - `LEXEME`, only if all of the following are true:
            - Q is a single word, a fixed phraseme, or a short, idiomatic, highly common sentence
            - Q contains no proper names of people
            - WORD_COUNT(Q) ≤ 5
          - `TEXT`, otherwise

        If `mode` = `LEXEME`, add Q in full to `cards` as a single entry,
        filling it according to the `Card` filling rules. Omit other root fields.

        Otherwise, if `mode` = `TEXT`, fill the root fields:
          - `translation` — to Russian if Q is in English, or to English if Q is in Russian
          - `rp` — British RP transcription without slashes, only if Q is in English and WORD_COUNT(Q) ≤ 10
          - `cards` — list all English lexemes at the level {{ level }} or higher that appear in Q if Q is in English.
            Extract multi-word lexemes (e.g. phrasal verbs and phrasemes) in addition to single words.
            Exclude proper names.
            Fill each card according to the `Card` filling rules.
            Treat different forms (e.g., verb and noun) as one lexeme

        `Card` filling rules (for the current card C):
          - Fill `C.lexeme` according to the `Lexeme` filling rules
          - Fill `C.translations` with an exhaustive list of translations,
            one translation per list item,
            including those outside Q's context, following the `Lexeme` filling rules
          - Fill `C.example` with the full sentence where the lexeme occurs in Q,
            correcting grammar and spelling beforehand;
            ensure proper sentence capitalisation and punctuation;
            don't correct contractions;
            don't correct local variants;
            leave the lexeme as is where possible;
            set to null if unavailable

        `Lexeme` filling rules (for the current lexeme L):
          - `text` — required; the lexeme in its base form
          - `language` — required
          - `rp` — list of possible British RP transcription of L, without slashes; only if L is English
          - `past_simple` and `past_participle` — only if Q is in English and L is an irregular word
    """).strip()
)

EXAMPLE_INSTRUCTIONS = JINJA_ENV.from_string(
    textwrap.dedent(r"""
        You are a translator and dictionary.
        Swear words are allowed when necessary.
        Informal language is allowed as well.
        Do not explain your actions. Output ONLY JSON matching the schema.
        Between UK and US variants, choose UK.

        Generate an example sentence using the provided lexeme and return it in the `example` field.
        Before doing so, correct grammar and spelling in the lexeme if needed.
        Do not expand or correct contractions.
        Keep the lexeme unchanged where possible.
        Keep the sentence concise—ideally around 5 words.
        Cite a film or book if appropriate, but avoid proper names.
        Ensure proper sentence capitalisation and punctuation.
        If unable to produce a meaningful example, set `example` to null.

        {%- if previous_examples -%}
            {{- '\n\n' -}}
            Don't use any of the following examples
            and try to produce one that doesn't resemble any of them:
            {%- for x in previous_examples -%}
                {{- '\n  ' -}}
                - {{ x }}
            {%- endfor -%}
        {%- endif -%}
    """).strip()
)


def _format_cli_card(card: Card, idx: int) -> Text:
    t = Text()
    t.append(f'[{idx}]', style='bold')
    t.append(' ')

    t.append(card.lexeme.text, style='bold')
    for rp in card.lexeme.rp:
        t.append(' ')
        t.append(f'/{rp}/', style='bold italic')

    def add_kv(k: str, v: str | None) -> None:
        if v:
            t.append('\n  ')
            t.append(f'{k}:', style='dim')
            t.append(' ')
            t.append(v)

    add_kv('PAST SIMPLE', card.lexeme.past_simple)
    add_kv('PAST PARTICIPLE', card.lexeme.past_participle)

    for x in card.translations:
        t.append('\n  - ')
        t.append(x.text)
        for rp in x.rp:
            t.append(' ')
            t.append(f'/{rp}/', style='italic')

    if card.example:
        t.append('\n\n  > ')
        t.append(card.example, style='italic')

    return t


def _format_header(cfg: argparse.Namespace, r: TranslatorResponse) -> Text | None:
    t = Text()
    first = True

    def add_skip() -> None:
        nonlocal first
        if not first:
            t.append('\n\n')
        first = False

    def add_kv(k: str, v: str | None) -> None:
        if v:
            add_skip()
            t.append(f'{k}:', style='dim')
            t.append(' ')
            t.append(v)

    if cfg.debug and r.mode is not None:
        add_kv('MODE', r.mode)

    if r.language == Language.OTHER:
        add_skip()
        t.append('UNSUPPORTED LANGUAGE')

    add_kv('TYPO NOTE', r.typo_note)

    if r.rp:
        add_skip()
        t.append(f'/{r.rp}/', style='italic')

    if r.translation:
        add_skip()
        t.append(r.translation)

    return t if not first else None


def print_response(cfg: argparse.Namespace, r: TranslatorResponse) -> None:
    first = True

    def print_skip() -> None:
        nonlocal first
        if not first:
            CON.print()
        first = False

    header = _format_header(cfg, r)
    if header is not None:
        print_skip()
        CON.print(header, markup=False)

    if len(r.cards) > 0:
        print_skip()
        CON.print('CARDS', style='bold underline')

    for idx, card in enumerate(r.cards, 1):
        print_skip()
        CON.print(_format_cli_card(card, idx), markup=False)


class ConfirmResult(Enum):
    YES = auto()
    NO = auto()
    EXAMPLE = auto()
    QUIT = auto()


def confirm(prompt: str) -> ConfirmResult:
    while True:
        try:
            ans = input(f'{prompt} [y/N/e/q] ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            return ConfirmResult.NO
        if ans == 'y':
            return ConfirmResult.YES
        if ans in ('n', ''):
            return ConfirmResult.NO
        if ans == 'e':
            return ConfirmResult.EXAMPLE
        if ans == 'q':
            return ConfirmResult.QUIT
        CON.print('Please enter y, n, e, or q')


def to_db_card(card) -> db_types.Card:
    return db_types.Card(
        lexeme=card.lexeme.text.strip(),
        rp=card.lexeme.rp,
        base_form=None,
        past_simple=card.lexeme.past_simple,
        past_participle=card.lexeme.past_participle,
        translations=[t.text.strip() for t in card.translations],
        example=[card.example] if card.example is not None else [],
    )


def make_example(cfg: argparse.Namespace, card: db_types.Card, previous_examples: list[str]) -> None:
    client = OpenAI(api_key=cfg.openai_api_key)
    instructions = EXAMPLE_INSTRUCTIONS.render(previous_examples=previous_examples)

    if cfg.debug:
        CON.print(f'INSTRUCTIONS:\n{instructions}')

    if cfg.reason == ReasoningLevel.UNSUPPORTED:
        resp = client.responses.parse(
            model=cfg.model,
            instructions=instructions,
            input=card.lexeme,
            text_format=ExampleResponse,
        )
    else:
        resp = client.responses.parse(
            model=cfg.model,
            reasoning={'effort': cfg.reason},
            instructions=instructions,
            input=card.lexeme,
            text_format=ExampleResponse,
        )
    if resp.output_parsed is None:
        raise SystemExit('OpenAI response could not be parsed')
    data: ExampleResponse = resp.output_parsed
    card.example = [data.example] if data.example is not None else []


def save_card(cfg: argparse.Namespace, card: db_types.Card) -> None:
    try:
        with psycopg.connect(cfg.db_conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                        insert into cards (
                            lexeme,
                            rp,
                            past_simple,
                            past_participle,
                            translations,
                            example
                        )
                        values (%s, %s, %s, %s, %s, %s)
                        on conflict (lower(lexeme)) do update set
                            translations = (
                                select coalesce(array_agg(distinct x), '{}')
                                from unnest(cards.translations || excluded.translations) as x
                            ),
                            rp = (
                                select coalesce(array_agg(distinct x), '{}')
                                from unnest(cards.rp || excluded.rp) as x
                            ),
                            example = (
                                select coalesce(array_agg(distinct x), '{}')
                                from unnest(cards.example || excluded.example) as x
                            )
                        returning (xmax = 0) as inserted;
                    """,
                    (
                        card.lexeme,
                        card.rp,
                        card.past_simple,
                        card.past_participle,
                        card.translations,
                        card.example,
                    ),
                )
                row = cur.fetchone()
                assert row is not None
                inserted = row[0]
                CON.print('Saved' if inserted else 'Merged')
            conn.commit()
    except psycopg.Error as e:
        print(f'Failed to save cards to postgres: {e}', file=sys.stderr)
        sys.exit(2)


def save_cards(cfg: argparse.Namespace, cards: list[db_types.Card]) -> None:
    for idx, card in enumerate(cards, 1):
        previous_examples = []
        proceed = True
        while proceed:
            proceed = False
            CON.print()
            CON.print(db_types.format_card(card, idx), markup=False)
            CON.print()
            res = confirm('Save?')
            if res == ConfirmResult.QUIT:
                CON.print('Skipping remaining cards')
                return
            if res == ConfirmResult.YES:
                save_card(cfg, card)
            if res == ConfirmResult.EXAMPLE:
                proceed = True
                if len(card.example) > 0:
                    previous_examples += card.example
                make_example(cfg, card, previous_examples)


def read_interactively() -> str:
    kb = KeyBindings()

    @kb.add('c-d')
    def _(event):
        event.app.exit(result=event.app.current_buffer.text)

    def cont(width: int, line_number: int, is_soft_wrap: int) -> str:
        return ' ' * width if is_soft_wrap else ' ' * (width - 2) + '… '

    session: PromptSession = PromptSession()
    return session.prompt('Ctrl-D> ', multiline=True, prompt_continuation=cont, key_bindings=kb)


def work() -> int:
    parser = get_parser(sections=[Sections.DB, Sections.VERNA, Sections.OPENAI], require_db=False)
    parser.add_argument('query', nargs='*')
    cfg = parser.parse_args()

    if cfg.print_config:
        print_config(cfg)
        return 0

    if cfg.show_schema:
        CON.print(json.dumps(TranslatorResponse.model_json_schema(), indent=2))
        return 0

    query = ' '.join(cfg.query).strip()
    no_query_error = SystemExit('You must provide a query or use --show-schema')
    if not query:
        if not sys.stdin.isatty():
            query = sys.stdin.read().strip()
            if not query:
                raise no_query_error
    if not query:
        query = read_interactively().strip()
    if not query:
        raise no_query_error

    client = OpenAI(api_key=cfg.openai_api_key)
    instructions = TRANSLATION_INSTRUCTIONS.render(
        word_count=len(query.split()),
        level=cfg.level,
    )
    if cfg.debug:
        CON.print(f'INSTRUCTIONS:\n{instructions}\n')

    start_time = time.perf_counter()
    if cfg.reason == ReasoningLevel.UNSUPPORTED:
        resp = client.responses.parse(
            model=cfg.model,
            instructions=instructions,
            input=query,
            text_format=TranslatorResponse,
        )
    else:
        resp = client.responses.parse(
            model=cfg.model,
            reasoning={'effort': cfg.reason},
            instructions=instructions,
            input=query,
            text_format=TranslatorResponse,
        )
    elapsed = time.perf_counter() - start_time
    CON.print(f'{cfg.model} responded in {elapsed:.1f}s\n')

    if resp.output_parsed is None:
        raise SystemExit('OpenAI response could not be parsed')

    data: TranslatorResponse = resp.output_parsed
    print_response(cfg, data)

    english_cards = [to_db_card(x) for x in data.cards if x.lexeme.language == Language.ENGLISH]
    if sys.stdin.isatty() and english_cards and cfg.db_conn_string:
        CON.print()
        CON.print('SAVING CARDS', style='bold underline')
        save_cards(cfg, english_cards)
    return 0


def main() -> int:
    try:
        return work()
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    sys.exit(main())
