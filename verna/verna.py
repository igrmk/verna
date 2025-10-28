import json
import sys
import textwrap
import argparse
import verna.db_types as db_types
from enum import Enum, auto
from string import Template

import psycopg
from openai import OpenAI
from pydantic import BaseModel, Field
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

from verna.upper_str_enum import UpperStrEnum
from verna.config import get_parser, Sections, print_config


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
    base_form: str | None = None
    past_simple: str | None = None
    past_participle: str | None = None


class Card(BaseModel):
    lexeme: Lexeme
    translations: list[Lexeme]
    context_sentence: str | None


class TranslatorResponse(BaseModel):
    mode: Mode | None = None
    language: Language
    typo_note: str | None = None
    rp: str | None = None
    translation: str | None = None
    cards: list[Card] = Field(default_factory=list)


INSTRUCTIONS = Template(
    textwrap.dedent("""
        You are a translator and dictionary. Swear words are allowed when necessary.
        Informal language is allowed as well.
        Do not explain your actions. Output ONLY JSON matching the schema.
        Between UK and US variants, choose UK.

        Let Q be the user input.

        Input variables:
          - WORD_COUNT(Q) = ${WORD_COUNT}

        Set `language` to Q's language.
        If `language` is `OTHER`, output ONLY {"language": "OTHER"}.

        If you suspect a genuine spelling typo, add a short note to `typo_note`.
        Not a typo:
          - informal usage
          - colloquial expressions
          - contractions such as "wanna" or "gonna"
          - UK-only spelling variants

        Fill `mode` to:
          - `LEXEME`, only if both of the following are true:
            - Q is a single word, a fixed phraseme, or a short, idiomatic sentence
            - WORD_COUNT(Q) ≤ 5
          - `TEXT`, otherwise

        If `mode` = `LEXEME`, add Q in full to `cards` as a single entry,
        filling it according to the `Card` filling rules. Omit other root fields.

        Otherwise, if `mode` = `TEXT`, fill the root fields:
          - `translation` — to Russian if Q is in English, or to English if Q is in Russian
          - `rp` — British RP transcription without slashes, only if Q is in English and WORD_COUNT(Q) ≤ 5
          - `cards` — list all English lexemes at the level ${LEVEL} or higher in Q.
            Prefer longer lexemes such as phrasal verbs or phrasemes when available.
            Don't list proper names.
            Fill each card according to the `Card` filling rules.
            Treat different forms (e.g., verb and noun) as one lexeme

        `Card` filling rules (for the current card C):
          - Fill `C.lexeme` according to the `Lexeme` filling rules
          - Fill `C.translations` with an exhaustive list of translations,
            including those outside Q's context, following the `Lexeme` filling rules
          - Fill `C.context_sentence` with the full sentence where the lexeme occurs in Q,
            correcting grammar and spelling beforehand;
            ensure proper sentence capitalization and punctuation;
            don't correct contractions;
            leave the lexeme as is where possible;
            set to null if unavailable

        `Lexeme` filling rules (for the current lexeme L):
          - `text` — required
          - `language` — required
          - `rp` — list of possible British RP transcription of L, without slashes, only if L is English
          - `base_form` — only if Q is in English and L is a word not in its base form
          - `past_simple` and `past_participle` — only if Q is in English and L is an irregular word
    """).strip()
)


def format_word(prefix: str, lex: Lexeme) -> str:
    parts = []

    def append_prefixed(s: str) -> None:
        parts.append(f'{prefix}{s}')

    if lex.text and lex.rp:
        append_prefixed(f'{lex.text} {", ".join(f"/{rp}/" for rp in lex.rp)}')
    elif lex.text:
        append_prefixed(f'{lex.text}')
    if lex.base_form:
        append_prefixed(f'  BASE FORM: {lex.base_form}')
    if lex.past_simple:
        append_prefixed(f'  PAST SIMPLE: {lex.past_simple}')
    if lex.past_participle:
        append_prefixed(f'  PAST PARTICIPLE: {lex.past_participle}')
    return '\n'.join(parts)


def format_card(card: Card) -> str:
    parts = []
    parts.append(format_word('', card.lexeme))
    if card.translations:
        for t in card.translations:
            parts.append(format_word('  - ', t))
    if card.context_sentence is not None:
        parts.append(f'\n  > {card.context_sentence}')
    return '\n'.join(parts)


def print_response(cfg: argparse.Namespace, r: TranslatorResponse) -> None:
    parts = []
    if cfg.debug:
        parts.append(f'MODE: {r.mode}')
    if r.language == Language.OTHER:
        parts.append('UNSUPPORTED LANGUAGE')
    if r.typo_note:
        parts.append(f'TYPO NOTE: {r.typo_note}')
    if r.rp:
        parts.append(f'/{r.rp}/')
    if r.translation:
        parts.append(r.translation)
    for idx, card in enumerate(r.cards, 1):
        parts.append(f'[{idx}] {format_card(card)}')
    print('\n\n'.join(parts))


class ConfirmResult(Enum):
    YES = auto()
    NO = auto()
    QUIT = auto()


def confirm(prompt: str) -> ConfirmResult:
    while True:
        try:
            ans = input(f'{prompt} [y/N/q] ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            return ConfirmResult.NO
        if ans == 'y':
            return ConfirmResult.YES
        if ans in ('n', ''):
            return ConfirmResult.NO
        if ans == 'q':
            return ConfirmResult.QUIT
        print('Please enter y, n, or q')


def to_db_card(card) -> db_types.Card:
    return db_types.Card(
        lexeme=card.lexeme.text.strip(),
        rp=card.lexeme.rp,
        base_form=card.lexeme.base_form,
        past_simple=card.lexeme.past_simple,
        past_participle=card.lexeme.past_participle,
        translations=[t.text.strip() for t in card.translations],
        context_sentence=[card.context_sentence] if card.context_sentence is not None else [],
    )


def save_cards(cfg, cards: list[db_types.Card]) -> None:
    for idx, card in enumerate(cards, 1):
        print()
        print(f'[{idx}] {db_types.format_card(card)}')
        print()
        res = confirm('Save?')
        if res == ConfirmResult.QUIT:
            print('Skipping remaining cards')
            break
        if res == ConfirmResult.NO:
            continue
        try:
            with psycopg.connect(cfg.db_conn_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                            insert into cards (
                                lexeme,
                                rp,
                                base_form,
                                past_simple,
                                past_participle,
                                translations,
                                context_sentence
                            )
                            values (%s, %s, %s, %s, %s, %s, %s)
                            on conflict (lower(lexeme)) do update set
                                translations = (
                                    select coalesce(array_agg(distinct x), '{}')
                                    from unnest(cards.translations || excluded.translations) as x
                                ),
                                rp = (
                                    select coalesce(array_agg(distinct x), '{}')
                                    from unnest(cards.rp || excluded.rp) as x
                                ),
                                context_sentence = (
                                    select coalesce(array_agg(distinct x), '{}')
                                    from unnest(cards.context_sentence || excluded.context_sentence) as x
                                )
                            returning (xmax = 0) as inserted;
                        """,
                        (
                            card.lexeme,
                            card.rp,
                            card.base_form,
                            card.past_simple,
                            card.past_participle,
                            card.translations,
                            card.context_sentence,
                        ),
                    )
                    row = cur.fetchone()
                    assert row is not None
                    inserted = row[0]
                    print('Saved' if inserted else 'Merged')
                conn.commit()
        except psycopg.Error as e:
            print(f'Failed to save cards to postgres: {e}')
            sys.exit(2)


def read_interactively() -> str:
    kb = KeyBindings()

    @kb.add('c-d')
    def _(event):
        event.app.exit(result=event.app.current_buffer.text)

    def cont(width: int, line_number: int, is_soft_wrap: int) -> str:
        return ' ' * width if is_soft_wrap else ' ' * (width - 2) + '… '

    session: PromptSession = PromptSession()
    return session.prompt('Ctrl-D> ', multiline=True, prompt_continuation=cont, key_bindings=kb)


def main() -> None:
    parser = get_parser(sections=[Sections.DB, Sections.VERNA, Sections.OPENAI], require_db=False)
    parser.add_argument('query', nargs='*')
    cfg = parser.parse_args()

    if cfg.print_config:
        print_config(cfg)
        sys.exit(0)

    if cfg.show_schema:
        print(json.dumps(TranslatorResponse.model_json_schema(), indent=2))
        sys.exit(0)

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
    instructions = INSTRUCTIONS.substitute(
        {
            'WORD_COUNT': len(query.split()),
            'LEVEL': cfg.level,
        }
    )
    if cfg.debug:
        print(f'INSTRUCTIONS:\n{instructions}\n')
    resp = client.responses.parse(
        model='gpt-5',
        reasoning={'effort': cfg.reason},
        instructions=instructions,
        input=query,
        text_format=TranslatorResponse,
    )

    if resp.output_parsed is None:
        raise SystemExit('OpenAI response could not be parsed')

    data: TranslatorResponse = resp.output_parsed
    print_response(cfg, data)

    english_cards = [to_db_card(x) for x in data.cards if x.lexeme.language == Language.ENGLISH]
    if sys.stdin.isatty() and english_cards and cfg.db_conn_string:
        print('\nSAVING CARDS')
        save_cards(cfg, english_cards)


if __name__ == '__main__':
    main()
