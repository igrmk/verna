import time
import json
import sys
import textwrap
import argparse
from concurrent.futures import ThreadPoolExecutor
from verna import db_types
from enum import Enum, auto
import psycopg
from openai import APITimeoutError, OpenAI
from openai.types.responses import ResponseInputParam
from openai.types.shared_params import Reasoning
from pydantic import BaseModel, Field
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

from verna.upper_str_enum import UpperStrEnum
from verna.config import get_parser, Sections, print_config, ReasoningLevel, CefrLevel

from rich.console import Console

from jinja2 import Environment, StrictUndefined

CON = Console()

JINJA_ENV = Environment(
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,
)


class Language(UpperStrEnum):
    ENGLISH = auto()
    RUSSIAN = auto()
    OTHER = auto()


class Lexeme(BaseModel):
    text: str
    rp: list[str] = Field(default_factory=list)
    past_simple: str | None = None
    past_participle: str | None = None


class Card(BaseModel):
    lexeme: Lexeme
    translations: list[str]
    example: str | None


class LanguageDetectionResponse(BaseModel):
    language: Language


class TranslationResponse(BaseModel):
    translation: str
    rp: str | None = None
    typo_note: str | None = None


class LexemeExtractionResponse(BaseModel):
    class Item(BaseModel):
        lexeme: str
        example: str | None = None
        cefr: CefrLevel

    items: list[Item] = Field(default_factory=list)


class LexemeTranslationResponse(BaseModel):
    lexeme: Lexeme
    translations: list[str] = Field(default_factory=list)


class ExampleResponse(BaseModel):
    example: str | None


TIMEOUT = 10


def _responses_parse(
    cfg: argparse.Namespace,
    client: OpenAI,
    *,
    step: str,
    instructions: str,
    user_input: str,
    text_format: type[BaseModel],
    model: str | None = None,
):
    model_id = model or cfg.model
    if cfg.debug:
        CON.print(step, style='underline dim')
        CON.print()
        CON.print('USER INPUT:', style='dim')
        CON.print(user_input, style='dim')
        CON.print()
        CON.print('SCHEMA:', style='dim')
        schema = text_format.model_json_schema()
        CON.print(json.dumps(schema, indent=2), style='dim')
        CON.print()
        CON.print(f'INSTRUCTIONS:\n{instructions}\n', style='dim')

    input_messages: ResponseInputParam = [
        {'role': 'system', 'content': instructions},
        {'role': 'user', 'content': user_input},
    ]
    reasoning: Reasoning | None = {'effort': cfg.reason} if cfg.reason != ReasoningLevel.UNSUPPORTED else None

    while True:
        start_time = time.perf_counter()
        try:
            resp = client.responses.parse(
                model=model_id,
                instructions=GENERAL_INSTRUCTIONS,
                input=input_messages,
                text_format=text_format,
                reasoning=reasoning,
                timeout=TIMEOUT,
            )
            break
        except APITimeoutError:
            CON.print(f'[{step}] Request timed out after {TIMEOUT}s', style='yellow')
            try:
                ans = input('Retry? [Y/n] ').strip().lower()
            except (EOFError, KeyboardInterrupt):
                raise SystemExit('Request timed out')
            if ans in ('', 'y'):
                continue
            raise SystemExit('Request timed out')

    elapsed = time.perf_counter() - start_time
    CON.print(f'[{step}] {model_id} responded in {elapsed:.1f}s', style='dim grey50')

    if resp.output_parsed is None:
        raise SystemExit(f'AI response could not be parsed ({step})')
    return resp.output_parsed


def detect_language(cfg: argparse.Namespace, client: OpenAI, query: str) -> LanguageDetectionResponse:
    instructions = LANGUAGE_DETECTION_INSTRUCTIONS
    return _responses_parse(
        cfg,
        client,
        step='LANGUAGE DETECTION',
        instructions=instructions,
        user_input=query,
        text_format=LanguageDetectionResponse,
        model=cfg.model_detect or cfg.model,
    )


def translate_text(
    cfg: argparse.Namespace,
    client: OpenAI,
    *,
    query: str,
    source_language: Language,
) -> TranslationResponse:
    word_count = len(query.split())
    target_language = 'Russian' if source_language == Language.ENGLISH else 'English'
    instructions = TRANSLATION_INSTRUCTIONS.render(
        target_language=target_language,
        word_count=word_count,
    )
    return _responses_parse(
        cfg,
        client,
        step='TRANSLATION',
        instructions=instructions,
        user_input=query,
        text_format=TranslationResponse,
        model=cfg.model_translate_text or cfg.model,
    )


def extract_lexemes(cfg: argparse.Namespace, client: OpenAI, *, query: str) -> LexemeExtractionResponse:
    instructions = LEXEME_EXTRACTION_INSTRUCTIONS.render()
    return _responses_parse(
        cfg,
        client,
        step='LEXEME EXTRACTION',
        instructions=instructions,
        user_input=query,
        text_format=LexemeExtractionResponse,
        model=cfg.model_extract or cfg.model,
    )


def translate_lexeme(
    cfg: argparse.Namespace,
    client: OpenAI,
    *,
    lexeme_text: str,
    example: str | None = None,
) -> LexemeTranslationResponse:
    instructions = LEXEME_TRANSLATION_INSTRUCTIONS.render(example=example)
    user_input = f'{lexeme_text}\nExample: {example}' if example else lexeme_text
    return _responses_parse(
        cfg,
        client,
        step='LEXEME TRANSLATION',
        instructions=instructions,
        user_input=user_input,
        text_format=LexemeTranslationResponse,
        model=cfg.model_translate_lexeme or cfg.model,
    )


GENERAL_INSTRUCTIONS = 'Do not explain your actions. Do not ask questions. Output ONLY JSON matching the schema'

LANGUAGE_DETECTION_INSTRUCTIONS = 'You are a language detector. Set `language` to the language of the user input.'

TRANSLATION_INSTRUCTIONS = JINJA_ENV.from_string(
    textwrap.dedent("""
        You are a translator.
        Informal language and swear words are allowed when necessary.
        Do not explain your actions. Output ONLY JSON matching the schema.
        Between UK and US variants, choose UK.

        Translate the user input into {{ target_language }} and fill these fields:
          - `translation` — translation of the user input into {{ target_language }}
          - `typo_note` — if you suspect a genuine spelling typo in the user input, add a short note;
            otherwise set to null.
            Not a typo:
              - informal usage
              - colloquial expressions
              - contractions such as "wanna" or "gonna"
              - UK-only spelling variants
          {% if target_language == "Russian" and word_count <= 10 %}
          - `rp` — British RP transcription of the user input without slashes
          {% else %}
          - `rp` — set to null
          {% endif %}
    """).strip()
)

LEXEME_EXTRACTION_INSTRUCTIONS = JINJA_ENV.from_string(
    textwrap.dedent("""
        You are a dictionary and lexeme extractor.
        Informal language and swear words are allowed when necessary.
        Prefer UK spelling over US spelling

        Extract all English lexemes that appear in the user input.
        Search for multi-word lexemes (e.g. phrasal verbs and phrasemes) in addition to single-word lexemes.
        Exclude proper names.
        Treat different forms (e.g., verb and noun) as one lexeme.

        Output `items`: a list of objects. For each extracted lexeme, create one item:
          - `item.lexeme` — the lexeme in its base form;
             use the plural if it is the standard form for the meaning in the sentence
             (e.g., scissors, or spoils as in "the spoils of victory")
          - `item.example` — the full sentence from the user input where the lexeme occurs,
             do not include it if it is not a sentence
          - `item.cefr` — estimate the lexeme's CEFR level.

        Before extracting the example sentence, correct its grammar and spelling first;
        ensure proper sentence capitalisation and punctuation;
        don't correct contractions;
        don't correct local variants;
        keep the lexeme unchanged where possible.
        If unavailable, set `item.example` to null.
    """).strip()
)

LEXEME_TRANSLATION_INSTRUCTIONS = JINJA_ENV.from_string(
    textwrap.dedent("""
        You are a translator and dictionary.
        Informal language and swear words are allowed when necessary.
        Do not explain your actions. Output ONLY JSON matching the schema.
        Between UK and US variants, choose UK.

        You will be given an English lexeme (L).
        {% if example %}
        You will also be given an example sentence showing how the lexeme is used.
        Ensure to include a translation matching the lexeme's meaning from this example.
        {% endif %}
        First, normalise it to its base form (use plural if it is the standard form for the meaning in the sentence)
        and fill the `lexeme` object according to the `Lexeme` filling rules.
        Then translate it to Russian.
        Provide an exhaustive list of translations.
        Prefer masculine adjectives over other genders.

        `Lexeme` filling rules (for the current lexeme L):
          - `text` — required; the lexeme in its base form
          - `rp` — list of possible British RP transcription of L, without slashes
          - `past_simple` and `past_participle` — only if L is an irregular verb

        `translations` — exhaustive list of Russian translations
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


def print_identified_language(lang_data: LanguageDetectionResponse) -> None:
    CON.print(f'Language detected: {lang_data.language}')
    CON.print()


def print_typo_note(typo_note: str | None) -> None:
    if not typo_note:
        return
    CON.print()
    CON.print('TYPO NOTE', style='bold')
    CON.print(typo_note)


def print_translation(translation_data: TranslationResponse) -> None:
    if translation_data.rp:
        CON.print(f'/{translation_data.rp}/', style='italic')
        CON.print()
    CON.print(translation_data.translation, markup=False)
    CON.print()


def print_extracted_lexemes(items: list[LexemeExtractionResponse.Item]) -> None:
    if not items:
        CON.print('No lexemes for memorisation found')
        return
    CON.print('LEXEMES', style='bold underline')
    CON.print()
    for idx, item in enumerate(items, 1):
        CON.print(f'[{idx}] {item.lexeme} ({item.cefr})', style='bold')
        if item.example:
            CON.print(f'  > {item.example}', style='italic')
        CON.print()


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
        past_simple=card.lexeme.past_simple,
        past_participle=card.lexeme.past_participle,
        translations=[t.strip() for t in card.translations],
        example=[card.example] if card.example is not None else [],
    )


def make_example(cfg: argparse.Namespace, card: db_types.Card, previous_examples: list[str]) -> None:
    client = OpenAI(base_url=cfg.api_base_url, api_key=cfg.api_key)
    instructions = EXAMPLE_INSTRUCTIONS.render(previous_examples=previous_examples)
    data: ExampleResponse = _responses_parse(
        cfg,
        client,
        step=f'EXAMPLE: {card.lexeme}',
        instructions=instructions,
        user_input=card.lexeme,
        text_format=ExampleResponse,
        model=cfg.model,
    )
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
                CON.print()
            conn.commit()
    except psycopg.Error as e:
        print(f'Failed to save cards to postgres: {e}', file=sys.stderr)
        sys.exit(2)


def prompt_card_selection(items: list[LexemeExtractionResponse.Item]) -> list[int]:
    while True:
        try:
            ans = input('Card number (a for all, q to quit): ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            return []
        if ans == 'q':
            return []
        if ans == 'a':
            return list(range(len(items)))
        try:
            num = int(ans)
            if 1 <= num <= len(items):
                return [num - 1]
            CON.print(f'Please enter a number between 1 and {len(items)}')
        except ValueError:
            CON.print('Please enter a valid number, a for all, or q to quit')


def save_single_lexeme(cfg: argparse.Namespace, client: OpenAI, item: LexemeExtractionResponse.Item, idx: int) -> bool:
    """Save a single lexeme. Returns False if user wants to quit."""
    CON.print()
    lexeme_text = item.lexeme.strip()
    tr = translate_lexeme(cfg, client, lexeme_text=lexeme_text, example=item.example)
    card = Card(
        lexeme=tr.lexeme,
        translations=tr.translations,
        example=item.example,
    )
    db_card = to_db_card(card)

    previous_examples: list[str] = []
    proceed = True
    while proceed:
        proceed = False
        CON.print(db_types.format_card(db_card, idx + 1), markup=False)
        CON.print()
        res = confirm('Save?')
        CON.print()
        if res == ConfirmResult.QUIT:
            return False
        if res == ConfirmResult.YES:
            save_card(cfg, db_card)
        if res == ConfirmResult.EXAMPLE:
            proceed = True
            if len(db_card.example) > 0:
                previous_examples += db_card.example
            make_example(cfg, db_card, previous_examples)
    return True


def save_extracted_lexemes(cfg: argparse.Namespace, client: OpenAI, items: list[LexemeExtractionResponse.Item]) -> None:
    if len(items) == 1:
        save_single_lexeme(cfg, client, items[0], 0)
        return
    while True:
        indices = prompt_card_selection(items)
        if not indices:
            return
        for idx in indices:
            if not save_single_lexeme(cfg, client, items[idx], idx):
                return


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
    parser = get_parser(sections=[Sections.DB, Sections.VERNA, Sections.AI], require_db=False)
    parser.add_argument('query', nargs='*')
    cfg = parser.parse_args()

    if cfg.print_config:
        print_config(cfg)
        return 0

    query = ' '.join(cfg.query).strip()
    no_query_error = SystemExit('You must provide a query')
    if not query:
        if not sys.stdin.isatty():
            query = sys.stdin.read().strip()
            if not query:
                raise no_query_error
    if not query:
        query = read_interactively().strip()
    if not query:
        raise no_query_error

    client = OpenAI(base_url=cfg.api_base_url, api_key=cfg.api_key)
    lang_data = detect_language(cfg, client, query)
    print_identified_language(lang_data)

    if lang_data.language == Language.OTHER:
        CON.print('UNSUPPORTED LANGUAGE')
        return 0

    if lang_data.language == Language.ENGLISH and len(query.split()) == 1:
        item = LexemeExtractionResponse.Item(lexeme=query, example=None, cefr=CefrLevel.C2)
        if sys.stdin.isatty() and cfg.db_conn_string:
            save_single_lexeme(cfg, client, item, 0)
        else:
            tr = translate_lexeme(cfg, client, lexeme_text=query)
            card = Card(lexeme=tr.lexeme, translations=tr.translations, example=None)
            db_card = to_db_card(card)
            CON.print(db_types.format_card(db_card, 1), markup=False)
        return 0

    if lang_data.language == Language.ENGLISH:
        with ThreadPoolExecutor(max_workers=2) as executor:
            translation_future = executor.submit(
                translate_text, cfg, client, query=query, source_language=lang_data.language
            )
            lexeme_future = executor.submit(extract_lexemes, cfg, client, query=query)
            translation_data = translation_future.result()
            lexeme_data = lexeme_future.result()
        lexeme_items = [item for item in lexeme_data.items if item.cefr >= cfg.level]
    else:
        translation_data = translate_text(cfg, client, query=query, source_language=lang_data.language)
        lexeme_items = None

    print_translation(translation_data)
    print_typo_note(translation_data.typo_note)

    if lexeme_items is not None:
        print_extracted_lexemes(lexeme_items)

    if sys.stdin.isatty() and cfg.db_conn_string and lexeme_items:
        CON.print('SAVING CARDS', style='bold underline')
        CON.print()
        save_extracted_lexemes(cfg, client, lexeme_items)
    return 0


def main() -> int:
    try:
        return work()
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    sys.exit(main())
