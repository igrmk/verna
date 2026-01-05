import asyncio
import time
import json
import sys
import textwrap
import argparse
from verna import db, db_types
from enum import Enum, auto
import psycopg
from openai import APITimeoutError, AsyncOpenAI
from openai.types.responses import ResponseInputParam
from openai.types.shared_params import Reasoning
from pydantic import BaseModel, Field
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout import Layout, HSplit, Window, FormattedTextControl, ScrollablePane
from prompt_toolkit.styles import Style

from verna.upper_str_enum import UpperStrEnum
from verna.config import get_parser, Sections, print_config, ReasoningLevel, CefrLevel
from verna import console, styles

from jinja2 import Environment, StrictUndefined

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


class Form(BaseModel):
    text: str
    rp: list[str] = Field(default_factory=list)


class Lexeme(BaseModel):
    base: Form
    past_simple: Form | None = None
    past_participle: Form | None = None


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


async def _responses_parse(
    cfg: argparse.Namespace,
    client: AsyncOpenAI,
    *,
    step: str,
    instructions: str,
    user_input: str,
    text_format: type[BaseModel],
    model: str | None = None,
):
    model_id = model or cfg.model
    if cfg.debug:
        console.print_debug_step(step)
        console.print_styled()
        console.print_debug('USER INPUT:')
        console.print_debug(user_input)
        console.print_styled()
        console.print_debug('SCHEMA:')
        schema = text_format.model_json_schema()
        console.print_debug(json.dumps(schema, indent=2))
        console.print_styled()
        console.print_debug(f'INSTRUCTIONS:\n{instructions}\n')

    input_messages: ResponseInputParam = [
        {'role': 'system', 'content': instructions},
        {'role': 'user', 'content': user_input},
    ]
    reasoning: Reasoning | None = {'effort': cfg.reason} if cfg.reason != ReasoningLevel.UNSUPPORTED else None

    while True:
        start_time = time.perf_counter()
        try:
            resp = await client.responses.parse(
                model=model_id,
                instructions=GENERAL_INSTRUCTIONS,
                input=input_messages,
                text_format=text_format,
                reasoning=reasoning,
                timeout=TIMEOUT,
            )
            break
        except APITimeoutError:
            console.print_warning(f'[{step}] Request timed out after {TIMEOUT}s')
            try:
                ans = input('Retry? [Y/n] ').strip().lower()
            except (EOFError, KeyboardInterrupt):
                raise SystemExit('Request timed out')
            if ans in ('', 'y'):
                continue
            raise SystemExit('Request timed out')

    elapsed = time.perf_counter() - start_time
    console.print_log(f'[{step}] {model_id} responded in {elapsed:.1f}s')

    if resp.output_parsed is None:
        raise SystemExit(f'AI response could not be parsed ({step})')
    return resp.output_parsed


async def detect_language(cfg: argparse.Namespace, client: AsyncOpenAI, query: str) -> LanguageDetectionResponse:
    return await _responses_parse(
        cfg,
        client,
        step='LANGUAGE DETECTION',
        instructions=LANGUAGE_DETECTION_INSTRUCTIONS,
        user_input=query,
        text_format=LanguageDetectionResponse,
        model=cfg.model_detect or cfg.model,
    )


async def translate_text(
    cfg: argparse.Namespace,
    client: AsyncOpenAI,
    *,
    query: str,
    source_language: Language,
) -> TranslationResponse:
    instructions = TRANSLATION_INSTRUCTIONS.render(
        target_language='Russian' if source_language == Language.ENGLISH else 'English',
        word_count=len(query.split()),
    )
    return await _responses_parse(
        cfg,
        client,
        step='TRANSLATION',
        instructions=instructions,
        user_input=query,
        text_format=TranslationResponse,
        model=cfg.model_translate_text or cfg.model,
    )


async def extract_lexemes(cfg: argparse.Namespace, client: AsyncOpenAI, *, query: str) -> LexemeExtractionResponse:
    return await _responses_parse(
        cfg,
        client,
        step='LEXEME EXTRACTION',
        instructions=LEXEME_EXTRACTION_INSTRUCTIONS,
        user_input=query,
        text_format=LexemeExtractionResponse,
        model=cfg.model_extract or cfg.model,
    )


async def translate_lexeme(
    cfg: argparse.Namespace,
    client: AsyncOpenAI,
    *,
    lexeme_text: str,
    example: str | None = None,
) -> LexemeTranslationResponse:
    instructions = LEXEME_TRANSLATION_INSTRUCTIONS.render(example=example)
    return await _responses_parse(
        cfg,
        client,
        step='LEXEME TRANSLATION',
        instructions=instructions,
        user_input=f'{lexeme_text}\nExample: {example}' if example else lexeme_text,
        text_format=LexemeTranslationResponse,
        model=cfg.model_translate_lexeme or cfg.model,
    )


GENERAL_INSTRUCTIONS = 'Do not explain your actions. Do not ask questions. Output ONLY JSON matching the schema'

LANGUAGE_DETECTION_INSTRUCTIONS = 'You are a language detector. Set `language` to the language of the user input.'

TRANSLATION_INSTRUCTIONS = JINJA_ENV.from_string(
    textwrap.dedent("""
        You are a translator.
        Informal language and swear words are allowed when necessary.
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

LEXEME_EXTRACTION_INSTRUCTIONS = textwrap.dedent("""
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

LEXEME_TRANSLATION_INSTRUCTIONS = JINJA_ENV.from_string(
    textwrap.dedent("""
        You are a translator and dictionary.
        Informal language and swear words are allowed when necessary.
        Between UK and US variants, choose UK.

        You will be given an English lexeme (L).
        {% if example %}
        You will also be given an example sentence showing how the lexeme is used.
        {% endif %}
        First, normalise it to its base form (use plural if it is the standard form for the meaning in the sentence)
        and fill the `lexeme` object:
          - `base` — required; Form with `text` (lexeme in base form) and `rp` (transcriptions without slashes)
          - `past_simple` and `past_participle` — only if L is an irregular verb; same Form structure

        Then translate it to Russian and fill
        `translations` — exhaustive list of Russian translations covering all parts of speech
        (nouns, verbs, adjectives, adverbs, etc.) but avoiding very close synonyms.
        Each item must be a single translation, not multiple translations separated by semicolons.
        Do not include part-of-speech labels like (verb), (noun), etc.
        {% if example %}
        Ensure to include a translation matching the lexeme's meaning from this example (but not only).
        {% endif %}
        Prefer masculine adjectives over other genders.
    """).strip()
)

EXAMPLE_INSTRUCTIONS = JINJA_ENV.from_string(
    textwrap.dedent(r"""
        You are a translator and dictionary.
        Swear words are allowed when necessary.
        Informal language is allowed as well.
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
    console.print_styled(f'Language detected: {lang_data.language}')
    console.print_styled()


def print_typo_note(typo_note: str | None) -> None:
    if not typo_note:
        return
    console.print_styled()
    console.print_styled('TYPO NOTE', 'class:note-header')
    console.print_styled(typo_note)


def print_translation(translation_data: TranslationResponse) -> None:
    if translation_data.rp:
        console.print_styled(f'/{translation_data.rp}/', 'class:transcription')
        console.print_styled()
    console.print_styled(translation_data.translation)
    console.print_styled()


def print_save_cards_header() -> None:
    header = 'SAVE CARDS'
    console.print_styled(header, 'class:section-header')
    console.print_styled('─' * len(header), 'class:section-header')
    console.print_styled()


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
        console.print_styled('Please enter y, n, e, or q')


def to_db_card(card) -> db_types.Card:
    return db_types.Card(
        lexeme=card.lexeme.base.text.strip(),
        rp=card.lexeme.base.rp,
        past_simple=card.lexeme.past_simple.text if card.lexeme.past_simple else None,
        past_simple_rp=card.lexeme.past_simple.rp if card.lexeme.past_simple else [],
        past_participle=card.lexeme.past_participle.text if card.lexeme.past_participle else None,
        past_participle_rp=card.lexeme.past_participle.rp if card.lexeme.past_participle else [],
        translations=[t.strip() for t in card.translations],
        example=[card.example] if card.example is not None else [],
    )


async def make_example(
    cfg: argparse.Namespace, client: AsyncOpenAI, card: db_types.Card, previous_examples: list[str]
) -> None:
    instructions = EXAMPLE_INSTRUCTIONS.render(previous_examples=previous_examples)
    data: ExampleResponse = await _responses_parse(
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
            inserted = db.save_card(conn, card)
            console.print_styled('Saved' if inserted else 'Merged')
            console.print_styled()
    except psycopg.Error as e:
        print(f'Failed to save cards to postgres: {e}', file=sys.stderr)
        sys.exit(2)


class SelectionResult(Enum):
    SELECTED = auto()
    ALL = auto()
    QUIT = auto()


class LexemeSelector:
    def __init__(self, items: list[LexemeExtractionResponse.Item]):
        self.items = items
        self.selected_idx = 0
        self.result: SelectionResult = SelectionResult.QUIT
        self._moved_down = False

    def _get_formatted_text(self) -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = []
        for idx, item in enumerate(self.items):
            is_selected = idx == self.selected_idx
            prefix = ' ▶ ' if is_selected else '   '
            style = 'class:selected class:lexeme' if is_selected else 'class:lexeme-dim'
            lines.append((style, f'{prefix}[{idx + 1}] {item.lexeme} ({item.cefr})'))
            if item.example:
                lines.append(('class:example', f'\n     > {item.example}'))
            lines.append(('', '\n\n'))
        lines.append(('class:dim', '↑/↓/j/k: navigate | Enter/1-9: select | a: all | Esc: quit'))
        return lines

    def _count_item_lines(self, item: LexemeExtractionResponse.Item) -> int:
        lines = 1  # lexeme line
        if item.example:
            lines += 1  # example line
        lines += 1  # separator
        return lines

    def _get_selected_line(self) -> int:
        line = 0
        for idx, item in enumerate(self.items):
            item_lines = self._count_item_lines(item)
            if idx == self.selected_idx:
                if self._moved_down:
                    return line + item_lines - 1  # show end of item when moving down
                return line  # show start of item when moving up
            line += item_lines
        return 0

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add('up')
        @kb.add('k')
        def _up(event):
            if self.selected_idx > 0:
                self.selected_idx -= 1
                self._moved_down = False

        @kb.add('down')
        @kb.add('j')
        def _down(event):
            if self.selected_idx < len(self.items) - 1:
                self.selected_idx += 1
                self._moved_down = True

        @kb.add('enter')
        def _select(event):
            self.result = SelectionResult.SELECTED
            event.app.exit()

        @kb.add('a')
        def _all(event):
            self.result = SelectionResult.ALL
            event.app.exit()

        @kb.add('escape')
        def _quit(event):
            self.result = SelectionResult.QUIT
            event.app.exit()

        for i in range(1, 10):

            @kb.add(str(i), eager=True)
            def _number(event, idx=i - 1):
                if idx < len(self.items):
                    self.selected_idx = idx
                    self.result = SelectionResult.SELECTED
                    event.app.exit()

        return kb

    async def run(self) -> tuple[SelectionResult, int]:
        control = FormattedTextControl(
            self._get_formatted_text,
            show_cursor=False,
            get_cursor_position=lambda: Point(0, self._get_selected_line()),
        )
        window = Window(control, wrap_lines=True, height=30)
        layout = Layout(HSplit([ScrollablePane(window)]))
        style = Style.from_dict(styles.PT_STYLES)
        app: Application = Application(
            layout=layout,
            key_bindings=self._create_key_bindings(),
            style=style,
            full_screen=False,
        )
        await app.run_async()
        return self.result, self.selected_idx


async def save_single_lexeme(
    cfg: argparse.Namespace, client: AsyncOpenAI, item: LexemeExtractionResponse.Item, idx: int
) -> bool:
    """Save a single lexeme. Returns False if user wants to quit."""
    console.print_styled()
    lexeme_text = item.lexeme.strip()
    tr = await translate_lexeme(cfg, client, lexeme_text=lexeme_text, example=item.example)
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
        parts = [('class:lexeme', f'[{idx + 1}] ')] + db_types.format_card(db_card, indent=2)
        console.print_formatted(parts)
        console.print_styled()
        res = confirm('Save?')
        console.print_styled()
        if res == ConfirmResult.QUIT:
            return False
        if res == ConfirmResult.YES:
            save_card(cfg, db_card)
        if res == ConfirmResult.EXAMPLE:
            proceed = True
            if len(db_card.example) > 0:
                previous_examples += db_card.example
            await make_example(cfg, client, db_card, previous_examples)
    return True


async def save_extracted_lexemes(
    cfg: argparse.Namespace, client: AsyncOpenAI, items: list[LexemeExtractionResponse.Item]
) -> None:
    if len(items) == 1:
        await save_single_lexeme(cfg, client, items[0], 0)
        return
    while True:
        selector = LexemeSelector(items)
        result, idx = await selector.run()
        if result == SelectionResult.QUIT:
            return
        if result == SelectionResult.ALL:
            for i in range(len(items)):
                if not await save_single_lexeme(cfg, client, items[i], i):
                    return
        elif result == SelectionResult.SELECTED:
            if not await save_single_lexeme(cfg, client, items[idx], idx):
                return


async def read_interactively() -> str:
    kb = KeyBindings()

    @kb.add('c-d')
    def _(event):
        event.app.exit(result=event.app.current_buffer.text)

    def cont(width: int, line_number: int, is_soft_wrap: int) -> str:
        return ' ' * width if is_soft_wrap else ' ' * (width - 2) + '… '

    session: PromptSession = PromptSession()
    return await session.prompt_async('Ctrl-D> ', multiline=True, prompt_continuation=cont, key_bindings=kb)


async def work() -> int:
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
        query = (await read_interactively()).strip()
    if not query:
        raise no_query_error

    client = AsyncOpenAI(base_url=cfg.api_base_url, api_key=cfg.api_key)
    lang_data = await detect_language(cfg, client, query)
    print_identified_language(lang_data)

    if lang_data.language == Language.OTHER:
        console.print_styled('UNSUPPORTED LANGUAGE')
        return 0

    if lang_data.language == Language.ENGLISH and len(query.split()) == 1:
        item = LexemeExtractionResponse.Item(lexeme=query, example=None, cefr=CefrLevel.C2)
        if sys.stdin.isatty() and cfg.db_conn_string:
            await save_single_lexeme(cfg, client, item, 0)
        else:
            tr = await translate_lexeme(cfg, client, lexeme_text=query)
            card = Card(lexeme=tr.lexeme, translations=tr.translations, example=None)
            db_card = to_db_card(card)
            parts = [('class:lexeme', '[1] ')] + db_types.format_card(db_card, indent=2)
            console.print_formatted(parts)
        return 0

    translation_task = asyncio.create_task(translate_text(cfg, client, query=query, source_language=lang_data.language))
    if lang_data.language == Language.ENGLISH:
        lexeme_task = asyncio.create_task(extract_lexemes(cfg, client, query=query))
    translation_data = await translation_task
    print_translation(translation_data)
    print_typo_note(translation_data.typo_note)
    if lang_data.language == Language.ENGLISH:
        lexeme_data = await lexeme_task
        lexeme_items = [item for item in lexeme_data.items if item.cefr >= cfg.level]
        if lexeme_items:
            print_save_cards_header()
            if sys.stdin.isatty() and cfg.db_conn_string:
                await save_extracted_lexemes(cfg, client, lexeme_items)
            else:
                for idx, item in enumerate(lexeme_items, 1):
                    console.print_styled(f'[{idx}] {item.lexeme} ({item.cefr})', 'class:lexeme')
                    if item.example:
                        console.print_styled(f'  > {item.example}', 'class:example')
                    console.print_styled()
        else:
            console.print_styled('No lexemes for memorisation found')
    return 0


def main() -> int:
    try:
        return asyncio.run(work())
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    sys.exit(main())
