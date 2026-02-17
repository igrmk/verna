import openai
import asyncio
import time
import json
import sys
import textwrap
import argparse
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any
from verna import db, db_types
from enum import Enum, auto
import psycopg
from openai import APIError, APITimeoutError, AsyncOpenAI
from openai.types.responses import ResponseInputParam
from openai.types.shared_params import Reasoning
import pydantic
from pydantic import BaseModel, Field
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application, get_app
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout import Layout, HSplit, Window, FormattedTextControl, ScrollablePane
from prompt_toolkit.styles import Style

from lingua import LanguageDetectorBuilder, Language as LinguaLanguage
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
    text_format: type[BaseModel] | openai.Omit,
    parse_as: type[BaseModel] | None = None,
    model: str | None = None,
):
    model_id = model or cfg.model
    if cfg.debug:
        console.print_debug_step(step)
        console.print_styled()
        console.print_debug('USER INPUT:')
        console.print_debug(user_input)
        console.print_styled()
        if isinstance(text_format, type) and issubclass(text_format, BaseModel):
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

    if cfg.debug:
        console.print_debug(f'Raw response: {resp.output_text}')
        console.print_styled()

    if parse_as is not None:
        try:
            return parse_as.model_validate_json(resp.output_text)
        except pydantic.ValidationError:
            raise SystemExit(f'AI response could not be parsed ({step})')
    if resp.output_parsed is None:
        raise SystemExit(f'AI response could not be parsed ({step})')
    return resp.output_parsed


_LINGUA_DETECTOR = LanguageDetectorBuilder.from_languages(LinguaLanguage.ENGLISH, LinguaLanguage.RUSSIAN).build()

_LINGUA_TO_LANGUAGE = {
    LinguaLanguage.ENGLISH: Language.ENGLISH,
    LinguaLanguage.RUSSIAN: Language.RUSSIAN,
}


def detect_language(query: str) -> Language:
    result = _LINGUA_DETECTOR.detect_language_of(query)
    if result is None:
        return Language.OTHER
    return _LINGUA_TO_LANGUAGE[result]


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
    Search for multi-word lexemes (e.g. phrasal verbs, idioms, and collocations) in addition to single-word lexemes.
    When a word is part of a phrasal verb or idiom, extract BOTH the multi-word lexeme AND the single-word lexeme separately.
    For example, "pick up" should yield both "pick up" and "pick".
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


def print_identified_language(language: Language) -> None:
    console.print_log(f'Language detected: {language}')
    console.print_styled()


def print_typo_note(typo_note: str | None) -> None:
    if not typo_note:
        return
    console.print_formatted([('class:note-header', 'Typo Note: '), ('', typo_note)])
    console.print_styled()


def print_translation(translation_data: TranslationResponse) -> None:
    if translation_data.rp:
        console.print_styled(f'/{translation_data.rp}/', 'class:transcription')
        console.print_styled()
    console.print_styled(translation_data.translation)
    console.print_styled()


def print_save_cards_header() -> None:
    console.print_formatted([('class:section-header', 'Save Cards'), ('', '\n')])


@dataclass
class SaveResult:
    stop: bool = False
    inserted: bool | None = None
    card: db_types.Card | None = None


class ConfirmResult(Enum):
    YES = auto()
    NO = auto()
    EXAMPLE = auto()
    QUIT = auto()


class ConfirmSelector:
    OPTIONS = [
        (ConfirmResult.YES, 'Save', 'y'),
        (ConfirmResult.EXAMPLE, 'Example', 'e'),
        (ConfirmResult.NO, 'Skip', 'Esc'),
        (ConfirmResult.QUIT, 'Quit', 'q'),
    ]

    def __init__(self):
        self.selected_idx = 0
        self.result = ConfirmResult.NO

    def _get_formatted_text(self) -> list[tuple[str, str]]:
        parts: list[tuple[str, str]] = []
        for idx, (_, label, key) in enumerate(self.OPTIONS):
            if idx > 0:
                parts.append(('', '  '))
            if idx == self.selected_idx:
                parts.append(('class:selected', f' {label} '))
            else:
                parts.append(('class:dim', f' {label} '))
            parts.append(('class:dim', f'({key})'))
        return parts

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add('left')
        @kb.add('h')
        def _left(event):
            if self.selected_idx > 0:
                self.selected_idx -= 1

        @kb.add('right')
        @kb.add('l')
        def _right(event):
            if self.selected_idx < len(self.OPTIONS) - 1:
                self.selected_idx += 1

        @kb.add('enter')
        def _select(event):
            self.result = self.OPTIONS[self.selected_idx][0]
            event.app.exit()

        @kb.add('y')
        def _yes(event):
            self.result = ConfirmResult.YES
            event.app.exit()

        @kb.add('e')
        def _example(event):
            self.result = ConfirmResult.EXAMPLE
            event.app.exit()

        @kb.add('escape')
        @kb.add('n')
        def _no(event):
            self.result = ConfirmResult.NO
            event.app.exit()

        @kb.add('q')
        def _quit(event):
            self.result = ConfirmResult.QUIT
            event.app.exit()

        return kb

    async def run(self) -> ConfirmResult:
        control = FormattedTextControl(self._get_formatted_text, show_cursor=False)
        window = Window(control, height=1)
        layout = Layout(window)
        style = Style.from_dict(styles.PT_STYLES)
        app: Application = Application(
            layout=layout,
            key_bindings=self._create_key_bindings(),
            style=style,
            full_screen=False,
            erase_when_done=True,
        )
        await app.run_async()
        return self.result


async def confirm() -> ConfirmResult:
    selector = ConfirmSelector()
    return await selector.run()


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


def save_card(cfg: argparse.Namespace, card: db_types.Card) -> bool:
    """Save card to database. Returns True if inserted, False if merged."""
    try:
        with psycopg.connect(cfg.db_conn_string) as conn:
            return db.save_card(conn, card)
    except psycopg.Error as e:
        print(f'Failed to save cards to postgres: {e}', file=sys.stderr)
        sys.exit(2)


class SelectionResult(Enum):
    SELECTED = auto()
    ALL = auto()
    QUIT = auto()


def _save_prefix(inserted: bool) -> str:
    return ' ✓ ' if inserted else ' ⊕ '


class LexemeSelector[T]:
    _SPINNER_FRAMES = ['⣾', '⣷', '⣯', '⣟', '⡿', '⢿', '⣻', '⣽']

    def __init__(
        self,
        items: list[LexemeExtractionResponse.Item],
        selected_idx: int = 0,
        saved: dict[int, bool] | None = None,
        on_select: Callable[[int], Coroutine[Any, Any, T]] | None = None,
    ):
        self.items = items
        self.selected_idx = min(selected_idx, len(items) - 1) if items else 0
        self.result: SelectionResult = SelectionResult.QUIT
        self._moved_down = False
        self.saved = saved or {}
        self._on_select = on_select
        self._loading_idx: int | None = None
        self._spinner_idx = 0
        self._select_result: T | None = None

    def _get_formatted_text(self) -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = []
        for idx, item in enumerate(self.items):
            is_selected = idx == self.selected_idx
            is_saved = idx in self.saved
            is_loading = idx == self._loading_idx
            if is_loading:
                prefix = f' {self._SPINNER_FRAMES[self._spinner_idx]} '
                prefix_style = 'class:lexeme'
                lexeme_style = 'class:lexeme'
            elif is_saved and is_selected:
                prefix = _save_prefix(self.saved[idx])
                prefix_style = 'class:selected class:success'
                lexeme_style = 'class:success'
            elif is_saved:
                prefix = _save_prefix(self.saved[idx])
                prefix_style = 'class:success'
                lexeme_style = 'class:success'
            elif is_selected:
                prefix = ' ▶ '
                prefix_style = 'class:selected class:lexeme'
                lexeme_style = 'class:lexeme'
            else:
                prefix = '   '
                prefix_style = 'class:lexeme-dim'
                lexeme_style = 'class:lexeme-dim'
            lines.append((prefix_style, prefix))
            lines.append((lexeme_style, f'[{idx + 1}] {item.lexeme} ({item.cefr})'))
            if item.example:
                lines.append(('class:example', f'\n     > {item.example}'))
            lines.append(('', '\n\n'))
        if self._loading_idx is None:
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

    async def _run_spinner(self, app: Application) -> None:
        while self._loading_idx is not None:
            await asyncio.sleep(0.08)
            self._spinner_idx = (self._spinner_idx + 1) % len(self._SPINNER_FRAMES)
            app.invalidate()

    async def _run_select(self, app: Application, idx: int) -> None:
        assert self._on_select is not None
        self._loading_idx = idx
        app.invalidate()
        spinner_task = asyncio.create_task(self._run_spinner(app))
        try:
            self._select_result = await self._on_select(idx)
        finally:
            self._loading_idx = None
            spinner_task.cancel()
            try:
                await spinner_task
            except asyncio.CancelledError:
                pass
        self.result = SelectionResult.SELECTED
        app.exit()

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add('up')
        @kb.add('k')
        def _up(event):
            if self._loading_idx is not None:
                return
            if self.selected_idx > 0:
                self.selected_idx -= 1
                self._moved_down = False

        @kb.add('down')
        @kb.add('j')
        def _down(event):
            if self._loading_idx is not None:
                return
            if self.selected_idx < len(self.items) - 1:
                self.selected_idx += 1
                self._moved_down = True

        @kb.add('enter')
        def _select(event):
            if self._loading_idx is not None:
                return
            if self._on_select:
                asyncio.create_task(self._run_select(event.app, self.selected_idx))
            else:
                self.result = SelectionResult.SELECTED
                event.app.exit()

        @kb.add('a')
        def _all(event):
            if self._loading_idx is not None:
                return
            self.result = SelectionResult.ALL
            event.app.exit()

        @kb.add('escape')
        def _quit(event):
            if self._loading_idx is not None:
                return
            self.result = SelectionResult.QUIT
            event.app.exit()

        for i in range(1, 10):

            @kb.add(str(i), eager=True)
            def _number(event, idx=i - 1):
                if self._loading_idx is not None:
                    return
                if idx < len(self.items):
                    self.selected_idx = idx
                    if self._on_select:
                        asyncio.create_task(self._run_select(event.app, idx))
                    else:
                        self.result = SelectionResult.SELECTED
                        event.app.exit()

        return kb

    async def run(self) -> tuple[SelectionResult, int, T | None]:
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
            erase_when_done=True,
        )
        await app.run_async()
        return self.result, self.selected_idx, self._select_result


def _count_lines(parts: list[tuple[str, str]]) -> int:
    """Count lines in formatted text parts."""
    text = ''.join(p[1] for p in parts)
    return text.count('\n') + 1


async def show_status_while(message: str, coro):
    """Show a status message while coroutine runs, then erase it."""
    result = None

    async def do_work():
        nonlocal result
        result = await coro
        get_app().exit()

    control = FormattedTextControl(lambda: [('class:dim', message)])
    window = Window(control, height=1)
    layout = Layout(window)
    style = Style.from_dict(styles.PT_STYLES)
    app: Application = Application(layout=layout, style=style, full_screen=False, erase_when_done=True)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(do_work())
        tg.create_task(app.run_async())

    return result


async def translate_single_lexeme(
    cfg: argparse.Namespace, client: AsyncOpenAI, item: LexemeExtractionResponse.Item
) -> LexemeTranslationResponse:
    """Translate a single lexeme."""
    lexeme_text = item.lexeme.strip()
    return await translate_lexeme(cfg, client, lexeme_text=lexeme_text, example=item.example)


async def confirm_and_save_lexeme(
    cfg: argparse.Namespace,
    client: AsyncOpenAI,
    item: LexemeExtractionResponse.Item,
    tr: LexemeTranslationResponse,
    idx: int,
) -> SaveResult:
    """Show confirm dialog and save lexeme."""
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
        res = await confirm()
        # Count lines: card lines + 1 empty + 1 debug log (prompt erases itself)
        lines_to_clear = _count_lines(parts) + 2
        if res == ConfirmResult.QUIT:
            console.clear_lines_above(lines_to_clear)
            return SaveResult(stop=True, card=db_card)
        if res == ConfirmResult.YES:
            inserted = save_card(cfg, db_card)
            console.clear_lines_above(lines_to_clear)
            return SaveResult(inserted=inserted, card=db_card)
        if res == ConfirmResult.NO:
            console.clear_lines_above(lines_to_clear)
            return SaveResult(card=db_card)
        if res == ConfirmResult.EXAMPLE:
            console.clear_lines_above(lines_to_clear)
            proceed = True
            if len(db_card.example) > 0:
                previous_examples += db_card.example
            example_coro = make_example(cfg, client, db_card, previous_examples)
            await show_status_while(f'Generating example for "{db_card.lexeme}"...', example_coro)
    return SaveResult(card=db_card)


async def save_single_lexeme(
    cfg: argparse.Namespace, client: AsyncOpenAI, item: LexemeExtractionResponse.Item, idx: int
) -> SaveResult:
    """Translate and save a single lexeme with status message."""
    lexeme_text = item.lexeme.strip()
    coro = translate_lexeme(cfg, client, lexeme_text=lexeme_text, example=item.example)
    tr = await show_status_while(f'Translating "{lexeme_text}"...', coro)
    return await confirm_and_save_lexeme(cfg, client, item, tr, idx)


async def save_extracted_lexemes(
    cfg: argparse.Namespace, client: AsyncOpenAI, items: list[LexemeExtractionResponse.Item]
) -> None:
    print_save_cards_header()
    saved: dict[int, bool] = {}
    cards: dict[int, db_types.Card] = {}

    if len(items) == 1:
        save_result = await save_single_lexeme(cfg, client, items[0], 0)
        assert save_result.card is not None
        cards[0] = save_result.card
        if save_result.inserted is not None:
            saved[0] = save_result.inserted
    else:

        async def on_select(idx: int) -> LexemeTranslationResponse:
            return await translate_single_lexeme(cfg, client, items[idx])

        selected_idx = 0
        quit_requested = False
        while not quit_requested:
            selector = LexemeSelector(items, selected_idx, saved, on_select=on_select)
            result, idx, tr = await selector.run()
            selected_idx = idx
            if result == SelectionResult.QUIT:
                break
            if result == SelectionResult.ALL:
                for i in range(len(items)):
                    if i not in saved:
                        save_result = await save_single_lexeme(cfg, client, items[i], i)
                        assert save_result.card is not None
                        cards[i] = save_result.card
                        if save_result.stop:
                            quit_requested = True
                            break
                        if save_result.inserted is not None:
                            saved[i] = save_result.inserted
            elif result == SelectionResult.SELECTED:
                assert tr is not None
                save_result = await confirm_and_save_lexeme(cfg, client, items[idx], tr, idx)
                assert save_result.card is not None
                cards[idx] = save_result.card
                if save_result.stop:
                    break
                if save_result.inserted is not None:
                    saved[idx] = save_result.inserted

    if cards:
        for i in range(len(items)):
            if i in cards:
                prefix_style = 'class:success' if i in saved else ''
                prefix = _save_prefix(saved[i]) if i in saved else '   '
                parts = [(prefix_style, prefix)] + db_types.format_card(cards[i], indent=3)
                console.print_formatted(parts)
                console.print_styled()
    else:
        console.print_styled('Nothing translated', 'class:dim')


def normalize_input(text: str) -> str:
    """Convert various apostrophes to ASCII ' and quotes to ASCII "."""
    apostrophes = '\u2018\u2019\u201a\u201b\u02bc\u02bb`'
    quotes = '\u201c\u201d\u201e\u201f\u00ab\u00bb'
    for ch in apostrophes:
        text = text.replace(ch, "'")
    for ch in quotes:
        text = text.replace(ch, '"')
    return text


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
    query = normalize_input(query)

    language = detect_language(query)
    print_identified_language(language)

    client = AsyncOpenAI(base_url=cfg.api_base_url, api_key=cfg.api_key)

    if language == Language.OTHER:
        console.print_styled('UNSUPPORTED LANGUAGE')
        return 0

    if language == Language.ENGLISH and len(query.split()) == 1:
        item = LexemeExtractionResponse.Item(lexeme=query, example=None, cefr=CefrLevel.C2)
        if sys.stdin.isatty() and cfg.db_conn_string:
            save_result = await save_single_lexeme(cfg, client, item, 0)
            assert save_result.card is not None
            prefix_style = 'class:success' if save_result.inserted is not None else ''
            prefix = _save_prefix(save_result.inserted) if save_result.inserted is not None else '   '
            parts = [(prefix_style, prefix)] + db_types.format_card(save_result.card, indent=3)
            console.print_formatted(parts)
            console.print_styled()
        else:
            tr = await translate_lexeme(cfg, client, lexeme_text=query)
            ai_card = Card(lexeme=tr.lexeme, translations=tr.translations, example=None)
            db_card = to_db_card(ai_card)
            parts = db_types.format_card(db_card, indent=3)
            console.print_formatted([('', '   ')] + parts)
            console.print_styled()
        return 0

    translation_task = asyncio.create_task(translate_text(cfg, client, query=query, source_language=language))
    if language == Language.ENGLISH:
        lexeme_task = asyncio.create_task(extract_lexemes(cfg, client, query=query))
    translation_data = await translation_task
    print_translation(translation_data)
    print_typo_note(translation_data.typo_note)
    if language == Language.ENGLISH:
        lexeme_data = await lexeme_task
        lexeme_items = [item for item in lexeme_data.items if item.cefr >= cfg.level]
        if lexeme_items:
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
    except APIError as e:
        print(f'API error: {e.message}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
