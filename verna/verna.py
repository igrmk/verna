import json
import sys
import textwrap
import argparse
from enum import Enum, auto

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
    rp: str | None = None
    base_form: str | None = None
    past_simple: str | None = None
    past_participle: str | None = None


class DictEntry(BaseModel):
    lexeme: Lexeme
    translations: list[Lexeme]


class TranslatorResponse(BaseModel):
    mode: Mode | None = None
    language: Language
    typo_note: str | None = None
    rp: str | None = None
    translation: str | None = None
    dict_entries: list[DictEntry] = Field(default_factory=list)


INSTRUCTIONS = textwrap.dedent("""
    You are a translator and dictionary. Swear words are allowed when necessary.
    Informal language is allowed as well.
    Do not explain your actions. Output ONLY JSON matching the schema.

    Let Q be the user input.
    Set `language` to Q's language.
    If `language` is `OTHER`, output ONLY {"language": "OTHER"}.

    If you suspect a genuine spelling typo
    (excluding informal usage, colloquial usage, or colloquial contractions such as "wanna" or "gonna"),
    add a short note to `typo_note`.

    Fill `mode` to:
      - `LEXEME`, if both of the following are true:
        - Q is a single word, a fixed phraseme, or a short everyday sentence
        - Q is ≤5 words
      - `TEXT`, otherwise

    If `mode` = `LEXEME`, add Q in full to `dict_entries` as a single entry,
    filling it according to the `DictEntry` filling rules.

    Otherwise, if `mode` = `TEXT`, fill the root fields:
      - `translation` — to Russian if Q is in English, or to English if Q is in Russian
      - `rp` — British RP transcription without slashes, only if Q is in English and ≤5 words
      - `dict_entries` — list all advanced English lexemes (C1+) in Q.
        Don't list beginner-level lexemes and proper names.
        Fill each entry according to the `DictEntry` filling rules.
        Treat different forms (e.g., verb and noun) as one lexeme

    `DictEntry` filling rules (for the current entry E):
      - Fill `E.lexeme` according to the `Lexeme` filling rules
      - Fill `E.translations` with an exhaustive list of translations,
        including those outside Q's context, following the `Lexeme` filling rules

    `Lexeme` filling rules (for the current lexeme L):
      - `text` — required
      - `language` — required
      - `rp` — British RP transcription without slashes, only if L is in English
      - `base_form` — only if Q is in English and L is not in its base form
      - `past_simple` and `past_participle` — only if Q is in English and L is irregular
""").strip()


def format_word(prefix: str, lex: Lexeme) -> str:
    parts = []

    def append_prefixed(s: str) -> None:
        parts.append(f'{prefix}{s}')

    if lex.text and lex.rp:
        append_prefixed(f'* {lex.text} /{lex.rp}/')
    elif lex.text:
        append_prefixed(f'* {lex.text}')
    if lex.base_form:
        append_prefixed(f'BASE FORM: {lex.base_form}')
    if lex.past_simple:
        append_prefixed(f'PAST SIMPLE: {lex.past_simple}')
    if lex.past_participle:
        append_prefixed(f'PAST PARTICIPLE: {lex.past_participle}')
    return '\n'.join(parts)


def format_dict_entry(e: DictEntry) -> str:
    parts = []
    parts.append(format_word('', e.lexeme))
    if e.translations:
        parts.append('TRANSLATIONS:')
        for t in e.translations:
            parts.append(format_word('  ', t))
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
        parts.append(f'TRANSLATION:\n{r.translation}')
    for e in r.dict_entries:
        parts.append(format_dict_entry(e))
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


def save_cards(conn, entries: list[DictEntry]) -> None:
    print()
    for e in entries:
        res = confirm(f'Save "{e.lexeme.text}"?')
        if res == ConfirmResult.QUIT:
            print('Skipping remaining cards')
            break
        if res == ConfirmResult.NO:
            continue
        translations = [t.text.strip() for t in e.translations]
        with conn.cursor() as cur:
            cur.execute(
                """
                    insert into cards (lexeme, rp, base_form, past_simple, past_participle, translations)
                    values (%s, %s, %s, %s, %s, %s)
                    on conflict (lower(lexeme)) do update
                    set translations = cards.translations || (
                        select array(
                            select x
                            from unnest(excluded.translations) as x
                            where not x = any(cards.translations)
                        )
                    )
                    returning (xmax = 0) as inserted;
                """,
                (
                    e.lexeme.text.strip(),
                    e.lexeme.rp,
                    e.lexeme.base_form,
                    e.lexeme.past_simple,
                    e.lexeme.past_participle,
                    translations,
                ),
            )
            inserted = cur.fetchone()[0]
            if not inserted:
                print('Merged translations')
        conn.commit()


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

    resp = client.responses.parse(
        model='gpt-5',
        reasoning={'effort': 'minimal'},
        instructions=INSTRUCTIONS,
        input=query,
        text_format=TranslatorResponse,
    )

    if resp.output_parsed is None:
        raise SystemExit('OpenAI response could not be parsed')

    data: TranslatorResponse = resp.output_parsed
    print_response(cfg, data)

    english_entries = [e for e in data.dict_entries if e.lexeme.language == Language.ENGLISH]
    if sys.stdin.isatty() and english_entries and cfg.db_conn_string:
        try:
            with psycopg.connect(cfg.db_conn_string) as conn:
                save_cards(conn, english_entries)
        except psycopg.Error as e:
            print(f'Failed to save cards to postgres: {e}')
            sys.exit(2)


if __name__ == '__main__':
    main()
