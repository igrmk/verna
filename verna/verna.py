import json
import sys
import textwrap
from enum import Enum, auto

import psycopg
from openai import OpenAI
from pydantic import BaseModel, Field

from verna.upper_str_enum import UpperStrEnum
from verna.config import get_parser, Sections, print_config

client = OpenAI()


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

    Let mode be:
      - "lexeme", if Q is a single word, a fixed phraseme, or a short everyday sentence (≤5 words)
      - "text", otherwise

    If mode = "lexeme", add a new entry for Q's text in `dict_entries`,
    filling it according to the `DictEntry` filling rules.

    Otherwise, if mode = "text", fill the root fields:
      - `translation` — to Russian if Q is in English, or to English if Q is in Russian
      - `rp` — British RP transcription without slashes, only if Q is in English and ≤5 words
      - `dict_entries` — include all and only English C1+ lexemes in Q,
        filling each entry according to the `DictEntry` filling rules.
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


def print_word(prefix: str, lex: Lexeme) -> None:
    def prefix_print(s: str) -> None:
        print(f'{prefix}{s}')
    if lex.text and lex.rp:
        prefix_print(f'* {lex.text} /{lex.rp}/')
    elif lex.text:
        prefix_print(f'* {lex.text}')
    if lex.base_form:
        prefix_print(f'BASE FORM: {lex.base_form}')
    if lex.past_simple:
        prefix_print(f'PAST SIMPLE: {lex.past_simple}')
    if lex.past_participle:
        prefix_print(f'PAST PARTICIPLE: {lex.past_participle}')


def print_dict_entry(e: DictEntry) -> None:
    print()
    print_word('', e.lexeme)
    if e.translations:
        print()
        print('TRANSLATIONS:')
        for t in e.translations:
            print_word('  ', t)


def print_response(r: TranslatorResponse) -> None:
    if r.language == Language.OTHER:
        print('UNSUPPORTED LANGUAGE')
    if r.typo_note:
        print(f'TYPO NOTE: {r.typo_note}')
    if r.translation:
        print()
        print('TRANSLATION:')
        print(r.translation)
    if r.rp:
        print(f'/{r.rp}/')
    for e in r.dict_entries:
        print_dict_entry(e)


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


def main() -> None:
    parser = get_parser(sections=[Sections.DB, Sections.VERNA], require_db=False)
    parser.add_argument('query', nargs='*')
    cfg = parser.parse_args()

    if cfg.print_config:
        print_config(cfg)
        sys.exit(0)

    if cfg.show_schema:
        print(json.dumps(TranslatorResponse.model_json_schema(), indent=2))
        sys.exit(0)

    query = ' '.join(cfg.query).strip()
    if not query:
        raise SystemExit('You must provide a query or use --show-schema')

    resp = client.responses.parse(
        model='gpt-5',
        reasoning={'effort': 'minimal'},
        instructions=INSTRUCTIONS,
        input=query,
        text_format=TranslatorResponse,
    )

    if resp.output_parsed is None:
        raise SystemExit("OpenAI response could not be parsed")

    data: TranslatorResponse = resp.output_parsed
    print_response(data)

    english_entries = [e for e in data.dict_entries if e.lexeme.language == Language.ENGLISH]
    if english_entries and cfg.db_conn_string:
        try:
            with psycopg.connect(cfg.db_conn_string) as conn:
                save_cards(conn, english_entries)
        except psycopg.Error as e:
            print(f'Failed to save cards to postgres: {e}')
            sys.exit(2)


if __name__ == '__main__':
    main()
