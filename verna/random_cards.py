import sys
import psycopg
import textwrap
import requests
from verna import db, db_types, console

from openai import OpenAI
from pydantic import BaseModel

from verna.config import get_parser, Sections, print_config, ReasoningLevel


class LexemeExamples(BaseModel):
    lexeme: str
    examples: list[str]


class GeneratedExamples(BaseModel):
    lexemes: list[LexemeExamples]


def send_telegram_message(
    text: str,
    *,
    bot_token: str,
    chat_id: str,
) -> None:
    url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    data = {'chat_id': chat_id, 'text': text, 'disable_web_page_preview': True}
    resp = requests.post(url, data=data, timeout=15)
    resp.raise_for_status()


def send_to_telegram_all(
    messages: list[str],
    *,
    bot_token: str,
    chat_id: str,
) -> None:
    for msg in messages:
        if msg.strip():
            send_telegram_message(msg, bot_token=bot_token, chat_id=chat_id)


GENERAL_INSTRUCTIONS = 'Do not explain your actions. Do not ask questions. Output ONLY JSON matching the schema.'

INSTRUCTIONS_TEMPLATE = textwrap.dedent("""
    Generate example sentences. Rules:
    - "lexeme" field: copy the input lexeme EXACTLY, nothing else
    - "examples" field: array of exactly {generate_examples} sentences, each showing a different meaning
    - Keep each sentence under 12 words
""").strip()


def main() -> None:
    parser = get_parser(
        sections=[Sections.DB, Sections.TELEGRAM, Sections.RANDOM, Sections.AI],
        require_db=True,
    )
    cfg = parser.parse_args()

    if cfg.print_config:
        print_config(cfg)
        sys.exit(0)

    if cfg.send_to_tg and (not cfg.tg_bot_token or not cfg.tg_chat_id):
        raise SystemExit('--send-to-tg requires --tg-bot-token and --tg-chat-id')

    with psycopg.connect(cfg.db_conn_string) as conn:
        cards = db.fetch_random_cards(conn, cfg.limit)

    if not cards:
        print('No lexemes found in cards', file=sys.stderr)
        return

    client = OpenAI(base_url=cfg.api_base_url, api_key=cfg.api_key)

    lexeme_list = '\n'.join(card.lexeme for card in cards)
    request_text = f'Lexemes:\n{lexeme_list}'

    instructions = INSTRUCTIONS_TEMPLATE.format(generate_examples=cfg.generate_examples)

    input_messages = [
        {'role': 'system', 'content': instructions},
        {'role': 'user', 'content': request_text},
    ]

    kwargs: dict = {
        'model': cfg.model,
        'instructions': GENERAL_INSTRUCTIONS,
        'input': input_messages,
        'text_format': GeneratedExamples,
    }

    if cfg.reason != ReasoningLevel.UNSUPPORTED:
        kwargs['reasoning'] = {'effort': cfg.reason}

    resp = client.responses.parse(**kwargs)

    if cfg.debug:
        console.print_debug(f'Raw output: {resp.output_text}')

    if resp.output_parsed is None:
        raise SystemExit('AI response could not be parsed')

    data: GeneratedExamples = resp.output_parsed

    if cfg.debug:
        console.print_debug(f'Parsed {len(data.lexemes)} lexemes:')
        for item in data.lexemes:
            console.print_debug(f'  {item.lexeme!r}: {item.examples!r}')

    examples_by_lexeme = {item.lexeme.lower(): item.examples for item in data.lexemes}

    tg_messages: list[str] = []
    for idx, card in enumerate(cards, 1):
        examples = examples_by_lexeme.get(card.lexeme.lower(), [])
        console.print_styled()
        parts = [('class:lexeme', f'[{idx}] ')] + db_types.format_card(card, indent=2)
        for ex in examples:
            parts.append(('', '\n  » '))
            parts.append(('class:example-generated', ex))
        console.print_formatted(parts)

        tg_lines = [f'[{idx}] ' + db_types.format_card_plain(card)]
        for ex in examples:
            tg_lines.append(f'   » {ex}')
        tg_messages.append('\n'.join(tg_lines))

    if cfg.send_to_tg:
        try:
            send_to_telegram_all(tg_messages, bot_token=cfg.tg_bot_token, chat_id=str(cfg.tg_chat_id))
            print('\n[Sent to Telegram]')
        except Exception as e:
            print(f'\n[Failed to send to Telegram: {e}]')


if __name__ == '__main__':
    main()
