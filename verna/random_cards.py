import sys
import psycopg
import textwrap
import requests
from rich.console import Console
from verna import db, db_types

from openai import OpenAI
from pydantic import BaseModel

from verna.config import get_parser, Sections, print_config, ReasoningLevel

CON = Console()


class Passage(BaseModel):
    english: str
    russian: str


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
    cards: list[str],
    english: str,
    russian: str,
    *,
    bot_token: str,
    chat_id: str,
) -> None:
    for msg in cards:
        if msg.strip():
            send_telegram_message(msg, bot_token=bot_token, chat_id=chat_id)
    if english.strip():
        send_telegram_message(english.strip(), bot_token=bot_token, chat_id=chat_id)
    if russian.strip():
        send_telegram_message(russian.strip(), bot_token=bot_token, chat_id=chat_id)


INSTRUCTIONS = textwrap.dedent("""
    You are a text generator. Output ONLY JSON that matches the given schema.
    Produce two fields:
      - `english`: a cohesive British English spy thriller passage (≤ 400 words).
        Use simple, natural language.
        Format each provided lexeme in the text as "[number] lexeme".
      - `russian`: a natural Russian translation of the entire English passage.
        Do not include numbers or English lexemes.
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

    tg_card_messages: list[str] = []
    for idx, card in enumerate(cards, 1):
        CON.print()
        card_text = db_types.format_card(card, idx)
        CON.print(card_text, markup=False)
        tg_card_messages.append(card_text.plain)

    client = OpenAI(base_url=cfg.api_base_url, api_key=cfg.api_key)

    lexeme_list = '\n'.join(f'[{idx}] {card.lexeme}' for idx, card in enumerate(cards, 1))
    request_text = textwrap.dedent(f"""
        Lexemes to use (any sensible form):
        {lexeme_list}
    """).strip()

    kwargs = {
        'model': cfg.model,
        'instructions': INSTRUCTIONS,
        'input': request_text,
        'text_format': Passage,
    }

    if cfg.reason != ReasoningLevel.UNSUPPORTED:
        kwargs['reasoning'] = {'effort': cfg.reason}

    resp = client.responses.parse(**kwargs)

    if resp.output_parsed is None:
        raise SystemExit('AI response could not be parsed')

    data: Passage = resp.output_parsed

    CON.print()
    CON.print()
    CON.print(data.english.strip(), markup=False, highlight=False)
    CON.print()
    CON.print()
    CON.print(data.russian.strip(), markup=False)

    if cfg.send_to_tg:
        try:
            send_to_telegram_all(
                tg_card_messages,
                data.english,
                data.russian,
                bot_token=cfg.tg_bot_token,
                chat_id=str(cfg.tg_chat_id),
            )
            print('\n[Sent to Telegram]')
        except Exception as e:
            print(f'\n[Failed to send to Telegram: {e}]')


if __name__ == '__main__':
    main()
