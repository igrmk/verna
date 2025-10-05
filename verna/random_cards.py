import sys
import psycopg
import textwrap
import requests
from dataclasses import dataclass

from openai import OpenAI
from pydantic import BaseModel

from verna.config import get_parser, Sections, print_config


@dataclass
class Card:
    lexeme: str
    rp: str | None
    past_simple: str | None
    past_participle: str | None
    translations: list[str]


class Passage(BaseModel):
    english: str
    russian: str


def send_telegram_message(
    text: str,
    *,
    bot_token: str,
    chat_id: str,
) -> None:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
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


def fetch_random_cards(conn, limit: int) -> list[Card]:
    with conn.cursor() as cur:
        cur.execute(
            """
                select lexeme, rp, past_simple, past_participle, translations
                from cards
                order by random()
                limit %s;
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [
            Card(
                lexeme=row[0],
                rp=row[1],
                past_simple=row[2],
                past_participle=row[3],
                translations=row[4],
            )
            for row in rows
        ]


def format_card(idx: int, card: Card) -> str:
    lines: list[str] = []
    header = f'[{idx}] {card.lexeme} /{card.rp}/' if card.rp else f'[{idx}] {card.lexeme}'
    lines.append(header)
    if card.past_simple:
        lines.append(f'PAST SIMPLE: {card.past_simple}')
    if card.past_participle:
        lines.append(f'PAST PARTICIPLE: {card.past_participle}')
    for t in card.translations:
        lines.append(f'  - {t}')
    return "\n".join(lines)


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
    parser = get_parser(sections=[Sections.DB, Sections.TELEGRAM, Sections.RANDOM, Sections.OPENAI], require_db=True)
    cfg = parser.parse_args()

    if cfg.print_config:
        print_config(cfg)
        sys.exit(0)

    if cfg.send_to_tg and (not cfg.tg_bot_token or not cfg.tg_chat_id):
        raise SystemExit('--send-to-tg requires --tg-bot-token and --tg-chat-id')

    with psycopg.connect(cfg.db_conn_string) as conn:
        cards = fetch_random_cards(conn, cfg.limit)

    if not cards:
        print('No lexemes found in cards')
        return

    tg_card_messages: list[str] = []
    for idx, card in enumerate(cards, 1):
        print()
        card_text = format_card(idx, card)
        print(card_text)
        tg_card_messages.append(card_text)

    client = OpenAI(api_key=cfg.openai_api_key)

    lexeme_list = '\n'.join(f'[{idx}] {c.lexeme}' for idx, c in enumerate(cards, 1))
    request_text = textwrap.dedent(f"""
        Lexemes to use (any sensible form):
        {lexeme_list}
    """).strip()

    resp = client.responses.parse(
        model='gpt-5',
        reasoning={'effort': 'minimal'},
        instructions=INSTRUCTIONS,
        input=request_text,
        text_format=Passage,
    )

    if resp.output_parsed is None:
        raise SystemExit("OpenAI response could not be parsed")

    data: Passage = resp.output_parsed

    print()
    print()
    print(data.english.strip())
    print()
    print()
    print(data.russian.strip())

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
