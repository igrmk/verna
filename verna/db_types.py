from dataclasses import dataclass
from rich.text import Text


@dataclass
class Card:
    lexeme: str
    rp: list[str]
    past_simple: str | None
    past_participle: str | None
    translations: list[str]
    example: list[str]


def format_card(card: Card, idx: int) -> Text:
    t = Text()
    t.append(f'[{idx}]', style='bold')
    t.append(' ')

    t.append(card.lexeme, style='bold')
    for rp in card.rp:
        t.append(' ')
        t.append(f'/{rp}/', style='bold italic')

    def add_kv(k: str, v: str | None) -> None:
        if v:
            t.append('\n  ')
            t.append(f'{k}:', style='dim')
            t.append(' ')
            t.append(v)

    add_kv('PAST SIMPLE', card.past_simple)
    add_kv('PAST PARTICIPLE', card.past_participle)

    for x in card.translations:
        t.append('\n  - ')
        t.append(x)

    if len(card.example) > 0:
        t.append('\n')
    for s in card.example:
        t.append('\n  > ')
        t.append(s, style='italic')

    return t
