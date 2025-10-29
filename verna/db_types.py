from dataclasses import dataclass
from rich.text import Text


@dataclass
class Card:
    lexeme: str
    rp: list[str]
    base_form: str | None
    past_simple: str | None
    past_participle: str | None
    translations: list[str]
    context_sentence: list[str]


def format_card(card: Card, idx: int) -> Text:
    t = Text()
    t.append(f'[{idx}]', style='bold')
    t.append(' ')

    t.append(card.lexeme)
    for rp in card.rp:
        t.append(' ')
        t.append(f'/{rp}/', style='italic')

    def add_kv(k: str, v: str | None) -> None:
        if v:
            t.append('\n  ')
            t.append(f'{k}:', style='dim')
            t.append(' ')
            t.append(v)

    add_kv('BASE FORM', card.base_form)
    add_kv('PAST SIMPLE', card.past_simple)
    add_kv('PAST PARTICIPLE', card.past_participle)

    for x in card.translations:
        t.append('\n  - ')
        t.append(x)

    if card.context_sentence:
        t.append('\n\n  > ')
        for s in card.context_sentence:
            t.append(s, style='italic')
            if s is not card.context_sentence[-1]:
                t.append('\n  > ')

    return t
