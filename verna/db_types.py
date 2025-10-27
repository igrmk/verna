from dataclasses import dataclass


@dataclass
class Card:
    lexeme: str
    rp: list[str]
    base_form: str | None
    past_simple: str | None
    past_participle: str | None
    translations: list[str]
    context_sentence: list[str]


def format_card(card: Card) -> str:
    lines: list[str] = []
    header = f'{card.lexeme} {", ".join(f"/{rp}/" for rp in card.rp)}' if card.rp else f'{card.lexeme}'
    lines.append(header)
    if card.base_form:
        lines.append(f'  BASE FORM: {card.base_form}')
    if card.past_simple:
        lines.append(f'  PAST SIMPLE: {card.past_simple}')
    if card.past_participle:
        lines.append(f'  PAST PARTICIPLE: {card.past_participle}')
    for x in card.translations:
        lines.append(f'  - {x}')
    if len(card.context_sentence) > 0:
        lines.append('')
    for x in card.context_sentence:
        lines.append(f'  > {x}')
    return '\n'.join(lines)
