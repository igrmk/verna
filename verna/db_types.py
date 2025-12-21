from dataclasses import dataclass

from prompt_toolkit.formatted_text import FormattedText


@dataclass
class Card:
    lexeme: str
    rp: list[str]
    past_simple: str | None
    past_participle: str | None
    translations: list[str]
    example: list[str]


def format_card(card: Card, idx: int) -> FormattedText:
    parts: list[tuple[str, str]] = []
    parts.append(('class:lexeme', f'[{idx}]'))
    parts.append(('', ' '))

    parts.append(('class:lexeme', card.lexeme))
    for rp in card.rp:
        parts.append(('', ' '))
        parts.append(('class:lexeme-italic', f'/{rp}/'))

    def add_kv(k: str, v: str | None) -> None:
        if v:
            parts.append(('', '\n  '))
            parts.append(('class:card-label', f'{k}:'))
            parts.append(('', ' '))
            parts.append(('', v))

    add_kv('PAST SIMPLE', card.past_simple)
    add_kv('PAST PARTICIPLE', card.past_participle)

    for x in card.translations:
        parts.append(('', '\n  - '))
        parts.append(('', x))

    if len(card.example) > 0:
        parts.append(('', '\n'))
    for s in card.example:
        parts.append(('', '\n  > '))
        parts.append(('class:example', s))

    return FormattedText(parts)


def format_card_plain(card: Card, idx: int) -> str:
    lines = []
    header = f'[{idx}] {card.lexeme}'
    if card.rp:
        header += ' /' + '/, /'.join(card.rp) + '/'
    lines.append(header)

    if card.past_simple:
        line = f'  PAST SIMPLE: {card.past_simple}'
        if card.past_participle:
            line += f'  PAST PARTICIPLE: {card.past_participle}'
        lines.append(line)

    for x in card.translations:
        lines.append(f'  - {x}')

    for s in card.example:
        lines.append(f'  > {s}')

    return '\n'.join(lines)
