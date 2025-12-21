from dataclasses import dataclass


@dataclass
class Card:
    lexeme: str
    rp: list[str]
    past_simple: str | None
    past_participle: str | None
    translations: list[str]
    example: list[str]


def format_card(card: Card, *, focused: bool = True, indent: int = 0) -> list[tuple[str, str]]:
    lexeme_style = 'class:lexeme' if focused else 'class:lexeme-dim'
    dim_style = 'class:card-label' if focused else 'class:dim'
    pad = ' ' * indent

    parts: list[tuple[str, str]] = []

    parts.append((lexeme_style, card.lexeme))
    for rp in card.rp:
        parts.append(('', ' '))
        parts.append((lexeme_style, f'/{rp}/'))

    if card.past_simple:
        parts.append(('', f'\n{pad}'))
        parts.append((dim_style, 'PAST:'))
        parts.append(('', f' {card.past_simple}'))
        if card.past_participle:
            parts.append(('', f' / {card.past_participle}'))

    for x in card.translations:
        parts.append(('', f'\n{pad}• '))
        parts.append(('', x))

    for s in card.example:
        parts.append(('', f'\n{pad}> '))
        parts.append(('class:example', s))

    return parts


def format_card_plain(card: Card) -> str:
    lines = []
    header = card.lexeme
    if card.rp:
        header += ' /' + '/, /'.join(card.rp) + '/'
    lines.append(header)

    if card.past_simple:
        line = f'   PAST: {card.past_simple}'
        if card.past_participle:
            line += f' / {card.past_participle}'
        lines.append(line)

    for x in card.translations:
        lines.append(f'   • {x}')

    for s in card.example:
        lines.append(f'   > {s}')

    return '\n'.join(lines)
