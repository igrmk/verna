from dataclasses import dataclass


@dataclass
class Card:
    lexeme: str
    rp: list[str]
    past_simple: str | None
    past_simple_rp: list[str]
    past_participle: str | None
    past_participle_rp: list[str]
    translations: list[str]
    example: list[str]


def format_card(card: Card, *, focused: bool = True, indent: int = 0) -> list[tuple[str, str]]:
    lexeme_style = 'class:lexeme' if focused else 'class:lexeme-dim'
    dim_style = 'class:card-label' if focused else 'class:dim'
    pad = ' ' * indent

    parts: list[tuple[str, str]] = []

    parts.append((lexeme_style, card.lexeme))
    for rp in card.rp:
        parts.append(('', ' /'))
        parts.append(('class:transcription', rp))
        parts.append(('', '/'))

    if card.past_simple or card.past_participle:
        parts.append(('', f'\n{pad}'))
        parts.append((dim_style, 'PAST:'))
        if card.past_simple:
            parts.append(('', f' {card.past_simple}'))
            for rp in card.past_simple_rp:
                parts.append(('', ' /'))
                parts.append(('class:transcription', rp))
                parts.append(('', '/'))
        if card.past_participle:
            parts.append(('', ' | ' if card.past_simple else ' '))
            parts.append(('', card.past_participle))
            for rp in card.past_participle_rp:
                parts.append(('', ' /'))
                parts.append(('class:transcription', rp))
                parts.append(('', '/'))

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

    if card.past_simple or card.past_participle:
        line = '   PAST:'
        if card.past_simple:
            line += f' {card.past_simple}'
            for rp in card.past_simple_rp:
                line += f' /{rp}/'
        if card.past_participle:
            line += ' | ' if card.past_simple else ' '
            line += card.past_participle
            for rp in card.past_participle_rp:
                line += f' /{rp}/'
        lines.append(line)

    for x in card.translations:
        lines.append(f'   • {x}')

    for s in card.example:
        lines.append(f'   > {s}')

    return '\n'.join(lines)
