from dataclasses import dataclass, field


@dataclass
class Translation:
    text: str
    rp: list[str] = field(default_factory=list)


@dataclass
class Card:
    lexeme: str
    past_simple: str | None
    past_simple_rp: list[str]
    past_participle: str | None
    past_participle_rp: list[str]
    translations: list[Translation]
    example: list[str]


def format_card(card: Card, *, focused: bool = True, indent: int = 0) -> list[tuple[str, str]]:
    lexeme_style = 'class:lexeme' if focused else 'class:lexeme-dim'
    dim_style = 'class:card-label' if focused else 'class:dim'
    pad = ' ' * indent

    parts: list[tuple[str, str]] = []

    parts.append((lexeme_style, card.lexeme))

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

    for t in card.translations:
        parts.append(('', f'\n{pad}• '))
        for rp in t.rp:
            parts.append(('', '/'))
            parts.append(('class:transcription', rp))
            parts.append(('', '/ '))
        parts.append(('', t.text))

    for s in card.example:
        parts.append(('', f'\n{pad}> '))
        parts.append(('class:example', s))

    return parts


def format_card_plain(card: Card) -> str:
    lines = []
    lines.append(card.lexeme)

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

    for t in card.translations:
        rp_str = ' '.join(f'/{rp}/' for rp in t.rp)
        if rp_str:
            lines.append(f'   • {rp_str} {t.text}')
        else:
            lines.append(f'   • {t.text}')

    for s in card.example:
        lines.append(f'   > {s}')

    return '\n'.join(lines)
