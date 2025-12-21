from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText, AnyFormattedText
from prompt_toolkit.styles import Style

from verna import styles


_style = Style.from_dict(styles.PT_STYLES)


def print_styled(text: str = '', style: str = '') -> None:
    if not text:
        print()
        return
    print_formatted_text(FormattedText([(style, text)]), style=_style)


def print_formatted(formatted_text: AnyFormattedText) -> None:
    print_formatted_text(formatted_text, style=_style)


def print_log(text: str) -> None:
    print_styled(text, 'class:log')


def print_debug(text: str) -> None:
    print_styled(text, 'class:debug')


def print_debug_step(text: str) -> None:
    print_styled(text, 'class:debug-step')


def print_warning(text: str) -> None:
    print_styled(text, 'class:warning')
