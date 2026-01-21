import sys

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style

from verna import styles


_style = Style.from_dict(styles.PT_STYLES)


def clear_lines_above(n: int) -> None:
    """Clear n lines above the cursor and move cursor up."""
    for _ in range(n):
        sys.stdout.write('\033[A\033[2K')  # move up, clear line
    sys.stdout.flush()


def print_styled(text: str = '', style: str = '') -> None:
    if not text:
        print()
        return
    print_formatted_text(FormattedText([(style, text)]), style=_style)


def print_formatted(parts: list[tuple[str, str]]) -> None:
    print_formatted_text(FormattedText(parts), style=_style)


def print_log(text: str) -> None:
    print_styled(text, 'class:log')


def print_debug(text: str) -> None:
    print_styled(text, 'class:debug')


def print_debug_step(text: str) -> None:
    print_styled(text, 'class:debug-step')


def print_warning(text: str) -> None:
    print_styled(text, 'class:warning')
