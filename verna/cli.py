import asyncio
import sys
from collections.abc import Iterable
from pathlib import Path

from platformdirs import user_data_dir
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

from verna.config import get_parser, Sections, print_config


def normalize_input(text: str) -> str:
    """Convert various apostrophes to ASCII ' and quotes to ASCII "."""
    apostrophes = '\u2018\u2019\u201a\u201b\u02bc\u02bb`'
    quotes = '\u201c\u201d\u201e\u201f\u00ab\u00bb'
    for ch in apostrophes:
        text = text.replace(ch, "'")
    for ch in quotes:
        text = text.replace(ch, '"')
    return text


class _CappedFileHistory(FileHistory):
    def __init__(self, filename: str, max_entries: int) -> None:
        self.max_entries = max_entries
        super().__init__(filename)

    def load_history_strings(self) -> Iterable[str]:
        strings = list(super().load_history_strings())
        if len(strings) > self.max_entries:
            strings = strings[: self.max_entries]
            self._rewrite(strings)
        return strings

    def store_string(self, string: str) -> None:
        super().store_string(string)
        if len(self._loaded_strings) > self.max_entries:
            self._loaded_strings = self._loaded_strings[: self.max_entries]
            self._rewrite(self._loaded_strings)

    def _rewrite(self, strings: list[str]) -> None:
        with open(self.filename, 'wb') as f:
            for entry in reversed(strings):
                for line in entry.split('\n'):
                    f.write(f'+{line}\n'.encode())
                f.write(b'\n')


async def read_interactively(history: _CappedFileHistory) -> str:
    kb = KeyBindings()

    @kb.add('c-d')
    def _(event):
        buf = event.app.current_buffer
        if buf.text:
            buf.append_to_history()
        event.app.exit(result=buf.text)

    def cont(width: int, line_number: int, is_soft_wrap: int) -> str:
        return ' ' * width if is_soft_wrap else ' ' * (width - 2) + '… '

    session: PromptSession = PromptSession(history=history)
    return await session.prompt_async('Ctrl-D> ', multiline=True, prompt_continuation=cont, key_bindings=kb)


async def work() -> int:
    parser = get_parser(sections=[Sections.DB, Sections.VERNA, Sections.AI], require_db=False)
    parser.add_argument('query', nargs='*')
    cfg = parser.parse_args()

    if cfg.print_config:
        print_config(cfg)
        return 0

    history_path = Path(user_data_dir('verna')) / 'history'
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history = _CappedFileHistory(str(history_path), cfg.max_history)

    query = ' '.join(cfg.query).strip()
    no_query_error = SystemExit('You must provide a query')
    if not query:
        if not sys.stdin.isatty():
            query = sys.stdin.read().strip()
            if not query:
                raise no_query_error
    if query:
        history.store_string(query)
    else:
        query = (await read_interactively(history)).strip()
    if not query:
        raise no_query_error
    query = normalize_input(query)

    # Imported lazily: openai, psycopg, pydantic and jinja2 are slow to load
    # and would delay the interactive prompt
    from verna import verna

    return await verna.run(cfg, query)


def main() -> int:
    try:
        return asyncio.run(work())
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    sys.exit(main())
