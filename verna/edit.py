import sys
import psycopg
from prompt_toolkit import Application
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window, FormattedTextControl, ScrollablePane
from prompt_toolkit.layout.containers import Container
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import Frame, TextArea
from prompt_toolkit.styles import Style

from verna.config import get_parser, Sections, print_config
from verna.db_types import Card


class CardEditor:
    def __init__(self, conn_string: str):
        self.conn_string = conn_string
        self.cards: list[tuple[int, Card]] = []
        self.selected_idx = 0
        self.editing_idx: int | None = None
        self.form_field_idx = 0
        self.message = ''
        self.app: Application | None = None

        self.search_area = TextArea(
            height=1,
            prompt='Search: ',
            multiline=False,
            wrap_lines=False,
        )

        self.results_control = FormattedTextControl(text=self._get_results_text, focusable=True, show_cursor=False)
        self.results_window = Window(content=self.results_control, wrap_lines=True)

        # Invisible control for form navigation (doesn't show cursor)
        self.form_nav_control = FormattedTextControl(text='', focusable=True, show_cursor=False)
        self.form_nav_window = Window(content=self.form_nav_control, height=0)

        # Create form fields with read_only filter (editable only when the field itself is focused)
        field_readonly = Condition(lambda: not self._in_any_form_field())
        self.field_lexeme = TextArea(height=1, multiline=False, wrap_lines=False, read_only=field_readonly)
        self.field_rp = TextArea(height=1, multiline=False, wrap_lines=False, read_only=field_readonly)
        self.field_past_simple = TextArea(height=1, multiline=False, wrap_lines=False, read_only=field_readonly)
        self.field_past_participle = TextArea(height=1, multiline=False, wrap_lines=False, read_only=field_readonly)
        self.field_translations = TextArea(height=4, multiline=True, wrap_lines=True, read_only=field_readonly)
        self.field_example = TextArea(height=4, multiline=True, wrap_lines=True, read_only=field_readonly)

        self.form_fields = [
            ('Lexeme', self.field_lexeme),
            ('RP', self.field_rp),
            ('Past simple', self.field_past_simple),
            ('Past participle', self.field_past_participle),
            ('Translations', self.field_translations),
            ('Example', self.field_example),
        ]

        self.message_control = FormattedTextControl(text=lambda: self.message)

        self.style = Style.from_dict(
            {
                'frame.border': 'fg:ansiblue',
                'selected': 'reverse',
                'lexeme': 'bold fg:ansigreen',
                'dim': 'fg:ansibrightblack',
                'label': 'fg:ansicyan',
                'label-selected': 'fg:ansicyan bold reverse',
                'field-editing': 'bg:#252525',
            }
        )

    def _in_form_nav(self) -> bool:
        if self.app is None:
            return False
        return self.app.layout.current_control == self.form_nav_control

    def _in_any_form_field(self) -> bool:
        if self.app is None:
            return False
        focused = self.app.layout.current_control
        for _, field in self.form_fields:
            if field.control == focused:
                return True
        return False

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        in_form_nav = Condition(self._in_form_nav)
        in_form_editing = Condition(self._in_any_form_field)
        in_results = has_focus(self.results_window)
        in_search = has_focus(self.search_area)

        @kb.add('c-c')
        @kb.add('c-q')
        def _exit(event):
            event.app.exit()

        # Search
        @kb.add('enter', filter=in_search)
        def _do_search(event):
            self._on_search()
            event.app.layout.focus(self.results_window)

        # Results navigation
        @kb.add('up', filter=in_results)
        @kb.add('k', filter=in_results)
        def _up(event):
            if self.selected_idx > 0:
                self.selected_idx -= 1
                self._update_preview()

        @kb.add('down', filter=in_results)
        @kb.add('j', filter=in_results)
        def _down(event):
            if self.selected_idx < len(self.cards) - 1:
                self.selected_idx += 1
                self._update_preview()

        @kb.add('enter', filter=in_results)
        def _start_edit(event):
            if self.cards:
                self._start_editing()

        @kb.add('escape', filter=in_results)
        def _focus_search_from_results(event):
            event.app.layout.focus(self.search_area)

        @kb.add('d', filter=in_results)
        def _delete(event):
            if self.cards:
                self._delete_card()

        @kb.add('/', filter=in_results)
        def _focus_search(event):
            event.app.layout.focus(self.search_area)

        # Form navigation mode (using invisible nav control - no cursor shown)
        @kb.add('up', filter=in_form_nav)
        @kb.add('k', filter=in_form_nav)
        def _form_up(event):
            self._form_navigate(-1)

        @kb.add('down', filter=in_form_nav)
        @kb.add('j', filter=in_form_nav)
        @kb.add('tab', filter=in_form_nav)
        def _form_down(event):
            self._form_navigate(1)

        @kb.add('s-tab', filter=in_form_nav)
        def _form_prev(event):
            self._form_navigate(-1)

        @kb.add('enter', filter=in_form_nav)
        @kb.add('i', filter=in_form_nav)
        def _form_enter_edit(event):
            # Focus the actual TextArea to show cursor and enable editing
            _, field = self.form_fields[self.form_field_idx]
            event.app.layout.focus(field)
            self.message = 'Editing field (Esc to stop)'

        @kb.add('escape', filter=in_form_nav)
        def _form_cancel(event):
            self._cancel_editing()
            event.app.layout.focus(self.results_window)

        @kb.add('c-s', filter=in_form_nav | in_form_editing)
        def _save(event):
            self._save_edit()
            event.app.layout.focus(self.results_window)

        # Form editing mode (actual TextArea focused - cursor shown)
        @kb.add('escape', filter=in_form_editing)
        def _form_exit_edit(event):
            event.app.layout.focus(self.form_nav_window)
            self.message = 'Navigate fields (j/k), edit (i/Enter)'

        @kb.add('tab', filter=in_form_editing)
        def _form_next_while_editing(event):
            self._form_navigate(1)

        @kb.add('s-tab', filter=in_form_editing)
        def _form_prev_while_editing(event):
            self._form_navigate(-1)

        return kb

    def _form_navigate(self, direction: int) -> None:
        self.form_field_idx = (self.form_field_idx + direction) % len(self.form_fields)
        if self.app:
            # Focus the invisible nav control (no cursor) rather than the actual TextArea
            self.app.layout.focus(self.form_nav_window)

    def _load_card_to_form(self, card: Card) -> None:
        self.form_fields[0][1].text = card.lexeme
        self.form_fields[1][1].text = ', '.join(card.rp)
        self.form_fields[2][1].text = card.past_simple or ''
        self.form_fields[3][1].text = card.past_participle or ''
        self.form_fields[4][1].text = '\n'.join(card.translations)
        self.form_fields[5][1].text = '\n'.join(card.example)

    def _form_to_card(self) -> Card:
        rp_text = self.form_fields[1][1].text.strip()
        return Card(
            lexeme=self.form_fields[0][1].text.strip(),
            rp=[x.strip() for x in rp_text.split(',') if x.strip()] if rp_text else [],
            past_simple=self.form_fields[2][1].text.strip() or None,
            past_participle=self.form_fields[3][1].text.strip() or None,
            translations=[x.strip() for x in self.form_fields[4][1].text.split('\n') if x.strip()],
            example=[x.strip() for x in self.form_fields[5][1].text.split('\n') if x.strip()],
        )

    def _start_editing(self) -> None:
        self.editing_idx = self.selected_idx
        self.form_field_idx = 0
        _, card = self.cards[self.editing_idx]
        self._load_card_to_form(card)
        self.message = 'Navigate fields (j/k), edit (i/Enter)'
        if self.app:
            # Focus the invisible nav control (no cursor shown during navigation)
            self.app.layout.focus(self.form_nav_window)

    def _cancel_editing(self) -> None:
        self.editing_idx = None
        self.message = 'Edit cancelled'
        self._update_preview()

    def _save_edit(self) -> None:
        if self.editing_idx is None:
            return

        card_id, _ = self.cards[self.editing_idx]
        card = self._form_to_card()

        try:
            with psycopg.connect(self.conn_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        update cards set
                            lexeme = %s,
                            rp = %s,
                            past_simple = %s,
                            past_participle = %s,
                            translations = %s,
                            example = %s
                        where id = %s
                        """,
                        (
                            card.lexeme,
                            card.rp,
                            card.past_simple,
                            card.past_participle,
                            card.translations,
                            card.example,
                            card_id,
                        ),
                    )
                conn.commit()
            self.cards[self.editing_idx] = (card_id, card)
            self.message = f'Saved: {card.lexeme}'
        except psycopg.Error as e:
            self.message = f'Error saving: {e}'

        self.editing_idx = None
        self._update_preview()

    def _delete_card(self) -> None:
        if not self.cards:
            return
        card_id, card = self.cards[self.selected_idx]
        try:
            with psycopg.connect(self.conn_string) as conn:
                with conn.cursor() as cur:
                    cur.execute('delete from cards where id = %s', (card_id,))
                conn.commit()
            self.message = f'Deleted: {card.lexeme}'
            del self.cards[self.selected_idx]
            if self.selected_idx >= len(self.cards) and self.cards:
                self.selected_idx = len(self.cards) - 1
            self._update_preview()
        except psycopg.Error as e:
            self.message = f'Error deleting: {e}'

    def _clear_form(self) -> None:
        for _, field in self.form_fields:
            field.text = ''

    def _update_preview(self) -> None:
        if self.cards and self.selected_idx < len(self.cards):
            _, card = self.cards[self.selected_idx]
            self._load_card_to_form(card)
        else:
            self._clear_form()

    def _on_search(self) -> None:
        query = self.search_area.text.strip()
        if not query:
            self.cards = []
            self.message = ''
            self._clear_form()
            return

        try:
            with psycopg.connect(self.conn_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        select id, lexeme, rp, past_simple, past_participle, translations, example
                        from cards
                        where lexeme ilike %s
                        order by lexeme
                        limit 100
                        """,
                        (f'%{query}%',),
                    )
                    rows = cur.fetchall()

            self.cards = [
                (
                    row[0],
                    Card(
                        lexeme=row[1],
                        rp=row[2] or [],
                        past_simple=row[3],
                        past_participle=row[4],
                        translations=row[5] or [],
                        example=row[6] or [],
                    ),
                )
                for row in rows
            ]
            self.selected_idx = 0
            self.message = f'Found {len(self.cards)} card(s)'
            self._update_preview()
        except psycopg.Error as e:
            self.message = f'Search error: {e}'
            self.cards = []

    def _get_results_text(self):
        if not self.cards:
            return [('class:dim', 'No results. Type a search query and press Enter.')]

        lines = []
        for idx, (card_id, card) in enumerate(self.cards):
            is_selected = idx == self.selected_idx

            prefix = '► ' if is_selected else '  '
            style = 'class:selected' if is_selected else ''

            lexeme_line = f'{prefix}{card.lexeme}'
            if card.rp:
                lexeme_line += ' /' + '/, /'.join(card.rp) + '/'

            lines.append((style + ' class:lexeme', lexeme_line))
            lines.append(('', '\n'))

            if card.past_simple:
                lines.append(('class:dim', f'    past: {card.past_simple}'))
                if card.past_participle:
                    lines.append(('class:dim', f' / {card.past_participle}'))
                lines.append(('', '\n'))

            for t in card.translations:
                lines.append(('', f'    • {t}\n'))

            for ex in card.example:
                lines.append(('class:dim', f'    > {ex}\n'))

            lines.append(('', '\n'))

        return lines

    def _get_help_text(self) -> str:
        if self._in_any_form_field():
            return 'Esc: stop editing field | Tab/S-Tab: next/prev field | Ctrl-S: save'
        if self._in_form_nav():
            return '↑/↓ or j/k: select field | i/Enter: edit field | Ctrl-S: save | Esc: cancel'
        return '↑/↓ or j/k: navigate | Enter: edit | d: delete | /: search | Esc: search | Ctrl-Q: quit'

    def _get_form_label_style(self, idx: int) -> str:
        if self.editing_idx is not None and idx == self.form_field_idx:
            return 'class:label-selected'
        return 'class:label'

    def _create_form_row(self, idx: int, label: str, field: TextArea) -> VSplit:
        # Wrap field with padding and dynamic background (only when editing this field)
        def get_style() -> str:
            return 'class:field-editing' if self._in_any_form_field() and self.form_field_idx == idx else ''

        def get_label_text() -> str:
            return f' {"►" if self.editing_idx is not None and idx == self.form_field_idx else " "} {label}: '

        def get_label_style() -> str:
            return self._get_form_label_style(idx)

        field_with_padding = VSplit(
            [
                Window(width=1, style=get_style),
                field,
                Window(width=1, style=get_style),
            ],
            style=get_style,
        )
        return VSplit(
            [
                Window(
                    FormattedTextControl(text=get_label_text),
                    width=Dimension.exact(20),
                    style=get_label_style,
                    dont_extend_width=True,
                ),
                field_with_padding,
            ]
        )

    def create_layout(self) -> Layout:
        help_control = FormattedTextControl(text=self._get_help_text)

        results_pane = ScrollablePane(self.results_window)

        edit_label_control = FormattedTextControl(
            text=lambda: ' EDITING' if self.editing_idx is not None else ' PREVIEW'
        )

        form_rows: list[Container] = [
            self.form_nav_window,  # Invisible focusable window for form navigation (no cursor)
            Window(edit_label_control, height=1, style='class:label'),
            Window(height=1, char='─', style='class:dim'),
        ]
        for idx, (label, field) in enumerate(self.form_fields):
            form_rows.append(self._create_form_row(idx, label, field))
            if idx == 3:  # After past_participle
                form_rows.append(Window(height=1))

        form = HSplit(form_rows)

        body = HSplit(
            [
                Frame(self.search_area, title='Search Lexemes', height=Dimension.exact(3)),
                Frame(results_pane, title='Results'),
                Frame(form, title='Editor'),
                Window(content=self.message_control, height=1),
                Window(content=help_control, height=1, style='class:dim'),
            ]
        )

        return Layout(body, focused_element=self.search_area)

    def run(self) -> None:
        self.app = Application(
            layout=self.create_layout(),
            key_bindings=self._create_key_bindings(),
            style=self.style,
            full_screen=True,
            mouse_support=True,
        )
        self.app.run()


def main() -> int:
    parser = get_parser(sections=[Sections.DB, Sections.EDITOR], require_db=True)
    cfg = parser.parse_args()

    if cfg.print_config:
        print_config(cfg)
        return 0

    editor = CardEditor(cfg.db_conn_string)
    editor.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
