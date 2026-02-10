import sys
import psycopg
from prompt_toolkit.application import Application
from verna import db
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.key_binding import KeyBindings, KeyBindingsBase, merge_key_bindings
from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout import (
    Layout,
    HSplit,
    VSplit,
    Window,
    FormattedTextControl,
    ConditionalContainer,
    Float,
    FloatContainer,
    WindowAlign,
)
from prompt_toolkit.layout.containers import Container
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import Frame, TextArea
from prompt_toolkit.styles import Style

import re

from verna.config import get_parser, Sections, print_config
from verna.db_types import Card, Translation, format_card
from verna import styles

from typing import Callable


class SearchPanel:
    def __init__(
        self,
        on_search: Callable[[str], None],
        on_focus_results: Callable[[], None],
        on_exit: Callable[[], None],
    ):
        self.on_search = on_search
        self.on_focus_results = on_focus_results
        self.on_exit = on_exit
        self.app: Application | None = None

        self.search_area = TextArea(
            height=1,
            prompt='Search: ',
            multiline=False,
            wrap_lines=False,
        )
        self.search_area.buffer.on_text_changed += lambda _: self.on_search(self.search_area.text.strip())

    def set_app(self, app: Application) -> None:
        self.app = app

    def is_focused(self) -> bool:
        if self.app is None:
            return False
        return self.app.layout.has_focus(self.search_area)

    def focus(self) -> None:
        if self.app:
            self.app.layout.focus(self.search_area)

    def get_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        in_search = has_focus(self.search_area)

        @kb.add('escape', filter=in_search)
        def _exit_from_search(event):
            self.on_exit()

        @kb.add('enter', filter=in_search)
        def _focus_results(event):
            self.on_focus_results()

        return kb

    def create_layout(self) -> Container:
        is_focused = Condition(self.is_focused)

        search_with_margin = VSplit(
            [
                Window(width=1),
                self.search_area,
                Window(width=1),
            ]
        )

        return HSplit(
            [
                ConditionalContainer(
                    Frame(
                        search_with_margin,
                        title='Search Lexemes',
                        height=Dimension.exact(3),
                        style='class:frame-focused',
                    ),
                    filter=is_focused,
                ),
                ConditionalContainer(
                    Frame(search_with_margin, title='Search Lexemes', height=Dimension.exact(3)),
                    filter=~is_focused,
                ),
            ]
        )


class ResultsPanel:
    def __init__(
        self,
        on_edit: Callable[[int, Card], None],
        on_delete: Callable[[int, Card], None],
        on_focus_search: Callable[[], None],
    ):
        self.on_edit = on_edit
        self.on_delete = on_delete
        self.on_focus_search = on_focus_search
        self.app: Application | None = None

        self.cards: list[tuple[int, Card]] = []
        self.selected_idx = 0
        self.show_delete_dialog = False
        self._moved_down = False

        self.results_control = FormattedTextControl(
            text=self._get_results_text,
            focusable=True,
            show_cursor=False,
            get_cursor_position=lambda: Point(0, self._get_selected_line()),
        )
        self.results_window = Window(content=self.results_control, wrap_lines=True)

    def set_app(self, app: Application) -> None:
        self.app = app

    def is_focused(self) -> bool:
        if self.app is None:
            return False
        return self.app.layout.has_focus(self.results_window)

    def focus(self) -> None:
        if self.app:
            self.app.layout.focus(self.results_window)

    def set_cards(self, cards: list[tuple[int, Card]]) -> None:
        self.cards = cards
        self.selected_idx = 0
        self._moved_down = False

    def update_card(self, card_id: int, card: Card) -> None:
        for i, (cid, _) in enumerate(self.cards):
            if cid == card_id:
                self.cards[i] = (card_id, card)
                break

    def remove_card(self, card_id: int) -> None:
        for i, (cid, _) in enumerate(self.cards):
            if cid == card_id:
                del self.cards[i]
                if self.selected_idx >= len(self.cards) and self.cards:
                    self.selected_idx = len(self.cards) - 1
                break

    def get_selected_card(self) -> tuple[int, Card] | None:
        if self.cards and self.selected_idx < len(self.cards):
            return self.cards[self.selected_idx]
        return None

    def _count_card_lines(self, card: Card) -> int:
        lines = 1  # lexeme line
        if card.past_simple or card.past_participle:
            lines += 1  # past tense line
        lines += len(card.translations)
        lines += len(card.example)
        lines += 1  # empty line separator
        return lines

    def _get_selected_line(self) -> int:
        if not self.cards:
            return 0
        line = 0
        for idx, (_, card) in enumerate(self.cards):
            card_lines = self._count_card_lines(card)
            if idx == self.selected_idx:
                if self._moved_down:
                    return line + card_lines - 1  # show end of card when moving down
                return line  # show start of card when moving up
            line += card_lines
        return 0

    def _get_results_text(self) -> list[tuple[str, str]]:
        if not self.cards:
            return [('class:dim', 'No results.')]

        results_focused = self.is_focused()
        lines: list[tuple[str, str]] = []
        for idx, (card_id, card) in enumerate(self.cards):
            is_selected = idx == self.selected_idx

            prefix = ' ▶ ' if is_selected else '   '
            if is_selected and not results_focused:
                prefix_style = 'class:selected-unfocused'
            elif is_selected:
                prefix_style = 'class:selected class:lexeme'
            elif not results_focused:
                prefix_style = 'class:lexeme-dim'
            else:
                prefix_style = 'class:lexeme'

            lines.append((prefix_style, prefix))
            lines.extend(format_card(card, focused=results_focused, indent=3))
            lines.append(('', '\n\n'))

        return lines

    def get_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        in_results = has_focus(self.results_window)
        in_delete_dialog = Condition(lambda: self.show_delete_dialog)

        @kb.add('up', filter=in_results & ~in_delete_dialog)
        @kb.add('k', filter=in_results & ~in_delete_dialog)
        def _up(event):
            if self.selected_idx > 0:
                self.selected_idx -= 1
                self._moved_down = False

        @kb.add('down', filter=in_results & ~in_delete_dialog)
        @kb.add('j', filter=in_results & ~in_delete_dialog)
        def _down(event):
            if self.selected_idx < len(self.cards) - 1:
                self.selected_idx += 1
                self._moved_down = True

        @kb.add('enter', filter=in_results & ~in_delete_dialog)
        def _start_edit(event):
            selected = self.get_selected_card()
            if selected:
                card_id, card = selected
                self.on_edit(card_id, card)

        @kb.add('escape', filter=in_results & ~in_delete_dialog)
        def _focus_search_from_results(event):
            self.on_focus_search()

        @kb.add('d', filter=in_results & ~in_delete_dialog)
        def _show_delete_dialog(event):
            if self.cards:
                self.show_delete_dialog = True

        @kb.add('y', filter=in_results & in_delete_dialog)
        def _confirm_delete(event):
            selected = self.get_selected_card()
            if selected:
                card_id, card = selected
                self.on_delete(card_id, card)
            self.show_delete_dialog = False

        @kb.add('n', filter=in_results & in_delete_dialog)
        @kb.add('escape', filter=in_results & in_delete_dialog)
        def _cancel_delete(event):
            self.show_delete_dialog = False

        @kb.add('/', filter=in_results & ~in_delete_dialog)
        def _focus_search(event):
            self.on_focus_search()

        return kb

    def create_layout(self) -> Container:
        is_focused = Condition(self.is_focused)

        results_pane = VSplit(
            [
                Window(width=1),
                self.results_window,
                Window(width=1),
            ]
        )

        return HSplit(
            [
                ConditionalContainer(
                    Frame(results_pane, title='Results', style='class:frame-focused'),
                    filter=is_focused,
                ),
                ConditionalContainer(
                    Frame(results_pane, title='Results'),
                    filter=~is_focused,
                ),
            ]
        )

    def create_delete_dialog(self) -> Float:
        def get_dialog_text():
            selected = self.get_selected_card()
            if selected:
                _, card = selected
                return f'Delete "{card.lexeme}"?\n\n\n(y) Yes        (n) No'
            return 'Delete this card?\n\n\n(y) Yes        (n) No'

        delete_dialog = Frame(
            body=HSplit(
                [
                    Window(height=1),
                    Window(
                        FormattedTextControl(text=get_dialog_text),
                        height=4,
                        width=Dimension(min=40),
                        align=WindowAlign.CENTER,
                    ),
                    Window(height=1),
                ]
            ),
            title='Confirm Delete',
            style='class:frame-focused',
        )

        show_delete = Condition(lambda: self.show_delete_dialog)

        return Float(content=ConditionalContainer(delete_dialog, filter=show_delete))


class EditorPanel:
    SINGLE_LINE_FIELD_COUNT = 5  # lexeme, past_simple, past_simple_rp, past_participle, past_participle_rp

    def __init__(
        self,
        on_save: Callable[[int, Card], None],
        on_cancel: Callable[[], None],
    ):
        self.on_save = on_save
        self.on_cancel = on_cancel
        self.app: Application | None = None

        self.editing_card_id: int | None = None
        self.form_field_idx = 0
        self.show_save_dialog = False
        self.original_card: Card | None = None

        # Invisible control for form navigation (doesn't show cursor)
        self.form_nav_control = FormattedTextControl(text='', focusable=True, show_cursor=False)
        self.form_nav_window = Window(content=self.form_nav_control, height=0)

        # Create form fields with read_only filter
        field_readonly = Condition(lambda: not self._in_any_form_field())
        self.field_lexeme = TextArea(height=1, multiline=False, wrap_lines=False, read_only=field_readonly)
        self.field_past_simple = TextArea(height=1, multiline=False, wrap_lines=False, read_only=field_readonly)
        self.field_past_simple_rp = TextArea(height=1, multiline=False, wrap_lines=False, read_only=field_readonly)
        self.field_past_participle = TextArea(height=1, multiline=False, wrap_lines=False, read_only=field_readonly)
        self.field_past_participle_rp = TextArea(height=1, multiline=False, wrap_lines=False, read_only=field_readonly)
        self.field_translations = TextArea(
            height=Dimension(min=1),
            multiline=True,
            wrap_lines=True,
            read_only=field_readonly,
            dont_extend_height=True,
        )
        self.field_example = TextArea(
            height=Dimension(min=1),
            multiline=True,
            wrap_lines=True,
            read_only=field_readonly,
            dont_extend_height=True,
        )

        self.form_fields = [
            ('Lexeme', self.field_lexeme),
            ('Past simple', self.field_past_simple),
            ('Past simple RP', self.field_past_simple_rp),
            ('Past participle', self.field_past_participle),
            ('Past part. RP', self.field_past_participle_rp),
            ('Translations', self.field_translations),
            ('Example', self.field_example),
        ]

    def set_app(self, app: Application) -> None:
        self.app = app

    def is_editing(self) -> bool:
        return self.editing_card_id is not None

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

    def focus(self) -> None:
        if self.app:
            self.app.layout.focus(self.form_nav_window)

    def start_editing(self, card_id: int, card: Card) -> None:
        self.editing_card_id = card_id
        self.form_field_idx = 0
        self.original_card = card
        self._load_card_to_form(card)
        self.focus()

    def _load_card_to_form(self, card: Card) -> None:
        self.field_lexeme.text = card.lexeme
        self.field_past_simple.text = card.past_simple or ''
        self.field_past_simple_rp.text = ', '.join(card.past_simple_rp)
        self.field_past_participle.text = card.past_participle or ''
        self.field_past_participle_rp.text = ', '.join(card.past_participle_rp)
        # Format translations as "/rp1/ /rp2/ text" or just "text" if no RP
        translation_lines = []
        for t in card.translations:
            rp_part = ' '.join(f'/{rp}/' for rp in t.rp)
            if rp_part:
                translation_lines.append(f'{rp_part} {t.text}')
            else:
                translation_lines.append(t.text)
        self.field_translations.text = '\n'.join(translation_lines)
        self.field_example.text = '\n'.join(card.example)

    def _parse_translation_line(self, line: str) -> Translation:
        """Parse a translation line in format '/rp1/ /rp2/ text' or just 'text'."""
        line = line.strip()
        rp_list: list[str] = []
        # Extract all /.../ patterns from the beginning
        rp_pattern = re.compile(r'^(/[^/]+/\s*)+')
        match = rp_pattern.match(line)
        if match:
            rp_part = match.group(0)
            text = line[len(rp_part) :].strip()
            # Extract individual RPs
            rp_list = [rp.strip() for rp in re.findall(r'/([^/]+)/', rp_part)]
        else:
            text = line
        return Translation(text=text, rp=rp_list)

    def _form_to_card(self) -> Card:
        past_simple_rp_text = self.field_past_simple_rp.text.strip()
        past_participle_rp_text = self.field_past_participle_rp.text.strip()
        # Parse translations from lines
        translations = [
            self._parse_translation_line(line) for line in self.field_translations.text.split('\n') if line.strip()
        ]
        return Card(
            lexeme=self.field_lexeme.text.strip(),
            past_simple=self.field_past_simple.text.strip() or None,
            past_simple_rp=[x.strip() for x in past_simple_rp_text.split(',') if x.strip()]
            if past_simple_rp_text
            else [],
            past_participle=self.field_past_participle.text.strip() or None,
            past_participle_rp=[x.strip() for x in past_participle_rp_text.split(',') if x.strip()]
            if past_participle_rp_text
            else [],
            translations=translations,
            example=[x.strip() for x in self.field_example.text.split('\n') if x.strip()],
        )

    def _has_changes(self) -> bool:
        if self.original_card is None:
            return False
        current = self._form_to_card()
        orig = self.original_card
        return (
            current.lexeme != orig.lexeme
            or current.past_simple != orig.past_simple
            or current.past_simple_rp != orig.past_simple_rp
            or current.past_participle != orig.past_participle
            or current.past_participle_rp != orig.past_participle_rp
            or current.translations != orig.translations
            or current.example != orig.example
        )

    def _form_navigate(self, direction: int) -> None:
        self.form_field_idx = (self.form_field_idx + direction) % len(self.form_fields)
        if self.app:
            self.app.layout.focus(self.form_nav_window)

    def _save_and_exit(self) -> None:
        if self.editing_card_id is None:
            return
        card = self._form_to_card()
        card_id = self.editing_card_id
        self.editing_card_id = None
        self.original_card = None
        self.on_save(card_id, card)

    def _cancel_and_exit(self) -> None:
        self.editing_card_id = None
        self.original_card = None
        self.on_cancel()

    def get_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        in_form_nav = Condition(self._in_form_nav)
        in_form_editing = Condition(self._in_any_form_field)
        in_save_dialog = Condition(lambda: self.show_save_dialog)
        in_single_line_field = Condition(lambda: self.form_field_idx < self.SINGLE_LINE_FIELD_COUNT)

        # Form navigation mode
        @kb.add('up', filter=in_form_nav & ~in_save_dialog)
        @kb.add('k', filter=in_form_nav & ~in_save_dialog)
        def _form_up(event):
            self._form_navigate(-1)

        @kb.add('down', filter=in_form_nav & ~in_save_dialog)
        @kb.add('j', filter=in_form_nav & ~in_save_dialog)
        @kb.add('tab', filter=in_form_nav & ~in_save_dialog)
        def _form_down(event):
            self._form_navigate(1)

        @kb.add('s-tab', filter=in_form_nav & ~in_save_dialog)
        def _form_prev(event):
            self._form_navigate(-1)

        @kb.add('i', filter=in_form_nav & ~in_save_dialog)
        @kb.add('enter', filter=in_form_nav & ~in_save_dialog)
        def _form_enter_edit(event):
            _, field = self.form_fields[self.form_field_idx]
            event.app.layout.focus(field)

        @kb.add('escape', filter=in_form_nav & ~in_save_dialog)
        def _exit_or_show_save_dialog(event):
            if self._has_changes():
                self.show_save_dialog = True
            else:
                self._cancel_and_exit()

        @kb.add('y', filter=in_form_nav & in_save_dialog)
        @kb.add('enter', filter=in_form_nav & in_save_dialog)
        def _confirm_save(event):
            self._save_and_exit()
            self.show_save_dialog = False

        @kb.add('n', filter=in_form_nav & in_save_dialog)
        def _discard_save(event):
            self._cancel_and_exit()
            self.show_save_dialog = False

        @kb.add('escape', filter=in_form_nav & in_save_dialog)
        def _back_to_editing(event):
            self.show_save_dialog = False

        # Form editing mode (actual TextArea focused)
        @kb.add('escape', filter=in_form_editing)
        def _form_exit_edit(event):
            # Remove empty lines from multiline fields
            _, field = self.form_fields[self.form_field_idx]
            if self.form_field_idx >= self.SINGLE_LINE_FIELD_COUNT:
                lines = [line for line in field.text.split('\n') if line.strip()]
                field.text = '\n'.join(lines)
            event.app.layout.focus(self.form_nav_window)

        @kb.add('enter', filter=in_form_editing & in_single_line_field)
        def _form_exit_single_line_field(event):
            event.app.layout.focus(self.form_nav_window)

        @kb.add('tab', filter=in_form_editing)
        def _form_next_while_editing(event):
            self._form_navigate(1)

        @kb.add('s-tab', filter=in_form_editing)
        def _form_prev_while_editing(event):
            self._form_navigate(-1)

        return kb

    def _get_form_label_style(self, idx: int) -> str:
        if self.is_editing() and idx == self.form_field_idx:
            return 'class:label-selected'
        return 'class:label'

    def _create_form_row(self, idx: int, label: str, field: TextArea) -> VSplit:
        def get_style() -> str:
            return 'class:field-editing' if self._in_any_form_field() and self.form_field_idx == idx else ''

        def get_label_text() -> str:
            return f' {"▶" if self.is_editing() and idx == self.form_field_idx else " "} {label}: '

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

    def create_layout(self) -> Container:
        form_rows: list[Container] = [self.form_nav_window]
        for idx, (label, field) in enumerate(self.form_fields):
            form_rows.append(self._create_form_row(idx, label, field))
            if idx == 4:  # After past_participle_rp
                form_rows.append(Window(height=1))
            if idx == 5:  # After translations
                form_rows.append(Window(height=1))

        form = VSplit(
            [
                Window(width=1),
                HSplit(form_rows),
                Window(width=1),
            ]
        )

        is_editing = Condition(self.is_editing)

        return ConditionalContainer(
            Frame(form, title='Editor', style='class:frame-focused'),
            filter=is_editing,
        )

    def create_save_dialog(self) -> Float:
        save_dialog = Frame(
            body=HSplit(
                [
                    Window(height=1),
                    Window(
                        FormattedTextControl(text='Save changes?\n\n\n(y) Save        (n) Discard'),
                        height=4,
                        width=Dimension(min=40),
                        align=WindowAlign.CENTER,
                    ),
                    Window(height=1),
                ]
            ),
            title='Save Changes',
            style='class:frame-focused',
        )

        show_save = Condition(lambda: self.show_save_dialog)

        return Float(content=ConditionalContainer(save_dialog, filter=show_save))


class CardEditor:
    def __init__(self, conn_string: str):
        self.conn_string = conn_string
        self.message = ''
        self.app: Application | None = None

        # Create panels with callbacks
        self.search_panel = SearchPanel(
            on_search=self._on_search,
            on_focus_results=lambda: self.results_panel.focus(),
            on_exit=lambda: self.app.exit() if self.app else None,
        )

        self.results_panel = ResultsPanel(
            on_edit=self._start_editing,
            on_delete=self._delete_card,
            on_focus_search=lambda: self.search_panel.focus(),
        )

        self.editor_panel = EditorPanel(
            on_save=self._save_card,
            on_cancel=self._cancel_editing,
        )

        self.message_control = FormattedTextControl(text=lambda: self.message)

        self.style = Style.from_dict(styles.PT_STYLES)

    def _start_editing(self, card_id: int, card: Card) -> None:
        self.editor_panel.start_editing(card_id, card)
        self.message = 'Navigate fields (j/k), edit (i/Enter), exit (Esc)'

    def _cancel_editing(self) -> None:
        self.results_panel.focus()
        self.message = 'Edit cancelled'

    def _save_card(self, card_id: int, card: Card) -> None:
        try:
            with psycopg.connect(self.conn_string) as conn:
                db.update_card(conn, card_id, card)
            self.results_panel.update_card(card_id, card)
            self.message = f'Saved: {card.lexeme}'
        except psycopg.Error as e:
            self.message = f'Error saving: {e}'
        self.results_panel.focus()

    def _delete_card(self, card_id: int, card: Card) -> None:
        try:
            with psycopg.connect(self.conn_string) as conn:
                db.delete_card_by_id(conn, card_id)
            self.results_panel.remove_card(card_id)
            self.message = f'Deleted: {card.lexeme}'
        except psycopg.Error as e:
            self.message = f'Error deleting: {e}'

    def _on_search(self, query: str) -> None:
        try:
            with psycopg.connect(self.conn_string) as conn:
                cards = db.search_cards(conn, query, limit=50)
            self.results_panel.set_cards(cards)
            if len(cards) == 50:
                self.message = 'Showing first 50 cards (refine search for more)'
            else:
                self.message = f'Found {len(cards)} card(s)'
        except psycopg.Error as e:
            self.message = f'Search error: {e}'
            self.results_panel.set_cards([])

    def _get_help_text(self) -> str:
        if self.editor_panel._in_any_form_field():
            return 'Esc/Enter: exit field | Tab/S-Tab: next/prev'
        if self.editor_panel.show_save_dialog:
            return '(y) Save | (n) Discard | Esc: back'
        if self.editor_panel._in_form_nav():
            return '↑/↓ or j/k: select field | i/Enter: edit | Esc: exit'
        if self.results_panel.is_focused():
            return '↑/↓ or j/k: navigate | Enter: edit | d: delete | /: search | Esc: back'
        return 'Type to search | Enter: search | Esc: quit'

    def _create_key_bindings(self) -> KeyBindingsBase:
        kb = KeyBindings()

        @kb.add('c-c')
        def _exit(event):
            event.app.exit()

        # Merge key bindings from all panels
        return merge_key_bindings(
            [
                kb,
                self.search_panel.get_key_bindings(),
                self.results_panel.get_key_bindings(),
                self.editor_panel.get_key_bindings(),
            ]
        )

    def _create_layout(self) -> Layout:
        help_control = FormattedTextControl(text=self._get_help_text)

        not_editing = Condition(lambda: not self.editor_panel.is_editing())

        main_content = HSplit(
            [
                ConditionalContainer(self.search_panel.create_layout(), filter=not_editing),
                ConditionalContainer(self.results_panel.create_layout(), filter=not_editing),
                self.editor_panel.create_layout(),
                Window(),  # Filler to push message/help to bottom
                Window(content=self.message_control, height=1),
                Window(content=help_control, height=1, style='class:dim'),
            ]
        )

        body = FloatContainer(
            content=main_content,
            floats=[
                self.results_panel.create_delete_dialog(),
                self.editor_panel.create_save_dialog(),
            ],
        )

        return Layout(body, focused_element=self.search_panel.search_area)

    def run(self) -> None:
        # Load initial list
        self._on_search('')

        self.app = Application(
            layout=self._create_layout(),
            key_bindings=self._create_key_bindings(),
            style=self.style,
            full_screen=True,
            mouse_support=True,
        )

        # Set app reference on all panels
        self.search_panel.set_app(self.app)
        self.results_panel.set_app(self.app)
        self.editor_panel.set_app(self.app)

        # Reduce escape key delay
        self.app.ttimeoutlen = 0.05
        self.app.timeoutlen = 0.05
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
