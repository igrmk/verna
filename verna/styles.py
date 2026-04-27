# prompt_toolkit styles for console output and TUI editor
PT_STYLES = {
    # Console output styles
    'lexeme': 'bold fg:ansicyan',
    'lexeme-italic': 'bold italic fg:ansicyan',
    'example': 'italic fg:#bbbbbb',
    'example-generated': 'italic fg:#888888',
    'section-header': 'fg:ansiwhite bold italic',
    'log': 'fg:#777777',
    'debug': 'fg:ansibrightblack',
    'debug-step': 'underline fg:ansibrightblack',
    'card-label': 'fg:ansibrightblack',
    'warning': 'fg:ansiyellow',
    'success': 'fg:ansigreen',
    'note-header': 'fg:ansiwhite bold italic',
    # No italic: italic `a` is single-storey and indistinguishable from `ɑ`,
    # which is a different vowel in Lindsey transcription (TRAP `a` vs START `ɑː`).
    'transcription': '',
    # TUI editor styles
    'frame.border': 'fg:ansibrightblack',
    'frame.label': 'fg:ansiwhite',
    'frame-focused frame.border': 'fg:ansicyan',
    'frame-focused frame.label': 'fg:ansicyan bold reverse',
    'selected': 'reverse',
    'selected-unfocused': 'fg:ansiblack bg:ansibrightblack',
    'lexeme-dim': 'fg:ansicyan',
    'dim': 'fg:ansibrightblack',
    'label': 'fg:ansicyan',
    'label-selected': 'fg:ansicyan bold reverse',
    'field-editing': 'bg:#252525',
    'dialog frame.border': 'fg:ansicyan',
}
