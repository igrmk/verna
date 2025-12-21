# prompt_toolkit styles for console output and TUI editor
PT_STYLES = {
    # Console output styles
    'lexeme': 'bold fg:ansicyan',
    'lexeme-italic': 'bold italic fg:ansicyan',
    'example': 'italic fg:#bbbbbb',
    'section-header': 'bold',
    'log': 'fg:#555555',
    'debug': 'fg:ansibrightblack',
    'debug-step': 'underline fg:ansibrightblack',
    'card-label': 'fg:ansibrightblack',
    'warning': 'fg:ansiyellow',
    'note-header': 'bold',
    'transcription': 'italic',
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
