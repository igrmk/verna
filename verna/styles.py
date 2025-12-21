# Rich styles (for Console output)
LEXEME_HEADER = 'cyan bold'
LEXEME_HEADER_ITALIC = 'cyan bold italic'
EXAMPLE = 'italic'
SECTION_HEADER = 'bold'
LOG = 'dim grey50'
DEBUG = 'dim'
DEBUG_STEP = 'underline dim'
CARD_LABEL = 'dim'
WARNING = 'yellow'
NOTE_HEADER = 'bold'
TRANSCRIPTION = 'italic'

# prompt_toolkit styles (for TUI editor)
PT_STYLES = {
    'frame.border': 'fg:ansibrightblack',
    'frame.label': 'fg:ansiwhite',
    'frame-focused frame.border': 'fg:ansicyan',
    'frame-focused frame.label': 'fg:ansicyan bold reverse',
    'selected': 'reverse',
    'selected-unfocused': 'fg:ansiblack bg:ansibrightblack',
    'lexeme': 'bold fg:ansicyan',
    'lexeme-dim': 'fg:ansicyan',
    'dim': 'fg:ansibrightblack',
    'label': 'fg:ansicyan',
    'label-selected': 'fg:ansicyan bold reverse',
    'field-editing': 'bg:#252525',
    'dialog frame.border': 'fg:ansicyan',
}
