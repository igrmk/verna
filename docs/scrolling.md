# Direction-Based Cursor Positioning for List Scrolling

## Problem

When navigating long lists of lexemes (in `verna`) or cards (in `verna-edit`),
items at the end of the list couldn't be fully seen
because the viewport didn't scroll far enough
to show the complete item content including translations and examples.

## Solution

Implement direction-aware cursor positioning that adjusts based on navigation direction:

- **Moving down**: Position cursor at the **end** of the selected item,
  ensuring examples/translations below the lexeme are visible
- **Moving up**: Position cursor at the **start** of the selected item,
  ensuring the lexeme header is visible
