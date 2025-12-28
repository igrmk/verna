# AGENTS.md

## Project Overview

**Verna** is a CLI tool for English-Russian translation and vocabulary learning. It translates words and texts between English and Russian, extracts and saves advanced English lexemes as flashcards to a PostgreSQL database, and can send daily vocabulary cards with AI-generated stories to Telegram.

## Tech Stack

- **Language**: Python >=3.12
- **AI**: OpenAI API (or compatible providers like OpenRouter) with structured outputs via Pydantic
- **Database**: PostgreSQL (via `psycopg`)
- **CLI**:
    `configargparse` for configuration,
    `prompt_toolkit` for terminal output and interactive input
- **Templating**: Jinja2 for AI prompt templates

## Project Structure

```
verna/
├── verna.py          # Main translation CLI (entry point: `verna`)
├── random_cards.py   # Daily cards + story generator (entry point: `verna-cards`)
├── edit.py           # TUI editor for lexemes (entry point: `verna-edit`)
├── config.py         # CLI argument parsing and configuration
├── console.py        # Console output helpers
├── db.py             # Database operations for cards table
├── db_types.py       # Database card dataclass and formatting
├── styles.py         # prompt_toolkit styles
├── migrator.py       # Database migration runner
├── create_spec.py    # DigitalOcean App Platform spec generator
├── verna.ini         # Default configuration file
└── migrations/       # SQL migration files (001_*, 002_*, ...)
```

## Entry Points

| Command        | Module                     | Description                                          |
|----------------|----------------------------|------------------------------------------------------|
| `verna`        | `verna.verna:main`         | Main translation tool                                |
| `verna-cards`  | `verna.random_cards:main`  | Fetch random cards, generate story, send to Telegram |
| `verna-edit`   | `verna.edit:main`          | TUI editor for searching and editing lexemes         |

## Main Workflow (`verna`)

1. **Language Detection** — Detect if input is English, Russian, or other
2. **Translation** — Translate the input text (English <-> Russian)
3. **Lexeme Extraction** — For English input, extract lexemes with CEFR levels
4. **Lexeme Translation** — Translate each extracted lexeme individually
5. **Card Saving** — Interactively save cards to PostgreSQL with examples

## Configuration

Configuration is loaded from (in order of precedence):
1. CLI arguments
2. Environment variables (`VERNA_*`)
3. Config files: `verna/verna.ini` and, e.g., `~/.config/verna/verna.ini` (OS dependent)

## Database Schema

The `cards` table stores vocabulary cards:

| Column              | Type        | Description                                |
|---------------------|-------------|--------------------------------------------|
| `id`                | bigserial   | Primary key                                |
| `created_at`        | timestamptz | Creation timestamp                         |
| `lexeme`            | text        | The word/phrase (unique, case-insensitive) |
| `rp`                | text[]      | British RP transcriptions                  |
| `past_simple`       | text        | Irregular verb past simple form            |
| `past_simple_rp`    | text[]      | Past simple RP transcriptions              |
| `past_participle`   | text        | Irregular verb past participle             |
| `past_participle_rp`| text[]      | Past participle RP transcriptions          |
| `translations`      | text[]      | Russian translations                       |
| `example`           | text[]      | Example sentences                          |

Run migrations with: `python -m verna.migrator`

## AI Integration

- Uses OpenAI's `responses.parse()` with Pydantic models for structured outputs
- Prompts are Jinja2 templates in `verna.py`
- Different models can be specified for different tasks (detection, translation, extraction)
- Supports reasoning effort levels via `--reason` or `--think` flags

## Dev Environment
- Virtual environments:
    created by github.com/igrmk/cavez
    managed by micromamba
    local package installation: pip
- Linting: Ruff
- Type checking: mypy
- Run locally: python -m verna
- Before committing, always run: `ruff check --fix verna/ && ruff format verna/ && mypy verna/`

## Code Style

- Line length: 120 characters
- Quote style: single quotes (') or triple double quotes (""")
- Type hints: Python 3.12+ style (`list[str]`, `str | None`)
- SQL: if select columns don't fit on a single line, put each column on a new line

## Commits

- Use conventional commit style
- Use simple `git commit -m "message"` without Co-Authored-By or other footers

## Testing

No test suite currently. Test manually by running commands with `--debug` flag.

## Common Development Tasks

### Adding a new CLI option
1. Add argument in `config.py` under the appropriate `_add_*` function
2. Use the option via `cfg.<option_name>` in the relevant module, don't use `getattr`