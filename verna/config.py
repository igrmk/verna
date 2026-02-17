import sys
import argparse
from pathlib import Path
import configargparse
from enum import StrEnum
from platformdirs import user_config_dir


class Sections(StrEnum):
    COMMON = 'common'
    DB = 'db'
    TELEGRAM = 'telegram'
    RANDOM = 'random'
    VERNA = 'verna'
    AI = 'ai'
    EDITOR = 'editor'


class ReasoningLevel(StrEnum):
    UNSUPPORTED = 'unsupported'
    NONE = 'none'
    MINIMAL = 'minimal'
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'


class CefrLevel(StrEnum):
    A1 = 'A1'
    A2 = 'A2'
    B1 = 'B1'
    B2 = 'B2'
    C1 = 'C1'
    C2 = 'C2'


def _add_common(p: configargparse.ArgParser) -> None:
    p.add_argument(
        '--config',
        required=False,
        is_config_file=True,
        help='path to a config file',
    )
    p.add_argument(
        '--print-config',
        action='store_true',
        help='print the loaded config and exit',
        default=False,
    )
    p.add_argument(
        '--debug',
        action=argparse.BooleanOptionalAction,
        help='debug mode',
        default=False,
    )


def _add_db(p: configargparse.ArgParser, *, require_db: bool) -> None:
    p.add_argument(
        '--db-conn-string',
        env_var='VERNA_DB_CONN_STRING',
        type=str,
        required=require_db,
        metavar='STR',
        help='PostgreSQL connection string',
    )
    p.add_argument(
        '--db-owner',
        env_var='VERNA_DB_OWNER',
        type=str,
        metavar='STR',
        help='PostgreSQL database owner (used for migrations)',
    )


def _add_ai(p: configargparse.ArgParser) -> None:
    p.add_argument(
        '--api-key',
        env_var='VERNA_API_KEY',
        type=str,
        required=True,
        help='AI API key',
    )
    p.add_argument(
        '--api-base-url',
        env_var='VERNA_API_BASE_URL',
        type=str,
        required=False,
        help='AI API base URL',
    )
    p.add_argument(
        '--model',
        env_var='VERNA_MODEL',
        type=str,
        required=True,
        help='AI model ID',
    )
    p.add_argument(
        '--model-detect',
        env_var='VERNA_MODEL_DETECT',
        type=str,
        required=False,
        help='AI model ID for language detection (defaults to --model)',
    )
    p.add_argument(
        '--structured-detect',
        env_var='VERNA_STRUCTURED_DETECT',
        action=argparse.BooleanOptionalAction,
        help='use structured outputs for language detection',
    )
    p.add_argument(
        '--model-translate-text',
        env_var='VERNA_MODEL_TRANSLATE_TEXT',
        type=str,
        required=False,
        help='AI model ID for whole-text translation (defaults to --model)',
    )
    p.add_argument(
        '--model-translate-lexeme',
        env_var='VERNA_MODEL_TRANSLATE_LEXEME',
        type=str,
        required=False,
        help='AI model ID for lexeme translation (defaults to --model)',
    )
    p.add_argument(
        '--model-extract',
        env_var='VERNA_MODEL_EXTRACT',
        type=str,
        required=False,
        help='AI model ID for lexeme extraction (defaults to --model)',
    )
    reason_group = p.add_mutually_exclusive_group(required=False)
    reason_group.add_argument(
        '--reason',
        env_var='VERNA_REASON',
        type=ReasoningLevel,
        choices=ReasoningLevel,
        help='AI reasoning effort',
    )
    default_think_reasoning_level = ReasoningLevel.MEDIUM
    reason_group.add_argument(
        '-t',
        '--think',
        dest='reason',
        action='store_const',
        const=default_think_reasoning_level,
        help=f'Make AI think (alias for --reason {default_think_reasoning_level})',
    )
    p.set_defaults(reason=ReasoningLevel.LOW)


def _add_tg(p: configargparse.ArgParser, *, require_tg: bool) -> None:
    p.add_argument(
        '--send-to-tg',
        env_var='VERNA_SEND_TO_TG',
        action=argparse.BooleanOptionalAction,
        required=True,
        help='send output to Telegram',
    )
    p.add_argument(
        '--tg-bot-token',
        env_var='VERNA_TG_BOT_TOKEN',
        required=require_tg,
        help='Telegram bot token',
    )
    p.add_argument(
        '--tg-chat-id',
        env_var='VERNA_TG_CHAT_ID',
        required=require_tg,
        help='Telegram chat ID',
    )


def _add_random(p: configargparse.ArgParser) -> None:
    p.add_argument(
        '--limit',
        type=int,
        metavar='N',
        required=True,
        help='number of random cards to fetch',
    )
    p.add_argument(
        '--generate-examples',
        type=int,
        metavar='N',
        required=True,
        help='number of AI-generated examples per lexeme',
    )


def _add_verna(p: configargparse.ArgParser) -> None:
    p.add_argument(
        '-l',
        '--level',
        type=CefrLevel,
        choices=CefrLevel,
        required=True,
        help='save lexemes at the specified level or higher',
    )


def get_parser(
    *,
    sections: list[Sections],
    require_db: bool = False,
    require_tg: bool = False,
) -> configargparse.ArgParser:
    config_paths = [
        Path(__file__).with_name('verna.ini').resolve(),
        Path(user_config_dir('verna')) / 'verna.ini',
        Path.home() / '.config' / 'verna' / 'verna.ini',
    ]
    p = configargparse.ArgParser(
        default_config_files=config_paths,
        config_file_parser_class=configargparse.IniConfigParser([Sections.COMMON, *sections], True),
        add_help=True,
    )
    _add_common(p)
    if Sections.DB in sections:
        _add_db(p, require_db=require_db)
    if Sections.TELEGRAM in sections:
        _add_tg(p, require_tg=require_tg)
    if Sections.RANDOM in sections:
        _add_random(p)
    if Sections.VERNA in sections:
        _add_verna(p)
    if Sections.AI in sections:
        _add_ai(p)
    return p


def print_config(cfg: argparse.Namespace) -> None:
    for k, v in vars(cfg).items():
        print(f'{k}: {v}')


def parse_config(
    *,
    sections: list[Sections],
    require_db: bool = False,
    require_tg: bool = False,
) -> argparse.Namespace:
    p = get_parser(sections=sections, require_db=require_db, require_tg=require_tg)
    cfg = p.parse_args()
    if cfg.print_config:
        print_config(cfg)
        sys.exit(0)
    return cfg
