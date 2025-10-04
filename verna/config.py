import sys
import argparse
from pathlib import Path
import configargparse
from enum import StrEnum
from platformdirs import user_config_dir


class Sections(StrEnum):
    DB = 'db'
    TELEGRAM = 'telegram'
    RANDOM = 'random'
    VERNA = 'verna'
    REMOVE = 'remove'


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


def _add_db(p: configargparse.ArgParser, *, require_db: bool) -> None:
    p.add_argument(
        '--db-conn-string',
        type=str,
        required=require_db,
        metavar='STR',
        env_var='DB_CONN_STRING',
        help='PostgreSQL connection string',
    )


def _add_telegram(p: configargparse.ArgParser) -> None:
    p.add_argument(
        '--send-to-tg',
        env_var='SEND_TO_TG',
        action=argparse.BooleanOptionalAction,
        help='send output to Telegram',
    )
    p.add_argument(
        '--tg-bot-token',
        env_var='TG_BOT_TOKEN',
        help='Telegram bot token',
    )
    p.add_argument(
        '--tg-chat-id',
        env_var='TG_CHAT_ID',
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


def _add_verna(p: configargparse.ArgParser) -> None:
    p.add_argument(
        '--show-schema',
        action=argparse.BooleanOptionalAction,
        help='show JSON schema and exit',
        default=False,
    )


def get_parser(
    *,
    sections: list[Sections],
    require_db: bool = False,
) -> configargparse.ArgParser:
    config_paths = [
        Path(__file__).with_name('appsettings.ini').resolve(),
        Path(user_config_dir('verna')) / 'verna.ini',
        Path.home() / '.config' / 'verna' / 'verna.ini',
        'appsettings.dev.ignore.ini',
    ]
    p = configargparse.ArgParser(
        default_config_files=config_paths,
        config_file_parser_class=configargparse.IniConfigParser(sections, True),
        add_help=True,
    )
    _add_common(p)
    if Sections.DB in sections:
        _add_db(p, require_db=require_db)
    if Sections.TELEGRAM in sections:
        _add_telegram(p)
    if Sections.RANDOM in sections:
        _add_random(p)
    if Sections.VERNA in sections:
        _add_verna(p)
    return p


def print_config(cfg: argparse.Namespace) -> None:
    for k, v in vars(cfg).items():
        print(f'{k}: {v}')


def parse_config(
    *,
    sections: list[Sections],
    require_db: bool = False,
) -> argparse.Namespace:
    p = get_parser(sections=sections, require_db=require_db)
    cfg = p.parse_args()
    if cfg.print_config:
        print_config(cfg)
        sys.exit(0)
    return cfg
