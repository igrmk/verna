import sys
import psycopg
from verna import db
from verna.config import get_parser, Sections, print_config


def main() -> int:
    parser = get_parser(sections=[Sections.DB, Sections.REMOVE], require_db=True)
    parser.add_argument('lexeme', nargs='+')
    cfg = parser.parse_args()

    if cfg.print_config:
        print_config(cfg)
        return 0

    q = ' '.join(cfg.lexeme).strip()
    if not q:
        raise SystemExit('You must provide a lexeme to remove')

    try:
        with psycopg.connect(cfg.db_conn_string) as conn:
            removed = db.delete_card_by_lexeme(conn, q)
            if removed:
                print(f'Removed lexeme: {q}')
            else:
                print(f'No lexeme found matching: {q}')
    except Exception as e:
        print(f'Failed to remove lexeme: {e}')
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
