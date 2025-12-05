import sys
import psycopg
from verna.config import get_parser, Sections, print_config


def remove_lexeme(conn, lexeme: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            delete from cards
            where lower(lexeme) = lower(%s)
            returning id;
            """,
            (lexeme,),
        )
        row = cur.fetchone()
    conn.commit()
    return row is not None


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
            removed = remove_lexeme(conn, q)
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
