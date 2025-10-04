import sys
import psycopg
from verna.config import parse_config, Sections


def ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
                create table if not exists cards (
                    id bigserial primary key,
                    lexeme text not null,
                    rp text,
                    base_form text,
                    past_simple text,
                    past_participle text,
                    translations text[] not null default '{}',
                    created_at timestamptz not null default now()
                );
            """
        )
        cur.execute('create unique index if not exists ix_cards_lexeme on cards (lower(lexeme));')
    conn.commit()


def main() -> None:
    cfg = parse_config(sections=[Sections.DB], require_db=True)
    try:
        with psycopg.connect(cfg.db_conn_string) as conn:
            ensure_schema(conn)
            print('Migration completed successfully')
    except Exception as e:
        print(f'Migration failed: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
