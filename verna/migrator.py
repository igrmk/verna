#!/usr/bin/env python3

import sys
from pathlib import Path

import psycopg
from verna.config import parse_config, Sections


def ensure_migrations_table(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            create table if not exists schema_migrations (
                version text primary key,
                applied_at timestamptz not null default now()
            );
        """)
    conn.commit()


def bootstrap_initial_version(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            insert into schema_migrations (version)
            values ('001_create_cards')
            on conflict do nothing;
        """)
    conn.commit()


def get_applied_versions(conn: psycopg.Connection) -> set[str]:
    with conn.cursor() as cur:
        cur.execute('select version from schema_migrations;')
        return {row[0] for row in cur.fetchall()}


def apply_sql_file(cur: psycopg.Cursor, sql_path: Path) -> None:
    sql = sql_path.read_text(encoding='utf-8')
    cur.execute(sql)


def apply_migrations(conn: psycopg.Connection) -> None:
    migrations_dir = Path(__file__).with_name('migrations')
    ensure_migrations_table(conn)
    bootstrap_initial_version(conn)
    applied = get_applied_versions(conn)
    pending = sorted(
        (p for p in migrations_dir.glob('*.sql') if p.stem not in applied),
        key=lambda p: p.stem,
    )

    if not pending:
        print('No migrations to apply')
        return

    for path in pending:
        version = path.stem
        print(f'Applying {version} ...', flush=True)
        with conn.transaction():
            with conn.cursor() as cur:
                apply_sql_file(cur, path)
                cur.execute('insert into schema_migrations (version) values (%s);', (version,))
        print(f'Applied {version}')

    print('All pending migrations applied')


def main() -> None:
    cfg = parse_config(sections=[Sections.DB], require_db=True)
    try:
        with psycopg.connect(cfg.db_conn_string) as conn:
            apply_migrations(conn)
            print('Migration completed successfully')
    except Exception as e:
        print(f'Migration failed: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
