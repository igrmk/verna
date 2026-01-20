from psycopg import Connection

from verna.db_types import Card


def save_card(conn: Connection, card: Card) -> bool:
    """Upsert a card. Returns True if inserted, False if merged with existing."""
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into cards (
                lexeme,
                rp,
                past_simple,
                past_simple_rp,
                past_participle,
                past_participle_rp,
                translations,
                example
            )
            values (%s, %s, %s, %s, %s, %s, %s, %s)
            on conflict (lower(lexeme)) do update set
                translations = (
                    select coalesce(array_agg(distinct x), '{}')
                    from unnest(cards.translations || excluded.translations) as x
                    where x is not null
                ),
                rp = (
                    select coalesce(array_agg(distinct x), '{}')
                    from unnest(cards.rp || excluded.rp) as x
                    where x is not null
                ),
                past_simple = coalesce(cards.past_simple, excluded.past_simple),
                past_simple_rp = (
                    select coalesce(array_agg(distinct x), '{}')
                    from unnest(cards.past_simple_rp || excluded.past_simple_rp) as x
                    where x is not null
                ),
                past_participle = coalesce(cards.past_participle, excluded.past_participle),
                past_participle_rp = (
                    select coalesce(array_agg(distinct x), '{}')
                    from unnest(cards.past_participle_rp || excluded.past_participle_rp) as x
                    where x is not null
                ),
                example = (
                    select coalesce(array_agg(distinct x), '{}')
                    from unnest(cards.example || excluded.example) as x
                    where x is not null
                )
            returning (xmax = 0) as inserted;
            """,
            (
                card.lexeme,
                card.rp,
                card.past_simple,
                card.past_simple_rp,
                card.past_participle,
                card.past_participle_rp,
                card.translations,
                card.example,
            ),
        )
        row = cur.fetchone()
        assert row is not None
        inserted: bool = row[0]
    conn.commit()
    return inserted


def update_card(conn: Connection, card_id: int, card: Card) -> None:
    """Update an existing card by ID."""
    with conn.cursor() as cur:
        cur.execute(
            """
            update cards set
                lexeme = %s,
                rp = %s,
                past_simple = %s,
                past_simple_rp = %s,
                past_participle = %s,
                past_participle_rp = %s,
                translations = %s,
                example = %s
            where id = %s
            """,
            (
                card.lexeme,
                card.rp,
                card.past_simple,
                card.past_simple_rp,
                card.past_participle,
                card.past_participle_rp,
                card.translations,
                card.example,
                card_id,
            ),
        )
    conn.commit()


def delete_card_by_id(conn: Connection, card_id: int) -> None:
    """Delete a card by ID."""
    with conn.cursor() as cur:
        cur.execute('delete from cards where id = %s', (card_id,))
    conn.commit()


def search_cards(conn: Connection, query: str, limit: int = 50) -> list[tuple[int, Card]]:
    """Search cards by lexeme with ILIKE. Returns list of (id, card) tuples."""
    with conn.cursor() as cur:
        if query:
            cur.execute(
                """
                select
                    id,
                    lexeme,
                    rp,
                    past_simple,
                    past_simple_rp,
                    past_participle,
                    past_participle_rp,
                    translations,
                    example
                from cards
                where lexeme ilike %s
                order by lexeme
                limit %s
                """,
                (f'%{query}%', limit),
            )
        else:
            cur.execute(
                """
                select
                    id,
                    lexeme,
                    rp,
                    past_simple,
                    past_simple_rp,
                    past_participle,
                    past_participle_rp,
                    translations,
                    example
                from cards
                order by lexeme
                limit %s
                """,
                (limit,),
            )
        rows = cur.fetchall()

    return [
        (
            row[0],
            Card(
                lexeme=row[1],
                rp=row[2] or [],
                past_simple=row[3],
                past_simple_rp=row[4] or [],
                past_participle=row[5],
                past_participle_rp=row[6] or [],
                translations=row[7] or [],
                example=row[8] or [],
            ),
        )
        for row in rows
    ]


def fetch_random_cards(conn: Connection, limit: int) -> list[Card]:
    """Fetch random cards up to the specified limit."""
    with conn.cursor() as cur:
        cur.execute(
            """
            select
                lexeme,
                rp,
                past_simple,
                past_simple_rp,
                past_participle,
                past_participle_rp,
                translations,
                example
            from cards
            order by random()
            limit %s;
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [
            Card(
                lexeme=row[0],
                rp=row[1] or [],
                past_simple=row[2],
                past_simple_rp=row[3] or [],
                past_participle=row[4],
                past_participle_rp=row[5] or [],
                translations=row[6] or [],
                example=row[7] or [],
            )
            for row in rows
        ]
