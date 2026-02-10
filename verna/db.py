from psycopg import Connection
from psycopg.types.json import Json

from verna.db_types import Card, Translation


def _translations_to_json(translations: list[Translation]) -> Json:
    return Json([{'text': t.text, 'rp': t.rp} for t in translations])


def _json_to_translations(data: list | None) -> list[Translation]:
    if not data:
        return []
    return [Translation(text=item['text'], rp=item.get('rp') or []) for item in data]


def save_card(conn: Connection, card: Card) -> bool:
    """Upsert a card. Returns True if inserted, False if merged with existing."""
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into cards (
                lexeme,
                past_simple,
                past_simple_rp,
                past_participle,
                past_participle_rp,
                translations,
                example
            )
            values (%s, %s, %s, %s, %s, %s, %s)
            on conflict (lower(lexeme)) do update set
                translations = (
                    select jsonb_agg(distinct_t)
                    from (
                        select distinct on ((elem->>'text')) elem as distinct_t
                        from jsonb_array_elements(cards.translations || excluded.translations) as elem
                        order by (elem->>'text')
                    ) as distinct_translations
                ),
                past_simple = coalesce(cards.past_simple, excluded.past_simple),
                past_simple_rp = (
                    select coalesce(array_agg(distinct x), '{}')
                    from unnest(cards.past_simple_rp || excluded.past_simple_rp) as x
                ),
                past_participle = coalesce(cards.past_participle, excluded.past_participle),
                past_participle_rp = (
                    select coalesce(array_agg(distinct x), '{}')
                    from unnest(cards.past_participle_rp || excluded.past_participle_rp) as x
                ),
                example = (
                    select coalesce(array_agg(distinct x), '{}')
                    from unnest(cards.example || excluded.example) as x
                )
            returning (xmax = 0) as inserted;
            """,
            (
                card.lexeme,
                card.past_simple,
                card.past_simple_rp,
                card.past_participle,
                card.past_participle_rp,
                _translations_to_json(card.translations),
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
                card.past_simple,
                card.past_simple_rp,
                card.past_participle,
                card.past_participle_rp,
                _translations_to_json(card.translations),
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
                past_simple=row[2],
                past_simple_rp=row[3] or [],
                past_participle=row[4],
                past_participle_rp=row[5] or [],
                translations=_json_to_translations(row[6]),
                example=row[7] or [],
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
                past_simple=row[1],
                past_simple_rp=row[2] or [],
                past_participle=row[3],
                past_participle_rp=row[4] or [],
                translations=_json_to_translations(row[5]),
                example=row[6] or [],
            )
            for row in rows
        ]
