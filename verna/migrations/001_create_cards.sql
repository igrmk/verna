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
