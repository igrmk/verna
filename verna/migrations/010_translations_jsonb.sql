alter table cards
alter column translations type jsonb using (
    select jsonb_agg(jsonb_build_object('text', t, 'rp', rp))
    from unnest(translations) as t
);

alter table cards drop column rp;
