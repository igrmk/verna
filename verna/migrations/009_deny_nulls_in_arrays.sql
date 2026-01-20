update cards set
    rp = array_remove(rp, null),
    past_simple_rp = array_remove(past_simple_rp, null),
    past_participle_rp = array_remove(past_participle_rp, null),
    translations = array_remove(translations, null),
    example = array_remove(example, null);

alter table cards add constraint cards_rp_no_nulls check (array_position(rp, null) is null);
alter table cards add constraint cards_past_simple_rp_no_nulls check (array_position(past_simple_rp, null) is null);
alter table cards add constraint cards_past_participle_rp_no_nulls check (array_position(past_participle_rp, null) is null);
alter table cards add constraint cards_translations_no_nulls check (array_position(translations, null) is null);
alter table cards add constraint cards_example_no_nulls check (array_position(example, null) is null);
