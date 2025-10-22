update cards set context_sentence = '{}' where context_sentence is null;

update cards set rp = '{}' where rp is null;

alter table cards
    alter column context_sentence set not null,
    alter column rp set not null;
