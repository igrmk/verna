alter table cards
alter column context_sentence type text[]
using case
    when context_sentence is null then '{}'
    else array[context_sentence]
end;
