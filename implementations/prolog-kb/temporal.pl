%% Temporal Reasoning Module - Time-Aware Knowledge Representation
%% This module provides sophisticated temporal reasoning capabilities for knowledge graphs

:- module(temporal, [
    temporal_fact/4,        % temporal_fact(Fact, StartTime, EndTime, Confidence)
    temporal_relation/5,    % temporal_relation(ID, Type, From, To, TimeInterval)
    valid_at/2,            % valid_at(Fact, Time)
    overlaps/2,            % overlaps(Interval1, Interval2)
    before/2,              % before(Interval1, Interval2)
    after/2,               % after(Interval1, Interval2)
    during/2,              % during(Interval1, Interval2)
    temporal_query/3,      % temporal_query(Pattern, Time, Results)
    temporal_evolution/3,  % temporal_evolution(Entity, Property, Timeline)
    causal_sequence/2,     % causal_sequence(Events, CausalChain)
    temporal_consistency/1 % temporal_consistency(Validation)
]).

:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(kb_core).

%% Dynamic predicates for temporal knowledge
:- dynamic temporal_fact/4.
:- dynamic temporal_relation/5.
:- dynamic temporal_event/4.
:- dynamic temporal_context/2.

%% =======================
%% TEMPORAL DATA STRUCTURES
%% =======================

%% Time intervals: interval(Start, End) where Start/End can be:
%% - specific dates: '2024-01-15'
%% - relative times: now, past, future
%% - infinity markers: '-infinity', '+infinity'
%% - null for unknown: null

%% Temporal facts with time validity
temporal_fact(works_at(john_doe, google), '2022-01-01', '2024-12-31', 0.95).
temporal_fact(works_at(john_doe, microsoft), '2020-01-01', '2021-12-31', 0.90).
temporal_fact(has_expertise(john_doe, machine_learning), '2018-01-01', '+infinity', 0.85).
temporal_fact(leads_project(alice_smith, project_alpha), '2023-06-01', '2024-06-01', 0.92).

%% Temporal relations with intervals
temporal_relation(temp_rel_001, collaboration, john_doe, alice_smith, 
    interval('2023-06-01', '2024-06-01')).
temporal_relation(temp_rel_002, mentorship, dr_smith, john_doe, 
    interval('2020-01-01', '2023-01-01')).
temporal_relation(temp_rel_003, competition, google, microsoft, 
    interval('1990-01-01', '+infinity')).

%% Temporal events for tracking changes
temporal_event(event_001, promotion, john_doe, '2023-01-15').
temporal_event(event_002, project_completion, project_alpha, '2024-06-01').
temporal_event(event_003, company_merger, microsoft_division, '2021-12-31').

%% =======================
%% TEMPORAL OPERATORS
%% =======================

%% Check if a fact is valid at a specific time
valid_at(Fact, Time) :-
    temporal_fact(Fact, Start, End, _),
    time_in_interval(Time, Start, End).

%% Time interval relationships
overlaps(interval(Start1, End1), interval(Start2, End2)) :-
    time_compare(Start1, '=<', End2),
    time_compare(Start2, '=<', End1).

before(interval(_, End1), interval(Start2, _)) :-
    time_compare(End1, '<', Start2).

after(interval(Start1, _), interval(_, End2)) :-
    time_compare(Start1, '>', End2).

during(interval(Start1, End1), interval(Start2, End2)) :-
    time_compare(Start2, '=<', Start1),
    time_compare(End1, '=<', End2).

meets(interval(_, End1), interval(Start2, _)) :-
    time_compare(End1, '=', Start2).

%% =======================
%% TIME UTILITIES
%% =======================

%% Check if time is within interval
time_in_interval(Time, Start, End) :-
    time_compare(Start, '=<', Time),
    time_compare(Time, '=<', End).

%% Compare temporal values
time_compare('+infinity', '>', _) :- !.
time_compare(_, '<', '+infinity') :- !.
time_compare('-infinity', '<', _) :- !.
time_compare(_, '>', '-infinity') :- !.
time_compare(Time1, Op, Time2) :-
    parse_time(Time1, T1),
    parse_time(Time2, T2),
    compare(Cmp, T1, T2),
    operator_mapping(Op, Cmp).

operator_mapping('<', '<').
operator_mapping('=<', '<').
operator_mapping('=<', '=').
operator_mapping('=', '=').
operator_mapping('>=', '>').
operator_mapping('>=', '=').
operator_mapping('>', '>').

%% Parse time strings to comparable format
parse_time(now, Now) :- 
    get_time(Now).
parse_time(TimeStr, Time) :-
    atom_string(TimeStr, TimeString),
    parse_time(TimeString, '%Y-%m-%d', Time).

%% =======================
%% TEMPORAL QUERIES
%% =======================

%% Query facts valid at specific time
temporal_query(Pattern, Time, Results) :-
    findall(Result, 
        (call(Pattern, Result), valid_at(Result, Time)), 
        Results).

%% Query relations active during time interval
active_relations_during(Type, Interval, Relations) :-
    findall(relation(ID, From, To), 
        (temporal_relation(ID, Type, From, To, RelInterval),
         overlaps(Interval, RelInterval)), 
        Relations).

%% Find all facts about entity at time
entity_state_at(Entity, Time, State) :-
    findall(fact(Property, Value), 
        (temporal_fact(property(Entity, Property, Value), Start, End, _),
         time_in_interval(Time, Start, End)), 
        State).

%% =======================
%% TEMPORAL EVOLUTION
%% =======================

%% Track how entity properties change over time
temporal_evolution(Entity, Property, Timeline) :-
    findall(change(Value, Start, End), 
        temporal_fact(property(Entity, Property, Value), Start, End, _), 
        Changes),
    sort(Changes, Timeline).

%% Temporal difference between two time points
entity_changes(Entity, Time1, Time2, Changes) :-
    entity_state_at(Entity, Time1, State1),
    entity_state_at(Entity, Time2, State2),
    subtract(State2, State1, Added),
    subtract(State1, State2, Removed),
    Changes = [added(Added), removed(Removed)].

%% =======================
%% CAUSAL REASONING
%% =======================

%% Define causal relationships between events
:- dynamic causal_rule/3.
causal_rule(promotion(Person), leads_to, increased_responsibility(Person)).
causal_rule(project_completion(Project), leads_to, team_available(Team)) :-
    worked_on(Team, Project).
causal_rule(company_merger(Company1, Company2), leads_to, resource_consolidation).

%% Find causal sequences
causal_sequence(Events, CausalChain) :-
    causal_sequence(Events, [], CausalChain).

causal_sequence([], Chain, Chain).
causal_sequence([Event|Rest], Acc, Chain) :-
    findall(Effect, causal_rule(Event, leads_to, Effect), Effects),
    append(Acc, [Event-Effects], NewAcc),
    causal_sequence(Rest, NewAcc, Chain).

%% Temporal causality with time constraints
temporal_causality(Cause, Effect, Evidence) :-
    temporal_event(_, Cause, _, CauseTime),
    temporal_event(_, Effect, _, EffectTime),
    time_compare(CauseTime, '<', EffectTime),
    causal_rule(Cause, leads_to, Effect),
    Evidence = [temporal_precedence(CauseTime, EffectTime), rule_based].

%% =======================
%% TEMPORAL AGGREGATION
%% =======================

%% Aggregate facts over time periods
temporal_aggregate(Entity, Property, Period, Aggregation) :-
    period_bounds(Period, Start, End),
    findall(Value, 
        (temporal_fact(property(Entity, Property, Value), S, E, _),
         overlaps(interval(S, E), interval(Start, End))), 
        Values),
    aggregate_values(Values, Aggregation).

period_bounds(year(Y), Start, End) :-
    format(atom(Start), '~w-01-01', [Y]),
    format(atom(End), '~w-12-31', [Y]).

period_bounds(month(Y, M), Start, End) :-
    format(atom(Start), '~w-~|~`0t~d~2+-01', [Y, M]),
    days_in_month(Y, M, Days),
    format(atom(End), '~w-~|~`0t~d~2+-~|~`0t~d~2+', [Y, M, Days]).

aggregate_values(Values, avg(Avg)) :-
    maplist(number, Values),
    sum_list(Values, Sum),
    length(Values, Len),
    Avg is Sum / Len.

aggregate_values(Values, count(Count)) :-
    length(Values, Count).

aggregate_values(Values, most_recent(Recent)) :-
    last(Values, Recent).

%% =======================
%% TEMPORAL PATTERNS
%% =======================

%% Find recurring patterns
recurring_pattern(Pattern, Frequency) :-
    findall(Time, 
        (temporal_event(_, Pattern, _, Time), 
         time_compare('2020-01-01', '=<', Time)), 
        Times),
    analyze_frequency(Times, Frequency).

analyze_frequency(Times, Frequency) :-
    length(Times, Count),
    (Count > 1 -> 
        sort(Times, SortedTimes),
        calculate_intervals(SortedTimes, Intervals),
        average_interval(Intervals, AvgInterval),
        Frequency = regular(AvgInterval, Count) ;
        Frequency = single(Count)).

calculate_intervals([_], []).
calculate_intervals([T1, T2|Rest], [Interval|Intervals]) :-
    parse_time(T2, PT2),
    parse_time(T1, PT1),
    Interval is PT2 - PT1,
    calculate_intervals([T2|Rest], Intervals).

average_interval(Intervals, Avg) :-
    sum_list(Intervals, Sum),
    length(Intervals, Len),
    Avg is Sum / Len.

%% =======================
%% TEMPORAL CONSISTENCY
%% =======================

%% Validate temporal consistency
temporal_consistency(ValidationResults) :-
    findall(inconsistency(Type, Details), 
        find_temporal_inconsistency(Type, Details), 
        ValidationResults).

find_temporal_inconsistency(overlapping_exclusive, Details) :-
    temporal_fact(Fact1, Start1, End1, _),
    temporal_fact(Fact2, Start2, End2, _),
    exclusive_facts(Fact1, Fact2),
    overlaps(interval(Start1, End1), interval(Start2, End2)),
    Details = [fact1(Fact1), fact2(Fact2)].

find_temporal_inconsistency(impossible_sequence, Details) :-
    temporal_event(_, Event1, _, Time1),
    temporal_event(_, Event2, _, Time2),
    requires_precedence(Event1, Event2),
    time_compare(Time2, '<', Time1),
    Details = [event1(Event1), event2(Event2), time1(Time1), time2(Time2)].

%% Define exclusive facts (can't be true simultaneously)
exclusive_facts(works_at(Person, Company1), works_at(Person, Company2)) :-
    Company1 \= Company2.

%% Define precedence requirements
requires_precedence(hired(Person, Company), works_at(Person, Company)).
requires_precedence(project_start(Project), project_completion(Project)).

%% =======================
%% TEMPORAL INDEXING
%% =======================

%% Create temporal indexes for efficient querying
:- dynamic temporal_index/3.

build_temporal_index :-
    retractall(temporal_index(_, _, _)),
    forall(
        temporal_fact(Fact, Start, End, Conf),
        (
            parse_time(Start, StartNum),
            parse_time(End, EndNum),
            assertz(temporal_index(Fact, StartNum-EndNum, Conf))
        )
    ).

%% Query using temporal index
indexed_temporal_query(Time, Facts) :-
    parse_time(Time, TimeNum),
    findall(Fact, 
        (temporal_index(Fact, Start-End, _),
         Start =< TimeNum, TimeNum =< End), 
        Facts).

%% =======================
%% UTILITY PREDICATES
%% =======================

%% Pretty print temporal information
print_temporal_fact(Fact) :-
    temporal_fact(Fact, Start, End, Confidence),
    format('~w: ~w to ~w (confidence: ~w)~n', [Fact, Start, End, Confidence]).

print_temporal_timeline(Entity) :-
    format('Timeline for ~w:~n', [Entity]),
    forall(
        temporal_fact(property(Entity, Property, Value), Start, End, _),
        format('  ~w: ~w (~w to ~w)~n', [Property, Value, Start, End])
    ).

%% Export temporal data
export_temporal_data(Format, Output) :-
    findall(temporal(Fact, Start, End, Conf), 
            temporal_fact(Fact, Start, End, Conf), 
            TemporalData),
    format_temporal_export(Format, TemporalData, Output).

format_temporal_export(json, Data, JSON) :-
    atomic_list_concat(Data, ', ', JSON).

%% =======================
%% INITIALIZATION
%% =======================

init_temporal :-
    format('Initializing temporal reasoning module...~n'),
    build_temporal_index,
    temporal_consistency(Inconsistencies),
    (Inconsistencies = [] -> 
        format('No temporal inconsistencies found.~n') ;
        format('Found temporal inconsistencies: ~w~n', [Inconsistencies])),
    format('Temporal module initialized.~n').

:- initialization(init_temporal).