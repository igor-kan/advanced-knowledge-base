%% Knowledge Base Core - Advanced Prolog Implementation
%% This module provides the foundation for a sophisticated knowledge representation system

:- module(kb_core, [
    entity/2,           % entity(ID, Type)
    property/3,         % property(Entity, Property, Value)
    relationship/5,     % relationship(ID, Type, From, To, Properties)
    add_entity/2,       % add_entity(ID, Type)
    add_relationship/5, % add_relationship(ID, Type, From, To, Properties)
    query_entities/2,   % query_entities(Type, Entities)
    query_relationships/3, % query_relationships(Type, From, To)
    infer/1,           % infer(Goal)
    kb_statistics/1     % kb_statistics(Stats)
]).

:- use_module(library(uuid)).
:- use_module(library(assoc)).
:- use_module(library(apply)).

%% Dynamic predicates for runtime knowledge modification
:- dynamic entity/2.
:- dynamic property/3.
:- dynamic relationship/5.
:- dynamic rule/2.
:- dynamic context/1.

%% =======================
%% CORE KNOWLEDGE ENTITIES
%% =======================

%% Sample entities - can be extended dynamically
entity(john_doe, person).
entity(google, company).
entity(microsoft, company).
entity(ai, field).
entity(machine_learning, field).
entity(natural_language_processing, field).
entity(project_alpha, project).
entity(project_beta, project).

%% Entity properties
property(john_doe, name, 'John Doe').
property(john_doe, age, 35).
property(john_doe, email, 'john.doe@email.com').
property(google, name, 'Google Inc.').
property(google, founded, 1998).
property(google, industry, technology).
property(ai, description, 'Artificial Intelligence').
property(machine_learning, parent_field, ai).

%% =======================
%% RELATIONSHIPS
%% =======================

%% Sophisticated relationships with properties
relationship(rel_001, works_at, john_doe, google, [
    start_date('2022-01-01'),
    role(senior_engineer),
    confidence(0.95),
    status(active)
]).

relationship(rel_002, expertise_in, john_doe, machine_learning, [
    level(expert),
    years_experience(8),
    confidence(0.9),
    certification(true)
]).

relationship(rel_003, develops, google, ai, [
    investment_level(high),
    strategic_priority(1),
    confidence(0.98)
]).

relationship(rel_004, specializes_in, machine_learning, ai, [
    relationship_type(subfield),
    confidence(1.0)
]).

%% =======================
%% INFERENCE RULES
%% =======================

%% Rule: Expert at company
expert_at_company(Person, Field, Company) :-
    relationship(_, works_at, Person, Company, WorkProps),
    relationship(_, expertise_in, Person, Field, ExpertProps),
    relationship(_, develops, Company, Field, _),
    member(status(active), WorkProps),
    member(level(expert), ExpertProps).

%% Rule: Collaborative potential
collaboration_potential(Person1, Person2, Field, Score) :-
    Person1 \= Person2,
    relationship(_, expertise_in, Person1, Field, Props1),
    relationship(_, expertise_in, Person2, Field, Props2),
    member(level(Level1), Props1),
    member(level(Level2), Props2),
    expertise_score(Level1, Score1),
    expertise_score(Level2, Score2),
    Score is (Score1 + Score2) / 2.

expertise_score(expert, 10).
expertise_score(advanced, 8).
expertise_score(intermediate, 6).
expertise_score(beginner, 3).

%% Rule: Knowledge flow
knowledge_flows(From, To, Field) :-
    relationship(_, mentors, From, To, _),
    relationship(_, expertise_in, From, Field, FromProps),
    relationship(_, learning, To, Field, ToProps),
    member(level(expert), FromProps),
    member(status(active), ToProps).

%% Rule: Technology adoption
adopts_technology(Company, Technology) :-
    relationship(_, develops, Company, Field, _),
    relationship(_, uses_technology, Field, Technology, TechProps),
    member(adoption_rate(Rate), TechProps),
    Rate > 0.7.

%% =======================
%% DYNAMIC KNOWLEDGE MANAGEMENT
%% =======================

%% Add new entity with validation
add_entity(ID, Type) :-
    \+ entity(ID, _),  % Ensure entity doesn't already exist
    assertz(entity(ID, Type)),
    format('Added entity ~w of type ~w~n', [ID, Type]).

add_entity(ID, _) :-
    entity(ID, ExistingType),
    format('Entity ~w already exists with type ~w~n', [ID, ExistingType]),
    fail.

%% Add new relationship with validation
add_relationship(ID, Type, From, To, Properties) :-
    entity(From, _),   % Ensure entities exist
    entity(To, _),
    \+ relationship(ID, _, _, _, _),  % Ensure relationship ID is unique
    assertz(relationship(ID, Type, From, To, Properties)),
    format('Added relationship ~w: ~w -> ~w~n', [Type, From, To]).

add_relationship(ID, Type, From, To, _) :-
    (\+ entity(From, _) ; \+ entity(To, _)),
    format('Error: Cannot create relationship ~w. Entity ~w or ~w does not exist~n', 
           [ID, From, To]),
    fail.

%% =======================
%% QUERYING SYSTEM
%% =======================

%% Query entities by type
query_entities(Type, Entities) :-
    findall(Entity, entity(Entity, Type), Entities).

%% Query relationships by type
query_relationships(Type, From, To) :-
    relationship(_, Type, From, To, _).

%% Advanced pattern matching
find_pattern(Pattern, Results) :-
    findall(Result, call(Pattern, Result), Results).

%% Multi-hop relationship queries
connected(From, To, Path) :-
    connected(From, To, [From], Path).

connected(From, To, Visited, [From|Path]) :-
    relationship(_, _, From, Next, _),
    \+ member(Next, Visited),
    connected(Next, To, [Next|Visited], Path).

connected(To, To, _, [To]).

%% =======================
%% CONTEXT MANAGEMENT
%% =======================

:- dynamic current_context/1.
current_context(default).

%% Context-aware facts
fact_in_context(Fact, Context) :-
    current_context(Context),
    call(Fact).

%% Switch context
switch_context(NewContext) :-
    retractall(current_context(_)),
    assertz(current_context(NewContext)),
    format('Switched to context: ~w~n', [NewContext]).

%% =======================
%% INFERENCE ENGINE
%% =======================

%% General inference predicate
infer(Goal) :-
    call(Goal).

%% Forward chaining inference
forward_chain :-
    forall(
        (rule(Condition, Conclusion), call(Condition), \+ call(Conclusion)),
        (assertz(Conclusion), format('Inferred: ~w~n', [Conclusion]))
    ).

%% Backward chaining (built into Prolog resolution)
prove(Goal) :-
    call(Goal).

%% =======================
%% STATISTICS AND ANALYTICS
%% =======================

%% Knowledge base statistics
kb_statistics(Stats) :-
    findall(_, entity(_, _), Entities),
    length(Entities, EntityCount),
    findall(_, relationship(_, _, _, _, _), Relationships),
    length(Relationships, RelationshipCount),
    findall(Type, entity(_, Type), TypesList),
    sort(TypesList, UniqueTypes),
    length(UniqueTypes, TypeCount),
    Stats = [
        entities(EntityCount),
        relationships(RelationshipCount),
        entity_types(TypeCount),
        unique_types(UniqueTypes)
    ].

%% Relationship statistics
relationship_stats(Type, Stats) :-
    findall(_, relationship(_, Type, _, _, _), Rels),
    length(Rels, Count),
    findall([From, To], relationship(_, Type, From, To, _), Pairs),
    length(Pairs, PairCount),
    Stats = [type(Type), count(Count), unique_pairs(PairCount)].

%% =======================
%% UTILITY PREDICATES
%% =======================

%% Pretty print entity
print_entity(ID) :-
    entity(ID, Type),
    format('Entity: ~w (Type: ~w)~n', [ID, Type]),
    forall(
        property(ID, Prop, Value),
        format('  ~w: ~w~n', [Prop, Value])
    ).

%% Pretty print relationship
print_relationship(ID) :-
    relationship(ID, Type, From, To, Props),
    format('Relationship ~w: ~w --[~w]--> ~w~n', [ID, From, Type, To]),
    forall(
        member(Prop, Props),
        format('  ~w~n', [Prop])
    ).

%% Validate knowledge base integrity
validate_kb :-
    format('Validating knowledge base...~n'),
    forall(
        relationship(_, _, From, To, _),
        (entity(From, _) -> true ; format('Warning: Missing entity ~w~n', [From])),
        (entity(To, _) -> true ; format('Warning: Missing entity ~w~n', [To]))
    ),
    format('Validation complete.~n').

%% =======================
%% INITIALIZATION
%% =======================

%% Initialize knowledge base
init_kb :-
    format('Initializing Advanced Prolog Knowledge Base...~n'),
    kb_statistics(Stats),
    format('Knowledge Base Statistics: ~w~n', [Stats]),
    format('Knowledge Base initialized successfully.~n').

%% Auto-initialize when loaded
:- initialization(init_kb).