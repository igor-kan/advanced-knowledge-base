/**
 * Universal Graph Engine - The Ultimate Graph Database
 * 
 * The most complex and flexible graph database ever conceived.
 * Supports infinite graph complexity while maintaining zero-cost abstractions
 * and effortless migration to any programming language.
 * 
 * Features:
 * - Universal node types (ANY data can be a node)
 * - N-ary hypergraph relationships (unlimited participants)
 * - Meta-relationships (relationships between relationships)
 * - Temporal and causal relationship tracking
 * - Quantum graph states with entanglement
 * - Graph evolution and genetic algorithms
 * - Real-time streaming updates
 * - Zero-copy operations
 * - Language-agnostic design for easy migration
 * 
 * Copyright (c) 2025 Universal Graph Engine Project
 * Licensed under MIT License
 */

#ifndef UNIVERSAL_GRAPH_H
#define UNIVERSAL_GRAPH_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <time.h>
#include <complex.h>

/* ============================================================================
 * UNIVERSAL TYPE SYSTEM
 * Supports literally ANY data type as graph nodes
 * ============================================================================ */

typedef enum {
    /* Primitive types */
    UG_TYPE_VOID = 0,
    UG_TYPE_BOOL,
    UG_TYPE_CHAR, UG_TYPE_UCHAR,
    UG_TYPE_SHORT, UG_TYPE_USHORT,
    UG_TYPE_INT, UG_TYPE_UINT,
    UG_TYPE_LONG, UG_TYPE_ULONG,
    UG_TYPE_LLONG, UG_TYPE_ULLONG,
    UG_TYPE_FLOAT, UG_TYPE_DOUBLE, UG_TYPE_LDOUBLE,
    
    /* String types */
    UG_TYPE_STRING,
    UG_TYPE_WSTRING,
    UG_TYPE_UTF8_STRING,
    UG_TYPE_UTF16_STRING,
    UG_TYPE_UTF32_STRING,
    
    /* Container types */
    UG_TYPE_ARRAY,
    UG_TYPE_VECTOR,
    UG_TYPE_LIST,
    UG_TYPE_SET,
    UG_TYPE_MAP,
    UG_TYPE_TREE,
    UG_TYPE_GRAPH,
    
    /* Mathematical types */
    UG_TYPE_COMPLEX,
    UG_TYPE_QUATERNION,
    UG_TYPE_MATRIX,
    UG_TYPE_TENSOR,
    UG_TYPE_POLYNOMIAL,
    UG_TYPE_RATIONAL,
    UG_TYPE_BIG_INTEGER,
    UG_TYPE_BIG_DECIMAL,
    
    /* Advanced types */
    UG_TYPE_FUNCTION,
    UG_TYPE_CLOSURE,
    UG_TYPE_COROUTINE,
    UG_TYPE_STREAM,
    UG_TYPE_FUTURE,
    UG_TYPE_PROMISE,
    
    /* Scientific types */
    UG_TYPE_QUANTUM_STATE,
    UG_TYPE_PROBABILITY_DISTRIBUTION,
    UG_TYPE_STATISTICAL_MODEL,
    UG_TYPE_NEURAL_NETWORK,
    UG_TYPE_GENETIC_SEQUENCE,
    UG_TYPE_MOLECULAR_STRUCTURE,
    
    /* Media types */
    UG_TYPE_IMAGE,
    UG_TYPE_AUDIO,
    UG_TYPE_VIDEO,
    UG_TYPE_3D_MODEL,
    UG_TYPE_ANIMATION,
    UG_TYPE_TEXTURE,
    
    /* Document types */
    UG_TYPE_TEXT_DOCUMENT,
    UG_TYPE_XML_DOCUMENT,
    UG_TYPE_JSON_DOCUMENT,
    UG_TYPE_BINARY_DOCUMENT,
    UG_TYPE_STRUCTURED_DATA,
    
    /* Temporal types */
    UG_TYPE_TIMESTAMP,
    UG_TYPE_TIME_INTERVAL,
    UG_TYPE_TIME_SERIES,
    UG_TYPE_TEMPORAL_PATTERN,
    UG_TYPE_CAUSAL_EVENT,
    
    /* Semantic types */
    UG_TYPE_CONCEPT,
    UG_TYPE_ENTITY,
    UG_TYPE_RELATIONSHIP,
    UG_TYPE_PROPERTY,
    UG_TYPE_CONTEXT,
    UG_TYPE_ANNOTATION,
    
    /* Custom types */
    UG_TYPE_CUSTOM_STRUCT,
    UG_TYPE_CUSTOM_UNION,
    UG_TYPE_CUSTOM_CLASS,
    UG_TYPE_OPAQUE_POINTER,
    UG_TYPE_FOREIGN_OBJECT,
    
    /* Meta types */
    UG_TYPE_TYPE_DESCRIPTOR,
    UG_TYPE_SCHEMA,
    UG_TYPE_CONSTRAINT,
    UG_TYPE_VALIDATOR,
    
    UG_TYPE_COUNT,
    UG_TYPE_UNKNOWN = 0xFFFF
} ug_type_t;

/* Universal value that can hold ANY type */
typedef struct {
    ug_type_t type;
    size_t size;
    void* data;
    void (*destructor)(void* data);
    void* (*clone)(const void* data);
    int (*compare)(const void* a, const void* b);
    size_t (*hash)(const void* data);
    char* (*to_string)(const void* data);
} ug_universal_value_t;

/* Complex number support */
typedef struct {
    double real;
    double imag;
} ug_complex_t;

/* Quaternion support */
typedef struct {
    double w, x, y, z;
} ug_quaternion_t;

/* Matrix support */
typedef struct {
    size_t rows;
    size_t cols;
    double* data;
    bool is_sparse;
    void* sparse_data;
} ug_matrix_t;

/* Tensor support */
typedef struct {
    size_t rank;
    size_t* dimensions;
    double* data;
    char* layout; /* "row_major", "column_major", "custom" */
} ug_tensor_t;

/* Quantum state support */
typedef struct {
    size_t qubit_count;
    double complex* amplitudes;
    bool is_entangled;
    struct ug_quantum_state** entangled_with;
    size_t entanglement_count;
} ug_quantum_state_t;

/* ============================================================================
 * IDENTIFIERS AND BASIC TYPES
 * ============================================================================ */

typedef uint64_t ug_node_id_t;
typedef uint64_t ug_relationship_id_t;
typedef uint64_t ug_hyperedge_id_t;
typedef uint64_t ug_graph_id_t;
typedef uint64_t ug_stream_id_t;
typedef uint64_t ug_event_id_t;

#define UG_INVALID_ID ((uint64_t)0)
#define UG_NULL_ID ((uint64_t)0)

/* Confidence and probability values */
typedef double ug_confidence_t;    /* 0.0 to 1.0 */
typedef double ug_probability_t;   /* 0.0 to 1.0 */
typedef double ug_weight_t;        /* Can be any real number */

/* ============================================================================
 * TEMPORAL SYSTEM
 * Advanced time-aware graph capabilities
 * ============================================================================ */

typedef int64_t ug_timestamp_t;    /* Nanoseconds since epoch */
typedef struct {
    ug_timestamp_t start;
    ug_timestamp_t end;
    bool is_infinite;
    bool is_point;
} ug_time_interval_t;

typedef enum {
    UG_CAUSALITY_NONE = 0,
    UG_CAUSALITY_FORWARD,      /* A causes B */
    UG_CAUSALITY_BACKWARD,     /* B causes A */
    UG_CAUSALITY_BIDIRECTIONAL,/* A and B cause each other */
    UG_CAUSALITY_UNCERTAIN     /* Causal direction unknown */
} ug_causality_t;

typedef struct {
    ug_timestamp_t timestamp;
    ug_universal_value_t event_data;
    ug_causality_t causality;
    ug_confidence_t confidence;
    ug_relationship_id_t* affected_relationships;
    size_t affected_count;
} ug_causal_event_t;

/* Temporal validity for relationships */
typedef struct {
    ug_time_interval_t validity;
    ug_causality_t causality;
    ug_confidence_t (*confidence_function)(ug_timestamp_t t);
    void* evolution_pattern;
} ug_temporal_validity_t;

/* ============================================================================
 * PROPERTY SYSTEM
 * Unlimited key-value properties with type safety
 * ============================================================================ */

typedef struct ug_property {
    char* key;
    ug_universal_value_t value;
    ug_time_interval_t validity;
    ug_confidence_t confidence;
    struct ug_property* next;
} ug_property_t;

typedef struct {
    ug_property_t* head;
    size_t count;
    bool is_sorted;
} ug_property_map_t;

/* ============================================================================
 * NODE SYSTEM
 * Universal nodes that can contain ANY data type
 * ============================================================================ */

typedef enum {
    UG_NODE_FLAG_NONE = 0,
    UG_NODE_FLAG_IMMUTABLE = 1 << 0,
    UG_NODE_FLAG_TEMPORARY = 1 << 1,
    UG_NODE_FLAG_INDEXED = 1 << 2,
    UG_NODE_FLAG_COMPRESSED = 1 << 3,
    UG_NODE_FLAG_ENCRYPTED = 1 << 4,
    UG_NODE_FLAG_REPLICATED = 1 << 5,
    UG_NODE_FLAG_QUANTUM = 1 << 6,
    UG_NODE_FLAG_TEMPORAL = 1 << 7
} ug_node_flags_t;

typedef struct ug_node {
    ug_node_id_t id;
    ug_universal_value_t data;
    ug_property_map_t properties;
    ug_temporal_validity_t temporal;
    ug_node_flags_t flags;
    
    /* Reference counting and memory management */
    uint32_t ref_count;
    void (*destructor)(struct ug_node* node);
    
    /* Relationship tracking */
    ug_relationship_id_t* incoming_relationships;
    ug_relationship_id_t* outgoing_relationships;
    ug_hyperedge_id_t* participating_hyperedges;
    size_t incoming_count;
    size_t outgoing_count;
    size_t hyperedge_count;
    
    /* Versioning and history */
    uint64_t version;
    struct ug_node* previous_version;
    ug_timestamp_t created_at;
    ug_timestamp_t updated_at;
    
    /* Quantum support */
    ug_quantum_state_t* quantum_state;
    bool is_quantum_entangled;
    
    /* Custom behaviors */
    void (*on_access)(struct ug_node* node);
    void (*on_modify)(struct ug_node* node);
    void (*on_delete)(struct ug_node* node);
} ug_node_t;

/* ============================================================================
 * RELATIONSHIP SYSTEM
 * Support for unlimited complexity relationships
 * ============================================================================ */

typedef enum {
    UG_REL_TYPE_SIMPLE = 0,        /* Traditional edge: A -> B */
    UG_REL_TYPE_HYPEREDGE,         /* N-ary relationship: A,B,C,... -> semantics */
    UG_REL_TYPE_META,              /* Relationship about relationships */
    UG_REL_TYPE_TEMPORAL,          /* Time-evolving relationship */
    UG_REL_TYPE_QUANTUM,           /* Quantum superposition relationship */
    UG_REL_TYPE_PROBABILISTIC,     /* Uncertain relationship */
    UG_REL_TYPE_CAUSAL,            /* Causal relationship */
    UG_REL_TYPE_STREAM             /* Streaming/dynamic relationship */
} ug_relationship_type_t;

typedef enum {
    UG_DIR_UNDIRECTED = 0,
    UG_DIR_DIRECTED,
    UG_DIR_BIDIRECTIONAL,
    UG_DIR_CUSTOM
} ug_direction_t;

/* Participant in a hyperedge relationship */
typedef struct {
    ug_node_id_t node_id;
    ug_weight_t weight;
    char* role;                    /* Semantic role in relationship */
    ug_confidence_t confidence;
    ug_temporal_validity_t temporal;
} ug_participant_t;

/* Universal relationship supporting all complexity */
typedef struct ug_relationship {
    ug_relationship_id_t id;
    ug_relationship_type_t type;
    char* semantic_type;           /* Human-readable relationship type */
    ug_direction_t direction;
    
    /* Participants (2 for simple edge, N for hyperedge) */
    ug_participant_t* participants;
    size_t participant_count;
    
    /* Relationship properties */
    ug_property_map_t properties;
    ug_weight_t weight;
    ug_confidence_t confidence;
    ug_temporal_validity_t temporal;
    
    /* Meta-relationships (relationships about this relationship) */
    struct ug_relationship** meta_relationships;
    size_t meta_count;
    
    /* Quantum support */
    struct {
        bool is_quantum;
        bool is_superposition;
        struct ug_relationship** quantum_states;
        ug_probability_t* state_probabilities;
        size_t state_count;
        
        /* Quantum entanglement */
        bool is_entangled;
        struct ug_relationship** entangled_with;
        size_t entanglement_count;
    } quantum;
    
    /* Causal properties */
    struct {
        ug_causality_t causality;
        ug_causal_event_t* causal_events;
        size_t causal_event_count;
        double causal_strength;
    } causal;
    
    /* Streaming properties */
    struct {
        bool is_streaming;
        ug_stream_id_t stream_id;
        void (*update_function)(struct ug_relationship* rel, ug_timestamp_t t);
        ug_timestamp_t last_update;
    } streaming;
    
    /* Versioning */
    uint64_t version;
    struct ug_relationship* previous_version;
    ug_timestamp_t created_at;
    ug_timestamp_t updated_at;
    
    /* Reference counting */
    uint32_t ref_count;
    void (*destructor)(struct ug_relationship* rel);
} ug_relationship_t;

/* ============================================================================
 * GRAPH SYSTEM
 * Universal graph container supporting infinite complexity
 * ============================================================================ */

typedef enum {
    UG_GRAPH_TYPE_SIMPLE = 0,
    UG_GRAPH_TYPE_MULTIGRAPH,
    UG_GRAPH_TYPE_HYPERGRAPH,
    UG_GRAPH_TYPE_METAGRAPH,
    UG_GRAPH_TYPE_TEMPORAL,
    UG_GRAPH_TYPE_QUANTUM,
    UG_GRAPH_TYPE_STREAMING,
    UG_GRAPH_TYPE_DISTRIBUTED
} ug_graph_type_t;

typedef enum {
    UG_CONSISTENCY_EVENTUAL = 0,
    UG_CONSISTENCY_WEAK,
    UG_CONSISTENCY_STRONG,
    UG_CONSISTENCY_LINEARIZABLE
} ug_consistency_model_t;

typedef struct ug_graph {
    ug_graph_id_t id;
    ug_graph_type_t type;
    char* name;
    
    /* Node storage */
    ug_node_t** nodes;
    size_t node_count;
    size_t node_capacity;
    
    /* Relationship storage */
    ug_relationship_t** relationships;
    size_t relationship_count;
    size_t relationship_capacity;
    
    /* Indexing structures */
    void* node_index;              /* Hash table: id -> node */
    void* relationship_index;      /* Hash table: id -> relationship */
    void* type_indexes;            /* Hash tables: type -> [nodes] */
    void* property_indexes;        /* Hash tables: property -> [nodes] */
    void* temporal_index;          /* Time-based index */
    void* spatial_index;           /* Spatial index if needed */
    
    /* Quantum state */
    struct {
        bool is_quantum;
        ug_quantum_state_t* graph_quantum_state;
        bool supports_superposition;
        bool supports_entanglement;
    } quantum;
    
    /* Temporal state */
    struct {
        bool is_temporal;
        ug_timestamp_t creation_time;
        ug_timestamp_t current_time;
        bool supports_time_travel;
        void* temporal_index;
    } temporal;
    
    /* Streaming state */
    struct {
        bool is_streaming;
        void* stream_processors;
        void* change_listeners;
        size_t listener_count;
    } streaming;
    
    /* Distributed state */
    struct {
        bool is_distributed;
        void* shard_manager;
        ug_consistency_model_t consistency_model;
        size_t replica_count;
    } distributed;
    
    /* Evolution and genetics */
    struct {
        bool supports_evolution;
        void* genetic_algorithms;
        double mutation_rate;
        void (*fitness_function)(struct ug_graph* graph);
        uint64_t generation;
    } evolution;
    
    /* Schema and constraints */
    void* schema;
    void* constraints;
    void* validators;
    
    /* Statistics and monitoring */
    struct {
        uint64_t operations_count;
        uint64_t queries_count;
        uint64_t mutations_count;
        double average_query_time;
        size_t memory_usage;
    } stats;
    
    /* Memory management */
    void* allocator;
    void* memory_pool;
    
    /* Thread safety */
    void* mutex;
    bool is_thread_safe;
    
    /* Versioning */
    uint64_t version;
    struct ug_graph* previous_version;
} ug_graph_t;

/* ============================================================================
 * QUERY SYSTEM
 * Universal query language supporting infinite complexity
 * ============================================================================ */

typedef enum {
    UG_QUERY_TYPE_PATTERN = 0,
    UG_QUERY_TYPE_TRAVERSAL,
    UG_QUERY_TYPE_TEMPORAL,
    UG_QUERY_TYPE_QUANTUM,
    UG_QUERY_TYPE_CAUSAL,
    UG_QUERY_TYPE_STREAM,
    UG_QUERY_TYPE_GENETIC,
    UG_QUERY_TYPE_SQL_LIKE,
    UG_QUERY_TYPE_SPARQL_LIKE,
    UG_QUERY_TYPE_CYPHER_LIKE,
    UG_QUERY_TYPE_CUSTOM
} ug_query_type_t;

typedef struct {
    ug_query_type_t type;
    char* query_string;
    void* compiled_query;
    void* parameters;
    
    /* Query constraints */
    size_t max_results;
    ug_timestamp_t timeout;
    ug_confidence_t min_confidence;
    
    /* Temporal constraints */
    ug_time_interval_t time_range;
    bool include_deleted;
    
    /* Quantum constraints */
    bool collapse_superposition;
    double observation_strength;
} ug_query_t;

typedef struct {
    ug_node_id_t* nodes;
    ug_relationship_id_t* relationships;
    size_t node_count;
    size_t relationship_count;
    
    /* Result metadata */
    double execution_time;
    size_t total_matches;
    bool is_partial;
    
    /* Confidence and probability */
    ug_confidence_t* node_confidences;
    ug_confidence_t* relationship_confidences;
    ug_probability_t result_probability;
} ug_query_result_t;

/* ============================================================================
 * CORE API FUNCTIONS
 * ============================================================================ */

/* Graph management */
ug_graph_t* ug_create_graph(void);
ug_graph_t* ug_create_graph_with_type(ug_graph_type_t type);
void ug_destroy_graph(ug_graph_t* graph);
ug_graph_t* ug_clone_graph(const ug_graph_t* graph);

/* Node operations */
ug_node_id_t ug_create_node(ug_graph_t* graph, ug_type_t type, const void* data);
ug_node_id_t ug_create_node_with_properties(ug_graph_t* graph, ug_type_t type, 
                                           const void* data, ug_property_map_t* properties);
ug_node_t* ug_get_node(ug_graph_t* graph, ug_node_id_t id);
bool ug_delete_node(ug_graph_t* graph, ug_node_id_t id);
bool ug_update_node(ug_graph_t* graph, ug_node_id_t id, const void* new_data);

/* Simple relationship operations */
ug_relationship_id_t ug_create_edge(ug_graph_t* graph, ug_node_id_t from, 
                                   ug_node_id_t to, const char* type, ug_weight_t weight);
ug_relationship_t* ug_get_relationship(ug_graph_t* graph, ug_relationship_id_t id);
bool ug_delete_relationship(ug_graph_t* graph, ug_relationship_id_t id);

/* Hyperedge operations */
ug_relationship_id_t ug_create_hyperedge(ug_graph_t* graph, ug_node_id_t* participants, 
                                        size_t count, const char* type);
ug_relationship_id_t ug_create_weighted_hyperedge(ug_graph_t* graph, 
                                                 ug_participant_t* participants, 
                                                 size_t count, const char* type);

/* Meta-relationship operations */
ug_relationship_id_t ug_create_meta_relationship(ug_graph_t* graph, 
                                                ug_relationship_id_t subject_rel,
                                                ug_relationship_id_t object_rel,
                                                const char* type);

/* Temporal operations */
ug_relationship_id_t ug_create_temporal_relationship(ug_graph_t* graph,
                                                   ug_node_id_t from, ug_node_id_t to,
                                                   const char* type,
                                                   ug_temporal_validity_t* temporal);
bool ug_add_causal_event(ug_graph_t* graph, ug_relationship_id_t rel_id,
                        ug_causal_event_t* event);

/* Quantum operations */
ug_relationship_id_t ug_create_quantum_relationship(ug_graph_t* graph,
                                                   ug_node_id_t from, ug_node_id_t to,
                                                   ug_relationship_t** quantum_states,
                                                   ug_probability_t* probabilities,
                                                   size_t state_count);
bool ug_entangle_relationships(ug_graph_t* graph, ug_relationship_id_t rel1,
                              ug_relationship_id_t rel2);
ug_relationship_t* ug_observe_quantum_relationship(ug_graph_t* graph,
                                                  ug_relationship_id_t rel_id);

/* Property operations */
bool ug_set_node_property(ug_graph_t* graph, ug_node_id_t id, 
                         const char* key, ug_universal_value_t* value);
ug_universal_value_t* ug_get_node_property(ug_graph_t* graph, ug_node_id_t id,
                                          const char* key);
bool ug_remove_node_property(ug_graph_t* graph, ug_node_id_t id, const char* key);

bool ug_set_relationship_property(ug_graph_t* graph, ug_relationship_id_t id,
                                 const char* key, ug_universal_value_t* value);
ug_universal_value_t* ug_get_relationship_property(ug_graph_t* graph, 
                                                  ug_relationship_id_t id, const char* key);

/* Query operations */
ug_query_result_t* ug_query(ug_graph_t* graph, const char* query_string);
ug_query_result_t* ug_query_with_params(ug_graph_t* graph, ug_query_t* query);
void ug_free_query_result(ug_query_result_t* result);

/* Traversal operations */
ug_node_id_t* ug_get_neighbors(ug_graph_t* graph, ug_node_id_t id, size_t* count);
ug_node_id_t* ug_bfs_traversal(ug_graph_t* graph, ug_node_id_t start, 
                              size_t max_depth, size_t* count);
ug_node_id_t* ug_dfs_traversal(ug_graph_t* graph, ug_node_id_t start,
                              size_t max_depth, size_t* count);
ug_node_id_t* ug_shortest_path(ug_graph_t* graph, ug_node_id_t from,
                              ug_node_id_t to, size_t* path_length);

/* Evolution operations */
bool ug_enable_evolution(ug_graph_t* graph, double mutation_rate,
                        void (*fitness_function)(ug_graph_t*));
bool ug_evolve_graph(ug_graph_t* graph, size_t generations);
ug_graph_t* ug_breed_graphs(ug_graph_t* parent1, ug_graph_t* parent2);

/* Streaming operations */
ug_stream_id_t ug_create_stream(ug_graph_t* graph, 
                               void (*callback)(ug_graph_t*, void*), void* user_data);
bool ug_destroy_stream(ug_graph_t* graph, ug_stream_id_t stream_id);

/* Utility functions */
size_t ug_get_node_count(ug_graph_t* graph);
size_t ug_get_relationship_count(ug_graph_t* graph);
void ug_print_graph_stats(ug_graph_t* graph);

/* Memory management utilities */
ug_universal_value_t* ug_create_value(ug_type_t type, const void* data, size_t size);
void ug_destroy_value(ug_universal_value_t* value);
ug_universal_value_t* ug_clone_value(const ug_universal_value_t* value);

/* Type conversion utilities */
bool ug_value_to_string(const ug_universal_value_t* value, char** result);
bool ug_string_to_value(const char* str, ug_type_t type, ug_universal_value_t* result);

/* Serialization */
bool ug_export_graph(ug_graph_t* graph, const char* format, const char* filename);
ug_graph_t* ug_import_graph(const char* format, const char* filename);

/* Thread safety */
bool ug_make_thread_safe(ug_graph_t* graph);
void ug_lock_graph(ug_graph_t* graph);
void ug_unlock_graph(ug_graph_t* graph);

#ifdef __cplusplus
}
#endif

#endif /* UNIVERSAL_GRAPH_H */