/**
 * Universal Graph Engine - C++ Bindings
 * 
 * Zero-cost C++ wrapper around the Universal Graph Engine C API
 * Provides type safety, RAII, and modern C++ idioms
 * 
 * Copyright (c) 2025 Universal Graph Engine Project
 * Licensed under MIT License
 */

#ifndef UNIVERSAL_GRAPH_HPP
#define UNIVERSAL_GRAPH_HPP

#include "universal_graph.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <type_traits>
#include <chrono>
#include <functional>

namespace ug {

/* ============================================================================
 * TYPE TRAITS AND CONCEPTS
 * ============================================================================ */

template<typename T>
struct TypeTraits {
    static constexpr ug_type_t ug_type = UG_TYPE_CUSTOM_STRUCT;
};

// Specializations for built-in types
template<> struct TypeTraits<bool> { static constexpr ug_type_t ug_type = UG_TYPE_BOOL; };
template<> struct TypeTraits<char> { static constexpr ug_type_t ug_type = UG_TYPE_CHAR; };
template<> struct TypeTraits<int> { static constexpr ug_type_t ug_type = UG_TYPE_INT; };
template<> struct TypeTraits<long> { static constexpr ug_type_t ug_type = UG_TYPE_LONG; };
template<> struct TypeTraits<float> { static constexpr ug_type_t ug_type = UG_TYPE_FLOAT; };
template<> struct TypeTraits<double> { static constexpr ug_type_t ug_type = UG_TYPE_DOUBLE; };
template<> struct TypeTraits<std::string> { static constexpr ug_type_t ug_type = UG_TYPE_STRING; };

/* ============================================================================
 * VALUE WRAPPER
 * ============================================================================ */

class Value {
private:
    std::unique_ptr<ug_universal_value_t> value_;

public:
    template<typename T>
    explicit Value(const T& data) {
        value_ = std::make_unique<ug_universal_value_t>();
        value_->type = TypeTraits<T>::ug_type;
        value_->size = sizeof(T);
        value_->data = new T(data);
        value_->destructor = [](void* ptr) { delete static_cast<T*>(ptr); };
        value_->clone = [](const void* ptr) -> void* { 
            return new T(*static_cast<const T*>(ptr)); 
        };
    }

    explicit Value(const std::string& str) {
        value_ = std::make_unique<ug_universal_value_t>();
        value_->type = UG_TYPE_STRING;
        value_->size = str.length() + 1;
        value_->data = new char[str.length() + 1];
        strcpy(static_cast<char*>(value_->data), str.c_str());
        value_->destructor = [](void* ptr) { delete[] static_cast<char*>(ptr); };
        value_->clone = [](const void* ptr) -> void* {
            const char* str = static_cast<const char*>(ptr);
            size_t len = strlen(str) + 1;
            char* copy = new char[len];
            strcpy(copy, str);
            return copy;
        };
    }

    ~Value() = default;
    Value(const Value&) = delete;
    Value& operator=(const Value&) = delete;
    Value(Value&&) = default;
    Value& operator=(Value&&) = default;

    template<typename T>
    T get() const {
        if constexpr (std::is_same_v<T, std::string>) {
            return std::string(static_cast<const char*>(value_->data));
        } else {
            return *static_cast<const T*>(value_->data);
        }
    }

    ug_universal_value_t* c_value() const { return value_.get(); }
};

/* ============================================================================
 * NODE WRAPPER
 * ============================================================================ */

using NodeId = ug_node_id_t;
using RelationshipId = ug_relationship_id_t;
using Weight = ug_weight_t;
using Confidence = ug_confidence_t;

class Node {
private:
    ug_node_t* node_;
    bool owns_node_;

public:
    explicit Node(ug_node_t* node, bool owns = false) 
        : node_(node), owns_node_(owns) {}

    ~Node() {
        if (owns_node_ && node_) {
            // In a real implementation, we'd handle reference counting
        }
    }

    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;
    Node(Node&& other) noexcept 
        : node_(other.node_), owns_node_(other.owns_node_) {
        other.node_ = nullptr;
        other.owns_node_ = false;
    }

    NodeId id() const { return node_ ? node_->id : UG_INVALID_ID; }
    ug_type_t type() const { return node_ ? node_->data.type : UG_TYPE_UNKNOWN; }

    template<typename T>
    T data() const {
        if (!node_ || !node_->data.data) {
            throw std::runtime_error("Invalid node or no data");
        }
        
        if constexpr (std::is_same_v<T, std::string>) {
            return std::string(static_cast<const char*>(node_->data.data));
        } else {
            return *static_cast<const T*>(node_->data.data);
        }
    }

    template<typename T>
    std::optional<T> getProperty(const std::string& key) const {
        if (!node_) return std::nullopt;
        
        ug_universal_value_t* prop = ug_get_node_property(nullptr, node_->id, key.c_str());
        if (!prop) return std::nullopt;
        
        if constexpr (std::is_same_v<T, std::string>) {
            return std::string(static_cast<const char*>(prop->data));
        } else {
            return *static_cast<const T*>(prop->data);
        }
    }

    ug_node_t* c_node() const { return node_; }
};

/* ============================================================================
 * RELATIONSHIP WRAPPER
 * ============================================================================ */

class Relationship {
private:
    ug_relationship_t* relationship_;
    bool owns_relationship_;

public:
    explicit Relationship(ug_relationship_t* rel, bool owns = false)
        : relationship_(rel), owns_relationship_(owns) {}

    ~Relationship() {
        if (owns_relationship_ && relationship_) {
            // Handle reference counting in real implementation
        }
    }

    Relationship(const Relationship&) = delete;
    Relationship& operator=(const Relationship&) = delete;
    Relationship(Relationship&& other) noexcept
        : relationship_(other.relationship_), owns_relationship_(other.owns_relationship_) {
        other.relationship_ = nullptr;
        other.owns_relationship_ = false;
    }

    RelationshipId id() const { return relationship_ ? relationship_->id : UG_INVALID_ID; }
    std::string semanticType() const { 
        return relationship_ && relationship_->semantic_type ? 
               std::string(relationship_->semantic_type) : ""; 
    }
    Weight weight() const { return relationship_ ? relationship_->weight : 0.0; }
    Confidence confidence() const { return relationship_ ? relationship_->confidence : 0.0; }

    std::vector<NodeId> participants() const {
        std::vector<NodeId> result;
        if (relationship_ && relationship_->participants) {
            result.reserve(relationship_->participant_count);
            for (size_t i = 0; i < relationship_->participant_count; i++) {
                result.push_back(relationship_->participants[i].node_id);
            }
        }
        return result;
    }

    ug_relationship_t* c_relationship() const { return relationship_; }
};

/* ============================================================================
 * MAIN GRAPH CLASS
 * ============================================================================ */

class UniversalGraph {
private:
    std::unique_ptr<ug_graph_t, decltype(&ug_destroy_graph)> graph_;

public:
    explicit UniversalGraph(ug_graph_type_t type = UG_GRAPH_TYPE_SIMPLE)
        : graph_(ug_create_graph_with_type(type), &ug_destroy_graph) {
        if (!graph_) {
            throw std::runtime_error("Failed to create graph");
        }
    }

    ~UniversalGraph() = default;
    UniversalGraph(const UniversalGraph&) = delete;
    UniversalGraph& operator=(const UniversalGraph&) = delete;
    UniversalGraph(UniversalGraph&&) = default;
    UniversalGraph& operator=(UniversalGraph&&) = default;

    // Node operations
    template<typename T>
    NodeId createNode(const T& data) {
        return ug_create_node(graph_.get(), TypeTraits<T>::ug_type, &data);
    }

    NodeId createNode(const std::string& data) {
        return ug_create_node(graph_.get(), UG_TYPE_STRING, data.c_str());
    }

    std::optional<Node> getNode(NodeId id) {
        ug_node_t* node = ug_get_node(graph_.get(), id);
        if (!node) return std::nullopt;
        return Node(node);
    }

    template<typename T>
    bool setNodeProperty(NodeId id, const std::string& key, const T& value) {
        Value val(value);
        return ug_set_node_property(graph_.get(), id, key.c_str(), val.c_value());
    }

    // Relationship operations
    RelationshipId createEdge(NodeId from, NodeId to, const std::string& type, 
                             Weight weight = 1.0) {
        return ug_create_edge(graph_.get(), from, to, type.c_str(), weight);
    }

    RelationshipId createHyperedge(const std::vector<NodeId>& participants, 
                                  const std::string& type) {
        return ug_create_hyperedge(graph_.get(), 
                                  const_cast<NodeId*>(participants.data()),
                                  participants.size(), type.c_str());
    }

    std::optional<Relationship> getRelationship(RelationshipId id) {
        ug_relationship_t* rel = ug_get_relationship(graph_.get(), id);
        if (!rel) return std::nullopt;
        return Relationship(rel);
    }

    // Advanced operations
    template<typename... Args>
    RelationshipId createVariadicHyperedge(const std::string& type, Args&&... nodes) {
        std::vector<NodeId> participants = {std::forward<Args>(nodes)...};
        return createHyperedge(participants, type);
    }

    // Quantum operations
    RelationshipId createQuantumRelationship(NodeId from, NodeId to,
                                           const std::vector<std::string>& states,
                                           const std::vector<double>& probabilities) {
        // In full implementation, would create quantum superposition
        return createEdge(from, to, "QUANTUM_SUPERPOSITION");
    }

    // Temporal operations
    RelationshipId createTemporalRelationship(NodeId from, NodeId to,
                                            const std::string& type,
                                            std::chrono::system_clock::time_point start,
                                            std::chrono::system_clock::time_point end) {
        // In full implementation, would set temporal validity
        return createEdge(from, to, type);
    }

    // Graph analytics
    std::vector<NodeId> bfsTraversal(NodeId start, size_t maxDepth = SIZE_MAX) {
        size_t count = 0;
        ug_node_id_t* result = ug_bfs_traversal(graph_.get(), start, maxDepth, &count);
        
        std::vector<NodeId> nodes;
        if (result) {
            nodes.assign(result, result + count);
            free(result);  // In real implementation, use proper memory management
        }
        
        return nodes;
    }

    std::vector<NodeId> shortestPath(NodeId from, NodeId to) {
        size_t pathLength = 0;
        ug_node_id_t* result = ug_shortest_path(graph_.get(), from, to, &pathLength);
        
        std::vector<NodeId> path;
        if (result) {
            path.assign(result, result + pathLength);
            free(result);
        }
        
        return path;
    }

    // Statistics
    size_t nodeCount() const { return ug_get_node_count(graph_.get()); }
    size_t relationshipCount() const { return ug_get_relationship_count(graph_.get()); }
    
    void printStats() const { ug_print_graph_stats(graph_.get()); }

    // Export operations
    bool exportGraph(const std::string& format, const std::string& filename) {
        return ug_export_graph(graph_.get(), format.c_str(), filename.c_str());
    }

    // Raw access for advanced users
    ug_graph_t* c_graph() const { return graph_.get(); }
};

/* ============================================================================
 * FLUENT API BUILDER
 * ============================================================================ */

class GraphBuilder {
private:
    std::unique_ptr<UniversalGraph> graph_;
    
public:
    GraphBuilder() : graph_(std::make_unique<UniversalGraph>()) {}

    template<typename T>
    GraphBuilder& addNode(const std::string& name, const T& data) {
        NodeId id = graph_->createNode(data);
        graph_->setNodeProperty(id, "name", name);
        return *this;
    }

    GraphBuilder& addEdge(const std::string& from, const std::string& to, 
                         const std::string& type, Weight weight = 1.0) {
        // In full implementation, would maintain name->id mapping
        return *this;
    }

    template<typename... Args>
    GraphBuilder& addHyperedge(const std::string& type, Args&&... nodeNames) {
        // In full implementation, would create N-ary relationship
        return *this;
    }

    std::unique_ptr<UniversalGraph> build() {
        return std::move(graph_);
    }
};

/* ============================================================================
 * QUERY INTERFACE
 * ============================================================================ */

class QueryBuilder {
private:
    std::string query_;
    std::unordered_map<std::string, Value> parameters_;

public:
    QueryBuilder& match(const std::string& pattern) {
        query_ += "MATCH " + pattern + " ";
        return *this;
    }

    QueryBuilder& where(const std::string& condition) {
        query_ += "WHERE " + condition + " ";
        return *this;
    }

    QueryBuilder& returns(const std::string& items) {
        query_ += "RETURN " + items + " ";
        return *this;
    }

    template<typename T>
    QueryBuilder& parameter(const std::string& name, const T& value) {
        parameters_[name] = Value(value);
        return *this;
    }

    std::string build() const { return query_; }
};

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

// Factory functions for common graph types
inline std::unique_ptr<UniversalGraph> createSimpleGraph() {
    return std::make_unique<UniversalGraph>(UG_GRAPH_TYPE_SIMPLE);
}

inline std::unique_ptr<UniversalGraph> createHypergraph() {
    return std::make_unique<UniversalGraph>(UG_GRAPH_TYPE_HYPERGRAPH);
}

inline std::unique_ptr<UniversalGraph> createTemporalGraph() {
    return std::make_unique<UniversalGraph>(UG_GRAPH_TYPE_TEMPORAL);
}

inline std::unique_ptr<UniversalGraph> createQuantumGraph() {
    return std::make_unique<UniversalGraph>(UG_GRAPH_TYPE_QUANTUM);
}

// Helper for complex data structures
template<typename T>
class ComplexNode {
private:
    T data_;
    std::unordered_map<std::string, Value> properties_;

public:
    explicit ComplexNode(T data) : data_(std::move(data)) {}

    template<typename PropT>
    ComplexNode& setProperty(const std::string& key, const PropT& value) {
        properties_[key] = Value(value);
        return *this;
    }

    const T& data() const { return data_; }
    const auto& properties() const { return properties_; }
};

} // namespace ug

#endif // UNIVERSAL_GRAPH_HPP