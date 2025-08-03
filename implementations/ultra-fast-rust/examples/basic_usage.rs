//! Basic usage example for the ultra-fast knowledge graph
//!
//! This example demonstrates:
//! - Creating nodes and edges
//! - Basic queries and traversals
//! - Pattern matching
//! - Performance monitoring

use ultra_fast_knowledge_graph::*;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Ultra-Fast Knowledge Graph - Basic Usage Example");
    println!("===================================================");
    
    // Create a new knowledge graph with optimized configuration
    let config = GraphConfig {
        initial_node_capacity: 10_000,
        initial_edge_capacity: 50_000,
        enable_simd: true,
        enable_gpu: false, // Set to true if CUDA is available
        thread_pool_size: None, // Use all available cores
        memory_limit_gb: Some(4),
    };
    
    let graph = UltraFastKnowledgeGraph::new(config)?;
    println!("✅ Knowledge graph created successfully");
    
    // Example 1: Create nodes representing people and organizations
    println!("\n📊 Creating nodes...");
    
    let alice_data = NodeData::new(
        "Alice Johnson".to_string(),
        serde_json::json!({
            "type": "Person",
            "age": 30,
            "occupation": "Data Scientist",
            "location": "San Francisco"
        })
    );
    
    let bob_data = NodeData::new(
        "Bob Smith".to_string(),
        serde_json::json!({
            "type": "Person", 
            "age": 35,
            "occupation": "Software Engineer",
            "location": "New York"
        })
    );
    
    let company_data = NodeData::new(
        "TechCorp Inc".to_string(),
        serde_json::json!({
            "type": "Organization",
            "industry": "Technology",
            "size": "Large",
            "founded": 2010
        })
    );
    
    let alice_id = graph.create_node(alice_data)?;
    let bob_id = graph.create_node(bob_data)?;
    let company_id = graph.create_node(company_data)?;
    
    println!("  👤 Alice: Node ID {}", alice_id);
    println!("  👤 Bob: Node ID {}", bob_id);
    println!("  🏢 TechCorp: Node ID {}", company_id);
    
    // Example 2: Create relationships between entities
    println!("\n🔗 Creating edges...");
    
    let alice_works_at = EdgeData::new(serde_json::json!({
        "type": "WORKS_AT",
        "start_date": "2022-01-15",
        "position": "Senior Data Scientist",
        "salary_range": "high"
    }));
    
    let bob_works_at = EdgeData::new(serde_json::json!({
        "type": "WORKS_AT", 
        "start_date": "2020-06-01",
        "position": "Principal Engineer",
        "salary_range": "high"
    }));
    
    let colleague_relationship = EdgeData::new(serde_json::json!({
        "type": "COLLEAGUE",
        "relationship_strength": 0.8,
        "collaboration_projects": ["ProjectX", "ProjectY"]
    }));
    
    let edge1 = graph.create_edge(alice_id, company_id, Weight(1.0), alice_works_at)?;
    let edge2 = graph.create_edge(bob_id, company_id, Weight(1.0), bob_works_at)?;
    let edge3 = graph.create_edge(alice_id, bob_id, Weight(0.8), colleague_relationship)?;
    
    println!("  🔗 Alice → TechCorp: Edge ID {}", edge1);
    println!("  🔗 Bob → TechCorp: Edge ID {}", edge2);
    println!("  🔗 Alice ↔ Bob: Edge ID {}", edge3);
    
    // Example 3: Batch operations for better performance
    println!("\n⚡ Demonstrating batch operations...");
    
    let batch_nodes: Vec<NodeData> = (0..1000).map(|i| {
        NodeData::new(
            format!("Employee_{}", i),
            serde_json::json!({
                "type": "Person",
                "employee_id": i,
                "department": match i % 4 {
                    0 => "Engineering",
                    1 => "Marketing", 
                    2 => "Sales",
                    _ => "Operations"
                },
                "level": i % 5 + 1
            })
        )
    }).collect();
    
    let start_time = std::time::Instant::now();
    let batch_node_ids = graph.batch_create_nodes(batch_nodes)?;
    let batch_duration = start_time.elapsed();
    
    println!("  📊 Created {} nodes in {:?}", batch_node_ids.len(), batch_duration);
    println!("  ⚡ Throughput: {:.0} nodes/second", 
             batch_node_ids.len() as f64 / batch_duration.as_secs_f64());
    
    // Create batch edges (reporting relationships)
    let batch_edges: Vec<(NodeId, NodeId, Weight, EdgeData)> = batch_node_ids
        .chunks(10)
        .enumerate()
        .flat_map(|(manager_idx, chunk)| {
            let manager_id = batch_node_ids[manager_idx * 10];
            chunk.iter().skip(1).map(move |&employee_id| {
                let edge_data = EdgeData::new(serde_json::json!({
                    "type": "REPORTS_TO",
                    "relationship": "manager"
                }));
                (employee_id, manager_id, Weight(1.0), edge_data)
            })
        })
        .collect();
    
    let start_time = std::time::Instant::now();
    let batch_edge_ids = graph.batch_create_edges(batch_edges)?;
    let edge_batch_duration = start_time.elapsed();
    
    println!("  📊 Created {} edges in {:?}", batch_edge_ids.len(), edge_batch_duration);
    println!("  ⚡ Throughput: {:.0} edges/second",
             batch_edge_ids.len() as f64 / edge_batch_duration.as_secs_f64());
    
    // Example 4: Graph traversal
    println!("\n🌐 Graph traversal examples...");
    
    let bfs_result = graph.traverse_bfs(alice_id, Some(3))?;
    println!("  🔍 BFS from Alice (depth 3): {} nodes visited in {:?}",
             bfs_result.nodes_visited, bfs_result.duration);
    
    let shortest_path = graph.shortest_path(alice_id, bob_id)?;
    if let Some(path) = shortest_path {
        println!("  🛤️  Shortest path Alice → Bob: {} steps, weight: {:.2}",
                 path.length, path.total_weight);
    }
    
    let neighborhood = graph.get_neighborhood(company_id, 2)?;
    println!("  🏘️  Company's 2-hop neighborhood: {} nodes", neighborhood.len());
    
    // Example 5: Centrality analysis
    println!("\n📈 Centrality analysis...");
    
    let degree_centrality = graph.compute_centrality(CentralityAlgorithm::Degree)?;
    println!("  📊 Top 5 nodes by degree centrality:");
    for (i, (node_id, centrality)) in degree_centrality.iter().take(5).enumerate() {
        println!("    {}. Node {}: {:.4}", i + 1, node_id, centrality);
    }
    
    let pagerank = graph.compute_centrality(CentralityAlgorithm::PageRank)?;
    println!("  📊 Top 5 nodes by PageRank:");
    for (i, (node_id, pr_score)) in pagerank.iter().take(5).enumerate() {
        println!("    {}. Node {}: {:.6}", i + 1, node_id, pr_score);
    }
    
    // Example 6: Pattern matching
    println!("\n🔍 Pattern matching examples...");
    
    // Pattern: Person → Organization
    let employment_pattern = Pattern {
        nodes: vec![
            PatternNode {
                id: "person".to_string(),
                type_filter: Some("Person".to_string()),
                property_filters: HashMap::new(),
            },
            PatternNode {
                id: "org".to_string(),
                type_filter: Some("Organization".to_string()),
                property_filters: HashMap::new(),
            },
        ],
        edges: vec![
            PatternEdge {
                from: "person".to_string(),
                to: "org".to_string(),
                type_filter: Some("WORKS_AT".to_string()),
                direction: EdgeDirection::Outgoing,
                weight_range: None,
            },
        ],
        constraints: PatternConstraints {
            max_results: Some(10),
            timeout: Some(std::time::Duration::from_secs(5)),
            min_confidence: Some(0.8),
        },
    };
    
    let start_time = std::time::Instant::now();
    let employment_matches = graph.find_pattern(&employment_pattern)?;
    let pattern_duration = start_time.elapsed();
    
    println!("  🎯 Employment pattern matches: {} found in {:?}",
             employment_matches.len(), pattern_duration);
    
    for (i, pattern_match) in employment_matches.iter().take(3).enumerate() {
        println!("    Match {}: Person {} → Org {}, Score: {:.3}",
                 i + 1,
                 pattern_match.node_bindings.get("person").unwrap_or(&0),
                 pattern_match.node_bindings.get("org").unwrap_or(&0),
                 pattern_match.score);
    }
    
    // Example 7: Hyperedge creation (N-ary relationships)
    println!("\n🕸️  Creating hyperedges...");
    
    let project_team_hyperedge = HyperedgeData::new(serde_json::json!({
        "type": "PROJECT_TEAM",
        "project_name": "AI Initiative",
        "start_date": "2024-01-01",
        "budget": 500000,
        "status": "active"
    }));
    
    let team_members = vec![alice_id, bob_id, batch_node_ids[0], batch_node_ids[1]];
    let hyperedge_id = graph.create_hyperedge(team_members, project_team_hyperedge)?;
    
    println!("  🕸️  Project team hyperedge created: ID {}", hyperedge_id);
    
    // Example 8: Performance monitoring
    println!("\n📊 Performance statistics...");
    
    let stats = graph.get_statistics();
    println!("  📈 Graph Statistics:");
    println!("    • Nodes: {}", stats.node_count);
    println!("    • Edges: {}", stats.edge_count);
    println!("    • Hyperedges: {}", stats.hyperedge_count);
    println!("    • Memory usage: {:.2} MB", stats.memory_usage.total as f64 / 1024.0 / 1024.0);
    println!("    • CSR compression ratio: {:.2}x", stats.csr_compression_ratio);
    
    let memory_usage = graph.get_memory_usage();
    println!("  💾 Memory Breakdown:");
    println!("    • Nodes: {:.2} MB", memory_usage.nodes as f64 / 1024.0 / 1024.0);
    println!("    • Edges: {:.2} MB", memory_usage.edges as f64 / 1024.0 / 1024.0);
    println!("    • CSR (outgoing): {:.2} MB", memory_usage.outgoing_csr as f64 / 1024.0 / 1024.0);
    println!("    • CSR (incoming): {:.2} MB", memory_usage.incoming_csr as f64 / 1024.0 / 1024.0);
    println!("    • Indices: {:.2} MB", memory_usage.indices as f64 / 1024.0 / 1024.0);
    
    // Example 9: Storage optimization
    println!("\n🗜️  Storage optimization...");
    
    let pre_optimization_memory = graph.get_memory_usage().total;
    let start_time = std::time::Instant::now();
    graph.optimize_storage()?;
    let optimization_duration = start_time.elapsed();
    let post_optimization_memory = graph.get_memory_usage().total;
    
    let memory_saved = pre_optimization_memory.saturating_sub(post_optimization_memory);
    let savings_percent = (memory_saved as f64 / pre_optimization_memory as f64) * 100.0;
    
    println!("  🗜️  Optimization completed in {:?}", optimization_duration);
    println!("  💾 Memory saved: {:.2} MB ({:.1}%)",
             memory_saved as f64 / 1024.0 / 1024.0, savings_percent);
    
    // Example 10: Advanced queries
    println!("\n🧠 Advanced query examples...");
    
    // Find all managers (people with incoming REPORTS_TO edges)
    let manager_pattern = Pattern {
        nodes: vec![
            PatternNode {
                id: "manager".to_string(),
                type_filter: Some("Person".to_string()),
                property_filters: HashMap::new(),
            },
            PatternNode {
                id: "employee".to_string(),
                type_filter: Some("Person".to_string()),
                property_filters: HashMap::new(),
            },
        ],
        edges: vec![
            PatternEdge {
                from: "employee".to_string(),
                to: "manager".to_string(),
                type_filter: Some("REPORTS_TO".to_string()),
                direction: EdgeDirection::Outgoing,
                weight_range: None,
            },
        ],
        constraints: PatternConstraints {
            max_results: Some(50),
            timeout: Some(std::time::Duration::from_secs(2)),
            min_confidence: Some(0.9),
        },
    };
    
    let manager_matches = graph.find_pattern(&manager_pattern)?;
    println!("  👥 Found {} manager-employee relationships", manager_matches.len());
    
    // Real-time performance monitoring
    println!("\n⚡ Performance monitoring...");
    
    let performance_start = std::time::Instant::now();
    
    // Simulate some operations
    for i in 0..100 {
        let node_data = NodeData::new(
            format!("temp_node_{}", i),
            serde_json::json!({"temp": true, "index": i})
        );
        graph.create_node(node_data)?;
    }
    
    let operations_duration = performance_start.elapsed();
    println!("  ⚡ 100 node creations: {:?} ({:.0} ops/sec)",
             operations_duration, 100.0 / operations_duration.as_secs_f64());
    
    // Final statistics
    let final_stats = graph.get_statistics();
    println!("\n🎉 Final Results:");
    println!("  📊 Total nodes: {}", final_stats.node_count);
    println!("  📊 Total edges: {}", final_stats.edge_count);
    println!("  📊 Total hyperedges: {}", final_stats.hyperedge_count);
    println!("  💾 Total memory: {:.2} MB", final_stats.memory_usage.total as f64 / 1024.0 / 1024.0);
    
    println!("\n✅ Ultra-Fast Knowledge Graph example completed successfully!");
    println!("   Visit https://github.com/igor-kan/advanced-knowledge-base for more examples");
    
    Ok(())
}

/// Helper function to demonstrate error handling
fn _demonstrate_error_handling() -> GraphResult<()> {
    let config = GraphConfig::default();
    let graph = UltraFastKnowledgeGraph::new(config)?;
    
    // This will return an error for non-existent node
    match graph.shortest_path(999999, 999998) {
        Ok(path) => println!("Found path: {:?}", path),
        Err(GraphError::NodeNotFound(node_id)) => {
            println!("Node {} not found - this is expected!", node_id);
        }
        Err(e) => {
            println!("Unexpected error: {:?}", e);
        }
    }
    
    Ok(())
}

/// Helper function to demonstrate SIMD operations
fn _demonstrate_simd_operations() -> GraphResult<()> {
    use ultra_fast_knowledge_graph::simd::SimdProcessor;
    
    let processor = SimdProcessor::new();
    
    // Distance update example
    let mut distances = vec![f32::INFINITY; 1000];
    let new_distances: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let mask = vec![true; 1000];
    
    let start_time = std::time::Instant::now();
    let updates = processor.simd_distance_update(&mut distances, &new_distances, &mask)?;
    let duration = start_time.elapsed();
    
    println!("SIMD distance update: {} updates in {:?}", updates, duration);
    
    Ok(())
}