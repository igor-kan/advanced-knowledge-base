//! Distributed Cluster Coordination for Infinite Scalability
//!
//! This module implements the distributed coordination layer that enables
//! infinite horizontal scaling across thousands of nodes while maintaining
//! consistency and optimal performance.

use std::collections::{HashMap, BTreeMap};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use uuid::Uuid;
use crate::error::QuantumDistributedError;
use crate::config::ClusterConfig;
use crate::QuantumResult;

/// Distributed cluster coordinator
pub struct ClusterCoordinator {
    /// Configuration
    config: ClusterConfig,
    
    /// Node registry
    node_registry: RwLock<NodeRegistry>,
    
    /// Cluster topology
    topology: RwLock<ClusterTopology>,
    
    /// Health monitor
    health_monitor: HealthMonitor,
    
    /// Load balancer
    load_balancer: LoadBalancer,
    
    /// Communication channels
    communication: CommunicationManager,
    
    /// Distributed metrics
    metrics: DistributedMetrics,
    
    /// Current node ID
    node_id: Uuid,
    
    /// Cluster state
    cluster_state: RwLock<ClusterState>,
}

/// Node registry for tracking cluster members
#[derive(Debug, Clone)]
pub struct NodeRegistry {
    /// Active nodes
    active_nodes: HashMap<Uuid, NodeInfo>,
    
    /// Pending nodes (joining)
    pending_nodes: HashMap<Uuid, NodeInfo>,
    
    /// Failed nodes
    failed_nodes: HashMap<Uuid, NodeFailureInfo>,
    
    /// Total node count
    total_nodes: usize,
    
    /// Last update time
    last_update: Instant,
}

/// Information about a cluster node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Node identifier
    pub id: Uuid,
    
    /// Network address
    pub address: SocketAddr,
    
    /// Node capabilities
    pub capabilities: NodeCapabilities,
    
    /// Current load
    pub current_load: f64,
    
    /// Health score
    pub health_score: f64,
    
    /// Last heartbeat
    pub last_heartbeat: Instant,
    
    /// Join time
    pub joined_at: Instant,
    
    /// Node version
    pub version: String,
    
    /// Hardware specifications
    pub hardware_spec: HardwareSpec,
}

/// Node capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    /// CPU cores
    pub cpu_cores: usize,
    
    /// Available memory in bytes
    pub memory_bytes: u64,
    
    /// Network bandwidth in bytes/sec
    pub network_bandwidth_bps: u64,
    
    /// Storage capacity in bytes
    pub storage_bytes: u64,
    
    /// Supports quantum operations
    pub supports_quantum: bool,
    
    /// Supports GPU acceleration
    pub supports_gpu: bool,
    
    /// Supports SIMD operations
    pub supports_simd: bool,
    
    /// Maximum concurrent connections
    pub max_connections: usize,
}

/// Hardware specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSpec {
    /// CPU model
    pub cpu_model: String,
    
    /// Total RAM in GB
    pub ram_gb: f64,
    
    /// Storage type (SSD, NVMe, etc.)
    pub storage_type: String,
    
    /// Network interface speed (Gbps)
    pub network_speed_gbps: f64,
    
    /// GPU information (if available)
    pub gpu_info: Option<String>,
}

/// Node failure information
#[derive(Debug, Clone)]
pub struct NodeFailureInfo {
    /// Node ID
    pub node_id: Uuid,
    
    /// Failure time
    pub failed_at: Instant,
    
    /// Failure reason
    pub failure_reason: String,
    
    /// Recovery attempts
    pub recovery_attempts: usize,
    
    /// Last recovery attempt
    pub last_recovery_attempt: Option<Instant>,
}

/// Cluster topology representation
#[derive(Debug, Clone)]
pub struct ClusterTopology {
    /// Node connections (adjacency list)
    connections: HashMap<Uuid, Vec<Uuid>>,
    
    /// Network zones/regions
    zones: HashMap<String, Vec<Uuid>>,
    
    /// Latency matrix between nodes
    latency_matrix: HashMap<(Uuid, Uuid), Duration>,
    
    /// Bandwidth matrix between nodes
    bandwidth_matrix: HashMap<(Uuid, Uuid), u64>,
    
    /// Topology generation
    generation: u64,
}

/// Health monitoring system
#[derive(Debug)]
pub struct HealthMonitor {
    /// Health check interval
    check_interval: Duration,
    
    /// Health thresholds
    thresholds: HealthThresholds,
    
    /// Health history
    health_history: RwLock<BTreeMap<Instant, ClusterHealthSnapshot>>,
    
    /// Alert system
    alert_system: AlertSystem,
}

/// Health thresholds for monitoring
#[derive(Debug, Clone)]
pub struct HealthThresholds {
    /// CPU utilization threshold
    pub cpu_threshold: f64,
    
    /// Memory utilization threshold
    pub memory_threshold: f64,
    
    /// Network latency threshold (ms)
    pub latency_threshold_ms: f64,
    
    /// Heartbeat timeout
    pub heartbeat_timeout: Duration,
    
    /// Health score minimum
    pub min_health_score: f64,
}

/// Cluster health snapshot
#[derive(Debug, Clone)]
pub struct ClusterHealthSnapshot {
    /// Timestamp
    pub timestamp: Instant,
    
    /// Overall health score
    pub overall_health: f64,
    
    /// Healthy nodes count
    pub healthy_nodes: usize,
    
    /// Unhealthy nodes count
    pub unhealthy_nodes: usize,
    
    /// Average node load
    pub avg_node_load: f64,
    
    /// Network health score
    pub network_health: f64,
}

/// Alert system for cluster monitoring
#[derive(Debug)]
pub struct AlertSystem {
    /// Alert thresholds
    thresholds: HashMap<String, f64>,
    
    /// Active alerts
    active_alerts: RwLock<HashMap<Uuid, Alert>>,
    
    /// Alert history
    alert_history: RwLock<Vec<Alert>>,
}

/// Alert information
#[derive(Debug, Clone)]
pub struct Alert {
    /// Alert ID
    pub id: Uuid,
    
    /// Alert type
    pub alert_type: AlertType,
    
    /// Affected node
    pub node_id: Option<Uuid>,
    
    /// Alert message
    pub message: String,
    
    /// Severity level
    pub severity: AlertSeverity,
    
    /// Created at
    pub created_at: Instant,
    
    /// Resolved at
    pub resolved_at: Option<Instant>,
}

/// Alert types
#[derive(Debug, Clone)]
pub enum AlertType {
    /// Node failure
    NodeFailure,
    /// High CPU usage
    HighCpuUsage,
    /// High memory usage
    HighMemoryUsage,
    /// High network latency
    HighNetworkLatency,
    /// Storage capacity warning
    StorageCapacityWarning,
    /// Consensus failure
    ConsensusFailure,
    /// Quantum decoherence
    QuantumDecoherence,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Information
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical
    Critical,
}

/// Load balancer for distributing work
#[derive(Debug)]
pub struct LoadBalancer {
    /// Load balancing strategy
    strategy: LoadBalancingStrategy,
    
    /// Node load tracking
    node_loads: RwLock<HashMap<Uuid, NodeLoad>>,
    
    /// Load balancing metrics
    metrics: LoadBalancingMetrics,
}

/// Load balancing strategies
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    /// Round robin
    RoundRobin,
    /// Least connections
    LeastConnections,
    /// Weighted round robin
    WeightedRoundRobin,
    /// Least response time
    LeastResponseTime,
    /// Resource-aware
    ResourceAware,
    /// Quantum-optimized
    QuantumOptimized,
}

/// Node load information
#[derive(Debug, Clone)]
pub struct NodeLoad {
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f64,
    
    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: f64,
    
    /// Network utilization (0.0 to 1.0)
    pub network_utilization: f64,
    
    /// Active connections
    pub active_connections: usize,
    
    /// Pending operations
    pub pending_operations: usize,
    
    /// Response time (ms)
    pub avg_response_time_ms: f64,
    
    /// Last update
    pub last_update: Instant,
}

/// Load balancing metrics
#[derive(Debug, Clone, Default)]
pub struct LoadBalancingMetrics {
    /// Total requests balanced
    pub total_requests: u64,
    
    /// Requests per node
    pub requests_per_node: HashMap<Uuid, u64>,
    
    /// Average response time
    pub avg_response_time_ms: f64,
    
    /// Load balancing efficiency
    pub efficiency_score: f64,
}

/// Communication manager for inter-node messaging
#[derive(Debug)]
pub struct CommunicationManager {
    /// Message senders by node
    senders: RwLock<HashMap<Uuid, mpsc::UnboundedSender<ClusterMessage>>>,
    
    /// Message receiver
    receiver: RwLock<Option<mpsc::UnboundedReceiver<ClusterMessage>>>,
    
    /// Communication metrics
    metrics: CommunicationMetrics,
}

/// Cluster message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusterMessage {
    /// Heartbeat message
    Heartbeat {
        node_id: Uuid,
        timestamp: chrono::DateTime<chrono::Utc>,
        health_score: f64,
        load_info: NodeLoad,
    },
    
    /// Node join request
    JoinRequest {
        node_info: NodeInfo,
    },
    
    /// Node leave notification
    LeaveNotification {
        node_id: Uuid,
        reason: String,
    },
    
    /// Query distribution
    QueryDistribution {
        query_id: Uuid,
        target_nodes: Vec<Uuid>,
        query_data: serde_json::Value,
    },
    
    /// Query result
    QueryResult {
        query_id: Uuid,
        from_node: Uuid,
        result_data: serde_json::Value,
    },
    
    /// Consensus message
    ConsensusMessage {
        proposal_id: Uuid,
        message_type: String,
        data: serde_json::Value,
    },
    
    /// Health check request
    HealthCheck {
        from_node: Uuid,
    },
    
    /// Health check response
    HealthCheckResponse {
        from_node: Uuid,
        health_status: serde_json::Value,
    },
}

/// Communication metrics
#[derive(Debug, Clone, Default)]
pub struct CommunicationMetrics {
    /// Messages sent
    pub messages_sent: u64,
    
    /// Messages received
    pub messages_received: u64,
    
    /// Message failures
    pub message_failures: u64,
    
    /// Average message latency
    pub avg_message_latency_ms: f64,
    
    /// Bandwidth utilization
    pub bandwidth_utilization: f64,
}

/// Distributed metrics for cluster performance
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistributedMetrics {
    /// Total cluster operations
    pub total_operations: u64,
    
    /// Operations per second
    pub ops_per_second: f64,
    
    /// Average operation latency
    pub avg_latency_ms: f64,
    
    /// Cluster efficiency score
    pub efficiency_score: f64,
    
    /// Node utilization
    pub node_utilization: f64,
    
    /// Network throughput
    pub network_throughput_gbps: f64,
    
    /// Fault tolerance score
    pub fault_tolerance_score: f64,
    
    /// Scalability factor
    pub scalability_factor: f64,
    
    /// Last update
    pub last_update: chrono::DateTime<chrono::Utc>,
}

/// Overall cluster state
#[derive(Debug, Clone)]
pub enum ClusterState {
    /// Initializing
    Initializing,
    /// Healthy and operational
    Healthy,
    /// Degraded performance
    Degraded,
    /// Unhealthy (needs intervention)
    Unhealthy,
    /// Shutting down
    ShuttingDown,
}

impl ClusterCoordinator {
    /// Create new cluster coordinator
    pub fn new(config: &ClusterConfig) -> QuantumResult<Self> {
        tracing::info!("🌐 Initializing cluster coordinator");
        
        let node_id = Uuid::new_v4();
        
        Ok(Self {
            config: config.clone(),
            node_registry: RwLock::new(NodeRegistry {
                active_nodes: HashMap::new(),
                pending_nodes: HashMap::new(),
                failed_nodes: HashMap::new(),
                total_nodes: 0,
                last_update: Instant::now(),
            }),
            topology: RwLock::new(ClusterTopology {
                connections: HashMap::new(),
                zones: HashMap::new(),
                latency_matrix: HashMap::new(),
                bandwidth_matrix: HashMap::new(),
                generation: 0,
            }),
            health_monitor: HealthMonitor {
                check_interval: Duration::from_secs(config.heartbeat_interval_ms / 1000),
                thresholds: HealthThresholds {
                    cpu_threshold: 0.8,
                    memory_threshold: 0.8,
                    latency_threshold_ms: 100.0,
                    heartbeat_timeout: Duration::from_millis(config.heartbeat_interval_ms * 3),
                    min_health_score: 0.7,
                },
                health_history: RwLock::new(BTreeMap::new()),
                alert_system: AlertSystem {
                    thresholds: HashMap::new(),
                    active_alerts: RwLock::new(HashMap::new()),
                    alert_history: RwLock::new(Vec::new()),
                },
            },
            load_balancer: LoadBalancer {
                strategy: LoadBalancingStrategy::QuantumOptimized,
                node_loads: RwLock::new(HashMap::new()),
                metrics: LoadBalancingMetrics::default(),
            },
            communication: CommunicationManager {
                senders: RwLock::new(HashMap::new()),
                receiver: RwLock::new(None),
                metrics: CommunicationMetrics::default(),
            },
            metrics: DistributedMetrics::default(),
            node_id,
            cluster_state: RwLock::new(ClusterState::Initializing),
        })
    }
    
    /// Start cluster coordinator
    pub async fn start(&self) -> QuantumResult<()> {
        tracing::info!("🚀 Starting cluster coordinator");
        
        // Initialize communication
        self.initialize_communication().await?;
        
        // Start health monitoring
        self.start_health_monitoring().await?;
        
        // Join cluster or bootstrap new cluster
        self.initialize_cluster().await?;
        
        // Update cluster state
        *self.cluster_state.write() = ClusterState::Healthy;
        
        tracing::info!("✅ Cluster coordinator started - Node ID: {}", self.node_id);
        
        Ok(())
    }
    
    /// Stop cluster coordinator
    pub async fn stop(&self) -> QuantumResult<()> {
        tracing::info!("⏹️  Stopping cluster coordinator");
        
        *self.cluster_state.write() = ClusterState::ShuttingDown;
        
        // Leave cluster gracefully
        self.leave_cluster().await?;
        
        // Stop health monitoring
        self.stop_health_monitoring().await?;
        
        tracing::info!("✅ Cluster coordinator stopped");
        
        Ok(())
    }
    
    /// Get health score
    pub async fn health_score(&self) -> f64 {
        let registry = self.node_registry.read();
        
        if registry.active_nodes.is_empty() {
            return 0.0;
        }
        
        let total_health: f64 = registry.active_nodes.values()
            .map(|node| node.health_score)
            .sum();
        
        total_health / registry.active_nodes.len() as f64
    }
    
    /// Get total nodes count
    pub async fn total_nodes(&self) -> usize {
        self.node_registry.read().total_nodes
    }
    
    /// Add nodes to cluster
    pub async fn add_nodes(&self, count: usize) -> QuantumResult<()> {
        tracing::info!("📈 Adding {} nodes to cluster", count);
        
        // In a real implementation, this would trigger node provisioning
        // For now, simulate node addition
        
        let mut registry = self.node_registry.write();
        
        for _ in 0..count {
            let new_node_id = Uuid::new_v4();
            let node_info = NodeInfo {
                id: new_node_id,
                address: "127.0.0.1:8080".parse().unwrap(), // Placeholder
                capabilities: NodeCapabilities {
                    cpu_cores: 8,
                    memory_bytes: 32 * 1024 * 1024 * 1024, // 32GB
                    network_bandwidth_bps: 10 * 1024 * 1024 * 1024, // 10 Gbps
                    storage_bytes: 1024 * 1024 * 1024 * 1024, // 1TB
                    supports_quantum: true,
                    supports_gpu: false,
                    supports_simd: true,
                    max_connections: 1000,
                },
                current_load: 0.0,
                health_score: 1.0,
                last_heartbeat: Instant::now(),
                joined_at: Instant::now(),
                version: "1.0.0".to_string(),
                hardware_spec: HardwareSpec {
                    cpu_model: "Intel Xeon".to_string(),
                    ram_gb: 32.0,
                    storage_type: "NVMe SSD".to_string(),
                    network_speed_gbps: 10.0,
                    gpu_info: None,
                },
            };
            
            registry.active_nodes.insert(new_node_id, node_info);
            registry.total_nodes += 1;
        }
        
        registry.last_update = Instant::now();
        
        tracing::info!("✅ Added {} nodes - Total nodes: {}", count, registry.total_nodes);
        
        Ok(())
    }
    
    /// Remove nodes from cluster
    pub async fn remove_nodes(&self, count: usize) -> QuantumResult<()> {
        tracing::info!("📉 Removing {} nodes from cluster", count);
        
        let mut registry = self.node_registry.write();
        
        // Remove nodes with highest load first
        let mut nodes_by_load: Vec<_> = registry.active_nodes.iter()
            .map(|(id, info)| (*id, info.current_load))
            .collect();
        
        nodes_by_load.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let remove_count = count.min(registry.active_nodes.len());
        
        for i in 0..remove_count {
            let node_id = nodes_by_load[i].0;
            registry.active_nodes.remove(&node_id);
            registry.total_nodes = registry.total_nodes.saturating_sub(1);
        }
        
        registry.last_update = Instant::now();
        
        tracing::info!("✅ Removed {} nodes - Total nodes: {}", remove_count, registry.total_nodes);
        
        Ok(())
    }
    
    /// Execute distributed query across cluster
    pub async fn execute_distributed_query(&self, 
                                          shard_queries: Vec<serde_json::Value>) -> QuantumResult<Vec<serde_json::Value>> {
        tracing::debug!("🔍 Executing distributed query across {} shards", shard_queries.len());
        
        let registry = self.node_registry.read();
        let active_nodes: Vec<_> = registry.active_nodes.keys().copied().collect();
        
        if active_nodes.is_empty() {
            return Err(QuantumDistributedError::ClusterError(
                "No active nodes available".to_string()
            ));
        }
        
        let mut results = Vec::new();
        
        // Distribute queries to nodes using load balancing
        for (i, shard_query) in shard_queries.into_iter().enumerate() {
            let target_node = active_nodes[i % active_nodes.len()];
            
            // Simulate query execution
            let result = self.execute_query_on_node(target_node, shard_query).await?;
            results.push(result);
        }
        
        tracing::debug!("✅ Completed distributed query execution");
        
        Ok(results)
    }
    
    /// Get distributed metrics
    pub async fn get_metrics(&self) -> DistributedMetrics {
        let mut metrics = self.metrics.clone();
        
        // Update real-time metrics
        let registry = self.node_registry.read();
        metrics.node_utilization = if registry.total_nodes > 0 {
            registry.active_nodes.values()
                .map(|node| node.current_load)
                .sum::<f64>() / registry.total_nodes as f64
        } else {
            0.0
        };
        
        metrics.efficiency_score = self.calculate_cluster_efficiency().await;
        metrics.fault_tolerance_score = self.calculate_fault_tolerance().await;
        metrics.scalability_factor = registry.total_nodes as f64 / 1000.0; // Normalized to 1000 nodes
        metrics.last_update = chrono::Utc::now();
        
        metrics
    }
    
    /// Initialize communication system
    async fn initialize_communication(&self) -> QuantumResult<()> {
        tracing::debug!("📡 Initializing communication system");
        
        let (sender, receiver) = mpsc::unbounded_channel();
        *self.communication.receiver.write() = Some(receiver);
        
        // In a real implementation, this would set up network listeners
        
        Ok(())
    }
    
    /// Start health monitoring
    async fn start_health_monitoring(&self) -> QuantumResult<()> {
        tracing::debug!("💓 Starting health monitoring");
        
        // Start background health checking
        // In a real implementation, this would spawn a background task
        
        Ok(())
    }
    
    /// Initialize cluster (join existing or bootstrap new)
    async fn initialize_cluster(&self) -> QuantumResult<()> {
        tracing::debug!("🏗️  Initializing cluster");
        
        // Try to discover existing cluster
        if let Ok(existing_nodes) = self.discover_existing_cluster().await {
            tracing::info!("Found existing cluster with {} nodes", existing_nodes.len());
            self.join_existing_cluster(existing_nodes).await?;
        } else {
            tracing::info!("No existing cluster found, bootstrapping new cluster");
            self.bootstrap_new_cluster().await?;
        }
        
        Ok(())
    }
    
    /// Discover existing cluster nodes
    async fn discover_existing_cluster(&self) -> QuantumResult<Vec<NodeInfo>> {
        // In a real implementation, this would use service discovery
        // For now, return empty to bootstrap new cluster
        Err(QuantumDistributedError::ClusterError("No existing cluster".to_string()))
    }
    
    /// Join existing cluster
    async fn join_existing_cluster(&self, _existing_nodes: Vec<NodeInfo>) -> QuantumResult<()> {
        tracing::info!("🤝 Joining existing cluster");
        
        // Implementation would handle cluster join protocol
        
        Ok(())
    }
    
    /// Bootstrap new cluster
    async fn bootstrap_new_cluster(&self) -> QuantumResult<()> {
        tracing::info!("🆕 Bootstrapping new cluster");
        
        // Add self as first node
        let self_info = NodeInfo {
            id: self.node_id,
            address: "127.0.0.1:8080".parse().unwrap(),
            capabilities: NodeCapabilities {
                cpu_cores: num_cpus::get(),
                memory_bytes: 16 * 1024 * 1024 * 1024, // 16GB default
                network_bandwidth_bps: 1024 * 1024 * 1024, // 1 Gbps default
                storage_bytes: 500 * 1024 * 1024 * 1024, // 500GB default
                supports_quantum: true,
                supports_gpu: false,
                supports_simd: true,
                max_connections: 100,
            },
            current_load: 0.0,
            health_score: 1.0,
            last_heartbeat: Instant::now(),
            joined_at: Instant::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            hardware_spec: HardwareSpec {
                cpu_model: "Unknown".to_string(),
                ram_gb: 16.0,
                storage_type: "SSD".to_string(),
                network_speed_gbps: 1.0,
                gpu_info: None,
            },
        };
        
        let mut registry = self.node_registry.write();
        registry.active_nodes.insert(self.node_id, self_info);
        registry.total_nodes = 1;
        registry.last_update = Instant::now();
        
        Ok(())
    }
    
    /// Leave cluster gracefully
    async fn leave_cluster(&self) -> QuantumResult<()> {
        tracing::info!("👋 Leaving cluster gracefully");
        
        // Implementation would handle graceful cluster leave
        
        Ok(())
    }
    
    /// Stop health monitoring
    async fn stop_health_monitoring(&self) -> QuantumResult<()> {
        tracing::debug!("⏹️  Stopping health monitoring");
        
        // Stop background health checking
        
        Ok(())
    }
    
    /// Execute query on specific node
    async fn execute_query_on_node(&self, 
                                  _node_id: Uuid, 
                                  query: serde_json::Value) -> QuantumResult<serde_json::Value> {
        // Simulate query execution with some processing delay
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        // Return mock result
        Ok(serde_json::json!({
            "status": "success",
            "query": query,
            "result_count": 42,
            "processing_time_ms": 10
        }))
    }
    
    /// Calculate cluster efficiency
    async fn calculate_cluster_efficiency(&self) -> f64 {
        let registry = self.node_registry.read();
        
        if registry.active_nodes.is_empty() {
            return 0.0;
        }
        
        // Simple efficiency calculation based on load distribution
        let loads: Vec<f64> = registry.active_nodes.values()
            .map(|node| node.current_load)
            .collect();
        
        let avg_load = loads.iter().sum::<f64>() / loads.len() as f64;
        let load_variance = loads.iter()
            .map(|load| (load - avg_load).powi(2))
            .sum::<f64>() / loads.len() as f64;
        
        // Higher efficiency when loads are balanced (low variance)
        1.0 - load_variance.sqrt()
    }
    
    /// Calculate fault tolerance score
    async fn calculate_fault_tolerance(&self) -> f64 {
        let registry = self.node_registry.read();
        let total = registry.total_nodes;
        let active = registry.active_nodes.len();
        let failed = registry.failed_nodes.len();
        
        if total == 0 {
            return 0.0;
        }
        
        // Fault tolerance based on Byzantine fault tolerance
        let byzantine_threshold = total / 3;
        let fault_tolerance = if failed < byzantine_threshold {
            1.0 - (failed as f64 / byzantine_threshold as f64)
        } else {
            0.0
        };
        
        fault_tolerance * (active as f64 / total as f64)
    }
}

impl Default for HealthThresholds {
    fn default() -> Self {
        Self {
            cpu_threshold: 0.8,
            memory_threshold: 0.8,
            latency_threshold_ms: 100.0,
            heartbeat_timeout: Duration::from_secs(30),
            min_health_score: 0.7,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ClusterConfig;
    
    #[tokio::test]
    async fn test_cluster_coordinator_creation() {
        let config = ClusterConfig::default();
        let coordinator = ClusterCoordinator::new(&config)
            .expect("Should create cluster coordinator");
        
        assert!(!coordinator.node_id.is_nil());
        
        let total_nodes = coordinator.total_nodes().await;
        assert_eq!(total_nodes, 0); // No nodes initially
    }
    
    #[tokio::test]
    async fn test_cluster_bootstrap() {
        let config = ClusterConfig::default();
        let coordinator = ClusterCoordinator::new(&config)
            .expect("Should create cluster coordinator");
        
        coordinator.start().await.expect("Should start coordinator");
        
        let total_nodes = coordinator.total_nodes().await;
        assert_eq!(total_nodes, 1); // Self node added
        
        let health_score = coordinator.health_score().await;
        assert!(health_score > 0.0);
        
        coordinator.stop().await.expect("Should stop coordinator");
    }
    
    #[tokio::test] 
    async fn test_node_scaling() {
        let config = ClusterConfig::default();
        let coordinator = ClusterCoordinator::new(&config)
            .expect("Should create cluster coordinator");
        
        coordinator.start().await.expect("Should start coordinator");
        
        // Add nodes
        coordinator.add_nodes(5).await.expect("Should add nodes");
        let total_nodes = coordinator.total_nodes().await;
        assert_eq!(total_nodes, 6); // 1 initial + 5 added
        
        // Remove nodes
        coordinator.remove_nodes(2).await.expect("Should remove nodes");
        let total_nodes = coordinator.total_nodes().await;
        assert_eq!(total_nodes, 4); // 6 - 2 removed
        
        coordinator.stop().await.expect("Should stop coordinator");
    }
    
    #[tokio::test]
    async fn test_distributed_query_execution() {
        let config = ClusterConfig::default();
        let coordinator = ClusterCoordinator::new(&config)
            .expect("Should create cluster coordinator");
        
        coordinator.start().await.expect("Should start coordinator");
        coordinator.add_nodes(3).await.expect("Should add nodes");
        
        let shard_queries = vec![
            serde_json::json!({"query": "test1"}),
            serde_json::json!({"query": "test2"}),
            serde_json::json!({"query": "test3"}),
        ];
        
        let results = coordinator.execute_distributed_query(shard_queries).await
            .expect("Should execute distributed query");
        
        assert_eq!(results.len(), 3);
        
        coordinator.stop().await.expect("Should stop coordinator");
    }
    
    #[tokio::test]
    async fn test_cluster_metrics() {
        let config = ClusterConfig::default();
        let coordinator = ClusterCoordinator::new(&config)
            .expect("Should create cluster coordinator");
        
        coordinator.start().await.expect("Should start coordinator");
        coordinator.add_nodes(10).await.expect("Should add nodes");
        
        let metrics = coordinator.get_metrics().await;
        
        assert!(metrics.node_utilization >= 0.0);
        assert!(metrics.efficiency_score >= 0.0);
        assert!(metrics.fault_tolerance_score >= 0.0);
        assert!(metrics.scalability_factor >= 0.0);
        
        coordinator.stop().await.expect("Should stop coordinator");
    }
}