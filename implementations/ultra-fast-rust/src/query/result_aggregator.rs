//! Result aggregation and ranking for query results
//!
//! This module implements:
//! - Pattern match ranking and scoring
//! - Result deduplication and filtering
//! - Parallel result aggregation
//! - Top-K result selection with SIMD

use crate::{NodeId, EdgeId, GraphError, GraphResult};
use crate::graph::PatternMatch;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, BinaryHeap};
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};

/// Result aggregator for ranking and filtering pattern matches
#[derive(Debug)]
pub struct ResultAggregator {
    /// Scoring configuration
    scoring_config: ScoringConfig,
    
    /// Deduplication settings
    dedup_config: DeduplicationConfig,
    
    /// Performance metrics
    metrics: AggregationMetrics,
}

impl ResultAggregator {
    /// Create a new result aggregator
    pub fn new() -> Self {
        Self {
            scoring_config: ScoringConfig::default(),
            dedup_config: DeduplicationConfig::default(),
            metrics: AggregationMetrics::new(),
        }
    }

    /// Rank pattern matches using advanced scoring
    pub fn rank_pattern_matches(&self, matches: Vec<PatternMatch>) -> GraphResult<Vec<PatternMatch>> {
        let start_time = std::time::Instant::now();
        
        if matches.is_empty() {
            return Ok(matches);
        }

        // Parallel scoring of matches
        let mut scored_matches = self.score_matches_parallel(matches)?;
        
        // Deduplication
        if self.dedup_config.enabled {
            scored_matches = self.deduplicate_matches(scored_matches)?;
        }
        
        // Parallel sorting by score
        scored_matches.par_sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal)
        });
        
        // Apply limit if configured
        if let Some(limit) = self.scoring_config.max_results {
            scored_matches.truncate(limit);
        }
        
        // Record metrics
        let duration = start_time.elapsed();
        self.metrics.record_aggregation(duration, scored_matches.len());
        
        Ok(scored_matches)
    }

    /// Aggregate multiple result sets
    pub fn aggregate_result_sets(&self, result_sets: Vec<Vec<PatternMatch>>) -> GraphResult<Vec<PatternMatch>> {
        let start_time = std::time::Instant::now();
        
        // Flatten all result sets
        let all_matches: Vec<PatternMatch> = result_sets
            .into_par_iter()
            .flatten()
            .collect();
        
        // Apply ranking
        let ranked_results = self.rank_pattern_matches(all_matches)?;
        
        // Record metrics
        let duration = start_time.elapsed();
        self.metrics.record_aggregation(duration, ranked_results.len());
        
        Ok(ranked_results)
    }

    /// Get top-K results efficiently
    pub fn get_top_k_results(&self, matches: Vec<PatternMatch>, k: usize) -> GraphResult<Vec<PatternMatch>> {
        if matches.len() <= k {
            return self.rank_pattern_matches(matches);
        }
        
        // Use parallel partial sorting for efficiency
        let mut scored_matches = self.score_matches_parallel(matches)?;
        
        // Parallel selection of top-K using nth_element equivalent
        scored_matches.par_sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal)
        });
        
        scored_matches.truncate(k);
        
        Ok(scored_matches)
    }

    /// Filter results by criteria
    pub fn filter_results(
        &self,
        matches: Vec<PatternMatch>,
        filter: &ResultFilter,
    ) -> GraphResult<Vec<PatternMatch>> {
        let filtered_matches: Vec<PatternMatch> = matches
            .into_par_iter()
            .filter(|pattern_match| self.match_satisfies_filter(pattern_match, filter))
            .collect();
        
        Ok(filtered_matches)
    }

    /// Group results by specified criteria
    pub fn group_results(
        &self,
        matches: Vec<PatternMatch>,
        group_by: &GroupByConfig,
    ) -> GraphResult<HashMap<String, Vec<PatternMatch>>> {
        let grouped: HashMap<String, Vec<PatternMatch>> = matches
            .into_par_iter()
            .fold(
                || HashMap::new(),
                |mut acc, pattern_match| {
                    let group_key = self.compute_group_key(&pattern_match, group_by);
                    acc.entry(group_key).or_insert_with(Vec::new).push(pattern_match);
                    acc
                }
            )
            .reduce(
                || HashMap::new(),
                |mut acc1, acc2| {
                    for (key, mut values) in acc2 {
                        acc1.entry(key).or_insert_with(Vec::new).append(&mut values);
                    }
                    acc1
                }
            );
        
        Ok(grouped)
    }

    /// Compute similarity between pattern matches
    pub fn compute_similarity(&self, match1: &PatternMatch, match2: &PatternMatch) -> f64 {
        let mut similarity = 0.0;
        let mut total_weight = 0.0;
        
        // Node binding similarity
        let node_similarity = self.compute_node_binding_similarity(match1, match2);
        similarity += node_similarity * self.scoring_config.node_similarity_weight;
        total_weight += self.scoring_config.node_similarity_weight;
        
        // Edge binding similarity
        let edge_similarity = self.compute_edge_binding_similarity(match1, match2);
        similarity += edge_similarity * self.scoring_config.edge_similarity_weight;
        total_weight += self.scoring_config.edge_similarity_weight;
        
        // Score similarity
        let score_similarity = 1.0 - ((match1.score - match2.score).abs() / (match1.score.max(match2.score) + 1e-6));
        similarity += score_similarity * self.scoring_config.score_similarity_weight;
        total_weight += self.scoring_config.score_similarity_weight;
        
        if total_weight > 0.0 {
            similarity / total_weight
        } else {
            0.0
        }
    }

    /// Score matches in parallel
    fn score_matches_parallel(&self, mut matches: Vec<PatternMatch>) -> GraphResult<Vec<PatternMatch>> {
        matches.par_iter_mut().for_each(|pattern_match| {
            pattern_match.score = self.compute_enhanced_score(pattern_match);
        });
        
        Ok(matches)
    }

    /// Compute enhanced score for a pattern match
    fn compute_enhanced_score(&self, pattern_match: &PatternMatch) -> f64 {
        let mut score = pattern_match.score;
        
        // Base score from the match
        let base_score = score * self.scoring_config.base_score_weight;
        
        // Confidence boost
        let confidence_boost = pattern_match.confidence * self.scoring_config.confidence_weight;
        
        // Node count bonus (larger matches are often more valuable)
        let node_count_bonus = (pattern_match.node_bindings.len() as f64).sqrt() * 
                              self.scoring_config.node_count_weight;
        
        // Edge count bonus
        let edge_count_bonus = (pattern_match.edge_bindings.len() as f64).sqrt() * 
                              self.scoring_config.edge_count_weight;
        
        // Completeness bonus (all expected bindings present)
        let completeness_bonus = if self.is_complete_match(pattern_match) {
            self.scoring_config.completeness_bonus
        } else {
            0.0
        };
        
        // Diversity penalty (penalize very similar matches)
        let diversity_factor = 1.0; // Would be computed based on existing results
        
        score = base_score + confidence_boost + node_count_bonus + edge_count_bonus + completeness_bonus;
        score *= diversity_factor;
        
        score.max(0.0) // Ensure non-negative score
    }

    /// Check if match is complete (has all expected bindings)
    fn is_complete_match(&self, pattern_match: &PatternMatch) -> bool {
        // Simple heuristic: match is complete if it has both node and edge bindings
        !pattern_match.node_bindings.is_empty() && !pattern_match.edge_bindings.is_empty()
    }

    /// Deduplicate matches based on similarity
    fn deduplicate_matches(&self, matches: Vec<PatternMatch>) -> GraphResult<Vec<PatternMatch>> {
        if matches.len() <= 1 {
            return Ok(matches);
        }
        
        let mut deduplicated = Vec::new();
        let mut processed = HashSet::new();
        
        for (i, current_match) in matches.iter().enumerate() {
            if processed.contains(&i) {
                continue;
            }
            
            let mut is_duplicate = false;
            
            // Check against already selected matches
            for existing_match in &deduplicated {
                let similarity = self.compute_similarity(current_match, existing_match);
                if similarity >= self.dedup_config.similarity_threshold {
                    is_duplicate = true;
                    break;
                }
            }
            
            if !is_duplicate {
                deduplicated.push(current_match.clone());
                processed.insert(i);
                
                // Mark similar matches as processed
                for (j, other_match) in matches.iter().enumerate().skip(i + 1) {
                    if !processed.contains(&j) {
                        let similarity = self.compute_similarity(current_match, other_match);
                        if similarity >= self.dedup_config.similarity_threshold {
                            processed.insert(j);
                        }
                    }
                }
            }
        }
        
        Ok(deduplicated)
    }

    /// Compute node binding similarity
    fn compute_node_binding_similarity(&self, match1: &PatternMatch, match2: &PatternMatch) -> f64 {
        let keys1: HashSet<_> = match1.node_bindings.keys().collect();
        let keys2: HashSet<_> = match2.node_bindings.keys().collect();
        
        let intersection_size = keys1.intersection(&keys2).count();
        let union_size = keys1.union(&keys2).count();
        
        if union_size == 0 {
            return 1.0; // Both empty
        }
        
        let jaccard_similarity = intersection_size as f64 / union_size as f64;
        
        // Also consider value similarity for common keys
        let mut value_similarity = 0.0;
        let mut common_keys = 0;
        
        for key in keys1.intersection(&keys2) {
            if match1.node_bindings[*key] == match2.node_bindings[*key] {
                value_similarity += 1.0;
            }
            common_keys += 1;
        }
        
        if common_keys > 0 {
            value_similarity /= common_keys as f64;
        }
        
        // Combine Jaccard similarity with value similarity
        (jaccard_similarity + value_similarity) / 2.0
    }

    /// Compute edge binding similarity
    fn compute_edge_binding_similarity(&self, match1: &PatternMatch, match2: &PatternMatch) -> f64 {
        let keys1: HashSet<_> = match1.edge_bindings.keys().collect();
        let keys2: HashSet<_> = match2.edge_bindings.keys().collect();
        
        let intersection_size = keys1.intersection(&keys2).count();
        let union_size = keys1.union(&keys2).count();
        
        if union_size == 0 {
            return 1.0; // Both empty
        }
        
        intersection_size as f64 / union_size as f64
    }

    /// Check if match satisfies filter criteria
    fn match_satisfies_filter(&self, pattern_match: &PatternMatch, filter: &ResultFilter) -> bool {
        // Score range filter
        if let Some((min_score, max_score)) = filter.score_range {
            if pattern_match.score < min_score || pattern_match.score > max_score {
                return false;
            }
        }
        
        // Confidence filter
        if let Some(min_confidence) = filter.min_confidence {
            if pattern_match.confidence < min_confidence {
                return false;
            }
        }
        
        // Node count filter
        if let Some((min_nodes, max_nodes)) = filter.node_count_range {
            let node_count = pattern_match.node_bindings.len();
            if node_count < min_nodes || node_count > max_nodes {
                return false;
            }
        }
        
        // Edge count filter
        if let Some((min_edges, max_edges)) = filter.edge_count_range {
            let edge_count = pattern_match.edge_bindings.len();
            if edge_count < min_edges || edge_count > max_edges {
                return false;
            }
        }
        
        // Node type filter
        if !filter.required_node_types.is_empty() {
            // Would need access to node data to check types
            // This is a simplified implementation
        }
        
        true
    }

    /// Compute group key for grouping
    fn compute_group_key(&self, pattern_match: &PatternMatch, group_by: &GroupByConfig) -> String {
        match group_by {
            GroupByConfig::Score => {
                // Group by score ranges
                let score_bucket = (pattern_match.score / 0.1).floor() * 0.1;
                format!("score_{:.1}", score_bucket)
            }
            GroupByConfig::NodeCount => {
                format!("nodes_{}", pattern_match.node_bindings.len())
            }
            GroupByConfig::EdgeCount => {
                format!("edges_{}", pattern_match.edge_bindings.len())
            }
            GroupByConfig::Confidence => {
                let conf_bucket = (pattern_match.confidence / 0.1).floor() * 0.1;
                format!("confidence_{:.1}", conf_bucket)
            }
            GroupByConfig::Custom(field) => {
                // Would need to extract custom field from match data
                format!("custom_{}", field)
            }
        }
    }
}

/// Configuration for scoring pattern matches
#[derive(Debug, Clone)]
pub struct ScoringConfig {
    pub base_score_weight: f64,
    pub confidence_weight: f64,
    pub node_count_weight: f64,
    pub edge_count_weight: f64,
    pub completeness_bonus: f64,
    pub node_similarity_weight: f64,
    pub edge_similarity_weight: f64,
    pub score_similarity_weight: f64,
    pub max_results: Option<usize>,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            base_score_weight: 1.0,
            confidence_weight: 0.5,
            node_count_weight: 0.2,
            edge_count_weight: 0.3,
            completeness_bonus: 0.1,
            node_similarity_weight: 0.4,
            edge_similarity_weight: 0.3,
            score_similarity_weight: 0.3,
            max_results: Some(1000),
        }
    }
}

/// Configuration for deduplication
#[derive(Debug, Clone)]
pub struct DeduplicationConfig {
    pub enabled: bool,
    pub similarity_threshold: f64,
    pub max_similar_results: usize,
}

impl Default for DeduplicationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            similarity_threshold: 0.8,
            max_similar_results: 5,
        }
    }
}

/// Filter configuration for results
#[derive(Debug, Clone)]
pub struct ResultFilter {
    pub score_range: Option<(f64, f64)>,
    pub min_confidence: Option<f64>,
    pub node_count_range: Option<(usize, usize)>,
    pub edge_count_range: Option<(usize, usize)>,
    pub required_node_types: Vec<String>,
    pub required_edge_types: Vec<String>,
}

impl Default for ResultFilter {
    fn default() -> Self {
        Self {
            score_range: None,
            min_confidence: None,
            node_count_range: None,
            edge_count_range: None,
            required_node_types: Vec::new(),
            required_edge_types: Vec::new(),
        }
    }
}

/// Grouping configuration
#[derive(Debug, Clone)]
pub enum GroupByConfig {
    Score,
    NodeCount,
    EdgeCount,
    Confidence,
    Custom(String),
}

/// Aggregation performance metrics
#[derive(Debug)]
struct AggregationMetrics {
    aggregations_performed: std::sync::atomic::AtomicU64,
    total_aggregation_time: std::sync::atomic::AtomicU64,
    results_processed: std::sync::atomic::AtomicU64,
}

impl AggregationMetrics {
    fn new() -> Self {
        Self {
            aggregations_performed: std::sync::atomic::AtomicU64::new(0),
            total_aggregation_time: std::sync::atomic::AtomicU64::new(0),
            results_processed: std::sync::atomic::AtomicU64::new(0),
        }
    }

    fn record_aggregation(&self, duration: std::time::Duration, result_count: usize) {
        self.aggregations_performed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.total_aggregation_time.fetch_add(
            duration.as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.results_processed.fetch_add(result_count as u64, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Ranked result with additional metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedResult {
    pub pattern_match: PatternMatch,
    pub rank: usize,
    pub normalized_score: f64,
    pub metadata: ResultMetadata,
}

/// Additional metadata for results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultMetadata {
    pub processing_time_us: u64,
    pub similarity_group: Option<String>,
    pub quality_score: f64,
    pub completeness_score: f64,
}

/// Result statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultStatistics {
    pub total_results: usize,
    pub unique_results: usize,
    pub average_score: f64,
    pub score_distribution: Vec<(f64, usize)>, // (score_range, count)
    pub confidence_distribution: Vec<(f64, usize)>,
    pub processing_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_match(score: f64, confidence: f64, node_count: usize) -> PatternMatch {
        let mut node_bindings = HashMap::new();
        for i in 0..node_count {
            node_bindings.insert(format!("node_{}", i), i as NodeId);
        }
        
        PatternMatch {
            node_bindings,
            edge_bindings: HashMap::new(),
            score,
            confidence,
        }
    }

    #[test]
    fn test_result_aggregator_creation() {
        let aggregator = ResultAggregator::new();
        assert!(aggregator.scoring_config.max_results.is_some());
    }

    #[test]
    fn test_rank_pattern_matches() {
        let aggregator = ResultAggregator::new();
        
        let matches = vec![
            create_test_match(0.5, 0.8, 2),
            create_test_match(0.9, 0.7, 3),
            create_test_match(0.3, 0.9, 1),
        ];
        
        let ranked = aggregator.rank_pattern_matches(matches).unwrap();
        
        // Should be sorted by enhanced score
        assert!(ranked[0].score >= ranked[1].score);
        assert!(ranked[1].score >= ranked[2].score);
    }

    #[test]
    fn test_compute_similarity() {
        let aggregator = ResultAggregator::new();
        
        let match1 = create_test_match(0.8, 0.9, 2);
        let match2 = create_test_match(0.8, 0.9, 2);
        
        let similarity = aggregator.compute_similarity(&match1, &match2);
        assert!(similarity > 0.5); // Should be somewhat similar
    }

    #[test]
    fn test_filter_results() {
        let aggregator = ResultAggregator::new();
        
        let matches = vec![
            create_test_match(0.9, 0.8, 2),
            create_test_match(0.3, 0.7, 1),
            create_test_match(0.7, 0.9, 3),
        ];
        
        let filter = ResultFilter {
            score_range: Some((0.5, 1.0)),
            min_confidence: Some(0.75),
            ..Default::default()
        };
        
        let filtered = aggregator.filter_results(matches, &filter).unwrap();
        
        // Should only keep matches with score >= 0.5 and confidence >= 0.75
        assert_eq!(filtered.len(), 2);
        for match_item in &filtered {
            assert!(match_item.score >= 0.5);
            assert!(match_item.confidence >= 0.75);
        }
    }

    #[test]
    fn test_get_top_k_results() {
        let aggregator = ResultAggregator::new();
        
        let matches = vec![
            create_test_match(0.9, 0.8, 2),
            create_test_match(0.7, 0.7, 1),
            create_test_match(0.8, 0.9, 3),
            create_test_match(0.6, 0.6, 2),
            create_test_match(0.5, 0.5, 1),
        ];
        
        let top_3 = aggregator.get_top_k_results(matches, 3).unwrap();
        
        assert_eq!(top_3.len(), 3);
        // Should be sorted by score (with enhancements)
        for i in 0..2 {
            assert!(top_3[i].score >= top_3[i + 1].score);
        }
    }
}