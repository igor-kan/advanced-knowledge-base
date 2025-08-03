"""
Advanced SPARQL Query Engine
Provides sophisticated query processing, optimization, and execution
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Iterator
from dataclasses import dataclass
from enum import Enum
import time
import re
from collections import defaultdict, deque

from rdflib import Graph, Namespace, URIRef, Literal, BNode, Variable
from rdflib.namespace import RDF, RDFS, OWL, XSD
from rdflib.plugins.sparql import prepareQuery, processUpdate
from rdflib.plugins.sparql.parser import parseQuery, parseUpdate
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.evaluate import evalQuery
from rdflib.term import Node
from rdflib.query import Result

logger = logging.getLogger(__name__)

class QueryType(Enum):
    SELECT = "SELECT"
    CONSTRUCT = "CONSTRUCT"
    ASK = "ASK"
    DESCRIBE = "DESCRIBE"
    INSERT = "INSERT"
    DELETE = "DELETE"

@dataclass
class QueryPlan:
    """Represents an optimized query execution plan"""
    query_type: QueryType
    triple_patterns: List[Dict]
    filters: List[Dict]
    joins: List[Dict]
    estimated_cost: float
    execution_order: List[int]

@dataclass
class QueryMetrics:
    """Query execution metrics"""
    execution_time: float
    parsing_time: float
    optimization_time: float
    evaluation_time: float
    result_count: int
    triples_processed: int
    join_operations: int
    filter_operations: int

class SPARQLQueryEngine:
    """
    Advanced SPARQL Query Engine with optimization and performance monitoring
    """
    
    def __init__(self, graph: Graph):
        """
        Initialize SPARQL query engine
        
        Args:
            graph: RDF graph to query against
        """
        self.graph = graph
        self.query_cache = {}
        self.plan_cache = {}
        self.statistics = defaultdict(int)
        
        # Query optimization settings
        self.enable_optimization = True
        self.enable_caching = True
        self.cache_max_size = 500
        
        # Performance monitoring
        self.query_history = deque(maxlen=1000)
        
        logger.info("SPARQL Query Engine initialized")
    
    def execute_query(self, query_string: str, 
                     bindings: Optional[Dict[str, Node]] = None) -> Tuple[Result, QueryMetrics]:
        """
        Execute SPARQL query with optimization and monitoring
        
        Args:
            query_string: SPARQL query string
            bindings: Variable bindings
            
        Returns:
            Tuple of (results, metrics)
        """
        start_time = time.time()
        
        # Parse query
        parse_start = time.time()
        try:
            parsed_query = parseQuery(query_string)
            query_type = self._detect_query_type(parsed_query)
        except Exception as e:
            logger.error(f"Query parsing failed: {e}")
            raise
        
        parsing_time = time.time() - parse_start
        
        # Query optimization
        opt_start = time.time()
        if self.enable_optimization:
            execution_plan = self._optimize_query(parsed_query, query_type)
        else:
            execution_plan = None
        optimization_time = time.time() - opt_start
        
        # Query execution
        eval_start = time.time()
        try:
            if bindings:
                prepared_query = prepareQuery(query_string)
                results = self.graph.query(prepared_query, initBindings=bindings)
            else:
                results = self.graph.query(query_string)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
        
        evaluation_time = time.time() - eval_start
        execution_time = time.time() - start_time
        
        # Collect metrics
        metrics = QueryMetrics(
            execution_time=execution_time,
            parsing_time=parsing_time,
            optimization_time=optimization_time,
            evaluation_time=evaluation_time,
            result_count=len(list(results)) if hasattr(results, '__len__') else 0,
            triples_processed=self._estimate_triples_processed(parsed_query),
            join_operations=self._count_joins(parsed_query),
            filter_operations=self._count_filters(parsed_query)
        )
        
        # Update statistics
        self._update_statistics(query_type, metrics)
        
        # Cache results if beneficial
        if self.enable_caching and self._should_cache_query(query_string, metrics):
            self._cache_query_result(query_string, results)
        
        logger.info(f"Query executed in {execution_time:.4f}s, "
                   f"{metrics.result_count} results")
        
        return results, metrics
    
    def _detect_query_type(self, parsed_query) -> QueryType:
        """Detect the type of SPARQL query"""
        # Simplified detection based on query structure
        query_str = str(parsed_query).upper()
        
        if 'SELECT' in query_str:
            return QueryType.SELECT
        elif 'CONSTRUCT' in query_str:
            return QueryType.CONSTRUCT
        elif 'ASK' in query_str:
            return QueryType.ASK
        elif 'DESCRIBE' in query_str:
            return QueryType.DESCRIBE
        elif 'INSERT' in query_str:
            return QueryType.INSERT
        elif 'DELETE' in query_str:
            return QueryType.DELETE
        else:
            return QueryType.SELECT  # Default
    
    def _optimize_query(self, parsed_query, query_type: QueryType) -> Optional[QueryPlan]:
        """
        Optimize query execution plan
        
        Args:
            parsed_query: Parsed SPARQL query
            query_type: Type of query
            
        Returns:
            Optimized execution plan
        """
        try:
            # Extract triple patterns
            triple_patterns = self._extract_triple_patterns(parsed_query)
            
            # Extract filters
            filters = self._extract_filters(parsed_query)
            
            # Estimate costs for different join orders
            join_plans = self._generate_join_plans(triple_patterns)
            
            # Select best plan
            best_plan = min(join_plans, key=lambda p: p.estimated_cost) if join_plans else None
            
            if best_plan:
                logger.debug(f"Selected execution plan with cost {best_plan.estimated_cost:.2f}")
            
            return best_plan
            
        except Exception as e:
            logger.warning(f"Query optimization failed: {e}")
            return None
    
    def _extract_triple_patterns(self, parsed_query) -> List[Dict]:
        """Extract triple patterns from parsed query"""
        patterns = []
        
        # This is a simplified extraction - real implementation would
        # need to traverse the query algebra tree properly
        query_str = str(parsed_query)
        
        # Simple pattern matching for demonstration
        # Real implementation would use proper AST traversal
        pattern_matches = re.findall(r'(\??\w+)\s+(\??\w+|\<[^>]+\>)\s+(\??\w+|\<[^>]+\>|"[^"]*")', query_str)
        
        for i, (subj, pred, obj) in enumerate(pattern_matches):
            patterns.append({
                'id': i,
                'subject': subj,
                'predicate': pred,
                'object': obj,
                'variables': [term for term in [subj, pred, obj] if term.startswith('?')],
                'estimated_selectivity': self._estimate_selectivity(subj, pred, obj)
            })
        
        return patterns
    
    def _extract_filters(self, parsed_query) -> List[Dict]:
        """Extract FILTER expressions from query"""
        filters = []
        query_str = str(parsed_query)
        
        # Simple filter extraction (real implementation would be more sophisticated)
        filter_matches = re.findall(r'FILTER\s*\(([^)]+)\)', query_str, re.IGNORECASE)
        
        for i, filter_expr in enumerate(filter_matches):
            filters.append({
                'id': i,
                'expression': filter_expr,
                'variables': re.findall(r'\?(\w+)', filter_expr),
                'estimated_selectivity': 0.5  # Default estimate
            })
        
        return filters
    
    def _estimate_selectivity(self, subject: str, predicate: str, obj: str) -> float:
        """
        Estimate selectivity of a triple pattern
        
        Args:
            subject: Subject (variable or constant)
            predicate: Predicate (variable or constant) 
            obj: Object (variable or constant)
            
        Returns:
            Estimated selectivity (0.0 to 1.0)
        """
        # Count variables vs constants
        var_count = sum(1 for term in [subject, predicate, obj] if term.startswith('?'))
        
        # More constants = higher selectivity (fewer results)
        if var_count == 0:
            return 0.001  # Very selective (exact match)
        elif var_count == 1:
            return 0.1    # Moderately selective
        elif var_count == 2:
            return 0.3    # Less selective
        else:
            return 1.0    # Not selective (all triples)
    
    def _generate_join_plans(self, triple_patterns: List[Dict]) -> List[QueryPlan]:
        """Generate possible join execution plans"""
        if not triple_patterns:
            return []
        
        plans = []
        
        # Generate different join orders
        # For simplicity, we'll just consider a few heuristics
        
        # Plan 1: Order by selectivity (most selective first)
        selectivity_order = sorted(range(len(triple_patterns)), 
                                 key=lambda i: triple_patterns[i]['estimated_selectivity'])
        
        cost1 = self._estimate_plan_cost(triple_patterns, selectivity_order)
        plans.append(QueryPlan(
            query_type=QueryType.SELECT,
            triple_patterns=triple_patterns,
            filters=[],
            joins=self._generate_join_sequence(triple_patterns, selectivity_order),
            estimated_cost=cost1,
            execution_order=selectivity_order
        ))
        
        # Plan 2: Order by variable connectivity
        connectivity_order = self._order_by_connectivity(triple_patterns)
        cost2 = self._estimate_plan_cost(triple_patterns, connectivity_order)
        plans.append(QueryPlan(
            query_type=QueryType.SELECT,
            triple_patterns=triple_patterns,
            filters=[],
            joins=self._generate_join_sequence(triple_patterns, connectivity_order),
            estimated_cost=cost2,
            execution_order=connectivity_order
        ))
        
        return plans
    
    def _order_by_connectivity(self, patterns: List[Dict]) -> List[int]:
        """Order patterns by variable connectivity"""
        if not patterns:
            return []
        
        # Build variable connectivity graph
        var_patterns = defaultdict(set)
        for i, pattern in enumerate(patterns):
            for var in pattern['variables']:
                var_patterns[var].add(i)
        
        # Start with most connected pattern
        remaining = set(range(len(patterns)))
        order = []
        
        # Find pattern with most shared variables
        start_pattern = max(remaining, 
                          key=lambda i: sum(len(var_patterns[var]) for var in patterns[i]['variables']))
        
        order.append(start_pattern)
        remaining.remove(start_pattern)
        
        # Greedily add most connected remaining pattern
        while remaining:
            current_vars = set()
            for i in order:
                current_vars.update(patterns[i]['variables'])
            
            if current_vars:
                # Find pattern with most overlap with current variables
                next_pattern = max(remaining,
                                 key=lambda i: len(set(patterns[i]['variables']) & current_vars))
            else:
                # No overlap, pick any
                next_pattern = next(iter(remaining))
            
            order.append(next_pattern)
            remaining.remove(next_pattern)
        
        return order
    
    def _generate_join_sequence(self, patterns: List[Dict], order: List[int]) -> List[Dict]:
        """Generate join operations for execution plan"""
        joins = []
        
        for i in range(1, len(order)):
            left_pattern = patterns[order[i-1]]
            right_pattern = patterns[order[i]]
            
            # Find common variables for join
            left_vars = set(left_pattern['variables'])
            right_vars = set(right_pattern['variables'])
            join_vars = left_vars & right_vars
            
            joins.append({
                'left': order[i-1],
                'right': order[i],
                'join_variables': list(join_vars),
                'join_type': 'hash_join' if join_vars else 'cross_product'
            })
        
        return joins
    
    def _estimate_plan_cost(self, patterns: List[Dict], order: List[int]) -> float:
        """Estimate execution cost for a plan"""
        if not patterns or not order:
            return 0.0
        
        cost = 0.0
        intermediate_size = 1.0
        
        for i in order:
            pattern = patterns[i]
            
            # Estimate pattern cost based on selectivity
            pattern_cost = pattern['estimated_selectivity'] * len(self.graph)
            
            # Join cost increases with intermediate result size
            if i > 0:
                join_cost = intermediate_size * pattern_cost * 0.1  # Join cost factor
                cost += join_cost
            
            cost += pattern_cost
            intermediate_size *= pattern_cost
        
        return cost
    
    def _estimate_triples_processed(self, parsed_query) -> int:
        """Estimate number of triples processed during query execution"""
        # Simplified estimation based on query structure
        query_str = str(parsed_query)
        pattern_count = len(re.findall(r'\??\w+\s+\??\w+\s+\??\w+', query_str))
        return max(pattern_count * 100, len(self.graph) // 10)  # Rough estimate
    
    def _count_joins(self, parsed_query) -> int:
        """Count number of join operations in query"""
        query_str = str(parsed_query)
        pattern_count = len(re.findall(r'\??\w+\s+\??\w+\s+\??\w+', query_str))
        return max(0, pattern_count - 1)
    
    def _count_filters(self, parsed_query) -> int:
        """Count number of filter operations in query"""
        query_str = str(parsed_query)
        return len(re.findall(r'FILTER\s*\(', query_str, re.IGNORECASE))
    
    def _should_cache_query(self, query_string: str, metrics: QueryMetrics) -> bool:
        """Determine if query result should be cached"""
        if not self.enable_caching:
            return False
        
        if len(self.query_cache) >= self.cache_max_size:
            return False
        
        # Cache expensive queries with small result sets
        return metrics.execution_time > 0.1 and metrics.result_count < 1000
    
    def _cache_query_result(self, query_string: str, result: Result):
        """Cache query result"""
        if len(self.query_cache) < self.cache_max_size:
            self.query_cache[query_string] = result
            logger.debug("Cached query result")
    
    def _update_statistics(self, query_type: QueryType, metrics: QueryMetrics):
        """Update query execution statistics"""
        self.statistics[f'{query_type.value}_count'] += 1
        self.statistics[f'{query_type.value}_total_time'] += metrics.execution_time
        self.statistics['total_queries'] += 1
        self.statistics['total_results'] += metrics.result_count
        
        # Add to history
        self.query_history.append({
            'type': query_type.value,
            'metrics': metrics,
            'timestamp': time.time()
        })
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get comprehensive query execution statistics"""
        if not self.query_history:
            return {}
        
        # Calculate averages
        total_queries = len(self.query_history)
        avg_execution_time = sum(q['metrics'].execution_time for q in self.query_history) / total_queries
        avg_result_count = sum(q['metrics'].result_count for q in self.query_history) / total_queries
        
        # Query type distribution
        type_counts = defaultdict(int)
        for query in self.query_history:
            type_counts[query['type']] += 1
        
        return {
            'total_queries': total_queries,
            'average_execution_time': avg_execution_time,
            'average_result_count': avg_result_count,
            'query_type_distribution': dict(type_counts),
            'cache_size': len(self.query_cache),
            'cache_hit_ratio': self.statistics.get('cache_hits', 0) / max(total_queries, 1),
            'optimization_enabled': self.enable_optimization,
            'recent_queries': list(self.query_history)[-10:]  # Last 10 queries
        }
    
    def execute_batch_queries(self, queries: List[str]) -> List[Tuple[Result, QueryMetrics]]:
        """
        Execute multiple queries with batch optimization
        
        Args:
            queries: List of SPARQL query strings
            
        Returns:
            List of (results, metrics) tuples
        """
        logger.info(f"Executing batch of {len(queries)} queries")
        start_time = time.time()
        
        results = []
        for i, query in enumerate(queries):
            try:
                result, metrics = self.execute_query(query)
                results.append((result, metrics))
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(queries)} queries")
                    
            except Exception as e:
                logger.error(f"Query {i} failed: {e}")
                results.append((None, None))
        
        total_time = time.time() - start_time
        logger.info(f"Batch execution completed in {total_time:.2f}s "
                   f"({len(queries)/total_time:.1f} queries/sec)")
        
        return results
    
    def explain_query(self, query_string: str) -> Dict[str, Any]:
        """
        Provide query execution plan explanation
        
        Args:
            query_string: SPARQL query to explain
            
        Returns:
            Query execution plan details
        """
        try:
            parsed_query = parseQuery(query_string)
            query_type = self._detect_query_type(parsed_query)
            
            triple_patterns = self._extract_triple_patterns(parsed_query)
            filters = self._extract_filters(parsed_query)
            
            execution_plan = None
            if self.enable_optimization:
                plans = self._generate_join_plans(triple_patterns)
                if plans:
                    execution_plan = min(plans, key=lambda p: p.estimated_cost)
            
            return {
                'query_type': query_type.value,
                'triple_patterns': triple_patterns,
                'filters': filters,
                'estimated_cost': execution_plan.estimated_cost if execution_plan else 0,
                'execution_order': execution_plan.execution_order if execution_plan else [],
                'join_sequence': execution_plan.joins if execution_plan else [],
                'optimization_enabled': self.enable_optimization
            }
            
        except Exception as e:
            logger.error(f"Query explanation failed: {e}")
            return {'error': str(e)}
    
    def optimize_settings(self, enable_optimization: bool = True, 
                         enable_caching: bool = True,
                         cache_max_size: int = 500):
        """Configure optimization settings"""
        self.enable_optimization = enable_optimization
        self.enable_caching = enable_caching
        self.cache_max_size = cache_max_size
        
        # Clear cache if size reduced
        if len(self.query_cache) > cache_max_size:
            # Keep most recent entries
            items = list(self.query_cache.items())[-cache_max_size:]
            self.query_cache = dict(items)
        
        logger.info(f"Updated optimization settings: "
                   f"optimization={enable_optimization}, "
                   f"caching={enable_caching}, "
                   f"cache_size={cache_max_size}")


# Example usage and testing
if __name__ == "__main__":
    from rdflib import Graph, Namespace
    from rdflib.namespace import FOAF
    
    # Create test graph
    g = Graph()
    ex = Namespace("http://example.org/")
    g.bind("ex", ex)
    g.bind("foaf", FOAF)
    
    # Add test data
    g.add((ex.JohnDoe, RDF.type, FOAF.Person))
    g.add((ex.JohnDoe, FOAF.name, Literal("John Doe")))
    g.add((ex.JohnDoe, FOAF.age, Literal(35)))
    g.add((ex.JohnDoe, ex.worksAt, ex.Google))
    
    g.add((ex.AliceSmith, RDF.type, FOAF.Person))
    g.add((ex.AliceSmith, FOAF.name, Literal("Alice Smith")))
    g.add((ex.AliceSmith, FOAF.age, Literal(28)))
    g.add((ex.AliceSmith, ex.worksAt, ex.Microsoft))
    
    # Create query engine
    engine = SPARQLQueryEngine(g)
    
    # Test queries
    queries = [
        """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?name ?age WHERE {
            ?person foaf:name ?name ;
                    foaf:age ?age .
            FILTER(?age > 30)
        }
        """,
        """
        PREFIX ex: <http://example.org/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?person ?company WHERE {
            ?person a foaf:Person ;
                    ex:worksAt ?company .
        }
        """
    ]
    
    print("Executing test queries...")
    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}:")
        result, metrics = engine.execute_query(query)
        print(f"Results: {len(list(result))}")
        print(f"Execution time: {metrics.execution_time:.4f}s")
        print(f"Metrics: {metrics}")
    
    print(f"\nQuery statistics: {engine.get_query_statistics()}")
    
    # Test query explanation
    print(f"\nQuery explanation:")
    explanation = engine.explain_query(queries[0])
    print(f"Explanation: {explanation}")