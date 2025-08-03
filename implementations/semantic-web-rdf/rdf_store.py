"""
Advanced RDF Triple Store Implementation
Provides sophisticated RDF storage, indexing, and querying capabilities
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Iterator, Union
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import hashlib
import pickle
import json

from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD, FOAF
from rdflib.plugins.stores.memory import Memory
from rdflib.term import Node
from rdflib.query import Result

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TriplePattern:
    """Represents a triple pattern for querying"""
    subject: Optional[Node] = None
    predicate: Optional[Node] = None
    object: Optional[Node] = None
    context: Optional[Node] = None

@dataclass
class QueryStats:
    """Query execution statistics"""
    execution_time: float
    triples_scanned: int
    results_returned: int
    cache_hits: int
    index_used: str

class RDFTripleStore:
    """
    Advanced RDF Triple Store with sophisticated indexing and query optimization
    """
    
    def __init__(self, store_type: str = "memory", store_path: str = None):
        """
        Initialize the RDF triple store
        
        Args:
            store_type: Type of store ("memory", "berkeleydb", "sqlite")
            store_path: Path for persistent stores
        """
        self.store_type = store_type
        self.store_path = store_path
        
        # Initialize RDFLib graph
        if store_type == "memory":
            self.graph = Graph()
        else:
            self.graph = Graph(store=store_type)
            if store_path:
                self.graph.open(store_path, create=True)
        
        # Initialize indexes for performance
        self.indexes = {
            'spo': defaultdict(lambda: defaultdict(set)),  # Subject-Predicate-Object
            'pos': defaultdict(lambda: defaultdict(set)),  # Predicate-Object-Subject  
            'osp': defaultdict(lambda: defaultdict(set)),  # Object-Subject-Predicate
            'pso': defaultdict(lambda: defaultdict(set)),  # Predicate-Subject-Object
            'sop': defaultdict(lambda: defaultdict(set)),  # Subject-Object-Predicate
            'ops': defaultdict(lambda: defaultdict(set))   # Object-Predicate-Subject
        }
        
        # Query cache for performance
        self.query_cache = {}
        self.cache_max_size = 1000
        
        # Statistics
        self.stats = {
            'total_triples': 0,
            'queries_executed': 0,
            'cache_hits': 0,
            'index_updates': 0
        }
        
        # Thread safety
        self.lock = threading.RWLock() if hasattr(threading, 'RWLock') else threading.Lock()
        
        # Namespace management
        self.namespaces = {
            'rdf': RDF,
            'rdfs': RDFS,
            'owl': OWL,
            'xsd': XSD,
            'foaf': FOAF
        }
        
        logger.info(f"Initialized RDF Triple Store with {store_type} backend")
    
    def add_namespace(self, prefix: str, uri: str):
        """Add a namespace prefix"""
        namespace = Namespace(uri)
        self.namespaces[prefix] = namespace
        self.graph.bind(prefix, namespace)
        logger.debug(f"Added namespace {prefix}: {uri}")
    
    def add_triple(self, subject: Node, predicate: Node, obj: Node, context: Node = None) -> bool:
        """
        Add a triple to the store with full indexing
        
        Args:
            subject: Subject node
            predicate: Predicate node  
            obj: Object node
            context: Optional context/graph
            
        Returns:
            True if triple was added, False if already exists
        """
        try:
            with self.lock:
                # Check if triple already exists
                if (subject, predicate, obj) in self.graph:
                    return False
                
                # Add to RDFLib graph
                if context:
                    self.graph.add((subject, predicate, obj, context))
                else:
                    self.graph.add((subject, predicate, obj))
                
                # Update all indexes
                self._update_indexes(subject, predicate, obj, context, add=True)
                
                # Update statistics
                self.stats['total_triples'] += 1
                self.stats['index_updates'] += 6  # Updated 6 indexes
                
                logger.debug(f"Added triple: {subject} {predicate} {obj}")
                return True
                
        except Exception as e:
            logger.error(f"Error adding triple: {e}")
            return False
    
    def remove_triple(self, subject: Node, predicate: Node, obj: Node, context: Node = None) -> bool:
        """Remove a triple from the store"""
        try:
            with self.lock:
                # Remove from RDFLib graph
                if context:
                    self.graph.remove((subject, predicate, obj, context))
                else:
                    self.graph.remove((subject, predicate, obj))
                
                # Update indexes
                self._update_indexes(subject, predicate, obj, context, add=False)
                
                # Update statistics
                self.stats['total_triples'] -= 1
                
                # Clear query cache (simplistic approach)
                self.query_cache.clear()
                
                return True
                
        except Exception as e:
            logger.error(f"Error removing triple: {e}")
            return False
    
    def _update_indexes(self, subject: Node, predicate: Node, obj: Node, context: Node, add: bool = True):
        """Update all indexes when triple is added/removed"""
        if add:
            # Add to indexes
            self.indexes['spo'][subject][predicate].add(obj)
            self.indexes['pos'][predicate][obj].add(subject)
            self.indexes['osp'][obj][subject].add(predicate)
            self.indexes['pso'][predicate][subject].add(obj)
            self.indexes['sop'][subject][obj].add(predicate)
            self.indexes['ops'][obj][predicate].add(subject)
        else:
            # Remove from indexes
            self.indexes['spo'][subject][predicate].discard(obj)
            self.indexes['pos'][predicate][obj].discard(subject)
            self.indexes['osp'][obj][subject].discard(predicate)
            self.indexes['pso'][predicate][subject].discard(obj)
            self.indexes['sop'][subject][obj].discard(predicate)
            self.indexes['ops'][obj][predicate].discard(subject)
    
    def query_triples(self, pattern: TriplePattern) -> Iterator[Tuple[Node, Node, Node]]:
        """
        Query triples using pattern matching with index optimization
        
        Args:
            pattern: Triple pattern to match
            
        Yields:
            Matching triples
        """
        start_time = time.time()
        triples_scanned = 0
        results_count = 0
        index_used = "sequential"
        
        try:
            with self.lock:
                # Choose optimal index based on pattern
                if pattern.subject and pattern.predicate:
                    # Use SPO index
                    index_used = "spo"
                    if pattern.object:
                        # Exact match
                        if pattern.object in self.indexes['spo'][pattern.subject][pattern.predicate]:
                            yield (pattern.subject, pattern.predicate, pattern.object)
                            results_count += 1
                    else:
                        # Subject-Predicate specified
                        for obj in self.indexes['spo'][pattern.subject][pattern.predicate]:
                            yield (pattern.subject, pattern.predicate, obj)
                            results_count += 1
                
                elif pattern.predicate and pattern.object:
                    # Use POS index
                    index_used = "pos"
                    for subject in self.indexes['pos'][pattern.predicate][pattern.object]:
                        yield (subject, pattern.predicate, pattern.object)
                        results_count += 1
                
                elif pattern.object and pattern.subject:
                    # Use OSP index
                    index_used = "osp"
                    for predicate in self.indexes['osp'][pattern.object][pattern.subject]:
                        yield (pattern.subject, predicate, pattern.object)
                        results_count += 1
                
                elif pattern.subject:
                    # Use SPO index, iterate all predicates
                    index_used = "spo_partial"
                    for predicate in self.indexes['spo'][pattern.subject]:
                        for obj in self.indexes['spo'][pattern.subject][predicate]:
                            if not pattern.predicate or predicate == pattern.predicate:
                                if not pattern.object or obj == pattern.object:
                                    yield (pattern.subject, predicate, obj)
                                    results_count += 1
                
                elif pattern.predicate:
                    # Use PSO index
                    index_used = "pso"
                    for subject in self.indexes['pso'][pattern.predicate]:
                        for obj in self.indexes['pso'][pattern.predicate][subject]:
                            if not pattern.object or obj == pattern.object:
                                yield (subject, pattern.predicate, obj)
                                results_count += 1
                
                elif pattern.object:
                    # Use OPS index
                    index_used = "ops"
                    for predicate in self.indexes['ops'][pattern.object]:
                        for subject in self.indexes['ops'][pattern.object][predicate]:
                            yield (subject, predicate, pattern.object)
                            results_count += 1
                
                else:
                    # No constraints, return all triples
                    index_used = "full_scan"
                    for triple in self.graph:
                        yield triple
                        results_count += 1
        
        finally:
            execution_time = time.time() - start_time
            self.stats['queries_executed'] += 1
            
            logger.debug(f"Query completed: {execution_time:.4f}s, "
                        f"{results_count} results, index: {index_used}")
    
    def sparql_query(self, query_string: str) -> Result:
        """
        Execute SPARQL query with caching
        
        Args:
            query_string: SPARQL query string
            
        Returns:
            Query results
        """
        query_hash = hashlib.md5(query_string.encode()).hexdigest()
        
        # Check cache
        if query_hash in self.query_cache:
            self.stats['cache_hits'] += 1
            logger.debug("Cache hit for SPARQL query")
            return self.query_cache[query_hash]
        
        start_time = time.time()
        
        try:
            with self.lock:
                results = self.graph.query(query_string)
                
                # Cache results (if cache not full)
                if len(self.query_cache) < self.cache_max_size:
                    self.query_cache[query_hash] = results
                
                execution_time = time.time() - start_time
                self.stats['queries_executed'] += 1
                
                logger.info(f"SPARQL query executed in {execution_time:.4f}s")
                return results
                
        except Exception as e:
            logger.error(f"SPARQL query error: {e}")
            raise
    
    def bulk_add_triples(self, triples: List[Tuple[Node, Node, Node]], batch_size: int = 1000):
        """
        Bulk add triples with batch processing for performance
        
        Args:
            triples: List of triples to add
            batch_size: Number of triples per batch
        """
        total_added = 0
        start_time = time.time()
        
        try:
            with self.lock:
                for i in range(0, len(triples), batch_size):
                    batch = triples[i:i + batch_size]
                    
                    for subject, predicate, obj in batch:
                        if self.add_triple(subject, predicate, obj):
                            total_added += 1
                    
                    if (i // batch_size + 1) % 10 == 0:
                        logger.info(f"Processed {i + len(batch)} triples...")
        
        except Exception as e:
            logger.error(f"Bulk add error: {e}")
            raise
        
        finally:
            execution_time = time.time() - start_time
            logger.info(f"Bulk added {total_added} triples in {execution_time:.2f}s "
                       f"({total_added/execution_time:.0f} triples/sec)")
    
    def get_subjects(self, predicate: Node = None, obj: Node = None) -> Set[Node]:
        """Get all subjects matching predicate and/or object"""
        subjects = set()
        
        if predicate and obj:
            subjects = self.indexes['pos'][predicate][obj]
        elif predicate:
            for obj_dict in self.indexes['pos'][predicate].values():
                subjects.update(obj_dict)
        elif obj:
            for pred_dict in self.indexes['osp'][obj].values():
                subjects.update(pred_dict)
        else:
            subjects = set(self.indexes['spo'].keys())
        
        return subjects
    
    def get_predicates(self, subject: Node = None, obj: Node = None) -> Set[Node]:
        """Get all predicates matching subject and/or object"""
        predicates = set()
        
        if subject and obj:
            predicates = self.indexes['sop'][subject][obj]
        elif subject:
            for pred_dict in self.indexes['spo'][subject].values():
                predicates.update(pred_dict)
        elif obj:
            predicates = set(self.indexes['ops'][obj].keys())
        else:
            predicates = set(self.indexes['pso'].keys())
        
        return predicates
    
    def get_objects(self, subject: Node = None, predicate: Node = None) -> Set[Node]:
        """Get all objects matching subject and/or predicate"""
        objects = set()
        
        if subject and predicate:
            objects = self.indexes['spo'][subject][predicate]
        elif subject:
            for obj_set in self.indexes['spo'][subject].values():
                objects.update(obj_set)
        elif predicate:
            for subj_dict in self.indexes['pso'][predicate].values():
                objects.update(subj_dict)
        else:
            objects = set(self.indexes['osp'].keys())
        
        return objects
    
    def get_statistics(self) -> Dict:
        """Get store statistics"""
        with self.lock:
            return {
                **self.stats,
                'cache_size': len(self.query_cache),
                'cache_hit_ratio': self.stats['cache_hits'] / max(self.stats['queries_executed'], 1),
                'index_sizes': {
                    index_name: len(index_data) 
                    for index_name, index_data in self.indexes.items()
                }
            }
    
    def optimize_indexes(self):
        """Optimize indexes by removing empty entries"""
        logger.info("Optimizing indexes...")
        
        for index_name, index_data in self.indexes.items():
            # Remove empty nested dictionaries
            keys_to_remove = []
            for key1, nested_dict in index_data.items():
                nested_keys_to_remove = []
                for key2, value_set in nested_dict.items():
                    if not value_set:
                        nested_keys_to_remove.append(key2)
                
                for key2 in nested_keys_to_remove:
                    del nested_dict[key2]
                
                if not nested_dict:
                    keys_to_remove.append(key1)
            
            for key1 in keys_to_remove:
                del index_data[key1]
        
        logger.info("Index optimization completed")
    
    def export_data(self, format: str = "turtle") -> str:
        """Export data in specified RDF format"""
        return self.graph.serialize(format=format)
    
    def import_data(self, data: str, format: str = "turtle"):
        """Import data from specified RDF format"""
        with self.lock:
            old_count = len(self.graph)
            self.graph.parse(data=data, format=format)
            new_count = len(self.graph)
            
            # Rebuild indexes after import
            self._rebuild_indexes()
            
            logger.info(f"Imported {new_count - old_count} triples from {format}")
    
    def _rebuild_indexes(self):
        """Rebuild all indexes from scratch"""
        logger.info("Rebuilding indexes...")
        
        # Clear existing indexes
        for index_data in self.indexes.values():
            index_data.clear()
        
        # Rebuild from graph
        for subject, predicate, obj in self.graph:
            self._update_indexes(subject, predicate, obj, None, add=True)
        
        self.stats['total_triples'] = len(self.graph)
        logger.info(f"Rebuilt indexes for {self.stats['total_triples']} triples")
    
    def close(self):
        """Close the store and clean up resources"""
        if hasattr(self.graph.store, 'close'):
            self.graph.store.close()
        logger.info("RDF Triple Store closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Example usage and testing
if __name__ == "__main__":
    # Create store
    store = RDFTripleStore()
    
    # Add namespaces
    store.add_namespace("ex", "http://example.org/")
    store.add_namespace("foaf", "http://xmlns.com/foaf/0.1/")
    
    # Create some test data
    ex = Namespace("http://example.org/")
    
    # Add triples
    store.add_triple(ex.JohnDoe, RDF.type, FOAF.Person)
    store.add_triple(ex.JohnDoe, FOAF.name, Literal("John Doe"))
    store.add_triple(ex.JohnDoe, FOAF.age, Literal(35))
    store.add_triple(ex.JohnDoe, ex.worksAt, ex.Google)
    
    store.add_triple(ex.Google, RDF.type, ex.Organization)
    store.add_triple(ex.Google, RDFS.label, Literal("Google Inc."))
    
    # Query examples
    print("All people:")
    pattern = TriplePattern(predicate=RDF.type, object=FOAF.Person)
    for triple in store.query_triples(pattern):
        print(f"  {triple}")
    
    print("\nAll properties of John Doe:")
    pattern = TriplePattern(subject=ex.JohnDoe)
    for triple in store.query_triples(pattern):
        print(f"  {triple}")
    
    # SPARQL query
    print("\nSPARQL query results:")
    results = store.sparql_query("""
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?name ?age WHERE {
            ?person foaf:name ?name ;
                    foaf:age ?age .
        }
    """)
    
    for row in results:
        print(f"  Name: {row.name}, Age: {row.age}")
    
    # Statistics
    print(f"\nStore statistics: {store.get_statistics()}")