"""
High-Performance Datalog Deductive Database Engine
Provides efficient evaluation of Datalog programs with recursive rules and complex queries
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Iterator, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class Atom:
    """Represents a Datalog atom (predicate with terms)"""
    predicate: str
    terms: List[str] = field(default_factory=list)
    
    def __str__(self):
        if self.terms:
            return f"{self.predicate}({', '.join(self.terms)})"
        return self.predicate
    
    def is_ground(self) -> bool:
        """Check if atom contains only constants (no variables)"""
        return all(not self.is_variable(term) for term in self.terms)
    
    @staticmethod
    def is_variable(term: str) -> bool:
        """Check if term is a variable (starts with uppercase or _)"""
        return term and (term[0].isupper() or term.startswith('_'))
    
    def get_variables(self) -> Set[str]:
        """Get all variables in this atom"""
        return {term for term in self.terms if self.is_variable(term)}
    
    def substitute(self, bindings: Dict[str, str]) -> 'Atom':
        """Apply variable substitutions to create new atom"""
        new_terms = [bindings.get(term, term) for term in self.terms]
        return Atom(self.predicate, new_terms)

@dataclass
class Rule:
    """Represents a Datalog rule (head :- body)"""
    head: Atom
    body: List[Atom] = field(default_factory=list)
    negated_body: List[Atom] = field(default_factory=list)
    
    def __str__(self):
        body_str = ", ".join(str(atom) for atom in self.body)
        if self.negated_body:
            neg_str = ", ".join(f"not {atom}" for atom in self.negated_body)
            if body_str:
                body_str += f", {neg_str}"
            else:
                body_str = neg_str
        
        if body_str:
            return f"{self.head} :- {body_str}."
        return f"{self.head}."
    
    def is_fact(self) -> bool:
        """Check if rule is just a ground fact"""
        return not self.body and not self.negated_body and self.head.is_ground()
    
    def is_recursive(self) -> bool:
        """Check if rule is recursive (head predicate appears in body)"""
        return any(atom.predicate == self.head.predicate for atom in self.body)
    
    def get_variables(self) -> Set[str]:
        """Get all variables used in this rule"""
        variables = self.head.get_variables()
        for atom in self.body + self.negated_body:
            variables.update(atom.get_variables())
        return variables

@dataclass
class Query:
    """Represents a Datalog query"""
    goals: List[Atom]
    
    def __str__(self):
        return "?- " + ", ".join(str(goal) for goal in self.goals) + "."

class DatalogEngine:
    """
    High-performance Datalog deductive database engine
    Supports both bottom-up and top-down evaluation strategies
    """
    
    def __init__(self, max_iterations: int = 1000, enable_parallel: bool = True):
        """
        Initialize Datalog engine
        
        Args:
            max_iterations: Maximum iterations for fixpoint computation
            enable_parallel: Enable parallel rule evaluation
        """
        self.max_iterations = max_iterations
        self.enable_parallel = enable_parallel
        
        # Knowledge base storage
        self.facts = set()  # Ground facts (EDB)
        self.rules = []     # Rules (IDB)
        self.derived_facts = set()  # Derived facts
        
        # Indexing for performance
        self.fact_index = defaultdict(lambda: defaultdict(list))  # predicate -> term_pos -> values
        self.rule_index = defaultdict(list)  # head_predicate -> rules
        
        # Statistics and monitoring
        self.statistics = {
            'facts_count': 0,
            'rules_count': 0,
            'derived_facts_count': 0,
            'evaluations_count': 0,
            'total_evaluation_time': 0.0
        }
        
        # Stratification for negation
        self.strata = []
        self.stratified = False
        
        # Thread pool for parallel execution
        if enable_parallel:
            self.thread_pool = ThreadPoolExecutor(max_workers=4)
        else:
            self.thread_pool = None
        
        logger.info("Datalog engine initialized")
    
    def add_fact(self, fact_str: str) -> bool:
        """
        Add a ground fact to the knowledge base
        
        Args:
            fact_str: Fact as string (e.g., "person(john)")
            
        Returns:
            True if fact was added, False if already exists
        """
        try:
            atom = self._parse_atom(fact_str.rstrip('.'))
            if not atom.is_ground():
                raise ValueError(f"Fact must be ground: {fact_str}")
            
            fact_tuple = (atom.predicate, tuple(atom.terms))
            if fact_tuple not in self.facts:
                self.facts.add(fact_tuple)
                self._update_fact_index(atom)
                self.statistics['facts_count'] += 1
                logger.debug(f"Added fact: {fact_str}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error adding fact '{fact_str}': {e}")
            raise
    
    def add_facts(self, facts: List[str]) -> int:
        """Add multiple facts"""
        added_count = 0
        for fact in facts:
            if self.add_fact(fact):
                added_count += 1
        return added_count
    
    def add_rule(self, rule_str: str) -> bool:
        """
        Add a rule to the knowledge base
        
        Args:
            rule_str: Rule as string (e.g., "employee(X) :- works_at(X, _)")
            
        Returns:
            True if rule was added
        """
        try:
            rule = self._parse_rule(rule_str)
            
            # Validate rule safety
            if not self._is_safe_rule(rule):
                raise ValueError(f"Unsafe rule: {rule_str}")
            
            self.rules.append(rule)
            self._update_rule_index(rule)
            self.statistics['rules_count'] += 1
            self.stratified = False  # Need to re-stratify
            
            logger.debug(f"Added rule: {rule_str}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding rule '{rule_str}': {e}")
            raise
    
    def add_rules(self, rules: List[str]) -> int:
        """Add multiple rules"""
        added_count = 0
        for rule in rules:
            if self.add_rule(rule):
                added_count += 1
        return added_count
    
    def _parse_atom(self, atom_str: str) -> Atom:
        """Parse atom string into Atom object"""
        atom_str = atom_str.strip()
        
        if '(' not in atom_str:
            return Atom(atom_str)
        
        # Extract predicate and terms
        pred_end = atom_str.index('(')
        predicate = atom_str[:pred_end].strip()
        
        # Extract terms between parentheses
        terms_str = atom_str[pred_end+1:-1].strip()
        if not terms_str:
            terms = []
        else:
            # Simple term splitting (handles basic cases)
            terms = [term.strip() for term in terms_str.split(',')]
        
        return Atom(predicate, terms)
    
    def _parse_rule(self, rule_str: str) -> Rule:
        """Parse rule string into Rule object"""
        rule_str = rule_str.strip().rstrip('.')
        
        if ':-' not in rule_str:
            # Just a fact
            head = self._parse_atom(rule_str)
            return Rule(head)
        
        # Split head and body
        head_str, body_str = rule_str.split(':-', 1)
        head = self._parse_atom(head_str.strip())
        
        # Parse body atoms
        body_atoms = []
        negated_atoms = []
        
        # Split body by commas (simple parsing)
        body_parts = [part.strip() for part in body_str.split(',')]
        
        for part in body_parts:
            if part.startswith('not '):
                neg_atom = self._parse_atom(part[4:].strip())
                negated_atoms.append(neg_atom)
            else:
                atom = self._parse_atom(part)
                body_atoms.append(atom)
        
        return Rule(head, body_atoms, negated_atoms)
    
    def _is_safe_rule(self, rule: Rule) -> bool:
        """
        Check if rule is safe (all variables in head appear in positive body)
        """
        if rule.is_fact():
            return True
        
        head_vars = rule.head.get_variables()
        positive_body_vars = set()
        
        for atom in rule.body:
            positive_body_vars.update(atom.get_variables())
        
        # All head variables must appear in positive body
        return head_vars.issubset(positive_body_vars)
    
    def _update_fact_index(self, atom: Atom):
        """Update fact index for efficient retrieval"""
        for i, term in enumerate(atom.terms):
            self.fact_index[atom.predicate][i].append(term)
    
    def _update_rule_index(self, rule: Rule):
        """Update rule index for efficient rule lookup"""
        self.rule_index[rule.head.predicate].append(rule)
    
    def evaluate_bottom_up(self, strategy: str = "semi_naive") -> Set[Tuple]:
        """
        Bottom-up evaluation using fixpoint computation
        
        Args:
            strategy: Evaluation strategy ("naive" or "semi_naive")
            
        Returns:
            Set of all derived facts
        """
        start_time = time.time()
        
        try:
            if strategy == "semi_naive":
                result = self._semi_naive_evaluation()
            else:
                result = self._naive_evaluation()
            
            evaluation_time = time.time() - start_time
            self.statistics['evaluations_count'] += 1
            self.statistics['total_evaluation_time'] += evaluation_time
            self.statistics['derived_facts_count'] = len(result)
            
            logger.info(f"Bottom-up evaluation completed in {evaluation_time:.4f}s, "
                       f"derived {len(result)} facts")
            
            return result
            
        except Exception as e:
            logger.error(f"Bottom-up evaluation failed: {e}")
            raise
    
    def _naive_evaluation(self) -> Set[Tuple]:
        """Naive bottom-up evaluation (recomputes all facts each iteration)"""
        current_facts = set(self.facts)
        
        for iteration in range(self.max_iterations):
            new_facts = set(current_facts)
            
            # Apply all rules
            for rule in self.rules:
                if rule.is_fact():
                    continue
                
                rule_results = self._apply_rule(rule, current_facts)
                new_facts.update(rule_results)
            
            # Check for fixpoint
            if new_facts == current_facts:
                logger.debug(f"Fixpoint reached in {iteration + 1} iterations")
                break
            
            current_facts = new_facts
        
        self.derived_facts = current_facts - self.facts
        return current_facts
    
    def _semi_naive_evaluation(self) -> Set[Tuple]:
        """Semi-naive evaluation (only considers new facts in each iteration)"""
        all_facts = set(self.facts)
        new_facts = set(self.facts)
        
        for iteration in range(self.max_iterations):
            if not new_facts:
                break
            
            iteration_facts = set()
            
            # Apply rules using new facts
            for rule in self.rules:
                if rule.is_fact():
                    continue
                
                # For recursive rules, use semi-naive approach
                if rule.is_recursive():
                    rule_results = self._apply_rule_semi_naive(rule, all_facts, new_facts)
                else:
                    rule_results = self._apply_rule(rule, all_facts)
                
                iteration_facts.update(rule_results)
            
            # Only keep truly new facts
            new_facts = iteration_facts - all_facts
            all_facts.update(new_facts)
            
            if not new_facts:
                logger.debug(f"Fixpoint reached in {iteration + 1} iterations")
                break
        
        self.derived_facts = all_facts - self.facts
        return all_facts
    
    def _apply_rule(self, rule: Rule, facts: Set[Tuple]) -> Set[Tuple]:
        """Apply a single rule to derive new facts"""
        if not rule.body:
            # Ground fact
            if rule.head.is_ground():
                return {(rule.head.predicate, tuple(rule.head.terms))}
            return set()
        
        # Generate all possible bindings
        bindings_list = self._generate_bindings(rule.body, facts)
        
        results = set()
        for bindings in bindings_list:
            # Check negated conditions
            if self._check_negated_conditions(rule.negated_body, bindings, facts):
                # Apply bindings to head
                ground_head = rule.head.substitute(bindings)
                if ground_head.is_ground():
                    results.add((ground_head.predicate, tuple(ground_head.terms)))
        
        return results
    
    def _apply_rule_semi_naive(self, rule: Rule, all_facts: Set[Tuple], 
                              new_facts: Set[Tuple]) -> Set[Tuple]:
        """Apply rule using semi-naive evaluation for recursive rules"""
        if not rule.is_recursive():
            return self._apply_rule(rule, all_facts)
        
        results = set()
        
        # For each body atom that matches the recursive predicate
        for i, body_atom in enumerate(rule.body):
            if body_atom.predicate == rule.head.predicate:
                # Use new_facts for this atom, all_facts for others
                modified_body = rule.body.copy()
                
                # Generate bindings with new facts for recursive atom
                recursive_bindings = self._generate_bindings_with_new_facts(
                    modified_body, all_facts, new_facts, i)
                
                for bindings in recursive_bindings:
                    if self._check_negated_conditions(rule.negated_body, bindings, all_facts):
                        ground_head = rule.head.substitute(bindings)
                        if ground_head.is_ground():
                            results.add((ground_head.predicate, tuple(ground_head.terms)))
        
        return results
    
    def _generate_bindings(self, body_atoms: List[Atom], facts: Set[Tuple]) -> List[Dict[str, str]]:
        """Generate all possible variable bindings for rule body"""
        if not body_atoms:
            return [{}]
        
        # Start with first atom
        first_atom = body_atoms[0]
        bindings_list = []
        
        # Find matching facts for first atom
        for fact in facts:
            if fact[0] == first_atom.predicate and len(fact[1]) == len(first_atom.terms):
                binding = {}
                valid = True
                
                # Try to bind variables
                for i, term in enumerate(first_atom.terms):
                    if Atom.is_variable(term):
                        if term in binding:
                            if binding[term] != fact[1][i]:
                                valid = False
                                break
                        else:
                            binding[term] = fact[1][i]
                    else:
                        # Constant must match
                        if term != fact[1][i]:
                            valid = False
                            break
                
                if valid:
                    bindings_list.append(binding)
        
        # Recursively handle remaining atoms
        if len(body_atoms) > 1:
            final_bindings = []
            for binding in bindings_list:
                remaining_bindings = self._extend_bindings(
                    body_atoms[1:], binding, facts)
                final_bindings.extend(remaining_bindings)
            return final_bindings
        
        return bindings_list
    
    def _extend_bindings(self, remaining_atoms: List[Atom], 
                        current_binding: Dict[str, str], 
                        facts: Set[Tuple]) -> List[Dict[str, str]]:
        """Extend current binding with remaining atoms"""
        if not remaining_atoms:
            return [current_binding]
        
        atom = remaining_atoms[0]
        extended_bindings = []
        
        for fact in facts:
            if fact[0] == atom.predicate and len(fact[1]) == len(atom.terms):
                new_binding = current_binding.copy()
                valid = True
                
                for i, term in enumerate(atom.terms):
                    if Atom.is_variable(term):
                        if term in new_binding:
                            if new_binding[term] != fact[1][i]:
                                valid = False
                                break
                        else:
                            new_binding[term] = fact[1][i]
                    else:
                        if term != fact[1][i]:
                            valid = False
                            break
                
                if valid:
                    # Recursively handle remaining atoms
                    final_bindings = self._extend_bindings(
                        remaining_atoms[1:], new_binding, facts)
                    extended_bindings.extend(final_bindings)
        
        return extended_bindings
    
    def _generate_bindings_with_new_facts(self, body_atoms: List[Atom], 
                                         all_facts: Set[Tuple],
                                         new_facts: Set[Tuple],
                                         new_fact_index: int) -> List[Dict[str, str]]:
        """Generate bindings using new facts for specific atom position"""
        # Use new_facts for the specified index, all_facts for others
        fact_sets = []
        for i, atom in enumerate(body_atoms):
            if i == new_fact_index:
                fact_sets.append(new_facts)
            else:
                fact_sets.append(all_facts)
        
        return self._generate_bindings_multi_sets(body_atoms, fact_sets)
    
    def _generate_bindings_multi_sets(self, body_atoms: List[Atom], 
                                     fact_sets: List[Set[Tuple]]) -> List[Dict[str, str]]:
        """Generate bindings using different fact sets for each atom"""
        if not body_atoms:
            return [{}]
        
        first_atom = body_atoms[0]
        first_facts = fact_sets[0]
        bindings_list = []
        
        for fact in first_facts:
            if fact[0] == first_atom.predicate and len(fact[1]) == len(first_atom.terms):
                binding = {}
                valid = True
                
                for i, term in enumerate(first_atom.terms):
                    if Atom.is_variable(term):
                        binding[term] = fact[1][i]
                    else:
                        if term != fact[1][i]:
                            valid = False
                            break
                
                if valid:
                    bindings_list.append(binding)
        
        if len(body_atoms) > 1:
            final_bindings = []
            for binding in bindings_list:
                remaining_bindings = self._extend_bindings_multi_sets(
                    body_atoms[1:], fact_sets[1:], binding)
                final_bindings.extend(remaining_bindings)
            return final_bindings
        
        return bindings_list
    
    def _extend_bindings_multi_sets(self, remaining_atoms: List[Atom],
                                   remaining_fact_sets: List[Set[Tuple]],
                                   current_binding: Dict[str, str]) -> List[Dict[str, str]]:
        """Extend bindings with remaining atoms using multiple fact sets"""
        if not remaining_atoms:
            return [current_binding]
        
        atom = remaining_atoms[0]
        facts = remaining_fact_sets[0]
        extended_bindings = []
        
        for fact in facts:
            if fact[0] == atom.predicate and len(fact[1]) == len(atom.terms):
                new_binding = current_binding.copy()
                valid = True
                
                for i, term in enumerate(atom.terms):
                    if Atom.is_variable(term):
                        if term in new_binding:
                            if new_binding[term] != fact[1][i]:
                                valid = False
                                break
                        else:
                            new_binding[term] = fact[1][i]
                    else:
                        if term != fact[1][i]:
                            valid = False
                            break
                
                if valid:
                    final_bindings = self._extend_bindings_multi_sets(
                        remaining_atoms[1:], remaining_fact_sets[1:], new_binding)
                    extended_bindings.extend(final_bindings)
        
        return extended_bindings
    
    def _check_negated_conditions(self, negated_atoms: List[Atom], 
                                 bindings: Dict[str, str], 
                                 facts: Set[Tuple]) -> bool:
        """Check that all negated conditions are satisfied"""
        for neg_atom in negated_atoms:
            ground_atom = neg_atom.substitute(bindings)
            if ground_atom.is_ground():
                fact_tuple = (ground_atom.predicate, tuple(ground_atom.terms))
                if fact_tuple in facts:
                    return False  # Negated condition violated
        return True
    
    def query(self, query_str: str) -> List[Dict[str, str]]:
        """
        Execute a query and return results
        
        Args:
            query_str: Query string (e.g., "employee(X)")
            
        Returns:
            List of variable bindings
        """
        try:
            # Ensure facts are derived
            all_facts = self.evaluate_bottom_up()
            
            # Parse query
            if query_str.startswith('?-'):
                query_str = query_str[2:].strip()
            
            query_atom = self._parse_atom(query_str.rstrip('.'))
            
            # Find matching facts
            results = []
            for fact in all_facts:
                if fact[0] == query_atom.predicate and len(fact[1]) == len(query_atom.terms):
                    binding = {}
                    valid = True
                    
                    for i, term in enumerate(query_atom.terms):
                        if Atom.is_variable(term):
                            binding[term] = fact[1][i]
                        else:
                            if term != fact[1][i]:
                                valid = False
                                break
                    
                    if valid:
                        results.append(binding)
            
            logger.info(f"Query '{query_str}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            **self.statistics,
            'avg_evaluation_time': (
                self.statistics['total_evaluation_time'] / 
                max(self.statistics['evaluations_count'], 1)
            ),
            'fact_index_size': sum(
                len(term_idx) for pred_idx in self.fact_index.values() 
                for term_idx in pred_idx.values()
            ),
            'rule_index_size': sum(len(rules) for rules in self.rule_index.values())
        }
    
    def clear(self):
        """Clear all facts, rules, and derived data"""
        self.facts.clear()
        self.rules.clear()
        self.derived_facts.clear()
        self.fact_index.clear()
        self.rule_index.clear()
        
        self.statistics = {
            'facts_count': 0,
            'rules_count': 0,
            'derived_facts_count': 0,
            'evaluations_count': 0,
            'total_evaluation_time': 0.0
        }
        
        logger.info("Datalog engine cleared")
    
    def close(self):
        """Close engine and cleanup resources"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        logger.info("Datalog engine closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Example usage and testing
if __name__ == "__main__":
    # Create engine
    engine = DatalogEngine()
    
    # Add facts
    facts = [
        "person(john)",
        "person(mary)",
        "person(bob)",
        "works_at(john, google)",
        "works_at(mary, microsoft)",
        "works_at(bob, google)",
        "expertise(john, ai)",
        "expertise(mary, databases)",
        "expertise(bob, systems)"
    ]
    
    print("Adding facts...")
    engine.add_facts(facts)
    
    # Add rules
    rules = [
        "employee(X) :- works_at(X, _)",
        "colleague(X, Y) :- works_at(X, Z), works_at(Y, Z), X != Y",
        "expert_employee(X) :- employee(X), expertise(X, _)"
    ]
    
    print("Adding rules...")
    engine.add_rules(rules)
    
    # Execute queries
    print("\nQuery results:")
    
    queries = [
        "employee(X)",
        "colleague(X, Y)",
        "expert_employee(X)",
        "works_at(john, Y)"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = engine.query(query)
        for i, result in enumerate(results):
            print(f"  {i+1}: {result}")
    
    # Show statistics
    print(f"\nEngine statistics: {engine.get_statistics()}")
    
    engine.close()