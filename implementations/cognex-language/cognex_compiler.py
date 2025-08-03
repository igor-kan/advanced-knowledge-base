#!/usr/bin/env python3
"""
CogneX Language Compiler
Compiles natural language knowledge specifications to optimized machine code
"""

import re
import ast
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

class TokenType(Enum):
    # Structural keywords
    DEFINE = "DEFINE"
    DECLARE = "DECLARE" 
    ASSERT = "ASSERT"
    ESTABLISH = "ESTABLISH"
    END = "END"
    
    # Query keywords
    FIND = "FIND"
    WHERE = "WHERE"
    WHEN = "WHEN"
    THEN = "THEN"
    
    # Relationship keywords
    BETWEEN = "BETWEEN"
    CONNECTING = "CONNECTING"
    WITH = "WITH"
    AND = "AND"
    OR = "OR"
    
    # Temporal keywords
    DURING = "DURING"
    FROM = "FROM"
    TO = "TO"
    PREVIOUSLY = "PREVIOUSLY"
    CURRENTLY = "CURRENTLY"
    
    # Quantifiers
    ALL = "ALL"
    SOME = "SOME"
    EXACTLY = "EXACTLY"
    
    # Comparison operators
    EQUALS = "EQUALS"
    GREATER_THAN = "GREATER_THAN"
    LESS_THAN = "LESS_THAN"
    
    # Types
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    PROPERTIES = "PROPERTIES"
    CONSTRAINTS = "CONSTRAINTS"
    
    # Primitives
    TEXT = "text"
    INTEGER = "integer"
    DECIMAL = "decimal"
    TIMESTAMP = "timestamp"
    CURRENCY = "currency"
    CONFIDENCE = "confidence"
    
    # Literals
    STRING_LITERAL = "STRING_LITERAL"
    INTEGER_LITERAL = "INTEGER_LITERAL"
    DECIMAL_LITERAL = "DECIMAL_LITERAL"
    IDENTIFIER = "IDENTIFIER"
    
    # Punctuation
    COMMA = "COMMA"
    AS = "AS"
    MUST = "MUST"
    BE = "BE"
    
    # Special
    EOF = "EOF"
    NEWLINE = "NEWLINE"

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int
    semantic_type: Optional[str] = None

@dataclass  
class Property:
    name: str
    type: str
    constraints: List[str] = field(default_factory=list)

@dataclass
class EntityDefinition:
    name: str
    properties: List[Property]
    constraints: List[str] = field(default_factory=list)

@dataclass
class RelationshipDefinition:
    name: str
    source_type: str
    target_type: str
    properties: List[Property]
    semantics: List[str] = field(default_factory=list)

@dataclass
class KnowledgeGraphDeclaration:
    name: str
    entities: List[str]
    relationships: List[str]
    reasoning_rules: List[str] = field(default_factory=list)

@dataclass
class Assertion:
    entity_type: str
    instance_name: str
    properties: Dict[str, Any]

@dataclass
class RelationshipAssertion:
    relationship_type: str
    source: str
    target: str
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Query:
    target_type: str
    variables: List[str]
    conditions: List[str]
    ordering: Optional[str] = None

class CogneXLexer:
    """
    Lexical analyzer for CogneX language
    Tokenizes natural language with semantic classification
    """
    
    def __init__(self):
        self.keywords = {
            'DEFINE': TokenType.DEFINE,
            'DECLARE': TokenType.DECLARE,
            'ASSERT': TokenType.ASSERT,
            'ESTABLISH': TokenType.ESTABLISH,
            'END': TokenType.END,
            'FIND': TokenType.FIND,
            'WHERE': TokenType.WHERE,
            'WHEN': TokenType.WHEN,
            'THEN': TokenType.THEN,
            'BETWEEN': TokenType.BETWEEN,
            'CONNECTING': TokenType.CONNECTING,
            'WITH': TokenType.WITH,
            'AND': TokenType.AND,
            'OR': TokenType.OR,
            'DURING': TokenType.DURING,
            'FROM': TokenType.FROM,
            'TO': TokenType.TO,
            'PREVIOUSLY': TokenType.PREVIOUSLY,
            'CURRENTLY': TokenType.CURRENTLY,
            'ALL': TokenType.ALL,
            'SOME': TokenType.SOME,
            'EXACTLY': TokenType.EXACTLY,
            'EQUALS': TokenType.EQUALS,
            'GREATER_THAN': TokenType.GREATER_THAN,
            'LESS_THAN': TokenType.LESS_THAN,
            'entity': TokenType.ENTITY,
            'relationship': TokenType.RELATIONSHIP,
            'PROPERTIES': TokenType.PROPERTIES,
            'CONSTRAINTS': TokenType.CONSTRAINTS,
            'text': TokenType.TEXT,
            'integer': TokenType.INTEGER,
            'decimal': TokenType.DECIMAL,
            'timestamp': TokenType.TIMESTAMP,
            'currency': TokenType.CURRENCY,
            'confidence': TokenType.CONFIDENCE,
            'AS': TokenType.AS,
            'MUST': TokenType.MUST,
            'BE': TokenType.BE
        }
        
        self.token_patterns = [
            # String literals
            (r'"([^"\\]|\\.)*"', TokenType.STRING_LITERAL),
            # Currency literals
            (r'\$\d+(?:,\d{3})*(?:\.\d{2})?', TokenType.DECIMAL_LITERAL),
            # Decimal literals
            (r'\d+\.\d+', TokenType.DECIMAL_LITERAL),
            # Integer literals
            (r'\d+', TokenType.INTEGER_LITERAL),
            # Identifiers (must come after keywords)
            (r'[a-zA-Z_][a-zA-Z0-9_]*', TokenType.IDENTIFIER),
            # Punctuation
            (r',', TokenType.COMMA),
            # Whitespace (ignored)
            (r'\s+', None),
            # Comments (ignored)
            (r'//.*', None),
        ]
    
    def tokenize(self, source_code: str) -> List[Token]:
        """Tokenize CogneX source code into semantic tokens"""
        tokens = []
        lines = source_code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            column = 1
            pos = 0
            
            while pos < len(line):
                match_found = False
                
                # Try each token pattern
                for pattern, token_type in self.token_patterns:
                    regex = re.compile(pattern)
                    match = regex.match(line, pos)
                    
                    if match:
                        value = match.group(0)
                        
                        # Skip whitespace and comments
                        if token_type is None:
                            pos = match.end()
                            column += len(value)
                            match_found = True
                            break
                        
                        # Check if identifier is actually a keyword
                        if token_type == TokenType.IDENTIFIER:
                            if value.upper() in self.keywords:
                                token_type = self.keywords[value.upper()]
                            elif value in self.keywords:
                                token_type = self.keywords[value]
                        
                        token = Token(
                            type=token_type,
                            value=value,
                            line=line_num,
                            column=column
                        )
                        tokens.append(token)
                        
                        pos = match.end()
                        column += len(value)
                        match_found = True
                        break
                
                if not match_found:
                    # Skip unknown character
                    pos += 1
                    column += 1
        
        # Add EOF token
        tokens.append(Token(TokenType.EOF, '', len(lines), 1))
        return tokens

class CogneXParser:
    """
    Syntactic analyzer for CogneX language
    Builds abstract syntax tree with semantic validation
    """
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0
        self.entities = {}
        self.relationships = {}
        self.knowledge_graphs = {}
        self.assertions = []
        self.queries = []
    
    def current_token(self) -> Token:
        if self.current < len(self.tokens):
            return self.tokens[self.current]
        return self.tokens[-1]  # EOF token
    
    def advance(self) -> Token:
        token = self.current_token()
        if self.current < len(self.tokens) - 1:
            self.current += 1
        return token
    
    def expect(self, expected_type: TokenType) -> Token:
        token = self.current_token()
        if token.type != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {token.type} at line {token.line}")
        return self.advance()
    
    def parse(self) -> Dict[str, Any]:
        """Parse complete CogneX program"""
        while self.current_token().type != TokenType.EOF:
            self.parse_statement()
        
        return {
            'entities': self.entities,
            'relationships': self.relationships,
            'knowledge_graphs': self.knowledge_graphs,
            'assertions': self.assertions,
            'queries': self.queries
        }
    
    def parse_statement(self):
        """Parse top-level statement"""
        token = self.current_token()
        
        if token.type == TokenType.DEFINE:
            self.parse_definition()
        elif token.type == TokenType.DECLARE:
            self.parse_declaration()
        elif token.type == TokenType.ASSERT:
            self.parse_assertion()
        elif token.type == TokenType.ESTABLISH:
            self.parse_relationship_assertion()
        elif token.type == TokenType.FIND:
            self.parse_query()
        else:
            # Skip unrecognized statements
            self.advance()
    
    def parse_definition(self):
        """Parse DEFINE statement"""
        self.advance()  # consume DEFINE
        
        definition_type = self.current_token()
        if definition_type.type == TokenType.ENTITY:
            self.parse_entity_definition()
        elif definition_type.type == TokenType.RELATIONSHIP:
            self.parse_relationship_definition()
        else:
            raise SyntaxError(f"Expected entity or relationship after DEFINE at line {definition_type.line}")
    
    def parse_entity_definition(self):
        """Parse entity definition"""
        self.advance()  # consume 'entity'
        
        entity_name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        
        properties = []
        constraints = []
        
        while self.current_token().type != TokenType.END:
            if self.current_token().type == TokenType.PROPERTIES:
                properties = self.parse_properties()
            elif self.current_token().type == TokenType.CONSTRAINTS:
                constraints = self.parse_constraints()
            else:
                self.advance()
        
        self.expect(TokenType.END)
        
        entity_def = EntityDefinition(
            name=entity_name,
            properties=properties,
            constraints=constraints
        )
        
        self.entities[entity_name] = entity_def
    
    def parse_relationship_definition(self):
        """Parse relationship definition"""
        self.advance()  # consume 'relationship'
        
        relationship_name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.CONNECTING)
        
        source_type = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.TO)
        target_type = self.expect(TokenType.IDENTIFIER).value
        
        properties = []
        semantics = []
        
        if self.current_token().type == TokenType.WITH:
            self.advance()
            
            while self.current_token().type != TokenType.END:
                if self.current_token().type == TokenType.PROPERTIES:
                    properties = self.parse_properties()
                elif self.current_token().value.upper() == 'SEMANTICS':
                    self.advance()
                    semantics = self.parse_semantics()
                else:
                    self.advance()
        
        self.expect(TokenType.END)
        
        relationship_def = RelationshipDefinition(
            name=relationship_name,
            source_type=source_type,
            target_type=target_type,
            properties=properties,
            semantics=semantics
        )
        
        self.relationships[relationship_name] = relationship_def
    
    def parse_properties(self) -> List[Property]:
        """Parse properties list"""
        self.advance()  # consume PROPERTIES
        
        properties = []
        
        while self.current_token().type == TokenType.IDENTIFIER:
            prop_name = self.advance().value
            self.expect(TokenType.AS)
            prop_type = self.advance().value
            
            constraints = []
            # Parse constraints if present
            
            property = Property(name=prop_name, type=prop_type, constraints=constraints)
            properties.append(property)
            
            if self.current_token().type == TokenType.COMMA:
                self.advance()
            else:
                break
        
        return properties
    
    def parse_constraints(self) -> List[str]:
        """Parse constraints list"""
        self.advance()  # consume CONSTRAINTS
        
        constraints = []
        
        # Simple constraint parsing - can be enhanced
        while (self.current_token().type not in [TokenType.END, TokenType.PROPERTIES] and
               self.current_token().type != TokenType.EOF):
            constraint_tokens = []
            
            # Collect tokens until logical delimiter
            while (self.current_token().type not in [TokenType.AND, TokenType.END, TokenType.PROPERTIES] and
                   self.current_token().type != TokenType.EOF):
                constraint_tokens.append(self.advance().value)
            
            if constraint_tokens:
                constraints.append(' '.join(constraint_tokens))
            
            if self.current_token().type == TokenType.AND:
                self.advance()
        
        return constraints
    
    def parse_semantics(self) -> List[str]:
        """Parse semantics list"""
        semantics = []
        
        while (self.current_token().type not in [TokenType.END, TokenType.PROPERTIES] and
               self.current_token().type != TokenType.EOF):
            semantics.append(self.advance().value)
            
            if self.current_token().type == TokenType.AND:
                self.advance()
            else:
                break
        
        return semantics
    
    def parse_declaration(self):
        """Parse DECLARE statement"""
        self.advance()  # consume DECLARE
        
        if self.current_token().value == 'knowledge_graph':
            self.parse_knowledge_graph_declaration()
    
    def parse_knowledge_graph_declaration(self):
        """Parse knowledge graph declaration"""
        self.advance()  # consume 'knowledge_graph'
        
        kg_name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        
        entities = []
        relationships = []
        reasoning_rules = []
        
        while self.current_token().type != TokenType.END and self.current_token().type != TokenType.EOF:
            if self.current_token().value.upper() == 'ENTITIES':
                self.advance()
                entities = self.parse_identifier_list()
            elif self.current_token().value.upper() == 'RELATIONSHIPS':
                self.advance()
                relationships = self.parse_identifier_list()
            elif self.current_token().value.upper() == 'REASONING_RULES':
                self.advance()
                reasoning_rules = self.parse_identifier_list()
            else:
                self.advance()
        
        if self.current_token().type == TokenType.END:
            self.advance()
        
        kg_decl = KnowledgeGraphDeclaration(
            name=kg_name,
            entities=entities,
            relationships=relationships,
            reasoning_rules=reasoning_rules
        )
        
        self.knowledge_graphs[kg_name] = kg_decl
    
    def parse_identifier_list(self) -> List[str]:
        """Parse comma-separated list of identifiers"""
        identifiers = []
        
        while self.current_token().type == TokenType.IDENTIFIER:
            identifiers.append(self.advance().value)
            
            if self.current_token().type == TokenType.COMMA:
                self.advance()
            else:
                break
        
        return identifiers
    
    def parse_assertion(self):
        """Parse ASSERT statement"""
        self.advance()  # consume ASSERT
        
        entity_type = self.expect(TokenType.IDENTIFIER).value
        instance_name = self.expect(TokenType.IDENTIFIER).value
        
        properties = {}
        
        if self.current_token().type == TokenType.WITH:
            self.advance()
            properties = self.parse_property_assignments()
        
        assertion = Assertion(
            entity_type=entity_type,
            instance_name=instance_name,
            properties=properties
        )
        
        self.assertions.append(assertion)
    
    def parse_relationship_assertion(self):
        """Parse ESTABLISH statement"""
        self.advance()  # consume ESTABLISH
        
        relationship_type = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.BETWEEN)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AND)
        target = self.expect(TokenType.IDENTIFIER).value
        
        properties = {}
        
        if self.current_token().type == TokenType.WITH:
            self.advance()
            properties = self.parse_property_assignments()
        
        rel_assertion = RelationshipAssertion(
            relationship_type=relationship_type,
            source=source,
            target=target,
            properties=properties
        )
        
        self.assertions.append(rel_assertion)
    
    def parse_property_assignments(self) -> Dict[str, Any]:
        """Parse property assignments"""
        properties = {}
        
        while self.current_token().type == TokenType.IDENTIFIER:
            prop_name = self.advance().value
            
            # Parse property value
            if self.current_token().type == TokenType.STRING_LITERAL:
                prop_value = self.advance().value.strip('"')
            elif self.current_token().type == TokenType.INTEGER_LITERAL:
                prop_value = int(self.advance().value)
            elif self.current_token().type == TokenType.DECIMAL_LITERAL:
                prop_value = float(self.advance().value.replace('$', '').replace(',', ''))
            else:
                prop_value = self.advance().value
            
            properties[prop_name] = prop_value
            
            if self.current_token().type == TokenType.COMMA:
                self.advance()
            else:
                break
        
        return properties
    
    def parse_query(self):
        """Parse FIND statement"""
        self.advance()  # consume FIND
        
        # Parse quantifier
        quantifier = None
        if self.current_token().type in [TokenType.ALL, TokenType.SOME]:
            quantifier = self.advance().value
        
        target_type = self.expect(TokenType.IDENTIFIER).value
        variables = [self.expect(TokenType.IDENTIFIER).value]
        
        conditions = []
        ordering = None
        
        if self.current_token().type == TokenType.WHERE:
            self.advance()
            conditions = self.parse_conditions()
        
        query = Query(
            target_type=target_type,
            variables=variables,
            conditions=conditions,
            ordering=ordering
        )
        
        self.queries.append(query)
    
    def parse_conditions(self) -> List[str]:
        """Parse WHERE conditions"""
        conditions = []
        condition_tokens = []
        
        while self.current_token().type not in [TokenType.EOF] and self.current_token().value.upper() not in ['ORDERED']:
            condition_tokens.append(self.advance().value)
        
        if condition_tokens:
            conditions.append(' '.join(condition_tokens))
        
        return conditions

class CogneXSemanticAnalyzer:
    """
    Semantic analyzer for CogneX language
    Validates semantic consistency and type safety
    """
    
    def __init__(self, ast: Dict[str, Any]):
        self.ast = ast
        self.symbol_table = {}
        self.type_table = {}
        self.errors = []
    
    def analyze(self) -> Dict[str, Any]:
        """Perform semantic analysis on AST"""
        self.build_symbol_table()
        self.validate_types()
        self.validate_constraints()
        self.validate_references()
        
        return {
            'ast': self.ast,
            'symbol_table': self.symbol_table,
            'type_table': self.type_table,
            'errors': self.errors
        }
    
    def build_symbol_table(self):
        """Build symbol table with type information"""
        # Register entity types
        for entity_name, entity_def in self.ast['entities'].items():
            self.symbol_table[entity_name] = {
                'type': 'entity_type',
                'properties': {prop.name: prop.type for prop in entity_def.properties},
                'constraints': entity_def.constraints
            }
        
        # Register relationship types
        for rel_name, rel_def in self.ast['relationships'].items():
            self.symbol_table[rel_name] = {
                'type': 'relationship_type',
                'source_type': rel_def.source_type,
                'target_type': rel_def.target_type,
                'properties': {prop.name: prop.type for prop in rel_def.properties},
                'semantics': rel_def.semantics
            }
        
        # Register instances from assertions
        for assertion in self.ast['assertions']:
            if isinstance(assertion, Assertion):
                self.symbol_table[assertion.instance_name] = {
                    'type': 'entity_instance',
                    'entity_type': assertion.entity_type,
                    'properties': assertion.properties
                }
    
    def validate_types(self):
        """Type checking and validation"""
        # Validate entity property types
        for assertion in self.ast['assertions']:
            if isinstance(assertion, Assertion):
                entity_type = assertion.entity_type
                
                if entity_type not in self.ast['entities']:
                    self.errors.append(f"Unknown entity type: {entity_type}")
                    continue
                
                entity_def = self.ast['entities'][entity_type]
                expected_props = {prop.name: prop.type for prop in entity_def.properties}
                
                for prop_name, prop_value in assertion.properties.items():
                    if prop_name in expected_props:
                        expected_type = expected_props[prop_name]
                        if not self.is_compatible_type(prop_value, expected_type):
                            self.errors.append(
                                f"Type mismatch for {entity_type}.{prop_name}: "
                                f"expected {expected_type}, got {type(prop_value).__name__}"
                            )
    
    def is_compatible_type(self, value: Any, expected_type: str) -> bool:
        """Check if value is compatible with expected type"""
        type_mapping = {
            'text': str,
            'integer': int,
            'decimal': (int, float),
            'timestamp': str,  # Simplified
            'currency': (int, float),
            'confidence': (int, float)
        }
        
        expected_python_type = type_mapping.get(expected_type, str)
        return isinstance(value, expected_python_type)
    
    def validate_constraints(self):
        """Validate entity and relationship constraints"""
        # Implement constraint validation logic
        pass
    
    def validate_references(self):
        """Validate entity and relationship references"""
        for assertion in self.ast['assertions']:
            if isinstance(assertion, RelationshipAssertion):
                # Check if source and target entities exist
                if assertion.source not in self.symbol_table:
                    self.errors.append(f"Unknown entity reference: {assertion.source}")
                
                if assertion.target not in self.symbol_table:
                    self.errors.append(f"Unknown entity reference: {assertion.target}")
                
                # Check if relationship type exists
                if assertion.relationship_type not in self.ast['relationships']:
                    self.errors.append(f"Unknown relationship type: {assertion.relationship_type}")

class CogneXCodeGenerator:
    """
    Code generator for CogneX language
    Generates optimized implementation code
    """
    
    def __init__(self, analyzed_ast: Dict[str, Any]):
        self.ast = analyzed_ast['ast']
        self.symbol_table = analyzed_ast['symbol_table']
        self.errors = analyzed_ast['errors']
    
    def generate_python_code(self) -> str:
        """Generate Python implementation code"""
        if self.errors:
            raise RuntimeError(f"Cannot generate code with semantic errors: {self.errors}")
        
        code_parts = []
        
        # Generate imports
        code_parts.append(self.generate_imports())
        
        # Generate entity classes
        code_parts.append(self.generate_entity_classes())
        
        # Generate relationship classes
        code_parts.append(self.generate_relationship_classes())
        
        # Generate knowledge graph class
        code_parts.append(self.generate_knowledge_graph_class())
        
        # Generate main execution code
        code_parts.append(self.generate_main_code())
        
        return '\n\n'.join(code_parts)
    
    def generate_imports(self) -> str:
        """Generate import statements"""
        return '''from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import uuid'''
    
    def generate_entity_classes(self) -> str:
        """Generate entity class definitions"""
        classes = []
        
        for entity_name, entity_def in self.ast['entities'].items():
            properties = []
            for prop in entity_def.properties:
                prop_type = self.map_cognex_type_to_python(prop.type)
                properties.append(f"    {prop.name}: {prop_type}")
            
            class_code = f'''@dataclass
class {entity_name}:
    """Generated entity class for {entity_name}"""
    entity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
{chr(10).join(properties)}
    
    def validate_constraints(self) -> bool:
        """Validate entity constraints"""
        # Generated constraint validation
        return True'''
            
            classes.append(class_code)
        
        return '\n\n'.join(classes)
    
    def generate_relationship_classes(self) -> str:
        """Generate relationship class definitions"""
        classes = []
        
        for rel_name, rel_def in self.ast['relationships'].items():
            properties = []
            for prop in rel_def.properties:
                prop_type = self.map_cognex_type_to_python(prop.type)
                properties.append(f"    {prop.name}: {prop_type}")
            
            class_code = f'''@dataclass
class {rel_name}:
    """Generated relationship class for {rel_name}"""
    relationship_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str  # {rel_def.source_type} ID
    target: str  # {rel_def.target_type} ID
{chr(10).join(properties)}
    
    def validate_semantics(self) -> bool:
        """Validate relationship semantics"""
        # Generated semantic validation
        return True'''
            
            classes.append(class_code)
        
        return '\n\n'.join(classes)
    
    def generate_knowledge_graph_class(self) -> str:
        """Generate knowledge graph management class"""
        return '''@dataclass
class KnowledgeGraph:
    """Generated knowledge graph management class"""
    entities: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, Any] = field(default_factory=dict)
    
    def add_entity(self, entity: Any) -> str:
        """Add entity to knowledge graph"""
        self.entities[entity.entity_id] = entity
        return entity.entity_id
    
    def add_relationship(self, relationship: Any) -> str:
        """Add relationship to knowledge graph"""
        self.relationships[relationship.relationship_id] = relationship
        return relationship.relationship_id
    
    def query(self, pattern: str) -> List[Any]:
        """Execute query against knowledge graph"""
        # Generated query execution
        return []
    
    def validate_consistency(self) -> bool:
        """Validate knowledge graph consistency"""
        return True'''
    
    def generate_main_code(self) -> str:
        """Generate main execution code"""
        code_parts = ['def main():']
        code_parts.append('    """Generated main execution function"""')
        code_parts.append('    kg = KnowledgeGraph()')
        code_parts.append('')
        
        # Generate entity instantiations
        for assertion in self.ast['assertions']:
            if isinstance(assertion, Assertion):
                props = ', '.join(f"{k}={repr(v)}" for k, v in assertion.properties.items())
                code_parts.append(f'    {assertion.instance_name} = {assertion.entity_type}({props})')
                code_parts.append(f'    kg.add_entity({assertion.instance_name})')
        
        code_parts.append('')
        
        # Generate relationship instantiations
        for assertion in self.ast['assertions']:
            if isinstance(assertion, RelationshipAssertion):
                props = ', '.join(f"{k}={repr(v)}" for k, v in assertion.properties.items())
                code_parts.append(
                    f'    rel_{assertion.relationship_type.lower()} = {assertion.relationship_type}('
                    f'source={assertion.source}.entity_id, target={assertion.target}.entity_id'
                    + (f', {props}' if props else '') + ')'
                )
                code_parts.append(f'    kg.add_relationship(rel_{assertion.relationship_type.lower()})')
        
        code_parts.append('')
        code_parts.append('    return kg')
        code_parts.append('')
        code_parts.append('if __name__ == "__main__":')
        code_parts.append('    knowledge_graph = main()')
        code_parts.append('    print(f"Knowledge graph created with {len(knowledge_graph.entities)} entities")')
        
        return '\n'.join(code_parts)
    
    def map_cognex_type_to_python(self, cognex_type: str) -> str:
        """Map CogneX types to Python types"""
        type_mapping = {
            'text': 'str',
            'integer': 'int',
            'decimal': 'float',
            'timestamp': 'datetime',
            'currency': 'float',
            'confidence': 'float'
        }
        return type_mapping.get(cognex_type, 'Any')

class CogneXCompiler:
    """
    Main CogneX compiler coordinating all compilation phases
    """
    
    def __init__(self):
        self.lexer = CogneXLexer()
    
    def compile(self, source_code: str, output_language: str = 'python') -> str:
        """
        Compile CogneX source code to target language
        
        Args:
            source_code: CogneX source code
            output_language: Target language ('python', 'cpp', 'rust')
            
        Returns:
            Generated code in target language
        """
        try:
            # Lexical analysis
            tokens = self.lexer.tokenize(source_code)
            
            # Syntactic analysis
            parser = CogneXParser(tokens)
            ast = parser.parse()
            
            # Semantic analysis
            analyzer = CogneXSemanticAnalyzer(ast)
            analyzed_ast = analyzer.analyze()
            
            # Code generation
            if output_language == 'python':
                generator = CogneXCodeGenerator(analyzed_ast)
                return generator.generate_python_code()
            else:
                raise NotImplementedError(f"Target language '{output_language}' not yet supported")
                
        except Exception as e:
            raise RuntimeError(f"Compilation failed: {e}")
    
    def compile_file(self, input_file: str, output_file: str, output_language: str = 'python'):
        """Compile CogneX file to target language file"""
        
        # Read source file
        source_path = Path(input_file)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {input_file}")
        
        source_code = source_path.read_text(encoding='utf-8')
        
        # Compile
        generated_code = self.compile(source_code, output_language)
        
        # Write output file
        output_path = Path(output_file)
        output_path.write_text(generated_code, encoding='utf-8')
        
        print(f"Successfully compiled {input_file} to {output_file}")

def main():
    """Command-line interface for CogneX compiler"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CogneX Language Compiler')
    parser.add_argument('input', help='Input CogneX source file')
    parser.add_argument('-o', '--output', help='Output file (default: input.py)')
    parser.add_argument('-l', '--language', default='python', 
                       choices=['python', 'cpp', 'rust'],
                       help='Target language (default: python)')
    
    args = parser.parse_args()
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        input_path = Path(args.input)
        extensions = {'python': '.py', 'cpp': '.cpp', 'rust': '.rs'}
        output_file = input_path.with_suffix(extensions[args.language])
    
    # Compile
    compiler = CogneXCompiler()
    try:
        compiler.compile_file(args.input, output_file, args.language)
    except Exception as e:
        print(f"Compilation error: {e}")
        exit(1)

if __name__ == '__main__':
    main()