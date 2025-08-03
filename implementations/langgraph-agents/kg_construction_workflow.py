"""
LangGraph Multi-Agent Knowledge Graph Construction Workflow
Advanced system for automated KG building using coordinated AI agents
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from pathlib import Path

from langgraph import StateGraph, START, END
from langgraph.graph import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingStep(Enum):
    EXTRACTION = "extraction"
    VALIDATION = "validation"
    DISAMBIGUATION = "disambiguation" 
    INTEGRATION = "integration"
    QUALITY_ASSURANCE = "quality_assurance"
    REASONING = "reasoning"
    COMPLETE = "complete"

@dataclass
class Entity:
    """Represents an extracted entity"""
    text: str
    entity_type: str
    confidence: float
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_span: tuple = None
    canonical_id: Optional[str] = None

@dataclass
class Relationship:
    """Represents an extracted relationship"""
    subject: Entity
    predicate: str
    object: Entity
    confidence: float
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_span: tuple = None

@dataclass
class ExtractionResult:
    """Results from entity/relationship extraction"""
    entities: List[Entity]
    relationships: List[Relationship] 
    source_document: str
    extraction_confidence: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    """Results from knowledge validation"""
    validated_entities: List[Entity]
    validated_relationships: List[Relationship]
    validation_scores: Dict[str, float]
    failed_validations: List[Dict] = field(default_factory=list)
    external_confirmations: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Document:
    """Input document for processing"""
    path: str
    content: str = ""
    content_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)

class KGConstructionState(TypedDict):
    """State passed between agents in the workflow"""
    messages: Annotated[list, add_messages]
    
    # Input data
    documents: List[Document]
    processing_context: Dict[str, Any]
    
    # Processing results
    extractions: List[ExtractionResult]
    validations: List[ValidationResult]
    disambiguated_entities: Dict[str, Entity]
    integrated_knowledge: Dict[str, Any]
    
    # Quality metrics
    quality_scores: Dict[str, float]
    confidence_levels: Dict[str, float]
    
    # Workflow control
    current_step: ProcessingStep
    errors: List[str]
    retry_count: int
    human_feedback_required: bool

class ExtractionAgent:
    """Agent responsible for extracting entities and relationships from text"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200
        )
        
    async def extract_knowledge(self, state: KGConstructionState) -> KGConstructionState:
        """Extract entities and relationships from documents"""
        logger.info("Starting knowledge extraction")
        
        extractions = []
        
        for doc in state["documents"]:
            try:
                start_time = time.time()
                
                # Split document into manageable chunks
                chunks = self.text_splitter.split_text(doc.content)
                
                doc_entities = []
                doc_relationships = []
                
                for chunk in chunks:
                    chunk_result = await self._extract_from_chunk(chunk, doc.metadata)
                    doc_entities.extend(chunk_result["entities"])
                    doc_relationships.extend(chunk_result["relationships"])
                
                # Deduplicate entities within document
                deduplicated_entities = self._deduplicate_entities(doc_entities)
                
                extraction = ExtractionResult(
                    entities=deduplicated_entities,
                    relationships=doc_relationships,
                    source_document=doc.path,
                    extraction_confidence=self._calculate_extraction_confidence(
                        deduplicated_entities, doc_relationships),
                    processing_time=time.time() - start_time
                )
                
                extractions.append(extraction)
                logger.info(f"Extracted {len(deduplicated_entities)} entities and "
                           f"{len(doc_relationships)} relationships from {doc.path}")
                
            except Exception as e:
                logger.error(f"Extraction failed for {doc.path}: {e}")
                state["errors"].append(f"Extraction error: {e}")
        
        state["extractions"] = extractions
        state["current_step"] = ProcessingStep.VALIDATION
        
        # Add summary message
        total_entities = sum(len(ext.entities) for ext in extractions)
        total_relationships = sum(len(ext.relationships) for ext in extractions)
        
        state["messages"].append(AIMessage(
            content=f"Extraction complete: {total_entities} entities, "
                   f"{total_relationships} relationships extracted from "
                   f"{len(state['documents'])} documents."
        ))
        
        return state
    
    async def _extract_from_chunk(self, text: str, context: Dict[str, Any]) -> Dict:
        """Extract knowledge from a single text chunk"""
        
        extraction_prompt = f"""
        You are an expert knowledge extraction system. Extract all significant entities and relationships from the following text.
        
        For entities, identify:
        1. Entity text (exact mention in text)
        2. Entity type (Person, Organization, Location, Concept, Product, Event, etc.)
        3. Confidence score (0.0-1.0)
        4. Key attributes (properties, descriptions, etc.)
        
        For relationships, identify:
        1. Subject entity
        2. Relationship type (works_at, located_in, founded, leads, etc.)
        3. Object entity
        4. Confidence score (0.0-1.0)
        5. Additional attributes
        
        Context: {context.get('domain', 'general knowledge')}
        
        Text: {text}
        
        Return results in this JSON format:
        {{
            "entities": [
                {{
                    "text": "entity mention",
                    "type": "entity_type",
                    "confidence": 0.95,
                    "attributes": {{"key": "value"}}
                }}
            ],
            "relationships": [
                {{
                    "subject": "subject_entity",
                    "predicate": "relationship_type", 
                    "object": "object_entity",
                    "confidence": 0.9,
                    "attributes": {{"key": "value"}}
                }}
            ]
        }}
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=extraction_prompt)])
            result = json.loads(response.content)
            
            # Convert to Entity and Relationship objects
            entities = [
                Entity(
                    text=e["text"],
                    entity_type=e["type"], 
                    confidence=e["confidence"],
                    attributes=e.get("attributes", {})
                )
                for e in result.get("entities", [])
            ]
            
            relationships = [
                Relationship(
                    subject=Entity(text=r["subject"], entity_type="Unknown", confidence=1.0),
                    predicate=r["predicate"],
                    object=Entity(text=r["object"], entity_type="Unknown", confidence=1.0),
                    confidence=r["confidence"],
                    attributes=r.get("attributes", {})
                )
                for r in result.get("relationships", [])
            ]
            
            return {"entities": entities, "relationships": relationships}
            
        except Exception as e:
            logger.error(f"Failed to extract from chunk: {e}")
            return {"entities": [], "relationships": []}
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities using similarity matching"""
        if not entities:
            return []
        
        # Simple deduplication based on text similarity
        deduplicated = []
        for entity in entities:
            is_duplicate = False
            for existing in deduplicated:
                if (entity.text.lower() == existing.text.lower() or 
                    self._calculate_text_similarity(entity.text, existing.text) > 0.9):
                    # Merge attributes and keep higher confidence
                    if entity.confidence > existing.confidence:
                        existing.confidence = entity.confidence
                    existing.attributes.update(entity.attributes)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(entity)
        
        return deduplicated
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        # Simple Jaccard similarity for now
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_extraction_confidence(self, entities: List[Entity], 
                                       relationships: List[Relationship]) -> float:
        """Calculate overall confidence for extraction results"""
        if not entities and not relationships:
            return 0.0
        
        total_confidence = 0.0
        total_items = 0
        
        for entity in entities:
            total_confidence += entity.confidence
            total_items += 1
        
        for relationship in relationships:
            total_confidence += relationship.confidence
            total_items += 1
        
        return total_confidence / total_items if total_items > 0 else 0.0

class ValidationAgent:
    """Agent responsible for validating extracted knowledge"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.0)
        self.external_kb_sources = ["wikidata", "dbpedia", "conceptnet"]
        
    async def validate_knowledge(self, state: KGConstructionState) -> KGConstructionState:
        """Validate extracted entities and relationships"""
        logger.info("Starting knowledge validation")
        
        validations = []
        
        for extraction in state["extractions"]:
            try:
                validation = await self._validate_extraction(extraction)
                validations.append(validation)
                
                logger.info(f"Validated {len(validation.validated_entities)} entities "
                           f"from {extraction.source_document}")
                
            except Exception as e:
                logger.error(f"Validation failed for {extraction.source_document}: {e}")
                state["errors"].append(f"Validation error: {e}")
        
        state["validations"] = validations
        state["current_step"] = ProcessingStep.DISAMBIGUATION
        
        # Calculate validation metrics
        total_validated = sum(len(v.validated_entities) for v in validations)
        total_extracted = sum(len(e.entities) for e in state["extractions"])
        validation_rate = total_validated / total_extracted if total_extracted > 0 else 0
        
        state["quality_scores"]["validation_rate"] = validation_rate
        
        state["messages"].append(AIMessage(
            content=f"Validation complete: {total_validated}/{total_extracted} entities validated "
                   f"({validation_rate:.2%} validation rate)"
        ))
        
        return state
    
    async def _validate_extraction(self, extraction: ExtractionResult) -> ValidationResult:
        """Validate a single extraction result"""
        
        validated_entities = []
        validated_relationships = []
        validation_scores = {}
        failed_validations = []
        
        # Validate entities
        for entity in extraction.entities:
            validation_result = await self._validate_entity(entity)
            
            if validation_result["is_valid"]:
                entity.confidence = min(entity.confidence, validation_result["confidence"])
                validated_entities.append(entity)
            else:
                failed_validations.append({
                    "entity": entity.text,
                    "reason": validation_result["reason"]
                })
        
        # Validate relationships
        for relationship in extraction.relationships:
            validation_result = await self._validate_relationship(relationship)
            
            if validation_result["is_valid"]:
                relationship.confidence = min(relationship.confidence, validation_result["confidence"])
                validated_relationships.append(relationship)
            else:
                failed_validations.append({
                    "relationship": f"{relationship.subject.text} {relationship.predicate} {relationship.object.text}",
                    "reason": validation_result["reason"]
                })
        
        # Calculate validation scores
        validation_scores["entity_validation_rate"] = len(validated_entities) / len(extraction.entities) if extraction.entities else 0
        validation_scores["relationship_validation_rate"] = len(validated_relationships) / len(extraction.relationships) if extraction.relationships else 0
        
        return ValidationResult(
            validated_entities=validated_entities,
            validated_relationships=validated_relationships,
            validation_scores=validation_scores,
            failed_validations=failed_validations
        )
    
    async def _validate_entity(self, entity: Entity) -> Dict[str, Any]:
        """Validate a single entity using LLM and external sources"""
        
        validation_prompt = f"""
        Validate the following entity extraction:
        
        Entity: {entity.text}
        Type: {entity.entity_type}
        Confidence: {entity.confidence}
        Attributes: {entity.attributes}
        
        Consider:
        1. Is this a real, well-known entity?
        2. Is the entity type appropriate?
        3. Are the attributes consistent with the entity?
        4. Does this seem like a plausible entity extraction?
        
        Return a JSON response:
        {{
            "is_valid": true/false,
            "confidence": 0.0-1.0,
            "reason": "explanation for validation decision",
            "suggested_corrections": {{"field": "corrected_value"}}
        }}
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=validation_prompt)])
            result = json.loads(response.content)
            return result
            
        except Exception as e:
            logger.error(f"Entity validation failed: {e}")
            return {
                "is_valid": True,  # Default to valid if validation fails
                "confidence": entity.confidence * 0.8,  # Reduce confidence
                "reason": f"Validation error: {e}"
            }
    
    async def _validate_relationship(self, relationship: Relationship) -> Dict[str, Any]:
        """Validate a single relationship using LLM reasoning"""
        
        validation_prompt = f"""
        Validate the following relationship extraction:
        
        Subject: {relationship.subject.text}
        Predicate: {relationship.predicate}
        Object: {relationship.object.text}
        Confidence: {relationship.confidence}
        
        Consider:
        1. Is this relationship logically consistent?
        2. Is the predicate appropriate for these entities?
        3. Does this relationship make sense in the real world?
        4. Are the subject and object compatible with this relationship type?
        
        Return a JSON response:
        {{
            "is_valid": true/false,
            "confidence": 0.0-1.0,
            "reason": "explanation for validation decision"
        }}
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=validation_prompt)])
            result = json.loads(response.content)
            return result
            
        except Exception as e:
            logger.error(f"Relationship validation failed: {e}")
            return {
                "is_valid": True,
                "confidence": relationship.confidence * 0.8,
                "reason": f"Validation error: {e}"
            }

class DisambiguationAgent:
    """Agent responsible for entity disambiguation and deduplication"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        self.clustering = HDBSCAN(min_cluster_size=2, metric='cosine')
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        
    async def disambiguate_entities(self, state: KGConstructionState) -> KGConstructionState:
        """Disambiguate and deduplicate entities across all validations"""
        logger.info("Starting entity disambiguation")
        
        # Collect all validated entities
        all_entities = []
        for validation in state["validations"]:
            all_entities.extend(validation.validated_entities)
        
        if not all_entities:
            state["disambiguated_entities"] = {}
            state["current_step"] = ProcessingStep.INTEGRATION
            return state
        
        try:
            # Perform disambiguation
            disambiguated = await self._perform_disambiguation(all_entities)
            
            # Create entity mapping
            entity_mapping = {}
            for canonical_id, entity_group in disambiguated.items():
                # Use the entity with highest confidence as canonical
                canonical_entity = max(entity_group, key=lambda e: e.confidence)
                canonical_entity.canonical_id = canonical_id
                entity_mapping[canonical_id] = canonical_entity
            
            state["disambiguated_entities"] = entity_mapping
            state["current_step"] = ProcessingStep.INTEGRATION
            
            logger.info(f"Disambiguated {len(all_entities)} entities into "
                       f"{len(entity_mapping)} canonical entities")
            
            state["messages"].append(AIMessage(
                content=f"Disambiguation complete: {len(all_entities)} entities "
                       f"resolved to {len(entity_mapping)} canonical entities"
            ))
            
        except Exception as e:
            logger.error(f"Disambiguation failed: {e}")
            state["errors"].append(f"Disambiguation error: {e}")
            # Fall back to no disambiguation
            entity_mapping = {f"entity_{i}": entity for i, entity in enumerate(all_entities)}
            state["disambiguated_entities"] = entity_mapping
            state["current_step"] = ProcessingStep.INTEGRATION
        
        return state
    
    async def _perform_disambiguation(self, entities: List[Entity]) -> Dict[str, List[Entity]]:
        """Perform entity disambiguation using embeddings and clustering"""
        
        # Generate embeddings for all entities
        entity_texts = [f"{e.text} ({e.entity_type})" for e in entities]
        embeddings = self.embedding_model.encode(entity_texts)
        
        # Perform clustering to find potential duplicates
        clusters = self.clustering.fit_predict(embeddings)
        
        # Group entities by cluster
        entity_groups = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id == -1:  # Noise/singleton
                # Create individual group
                group_id = f"singleton_{i}"
                entity_groups[group_id] = [entities[i]]
            else:
                # Add to cluster group
                group_id = f"cluster_{cluster_id}"
                if group_id not in entity_groups:
                    entity_groups[group_id] = []
                entity_groups[group_id].append(entities[i])
        
        # For clusters with multiple entities, use LLM to confirm they're the same
        final_groups = {}
        group_counter = 0
        
        for group_id, entity_group in entity_groups.items():
            if len(entity_group) == 1:
                # Single entity, no disambiguation needed
                final_groups[f"entity_{group_counter}"] = entity_group
                group_counter += 1
            else:
                # Multiple entities, confirm they're duplicates
                confirmed_groups = await self._confirm_entity_duplicates(entity_group)
                for confirmed_group in confirmed_groups:
                    final_groups[f"entity_{group_counter}"] = confirmed_group
                    group_counter += 1
        
        return final_groups
    
    async def _confirm_entity_duplicates(self, entity_group: List[Entity]) -> List[List[Entity]]:
        """Use LLM to confirm whether entities in a group are actually duplicates"""
        
        entity_descriptions = []
        for i, entity in enumerate(entity_group):
            entity_descriptions.append(f"{i}: {entity.text} (type: {entity.entity_type}, confidence: {entity.confidence})")
        
        confirmation_prompt = f"""
        Analyze the following entities that were clustered together as potential duplicates:
        
        {chr(10).join(entity_descriptions)}
        
        Determine which entities refer to the same real-world entity and group them together.
        Consider:
        1. Do they refer to the same person, organization, location, or concept?
        2. Are variations in spelling/naming acceptable (e.g., "NYC" vs "New York City")?
        3. Are they different entities that happen to have similar names?
        
        Return a JSON response with groups of entity indices that refer to the same entity:
        {{
            "groups": [
                [0, 2],  // entities 0 and 2 are the same
                [1],     // entity 1 is unique
                [3, 4]   // entities 3 and 4 are the same
            ]
        }}
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=confirmation_prompt)])
            result = json.loads(response.content)
            
            # Convert index groups back to entity groups
            confirmed_groups = []
            for index_group in result.get("groups", []):
                entity_subset = [entity_group[i] for i in index_group if i < len(entity_group)]
                if entity_subset:
                    confirmed_groups.append(entity_subset)
            
            return confirmed_groups
            
        except Exception as e:
            logger.error(f"Entity duplicate confirmation failed: {e}")
            # Fall back to treating each entity as separate
            return [[entity] for entity in entity_group]

class KGConstructionWorkflow:
    """Main workflow orchestrator for knowledge graph construction"""
    
    def __init__(self):
        self.extraction_agent = ExtractionAgent()
        self.validation_agent = ValidationAgent()
        self.disambiguation_agent = DisambiguationAgent()
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
        
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        workflow = StateGraph(KGConstructionState)
        
        # Add agent nodes
        workflow.add_node("extract", self.extraction_agent.extract_knowledge)
        workflow.add_node("validate", self.validation_agent.validate_knowledge)
        workflow.add_node("disambiguate", self.disambiguation_agent.disambiguate_entities)
        workflow.add_node("complete", self._complete_workflow)
        
        # Define workflow edges
        workflow.add_edge(START, "extract")
        workflow.add_edge("extract", "validate")
        workflow.add_edge("validate", "disambiguate")
        workflow.add_edge("disambiguate", "complete")
        workflow.add_edge("complete", END)
        
        return workflow.compile()
    
    async def _complete_workflow(self, state: KGConstructionState) -> KGConstructionState:
        """Complete the workflow and generate final results"""
        
        state["current_step"] = ProcessingStep.COMPLETE
        
        # Calculate final quality metrics
        total_entities = len(state["disambiguated_entities"])
        total_extractions = sum(len(ext.entities) for ext in state["extractions"])
        
        state["quality_scores"]["final_entity_count"] = total_entities
        state["quality_scores"]["extraction_efficiency"] = total_entities / total_extractions if total_extractions > 0 else 0
        
        # Generate completion message
        state["messages"].append(AIMessage(
            content=f"Knowledge graph construction complete! "
                   f"Final result: {total_entities} canonical entities "
                   f"with quality score: {state['quality_scores'].get('validation_rate', 0):.2%}"
        ))
        
        logger.info("Knowledge graph construction workflow completed")
        return state
    
    async def process_documents(self, documents: List[Document], 
                               context: Dict[str, Any] = None) -> KGConstructionState:
        """Process documents through the complete KG construction pipeline"""
        
        # Initialize state
        initial_state = KGConstructionState(
            messages=[HumanMessage(content=f"Starting KG construction for {len(documents)} documents")],
            documents=documents,
            processing_context=context or {},
            extractions=[],
            validations=[],
            disambiguated_entities={},
            integrated_knowledge={},
            quality_scores={},
            confidence_levels={},
            current_step=ProcessingStep.EXTRACTION,
            errors=[],
            retry_count=0,
            human_feedback_required=False
        )
        
        # Execute the workflow
        result = await self.workflow.ainvoke(initial_state)
        
        return result

# Example usage
async def main():
    """Example usage of the KG construction workflow"""
    
    # Create sample documents
    documents = [
        Document(
            path="sample_doc_1.txt",
            content="John Smith works at Google as a software engineer. Google was founded by Larry Page and Sergey Brin in 1998. The company is headquartered in Mountain View, California.",
            content_type="text",
            metadata={"domain": "technology", "source": "company_info"}
        ),
        Document(
            path="sample_doc_2.txt", 
            content="Apple Inc. is led by Tim Cook as CEO. The company was co-founded by Steve Jobs, Steve Wozniak, and Ronald Wayne. Apple is known for products like the iPhone and MacBook.",
            content_type="text",
            metadata={"domain": "technology", "source": "company_info"}
        )
    ]
    
    # Initialize workflow
    workflow = KGConstructionWorkflow()
    
    # Process documents
    result = await workflow.process_documents(documents)
    
    # Display results
    print("=== Knowledge Graph Construction Results ===")
    print(f"Final entities: {len(result['disambiguated_entities'])}")
    print(f"Quality scores: {result['quality_scores']}")
    print(f"Errors: {result['errors']}")
    
    print("\n=== Extracted Entities ===")
    for entity_id, entity in result["disambiguated_entities"].items():
        print(f"- {entity.text} ({entity.entity_type}) - Confidence: {entity.confidence:.2f}")
    
    print("\n=== Workflow Messages ===")
    for msg in result["messages"]:
        print(f"- {msg.content}")

if __name__ == "__main__":
    asyncio.run(main())