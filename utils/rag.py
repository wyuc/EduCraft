#!/usr/bin/env python3
"""
RAG Utilities for Knowledge Base Retrieval

This module provides utilities for Retrieval-Augmented Generation (RAG) systems.
It includes functionality for embedding documents, building and searching vector indices using Milvus,
and retrieving relevant context for presentations.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple
from tqdm import tqdm

# Import Milvus client instead of FAISS
from pymilvus import MilvusClient

from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagModel

# Import project modules
from utils import get_temp_dir

# RAG settings
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_VECTOR_DIM = 1024  # BGE-M3 embedding dimension

# Configure logging
logger = logging.getLogger(__name__)

# Global model cache to avoid reinitializing models
_MODEL_CACHE = {}

def get_embedding_model(model_name_or_path=DEFAULT_EMBEDDING_MODEL):
    """
    Get an embedding model instance from cache or create a new one.
    
    Args:
        model_name_or_path: Name or path of the embedding model
        
    Returns:
        RAGUtils instance with the specified embedding model
    """
    # Handle None values by using the default model
    if model_name_or_path is None:
        model_name_or_path = DEFAULT_EMBEDDING_MODEL
    
    # Check if model is already in cache
    if model_name_or_path in _MODEL_CACHE:
        logger.info(f"Using cached embedding model: {model_name_or_path}")
        return _MODEL_CACHE[model_name_or_path]
    
    # Create new model instance
    logger.info(f"Initializing new embedding model: {model_name_or_path}")
    model = RAGUtils(model_name_or_path)
    
    # Cache the model for future use
    _MODEL_CACHE[model_name_or_path] = model
    
    return model

class RAGUtils:
    """Utility class for RAG functionality"""
    
    def __init__(self, model_name_or_path=DEFAULT_EMBEDDING_MODEL):
        """Initialize RAG utilities with specified embedding model."""
        # Handle None values by using the default model
        if model_name_or_path is None:
            model_name_or_path = DEFAULT_EMBEDDING_MODEL
            
        logger.info(f"Initializing RAG utilities with model: {model_name_or_path}")
        
        self.model_name = model_name_or_path
        
        # Check if using BGE-M3 or another model
        if 'bge-m3' in model_name_or_path.lower():
            logger.info("Using BGE-M3 embedding model")
            self.model = FlagModel(model_name_or_path, 
                                query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                                use_fp16=True)  # Use FP16 for faster inference
            self.is_bge = True
        else:
            logger.info("Using SentenceTransformer embedding model")
            self.model = SentenceTransformer(model_name_or_path)
            self.is_bge = False
        
        # Get embedding dimension
        if self.is_bge:
            self.dimension = 1024  # BGE-M3 embedding dimension
        else:
            # For other models, get dimension from a test embedding
            test_embedding = self.embed_query("test")
            self.dimension = len(test_embedding)
            
        logger.info(f"Embedding dimension: {self.dimension}")
        
        # Initialize Milvus client
        self.milvus_client = None
    
    def embed_query(self, query):
        """Embed a query text."""
        if self.is_bge:
            embeddings = self.model.encode_queries([query])
            return embeddings[0]
        else:
            return self.model.encode(query)
    
    def embed_documents(self, documents):
        """Embed a list of documents."""
        if not documents:
            return []
            
        if self.is_bge:
            return self.model.encode(documents)
        else:
            return self.model.encode(documents, show_progress_bar=True)
    
    def connect_to_milvus(self, milvus_uri="./milvus.db"):
        """Connect to Milvus server or initialize Milvus Lite."""
        try:
            self.milvus_client = MilvusClient(uri=milvus_uri)
            logger.info(f"Connected to Milvus at {milvus_uri}")
            return self.milvus_client
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def build_milvus_collection(self, collection_name, embeddings, texts=None, metadata=None):
        """Build a Milvus collection from embeddings.
        
        Args:
            collection_name: Name of the collection to create
            embeddings: Numpy array of embeddings or list of embeddings
            texts: Optional list of original texts corresponding to embeddings
            metadata: Optional list of metadata dicts for each embedding
        
        Returns:
            collection_name: Name of the created collection
        """
        if self.milvus_client is None:
            self.connect_to_milvus()
            
        # Convert embeddings to numpy array if not already
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings).astype('float32')
            
        # Drop existing collection if it exists
        if self.milvus_client.has_collection(collection_name):
            logger.info(f"Dropping existing collection: {collection_name}")
            self.milvus_client.drop_collection(collection_name)
            
        # Create new collection
        logger.info(f"Creating new Milvus collection: {collection_name}")
        self.milvus_client.create_collection(
            collection_name=collection_name,
            dimension=self.dimension,
            metric_type="IP",  # Inner product (cosine similarity)
        )
        
        # Prepare data for insertion
        data = []
        for i in range(len(embeddings)):
            entity = {
                "id": i,
                "vector": embeddings[i].astype('float32')
            }
            
            # Add text if provided
            if texts is not None and i < len(texts):
                entity["text"] = texts[i]
                
            # Add metadata if provided
            if metadata is not None and i < len(metadata):
                for key, value in metadata[i].items():
                    entity[key] = value
                    
            data.append(entity)
        
        # Insert data
        logger.info(f"Inserting {len(data)} vectors into collection: {collection_name}")
        self.milvus_client.insert(collection_name=collection_name, data=data)
        
        return collection_name
    
    def search(self, collection_name, query_embedding, k=5, filter_expr=None):
        """Search Milvus collection for similar embeddings.
        
        Args:
            collection_name: Name of collection to search
            query_embedding: Query vector
            k: Number of results to return
            filter_expr: Optional filter expression
            
        Returns:
            (distances, indices): Tuple of distances and indices of nearest neighbors
        """
        if self.milvus_client is None:
            self.connect_to_milvus()
            
        # Ensure query embedding is properly shaped
        if isinstance(query_embedding, np.ndarray) and len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Convert query embedding to list if it's a numpy array
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
            
        # Ensure we have a list of embeddings
        if not isinstance(query_embedding[0], list):
            query_embedding = [query_embedding]
            
        # Execute the search
        search_params = {
            "metric_type": "IP",
            "params": {}
        }
        
        result = self.milvus_client.search(
            collection_name=collection_name,
            data=query_embedding,
            limit=k,
            search_params=search_params,
            filter=filter_expr,
            output_fields=["*"]  # Return all fields
        )
        
        # Extract distances and indices
        distances = []
        indices = []
        
        for hits in result:
            distances_per_query = []
            indices_per_query = []
            for hit in hits:
                distances_per_query.append(hit["distance"])
                indices_per_query.append(hit["id"])
            distances.append(distances_per_query)
            indices.append(indices_per_query)
            
        return np.array(distances), np.array(indices)
    
    def save_milvus_collection(self, collection_name, filepath):
        """Save Milvus collection information to disk.
        
        This method doesn't actually save the collection data, as Milvus
        handles persistence itself. It only saves metadata about the collection
        to make it easier to reconnect later.
        """
        if self.milvus_client is None:
            self.connect_to_milvus()
            
        metadata = {
            "collection_name": collection_name,
            "dimension": self.dimension,
            "model_name": self.model_name
        }
        
        # Save metadata
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved Milvus collection metadata to {filepath}")
        
    def load_milvus_collection(self, filepath):
        """Load Milvus collection information from disk."""
        with open(filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        collection_name = metadata.get("collection_name")
        
        # Check if collection exists
        if self.milvus_client is None:
            self.connect_to_milvus()
            
        if not self.milvus_client.has_collection(collection_name):
            logger.error(f"Collection {collection_name} does not exist")
            return None
            
        logger.info(f"Loaded Milvus collection metadata from {filepath}")
        return collection_name

def build_vector_index(kb_dir: Path, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
    """Build vector index from all chunks in the knowledge base."""
    logger.info("Building vector index for the knowledge base")
    
    # Get RAG utils from cache or initialize new instance
    rag_utils = get_embedding_model(embedding_model)
    
    # Connect to Milvus
    rag_utils.connect_to_milvus(milvus_uri=str(kb_dir / "milvus.db"))
    
    # Get index data
    index_path = kb_dir / "index.json"
    if not index_path.exists():
        logger.error(f"Index file not found: {index_path}. Build knowledge base first.")
        return None
    
    with open(index_path, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
    
    # Collect all chunks
    chunks = []
    chunk_metadata = []
    
    logger.info("Loading chunks from knowledge base")
    for article in tqdm(index_data["articles"], desc="Loading chunks"):
        title = article["title"]
        
        for chunk_info in article["chunks"]:
            chunk_path = kb_dir / chunk_info["path"]
            
            try:
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                chunks.append(content)
                
                # Store metadata for this chunk
                chunk_metadata.append({
                    "title": title,
                    "original_topic": article.get("original_topic", ""),
                    "chunk_path": str(chunk_path),
                    "url": article["url"]
                })
            except Exception as e:
                logger.error(f"Error loading chunk {chunk_path}: {e}")
    
    if not chunks:
        logger.error("No chunks found in the knowledge base")
        return None
    
    # Create embeddings
    logger.info(f"Creating embeddings for {len(chunks)} chunks")
    embeddings = rag_utils.embed_documents(chunks)
    
    # Build Milvus collection
    collection_name = "kb_vector_index"
    rag_utils.build_milvus_collection(
        collection_name=collection_name,
        embeddings=embeddings,
        texts=chunks,
        metadata=chunk_metadata
    )
    
    # Save collection metadata
    metadata_file = kb_dir / "milvus_metadata.json"
    rag_utils.save_milvus_collection(collection_name, metadata_file)
    
    # Save chunk metadata separately (for compatibility with old code)
    chunk_metadata_file = kb_dir / "chunk_metadata.json"
    with open(chunk_metadata_file, 'w', encoding='utf-8') as f:
        json.dump(chunk_metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Vector index built with {len(chunks)} chunks")
    
    return {
        "collection_name": collection_name,
        "chunks": chunks,
        "metadata": chunk_metadata
    }

def search_knowledge_base(kb_dir: Union[str, Path], query: str, top_k: int = 5, embedding_model: Optional[str] = None):
    """Search the knowledge base for relevant chunks."""
    # Convert to Path object if string
    if isinstance(kb_dir, str):
        kb_dir = Path(kb_dir)
        
    # Check if vector index exists
    metadata_file = kb_dir / "milvus_metadata.json"
    chunk_metadata_file = kb_dir / "chunk_metadata.json"
    
    if not metadata_file.exists() or not chunk_metadata_file.exists():
        logger.error("Vector index not found. Build index first.")
        return None
    
    # If embedding_model is None, use the default
    if embedding_model is None:
        embedding_model = DEFAULT_EMBEDDING_MODEL
    
    # Get RAG utils from cache or initialize new instance
    rag_utils = get_embedding_model(embedding_model)
    
    # Connect to Milvus and load collection
    rag_utils.connect_to_milvus(milvus_uri=str(kb_dir / "milvus.db"))
    collection_name = rag_utils.load_milvus_collection(metadata_file)
    
    if not collection_name:
        logger.error("Failed to load Milvus collection")
        return None
    
    # Load chunk metadata
    with open(chunk_metadata_file, 'r', encoding='utf-8') as f:
        chunk_metadata = json.load(f)
    
    # Embed query
    query_embedding = rag_utils.embed_query(query)
    
    # Search
    distances, indices = rag_utils.search(collection_name, query_embedding, k=top_k)
    
    # Collect results
    results = []
    for i, idx in enumerate(indices[0]):
        idx = int(idx)  # Convert to int to handle numpy types
        
        # Load chunk content
        chunk_path = chunk_metadata[idx]["chunk_path"]
        with open(chunk_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        results.append({
            "score": float(distances[0][i]),  # Convert to Python float for JSON serialization
            "metadata": chunk_metadata[idx],
            "content": content
        })
    
    return results


if __name__ == "__main__":
    kb_dir = Path("data/wiki_knowledge_base")
    query = "What is the capital of France?"
    print(search_knowledge_base(kb_dir, query))
