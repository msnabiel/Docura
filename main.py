import os
import json
import time
import re
import random
import httpx
from typing import List, Union, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from contextlib import asynccontextmanager
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import torch
from utils import  (extract_final_answer, 
                    clean_string_post, 
                    classify_url_by_response, 
                    clean_text_for_gemini,
                    extract_json_from_response, 
                    is_url)
from cache import DocumentCache
from models import (
    HackRxRunRequest,
    ParseRequest,
    ParseBatchRequest,
    SearchRequest,
    MultiSearchRequest,
    IngestionResult
)
from google.genai import types
import google
from config import (
    generation_config
)
# Import the direct text extraction functions
from text_extractor import (
    extract_text_from_url, 
    extract_text_from_multiple_urls, 
    extract_text_from_bytes,
    extract_text_from_web_url,
    extract_text_from_api_url,
    CleaningOptions
)
import logging
# Simple logging setup that definitely works
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'app.log'), mode='a'),
        logging.StreamHandler()
    ],
    force=True
)
# Get logger
logger = logging.getLogger("app")

# Test logging is working
logger.info("=== LOGGING SYSTEM INITIALIZED ===")
logger.info(f"Log file: app.log")
logger.info(f"Log level: {logging.getLevelName(logger.level)}")
logger.info("=== LOGGING SYSTEM READY ===\n")

# GPU detection and optimization
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Configuration
UPLOAD_DIR = "./uploaded_docs"
CACHE_DIR = "/tmp/document_cache"  # Cache directory for processed documents
BEARER_API_TOKEN = "Bearer d36115d114e821579e99ec6c83bad42017fc11f9cca5a8cbeb1b471c3ae493e4"
# Load from .env.local
load_dotenv(dotenv_path=".env.local")
#GEMINI_API_KEY_1 = os.getenv("GEMINI_API_KEY_1")
#GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2")
#GEMINI_API_KEY_3 = os.getenv("GEMINI_API_KEY_3")
#GEMINI_API_KEY_4 = os.getenv("GEMINI_API_KEY_4")
#GEMINI_API_KEY_5 = os.getenv("GEMINI_API_KEY_5")
#GEMINI_API_KEY_6 = os.getenv("GEMINI_API_KEY_6")
#GEMINI_API_KEY_7 = os.getenv("GEMINI_API_KEY_7")
#GEMINI_API_KEY_8 = os.getenv("GEMINI_API_KEY_8")
#GEMINI_API_KEY_9 = os.getenv("GEMINI_API_KEY_9")
#GEMINI_API_KEY_10 = os.getenv("GEMINI_API_KEY_10")
GEMINI_API_KEY_PAID = os.getenv("GEMINI_API_KEY_PAID")
client = google.genai.Client(api_key=GEMINI_API_KEY_PAID)

# Validate API keys and create a list of valid keys
def get_valid_gemini_keys():
    """Get a list of valid Gemini API keys"""
    """all_keys = [
        GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3, GEMINI_API_KEY_4, GEMINI_API_KEY_5,
        GEMINI_API_KEY_6, GEMINI_API_KEY_7, GEMINI_API_KEY_8, GEMINI_API_KEY_9, GEMINI_API_KEY_10,
        GEMINI_API_KEY_PAID
    ]"""
    all_keys = [GEMINI_API_KEY_PAID]
    valid_keys = [key for key in all_keys if key and key.strip()]
    
    if not valid_keys:
        logger.warning("No valid Gemini API keys found. Gemini functionality will be disabled.")
        return []
    
    logger.info(f"Found {len(valid_keys)} valid Gemini API keys")
    return valid_keys

VALID_GEMINI_KEYS = get_valid_gemini_keys()
MAX_WORKERS = 10
CHUNK_SIZE = 512
OVERLAP_SIZE = 50

# Embedding optimization settings
EMBEDDING_BATCH_SIZE = 64 if DEVICE == "cuda" else 32  # Larger batches for GPU
EMBEDDING_WORKERS = 2 if DEVICE == "cuda" else 4  # Fewer workers for GPU to avoid memory issues

# Text extraction is now handled directly via imported functions

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY_PAID)
def load_prompt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    
system_prompt = load_prompt("prompts/system_prompt.txt")
"""cache_system_prompt = client.caches.create(
    model="gemini-2.5-flash",
    config=types.CreateCachedContentConfig(
        system_instruction=system_prompt
    )
)"""
#generation_config["cached_content"] = cache_system_prompt.name
generation_config['system_instruction'] = system_prompt

#print(generation_config)
generation_config = types.GenerateContentConfig(**generation_config)
class DocumentChunk:
    def __init__(self, text: str, metadata: Dict[str, Any] = None):
        self.text = text
        self.metadata = metadata or {}
        self.chunk_id = None
        self.embedding = None
        self.semantic_score = 0.0
        self.lexical_score = 0.0
        self.combined_score = 0.0
        self.relevance_score = 0.0

class SearchResult:
    def __init__(self, chunk: DocumentChunk, semantic_score: float = 0.0, 
                 lexical_score: float = 0.0, combined_score: float = 0.0,
                 search_strategy: str = "ensemble"):
        self.chunk = chunk
        self.semantic_score = semantic_score
        self.lexical_score = lexical_score
        self.combined_score = combined_score
        self.search_strategy = search_strategy
        self.rank = 0

class TextExtractionSystem:
    def __init__(self):
        # Multiple embedding models for ensemble search
        self.bge_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        self.all_mini_model = SentenceTransformer('all-MiniLM-L6-v2')  # Alternative model
        #self.bge_model_1024 = SentenceTransformer('BAAI/bge-large-en')
        #self.intfloat_model_1024 = SentenceTransformer('intfloat/e5-large')
    
        # Move models to appropriate device
        if DEVICE == "cuda":
            self.bge_model = self.bge_model.to(DEVICE)
            self.all_mini_model = self.all_mini_model.to(DEVICE)
            logger.info("Models moved to GPU for faster inference")
        
        # Search indices
        self.faiss_index = None
        self.bm25 = None
        self.chunks = []
        self.bge_embeddings = None  # Separate BGE embeddings
        self.all_mini_embeddings = None  # Separate All-MiniLM embeddings
        self.bge_embeddings_1024 = None
        self.intfloat_embeddings_1024 = None
        
        # Search configuration
        self.search_strategies = {
            'semantic': 0.6,
            'lexical': 0.4,
            'hybrid': 1.0,
            'ensemble': 1.0
        }
        
        # Reranking configuration
        self.rerank_top_k = 20
        self.final_top_k = 10
        
        logger.info("Enhanced TextExtractionSystem initialized with multiple search strategies")
    
    async def extract_text_from_url(self, url: str) -> str:
        """Extract text from document URL using direct extraction with caching"""
        try:
            logger.info(f"Extracting text from URL: {url}")

            # === Check for .bin file and skip ===
            if url.lower().endswith('.bin'):
                logger.warning(f"Skipping .bin file: {url}")
                raise HTTPException(status_code=400, detail="Unsupported file type: .bin files are not supported.")

            # === Classify URL type ===
            url_type = classify_url_by_response(url)
            logger.info(f"URL {url} classified as: {url_type}")

            # === Check cache first ===
            cached_data = document_cache.get(url)
            if cached_data:
                logger.info(f"Using cached data for {url}")
                return cached_data["data"]["text"]

            # === Handle different URL types ===
            if url_type == 'web':
                # Use web parsing for HTML pages
                logger.info(f"Processing web page: {url}")
                text, metadata = extract_text_from_web_url(
                    url,
                    cleaning_options=CleaningOptions(
                        normalize_unicode=True,
                        remove_urls=False,
                        remove_emails=False,
                        clean_whitespace=True,
                        preserve_structure=True,
                        max_length=100000,
                        enable_ocr=False,  # No OCR for web pages
                        aggressive_cleaning=True
                    )
                )
            elif url_type == 'document':
                # Use document extraction for files
                logger.info(f"Processing document: {url}")
                text, metadata = extract_text_from_url(
                    url, 
                    enable_ocr=True,
                    cleaning_options=CleaningOptions(
                        normalize_unicode=True,
                        remove_urls=False,
                        remove_emails=False,
                        clean_whitespace=True,
                        preserve_structure=True,
                        max_length=100000,
                        enable_ocr=True,
                        aggressive_cleaning=True
                    )
                )
            elif url_type == 'api':
                # Handle API responses
                logger.info(f"Processing API response: {url}")
                text, metadata = extract_text_from_api_url(
                    url,
                    cleaning_options=CleaningOptions(
                        normalize_unicode=True,
                        remove_urls=False,
                        remove_emails=False,
                        clean_whitespace=True,
                        preserve_structure=True,
                        max_length=100000,
                        enable_ocr=False,
                        aggressive_cleaning=True
                    )
                )
            else:
                # Fallback to document extraction
                logger.warning(f"Unknown URL type '{url_type}', trying document extraction: {url}")
                text, metadata = extract_text_from_url(
                    url, 
                    enable_ocr=True,
                    cleaning_options=CleaningOptions(
                        normalize_unicode=True,
                        remove_urls=False,
                        remove_emails=False,
                        clean_whitespace=True,
                        preserve_structure=True,
                        max_length=100000,
                        enable_ocr=True,
                        aggressive_cleaning=True
                    )
                )

            # === Cache result ===
            cache_data = {
                "text": text,
                "metadata": metadata,
                "processing_time": metadata.get('processing_time', 0),
                "url_type": url_type
            }
            document_cache.set(url, cache_data)

            logger.info(f"Successfully extracted text from {url} (type: {url_type}, processing time: {metadata.get('processing_time', 0):.2f}s)")
            return text

        except HTTPException:
            raise  # re-raise known HTTP errors
        except Exception as e:
            logger.error(f"Error extracting text from URL {url}: {e}")
            raise HTTPException(status_code=500, detail=f"Text extraction error: {e}")
    async def extract_text_from_urls_batch(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Extract text from multiple document URLs using direct batch extraction"""
        try:
            logger.info(f"Extracting text from {len(urls)} URLs using direct batch extraction")
            
            # Classify URLs and process them accordingly
            formatted_results = []
            
            for url in urls:
                try:
                    url_type = classify_url_by_response(url)
                    logger.info(f"URL {url} classified as: {url_type}")
                    
                    if url_type == 'web':
                        text, metadata = extract_text_from_web_url(
                            url,
                            cleaning_options=CleaningOptions(
                                normalize_unicode=True,
                                remove_urls=False,
                                remove_emails=False,
                                clean_whitespace=True,
                                preserve_structure=True,
                                max_length=100000,
                                enable_ocr=False,
                                aggressive_cleaning=True
                            )
                        )
                    elif url_type == 'api':
                        text, metadata = extract_text_from_api_url(
                            url,
                            cleaning_options=CleaningOptions(
                                normalize_unicode=True,
                                remove_urls=False,
                                remove_emails=False,
                                clean_whitespace=True,
                                preserve_structure=True,
                                max_length=100000,
                                enable_ocr=False,
                                aggressive_cleaning=True
                            )
                        )
                    elif url_type == 'document':
                        text, metadata = extract_text_from_url(
                            url,
                            enable_ocr=True,
                            cleaning_options=CleaningOptions(
                                normalize_unicode=True,
                                remove_urls=False,
                                remove_emails=False,
                                clean_whitespace=True,
                                preserve_structure=True,
                                max_length=100000,
                                enable_ocr=True,
                                aggressive_cleaning=True
                            )
                        )
                    else:
                        # Fallback to document extraction
                        text, metadata = extract_text_from_url(
                            url,
                            enable_ocr=True,
                            cleaning_options=CleaningOptions(
                                normalize_unicode=True,
                                remove_urls=False,
                                remove_emails=False,
                                clean_whitespace=True,
                                preserve_structure=True,
                                max_length=100000,
                                enable_ocr=True,
                                aggressive_cleaning=True
                            )
                        )
                    
                    if text.strip():  # Only include successful extractions
                        formatted_results.append({
                            "url": url,
                            "text": text,
                            "metadata": metadata,
                            "status": "success",
                            "url_type": url_type
                        })
                    else:
                        formatted_results.append({
                            "url": url,
                            "text": "",
                            "metadata": metadata,
                            "status": "failed",
                            "error": metadata.get("error", "No text extracted"),
                            "url_type": url_type
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {e}")
                    formatted_results.append({
                        "url": url,
                        "text": "",
                        "metadata": {"error": str(e)},
                        "status": "failed",
                        "error": str(e),
                        "url_type": "error"
                    })
            
            logger.info(f"Successfully processed {len(formatted_results)} URLs")
            return formatted_results
                        
        except Exception as e:
            logger.error(f"Error in batch text extraction: {e}")
            raise HTTPException(status_code=500, detail=f"Batch text extraction error: {e}")
    
    def extract_text_from_local_file(self, file_path: str) -> str:
        """Extract text from local file using direct extraction with caching"""
        try:
            logger.info(f"Extracting text from local file: {file_path}")
            
            # Check cache first
            cached_data = document_cache.get(file_path)
            if cached_data:
                logger.info(f"Using cached data for {file_path}")
                return cached_data["data"]["text"]
            
            # Read file bytes
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            
            # Use direct extraction function
            text, metadata = extract_text_from_bytes(
                file_bytes,
                os.path.basename(file_path),
                enable_ocr=True,  # Enable OCR for better results
                cleaning_options=CleaningOptions(
                    normalize_unicode=True,
                    remove_urls=False,
                    remove_emails=False,
                    clean_whitespace=True,
                    preserve_structure=True,
                    max_length=100000,
                    enable_ocr=True,
                    aggressive_cleaning=True
                )
            )
            
            # Cache the result
            cache_data = {
                "text": text,
                "metadata": metadata,
                "processing_time": metadata.get('processing_time', 0)
            }
            document_cache.set(file_path, cache_data)
            
            logger.info(f"Successfully extracted text from {file_path} (processing time: {metadata.get('processing_time', 0):.2f}s)")
            return text
            
        except Exception as e:
            logger.error(f"Error processing local file: {e}")
            raise HTTPException(status_code=500, detail=f"Local file processing error: {e}")
    
    def chunk_text(self, text: str) -> List[DocumentChunk]:
        """Enhanced text chunking with multiple strategies"""
        chunks = []
        
        # Strategy 1: Semantic chunking (sentence-based)
        semantic_chunks = self._semantic_chunking(text)
        chunks.extend(semantic_chunks)
        
        # Strategy 2: Hierarchical chunking (paragraph-based)
        hierarchical_chunks = self._hierarchical_chunking(text)
        chunks.extend(hierarchical_chunks)
        
        # Strategy 3: Fixed-size chunking (fallback)
        if len(chunks) < 3:  # If other strategies didn't produce enough chunks
            fixed_chunks = self._fixed_size_chunking(text)
            chunks.extend(fixed_chunks)
        
        # Remove duplicates and sort by position
        unique_chunks = self._deduplicate_chunks(chunks)
        unique_chunks.sort(key=lambda x: x.metadata.get("start_idx", 0))
        
        return unique_chunks
    
    def _semantic_chunking(self, text: str) -> List[DocumentChunk]:
        """Chunk text based on semantic boundaries (sentences)"""
        
        # Split by sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        
        current_chunk = []
        current_length = 0
        start_idx = 0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > CHUNK_SIZE and current_chunk:
                # Create chunk from current sentences
                chunk_text = " ".join(current_chunk)
                chunk = DocumentChunk(
                    text=chunk_text,
                    metadata={
                        "start_idx": start_idx,
                        "end_idx": start_idx + current_length,
                        "chunk_type": "semantic",
                        "num_sentences": len(current_chunk)
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
                start_idx = start_idx + len(overlap_sentences) * 10  # Approximate
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk = DocumentChunk(
                text=chunk_text,
                metadata={
                    "start_idx": start_idx,
                    "end_idx": start_idx + current_length,
                    "chunk_type": "semantic",
                    "num_sentences": len(current_chunk)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _hierarchical_chunking(self, text: str) -> List[DocumentChunk]:
        """Chunk text based on hierarchical structure (paragraphs, sections)"""
        # Split by paragraph boundaries
        paragraphs = text.split('\n\n')
        chunks = []
        
        current_chunk = []
        current_length = 0
        start_idx = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            paragraph_length = len(paragraph.split())
            
            if current_length + paragraph_length > CHUNK_SIZE * 1.5 and current_chunk:
                # Create chunk from current paragraphs
                chunk_text = "\n\n".join(current_chunk)
                chunk = DocumentChunk(
                    text=chunk_text,
                    metadata={
                        "start_idx": start_idx,
                        "end_idx": start_idx + current_length,
                        "chunk_type": "hierarchical",
                        "num_paragraphs": len(current_chunk)
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk = [paragraph]
                current_length = paragraph_length
                start_idx = start_idx + current_length
            else:
                current_chunk.append(paragraph)
                current_length += paragraph_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunk = DocumentChunk(
                text=chunk_text,
                metadata={
                    "start_idx": start_idx,
                    "end_idx": start_idx + current_length,
                    "chunk_type": "hierarchical",
                    "num_paragraphs": len(current_chunk)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _fixed_size_chunking(self, text: str) -> List[DocumentChunk]:
        """Traditional fixed-size chunking with overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), CHUNK_SIZE - OVERLAP_SIZE):
            chunk_words = words[i:i + CHUNK_SIZE]
            chunk_text = " ".join(chunk_words)
            
            chunk = DocumentChunk(
                text=chunk_text,
                metadata={
                    "start_idx": i,
                    "end_idx": i + len(chunk_words),
                    "chunk_type": "fixed_size"
                }
            )
            chunks.append(chunk)
            
        return chunks
    
    def _deduplicate_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Remove duplicate chunks based on content similarity"""
        unique_chunks = []
        seen_texts = set()
        
        for chunk in chunks:
            # Normalize text for comparison
            normalized_text = chunk.text.strip().lower()
            if normalized_text not in seen_texts and len(normalized_text) > 10:
                seen_texts.add(normalized_text)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _create_chunks(self, content: str, file_path: str) -> List[DocumentChunk]:
        """Create document chunks from content with enhanced metadata"""
        if not content.strip():
            logger.warning(f"No content to chunk from {file_path}")
            return []
        
        # Use enhanced chunking strategies
        chunks = self.chunk_text(content)
        
        # Add source metadata to all chunks
        for chunk in chunks:
            chunk.metadata["source"] = file_path
            chunk.metadata["file_type"] = os.path.splitext(file_path)[1] if "." in file_path else "unknown"
            chunk.metadata["processing_timestamp"] = time.time()
        
        logger.info(f"Created {len(chunks)} enhanced chunks from {file_path}")
        return chunks
    
    def build_indices(self, chunks: List[DocumentChunk]):
        """Build enhanced indices with parallel embedding generation for multiple models"""
        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        
        logger.info(f"Building indices for {len(chunks)} chunks with parallel embedding generation...")
        start_time = time.time()
        
        # Parallel embedding generation for both models
        with ThreadPoolExecutor(max_workers=EMBEDDING_WORKERS) as executor:
            # Submit both embedding tasks
            bge_future = executor.submit(self._generate_bge_embeddings, texts)
            all_mini_future = executor.submit(self._generate_all_mini_embeddings, texts)
            #bge_future_1024 = executor.submit(self._generate_bge_embeddings_1024, texts)
            #intfloat_future_1024 = executor.submit(self._generate_intfloat_embeddings_1024,texts)
            # Wait for both to complete
            self.bge_embeddings = bge_future.result()
            self.all_mini_embeddings = all_mini_future.result()
            #self.bge_embeddings_1024 = bge_future_1024.result()
            #self.intfloat_embeddings_1024 = intfloat_future_1024.result()
        
        # === Concatenate both embeddings ===
        logger.info("Concatenating BGE and AllMiniLM embeddings...")
        #self.combined_embeddings = np.concatenate(
        #        [self.bge_embeddings_1024, self.intfloat_embeddings_1024], axis=1
        #    )
        # Concatenate BGE + AllMiniLM (both 384-dim) → 768-dim
        self.combined_embeddings = np.concatenate(
            [self.bge_embeddings, self.all_mini_embeddings], axis=1
        )

        #self.combined_embeddings = (0.5 * self.bge_embeddings_1024 + 0.5 * self.intfloat_embeddings_1024)

        # Store embeddings for ensemble search
        for i, chunk in enumerate(chunks):
            #chunk.embedding_bge = self.bge_embeddings_1024[i]
            #chunk.embedding_intfloat = self.intfloat_embeddings_1024[i]
            chunk.embedding = self.combined_embeddings[i]  # Store combined embedding
            chunk.chunk_id = f"chunk_{i}"

        logger.info("Building unified FAISS index with combined embeddings...")
        dimension = self.combined_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for normalized embeddings
        self.faiss_index.add(self.combined_embeddings.astype('float32'))
            
        
        """# Build FAISS index for BGE
        logger.info("Building FAISS index for BGE...")
        dim_bge = self.bge_embeddings_1024.shape[1]
        self.faiss_index_bge = faiss.IndexFlatIP(dim_bge)
        self.faiss_index_bge.add(self.bge_embeddings_1024.astype('float32'))

        # Build FAISS index for Intfloat
        logger.info("Building FAISS index for Intfloat...")
        dim_intfloat = self.intfloat_embeddings_1024.shape[1]
        self.faiss_index_intfloat = faiss.IndexFlatIP(dim_intfloat)
        self.faiss_index_intfloat.add(self.intfloat_embeddings_1024.astype('float32'))"""
        
        # Build BM25 index with enhanced tokenization
        logger.info("Building BM25 index...")
        tokenized_texts = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        embedding_time = time.time() - start_time
        logger.info(f"Enhanced indices built successfully with {len(chunks)} chunks in {embedding_time:.2f} seconds")
    
    def _generate_bge_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate BGE embeddings with optimized batch processing"""
        logger.info(f"Generating BGE embeddings for {len(texts)} texts...")
        start_time = time.time()
        
        # Use batch processing for better performance
        embeddings = self.bge_model.encode(
            texts, 
            normalize_embeddings=True,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True
        )
        
        generation_time = time.time() - start_time
        logger.info(f"BGE embeddings generated in {generation_time:.2f} seconds")
        return embeddings
    
    def _generate_all_mini_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate All-MiniLM embeddings with optimized batch processing"""
        logger.info(f"Generating All-MiniLM embeddings for {len(texts)} texts...")
        start_time = time.time()
        
        # Use batch processing for better performance
        embeddings = self.all_mini_model.encode(
            texts, 
            normalize_embeddings=True,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True
        )
        
        generation_time = time.time() - start_time
        logger.info(f"All-MiniLM embeddings generated in {generation_time:.2f} seconds")
        return embeddings
    
    def _generate_bge_embeddings_1024(self, texts: List[str]) -> np.ndarray:
        """Generate BGE embeddings with optimized batch processing"""
        logger.info(f"Generating BGE 1024 embeddings for {len(texts)} texts...")
        start_time = time.time()
        
        # Use batch processing for better performance
        embeddings = self.bge_model_1024.encode(
            texts, 
            normalize_embeddings=True,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True
        )
        
        generation_time = time.time() - start_time
        logger.info(f"BGE 1024 embeddings generated in {generation_time:.2f} seconds")
        return embeddings

    def _generate_intfloat_embeddings_1024(self, texts: List[str]) -> np.ndarray:
        """Generate BGE embeddings with optimized batch processing"""
        logger.info(f"Generating IntFloat 1024 embeddings for {len(texts)} texts...")
        start_time = time.time()
        
        # Use batch processing for better performance
        embeddings = self.intfloat_model_1024.encode(
            [f"passage: {t}" for t in texts],
            normalize_embeddings=True,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True
        )

        
        generation_time = time.time() - start_time
        logger.info(f"IntFloat embeddings generated in {generation_time:.2f} seconds")
        return embeddings

    """    def semantic_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        if not self.faiss_index:
            raise HTTPException(status_code=500, detail="FAISS index not built")
        
        query_embedding = self.bge_model_1024.encode([query], normalize_embeddings=True)
        scores, indices = self.faiss_index.search(
            query_embedding.astype('float32'), min(top_k, len(self.chunks))
        )
        
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                result = SearchResult(
                    chunk=chunk,
                    semantic_score=float(score),
                    search_strategy="semantic"
                )
                results.append(result)
        
        return results
        """
    def semantic_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Pure semantic search using FAISS (combined BGE + Intfloat)"""
        if not self.faiss_index:
            raise HTTPException(status_code=500, detail="FAISS index not built")

        # Generate both embeddings
        #embedding_bge = self.bge_model_1024.encode([query], normalize_embeddings=True)
        #embedding_intfloat = self.intfloat_model_1024.encode([f"query: {query}"], normalize_embeddings=True)

        embedding_bge = self.bge_model.encode([query], normalize_embeddings=True)
        embedding_all_mini = self.all_mini_model.encode([query], normalize_embeddings=True)

        # Concatenate to match the FAISS index format
        #query_embedding = np.concatenate([embedding_bge, embedding_intfloat], axis=1)
        query_embedding = np.concatenate([embedding_bge, embedding_all_mini], axis=1)
        #query_embedding = 0.5 * embedding_bge + 0.5 * embedding_all_mini

        # Search
        scores, indices = self.faiss_index.search(
            query_embedding.astype('float32'), min(top_k*3, len(self.chunks))
        )

        # Prepare results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                result = SearchResult(
                    chunk=chunk,
                    semantic_score=float(score),
                    search_strategy="semantic"
                )
                results.append(result)

        return results

    def lexical_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Pure lexical search using BM25"""
        if not self.bm25:
            raise HTTPException(status_code=500, detail="BM25 index not built")
        
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            chunk = self.chunks[idx]
            result = SearchResult(
                chunk=chunk,
                lexical_score=float(scores[idx]),
                search_strategy="lexical"
            )
            results.append(result)
        
        return results
    
    def ensemble_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Ensemble search using multiple embedding models"""
        start_time = time.time()
        if not self.faiss_index or not self.bm25:
            raise HTTPException(status_code=500, detail="Indices not built")
        
        # Get results from different models
        bge_results = self.semantic_search(query, top_k * 3) # Semantic (BGE+AllMini) weight
        #all_mini_results = self._all_mini_search(query, top_k * 2)
        bm25_results = self.lexical_search(query, top_k * 3)
        
        # Combine results using reciprocal rank fusion
        combined_scores = {}
        
        for i, result in enumerate(bge_results):
            chunk_id = result.chunk.chunk_id
            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = {"chunk": result.chunk, "scores": []}
            combined_scores[chunk_id]["scores"].append(1.0 / (i + 20))  # BGE 1024 + Intfloat 1024 weight
        
        """for i, result in enumerate(all_mini_results):
            chunk_id = result.chunk.chunk_id
            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = {"chunk": result.chunk, "scores": []}
            combined_scores[chunk_id]["scores"].append(1.0 / (i + 60))  # All-MiniLM weight"""
        
        for i, result in enumerate(bm25_results):
            chunk_id = result.chunk.chunk_id
            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = {"chunk": result.chunk, "scores": []}
            combined_scores[chunk_id]["scores"].append(1.0 / (i + 40))  # BM25 weight
        
        # Calculate final scores
        results = []
        for chunk_id, data in combined_scores.items():
            final_score = sum(data["scores"])
            result = SearchResult(
                chunk=data["chunk"],
                combined_score=final_score,
                search_strategy="ensemble"
            )
            results.append(result)
        
        # Sort by final score and return top-k
        results.sort(key=lambda x: x.combined_score, reverse=True)
        logger.info(f"Ensemble search took {time.time() - start_time:.2f} seconds")
        return results[:top_k]

    
    def _all_mini_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search using All-MiniLM model with pre-computed embeddings"""
        if not self.chunks or self.all_mini_embeddings is None:
            return []
        
        # Encode query with All-MiniLM
        query_embedding = self.all_mini_model.encode([query], normalize_embeddings=True)
        
        # Calculate similarities with pre-computed All-MiniLM embeddings
        similarities = np.dot(self.all_mini_embeddings, query_embedding[0])
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            result = SearchResult(
                chunk=chunk,
                semantic_score=float(similarities[idx]),
                search_strategy="all_mini"
            )
            results.append(result)
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 10) -> List[DocumentChunk]:
        """Enhanced hybrid search with sophisticated reranking"""
        if not self.faiss_index or not self.bm25:
            raise HTTPException(status_code=500, detail="Indices not built")
        
        # Get initial candidates from multiple strategies
        semantic_results = self.semantic_search(query, self.rerank_top_k)
        lexical_results = self.lexical_search(query, self.rerank_top_k)
        
        # Combine and rerank
        all_results = self._combine_and_rerank(query, semantic_results, lexical_results)
        
        # Return top-k chunks
        return [result.chunk for result in all_results[:top_k]]
    
    def advanced_search(self, query: str, strategy: str = "ensemble", top_k: int = 10) -> Dict[str, Any]:
        """Advanced search with multiple strategies"""
        if not self.chunks:
            raise HTTPException(status_code=400, detail="No documents indexed")
        
        try:
            if strategy == "semantic":
                results = self.semantic_search(query, top_k)
                chunks = [result.chunk for result in results]
                scores = [result.semantic_score for result in results]
            elif strategy == "lexical":
                results = self.lexical_search(query, top_k)
                chunks = [result.chunk for result in results]
                scores = [result.lexical_score for result in results]
            elif strategy == "ensemble":
                results = self.ensemble_search(query, top_k)
                chunks = [result.chunk for result in results]
                scores = [result.combined_score for result in results]
            else:  # hybrid
                results = self.hybrid_search(query, top_k)
                chunks = results
                scores = [1.0] * len(results)  # Default scores for hybrid
            
            # Format results
            formatted_results = []
            for i, (chunk, score) in enumerate(zip(chunks, scores)):
                formatted_results.append({
                    "rank": i + 1,
                    "content": chunk.text,
                    "metadata": chunk.metadata,
                    "score": float(score),
                    "chunk_id": chunk.chunk_id
                })
            
            return {
                "query": query,
                "strategy": strategy,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "search_time": 0.0  # Could add timing if needed
            }
            
        except Exception as e:
            logger.error(f"Error in advanced search: {e}")
            raise HTTPException(status_code=500, detail=f"Search error: {e}")
    
    def _combine_and_rerank(self, query: str, semantic_results: List[SearchResult], 
                           lexical_results: List[SearchResult]) -> List[SearchResult]:
        """Combine and rerank search results"""
        # Create a mapping of chunk_id to results
        combined_results = {}
        
        # Add semantic results
        for i, result in enumerate(semantic_results):
            chunk_id = result.chunk.chunk_id
            if chunk_id not in combined_results:
                combined_results[chunk_id] = result
            else:
                # Update with better semantic score
                if result.semantic_score > combined_results[chunk_id].semantic_score:
                    combined_results[chunk_id].semantic_score = result.semantic_score
        
        # Add lexical results
        for i, result in enumerate(lexical_results):
            chunk_id = result.chunk.chunk_id
            if chunk_id not in combined_results:
                combined_results[chunk_id] = result
            else:
                # Update with better lexical score
                if result.lexical_score > combined_results[chunk_id].lexical_score:
                    combined_results[chunk_id].lexical_score = result.lexical_score
        
        # Calculate combined scores with normalization
        results = list(combined_results.values())
        
        # Normalize semantic scores
        if results:
            max_semantic = max(r.semantic_score for r in results) if any(r.semantic_score > 0 for r in results) else 1.0
            min_semantic = min(r.semantic_score for r in results)
            semantic_range = max_semantic - min_semantic if max_semantic > min_semantic else 1.0
            
            # Normalize lexical scores
            max_lexical = max(r.lexical_score for r in results) if any(r.lexical_score > 0 for r in results) else 1.0
            min_lexical = min(r.lexical_score for r in results)
            lexical_range = max_lexical - min_lexical if max_lexical > min_lexical else 1.0
            
            for result in results:
                # Normalize scores
                norm_semantic = (result.semantic_score - min_semantic) / semantic_range if semantic_range > 0 else 0.0
                norm_lexical = (result.lexical_score - min_lexical) / lexical_range if lexical_range > 0 else 0.0
                
                # Calculate combined score with weights
                result.combined_score = (norm_semantic * 0.6) + (norm_lexical * 0.4)
                result.search_strategy = "hybrid"
        
        # Apply additional reranking factors
        results = self._apply_reranking_factors(query, results)
        
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return results
    
    def _apply_reranking_factors(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Apply additional reranking factors"""
        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        
        for result in results:
            chunk_text_lower = result.chunk.text.lower()
            chunk_tokens = set(chunk_text_lower.split())
            
            # Factor 1: Query term density
            overlap_tokens = query_tokens.intersection(chunk_tokens)
            term_density = len(overlap_tokens) / len(query_tokens) if query_tokens else 0.0
            
            # Factor 2: Chunk length penalty (prefer medium-length chunks)
            chunk_length = len(result.chunk.text.split())
            length_penalty = 1.0
            if chunk_length < 10:
                length_penalty = 0.7  # Too short
            elif chunk_length > 200:
                length_penalty = 0.8  # Too long
            
            # Factor 3: Position bonus (prefer chunks from beginning of document)
            position_bonus = 1.0
            if result.chunk.metadata.get("start_idx", 0) < 1000:
                position_bonus = 1.1  # Slight bonus for early chunks
            
            # Apply factors to combined score
            result.combined_score *= (1.0 + term_density * 0.2) * length_penalty * position_bonus
        
        return results
    
    def process_documents_parallel(self, inputs: List[str], max_workers: int = 10) -> List[DocumentChunk]:
        """Process multiple documents (local files or URLs) in parallel using direct extraction"""

        def process_single_input(input_source: str) -> List[DocumentChunk]:
            try:
                # Check cache first for both URLs and local files
                cached_data = document_cache.get(input_source)
                if cached_data:
                    logger.info(f"🔄 Using cached data for {input_source}")
                    text = cached_data["data"]["text"]
                    metadata = cached_data["data"]["metadata"]
                else:
                    # Process the document
                    if input_source.startswith(("http://", "https://")):
                        logger.info(f"🔗 Processing URL: {input_source}")
                        
                        # Use direct URL extraction
                        text, metadata = extract_text_from_url(
                            input_source,
                            enable_ocr=True,
                            cleaning_options=CleaningOptions(
                                normalize_unicode=True,
                                remove_urls=False,
                                remove_emails=False,
                                clean_whitespace=True,
                                preserve_structure=True,
                                max_length=100000,
                                enable_ocr=True,
                                aggressive_cleaning=True
                            )
                        )
                        
                        # Cache the result
                        cache_data = {
                            "text": text,
                            "metadata": metadata,
                            "processing_time": metadata.get('processing_time', 0)
                        }
                        document_cache.set(input_source, cache_data)
                                
                    elif input_source.startswith("file://") or os.path.exists(input_source):
                        # normalize local file path
                        local_path = input_source.replace("file:/", "")
                        logger.info(f"📄 Processing local file: {local_path}")

                        with open(local_path, 'rb') as f:
                            file_bytes = f.read()

                        text, metadata = extract_text_from_bytes(
                            file_bytes,
                            os.path.basename(local_path),
                            enable_ocr=True,
                            cleaning_options=CleaningOptions(
                                normalize_unicode=True,
                                remove_urls=False,
                                remove_emails=False,
                                clean_whitespace=True,
                                preserve_structure=True,
                                max_length=100000,
                                enable_ocr=True,
                                aggressive_cleaning=True
                            )
                        )

                        cache_data = {
                            "text": text,
                            "metadata": metadata,
                            "processing_time": metadata.get('processing_time', 0)
                        }
                        document_cache.set(input_source, cache_data)

                    else:
                        logger.error(f"Invalid input: {input_source}")
                        return []

                logger.info(f"✅ Extracted {len(text)} characters from {input_source}")

                if not text.strip():
                    logger.warning(f"No content extracted from {input_source}")
                    return []

                return self._create_chunks(text, input_source)

            except Exception as e:
                logger.error(f"Processing failed for {input_source}: {e}")
                return []

        # Parallel execution
        all_chunks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_input = {executor.submit(process_single_input, inp): inp for inp in inputs}
            for future in as_completed(future_to_input):
                input_source = future_to_input[future]
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                    logger.info(f"✅ Finished processing {input_source} → {len(chunks)} chunks")
                except Exception as e:
                    logger.error(f"Unhandled exception for {input_source}: {e}")

        return all_chunks

    async def ingest_documents_async(self, doc_inputs: List[str]) -> List[IngestionResult]:
        """Ingest documents using the text extraction service"""
        all_chunks = []
        results: List[IngestionResult] = []

        # Separate URLs and local files
        urls = [doc for doc in doc_inputs if is_url(doc)]
        local_files = [doc for doc in doc_inputs if not is_url(doc)]

        # Process URLs
        if urls:
            logger.info(f"Processing {len(urls)} URLs using batch service...")
            try:
                if len(urls) == 1:
                    url = urls[0]
                    if url.lower().endswith(".bin"):
                        logger.warning(f"Skipping .bin URL: {url}")
                        results.append(IngestionResult(source=url, success=False))
                    else:
                        text = await self.extract_text_from_url(url)
                        if text.strip():
                            chunks = self.chunk_text(text)
                            for chunk in chunks:
                                chunk.metadata["source"] = url
                            all_chunks.extend(chunks)
                            results.append(IngestionResult(source=url, success=True))
                        else:
                            results.append(IngestionResult(source=url, success=False))
                else:
                    valid_urls = [url for url in urls if not url.lower().endswith(".bin")]
                    skipped_urls = [url for url in urls if url.lower().endswith(".bin")]

                    for url in skipped_urls:
                        logger.warning(f"Skipping .bin URL: {url}")
                        results.append(IngestionResult(source=url, success=False))

                    if valid_urls:
                        batch_results = await self.extract_text_from_urls_batch(valid_urls)
                        for result in batch_results:
                            url = result.get("url", "unknown")
                            if result.get("status") == "success":
                                text = result.get("text", "")
                                if text.strip():
                                    chunks = self.chunk_text(text)
                                    for chunk in chunks:
                                        chunk.metadata["source"] = url
                                    all_chunks.extend(chunks)
                                    results.append(IngestionResult(source=url, success=True))
                                else:
                                    results.append(IngestionResult(source=url, success=False))
                            else:
                                logger.error(f"Failed to extract text from URL: {url}")
                                results.append(IngestionResult(source=url, success=False))
            except Exception as e:
                logger.error(f"Error processing URLs: {e}")
                raise

        # Process local files
        if local_files:
            logger.info(f"Processing {len(local_files)} local files using parallel method...")
            try:
                local_chunks = self.process_documents_parallel(local_files)
                all_chunks.extend(local_chunks)
                for file in local_files:
                    results.append(IngestionResult(source=file, success=True))
            except Exception as e:
                logger.error(f"Failed to process local files: {e}")
                for file in local_files:
                    results.append(IngestionResult(source=file, success=False))

        # Check if anything worked
        if not all_chunks:
            logger.warning("No chunks extracted from any document.")
            # Don't raise; let caller handle it
        else:
            self.build_indices(all_chunks)
            logger.info(f"Successfully ingested {len(all_chunks)} chunks from {len(doc_inputs)} documents")

        return results

    def ingest_documents(self, file_paths: List[str]):
        """Synchronous wrapper for document ingestion"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.ingest_documents_async(file_paths))
        finally:
            loop.close()

    async def handle_need_api(self, api_data: dict) -> str:
        try:
            method = api_data.get("type", "GET").upper()
            url = api_data["url"]
            headers = api_data.get("headers", {})
            body = api_data.get("body", {})

            async with httpx.AsyncClient(timeout=10.0) as client:
                if method == "GET":
                    response = await client.get(url, headers=headers, params=body)
                elif method == "POST":
                    response = await client.post(url, headers=headers, json=body)
                else:
                    return f"Unsupported API method: {method}"

            if response.status_code == 200:
                return response.text
            else:
                return f"API call failed with status {response.status_code}: {response.text}"

        except Exception as e:
            logger.error(f"Error in handle_need_api: {e}")
            return f"API call error: {e}"
    async def process_query_with_gemini(self, query: str, search_strategy: str = "ensemble") -> str:
        try:
            # Step 1: Retrieve relevant chunks
            if search_strategy == "semantic":
                search_results = self.semantic_search(query, top_k=8)
            elif search_strategy == "lexical":
                search_results = self.lexical_search(query, top_k=8)
            elif search_strategy == "ensemble":
                search_results = self.ensemble_search(query, top_k=16)
            else:
                search_results = self.hybrid_search(query, top_k=8)

            relevant_chunks = [result.chunk for result in search_results]

            # Deduplicate
            seen_texts = set()
            unique_chunks = []
            for chunk in relevant_chunks:
                if chunk.text not in seen_texts:
                    unique_chunks.append(chunk)
                    seen_texts.add(chunk.text)
            relevant_chunks = unique_chunks

            # Initial context
            context_parts = [
                f"[Chunk {i+1} - {chunk.metadata.get('chunk_type', 'unknown')}]: {chunk.text}"
                for i, chunk in enumerate(relevant_chunks)
            ]
            base_context = clean_text_for_gemini("\n\n".join(context_parts))

            # Load prompt
            try:
                prompt_template = load_prompt("prompts/phase1_prompt.txt")
                logger.info("Loaded prompt from prompts/phase1_prompt.txt")
            except FileNotFoundError:
                logger.warning("Prompt file not found, using default")
                prompt_template = """
                Based on the following context and any intermediate results, answer the question in the following JSON format:

                {{
                "answer": "<your answer>",
                "confidence_score": <float between 0 and 1>,
                "need_api": false | {{
                    "type": "GET" | "POST",
                    "url": "<api or webpage URL>",
                    "headers": {{ ...optional... }},
                    "body": {{ ...optional... }}
                }}
                }}

                Context: {context}

                Question: {query}

                Answer:
                """
            # ✅ Format the prompt using actual context and query
            #prompt = prompt_template.format(context=base_context, query=query)
            max_attempts = 5
            #keys = [GEMINI_API_KEY_PAID,GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3, GEMINI_API_KEY_4, GEMINI_API_KEY_5, GEMINI_API_KEY_6, GEMINI_API_KEY_7, GEMINI_API_KEY_8, GEMINI_API_KEY_9, GEMINI_API_KEY_10]
            keys = [GEMINI_API_KEY_PAID]
            MODEL_FALLBACK_ORDER = ['gemini-2.5-flash', 'gemini-2.0-flash']

            intermediate_results = []
            answers_history = []        
            for outer_attempt in range(max_attempts):
                random.shuffle(keys)
                for api_key in keys:
                    key_label = f"KEY_{keys.index(api_key) + 1}"
                    logger.info(f"Attempt {outer_attempt + 1}: Trying Gemini key: {key_label}")
                    #Old method
                    #genai.configure(api_key=api_key)
                    # New method
                    client = google.genai.Client(api_key=api_key)
                    for model_name in MODEL_FALLBACK_ORDER:
                        try:
                            logger.info(f"Trying model: {model_name}")
                            # Old method
                            #model = genai.GenerativeModel(model_name)
                            
                            current_context = base_context
                            prompt_query = query
                            
                            # Loop until need_api == false
                            for step in range(5):  # limit to avoid infinite loops
                                full_context = current_context
                                if intermediate_results:
                                    full_context += "\n\nIntermediate Results:\n" + "\n".join(intermediate_results)
                                if answers_history:
                                    full_context += "\n\nPrevious Answers:\n" + "\n".join(answers_history)
                                api_result_str = "\n".join(intermediate_results) if intermediate_results else "None"
                                prompt = prompt_template.format(context=full_context, query=prompt_query,api_response= api_result_str)

                                logger.info(f"Gemini loop step {step + 1}, model: {model_name}")
                                # Old method
                                #response = await asyncio.to_thread(model.generate_content, prompt,generation_config=generation_config,system_prompt=system_prompt)
                                # Without asyncio
                                #response = client.models.generate_content(
                                #    model=model_name,
                                #    config=generation_config,
                                #    contents=prompt
                                #)
                                # With asyncio
                                response = await asyncio.to_thread(
                                    client.models.generate_content,
                                    model=model_name,
                                    config=generation_config,
                                    contents=prompt
                                )

                                #response = await client.aio.models.generate_content(
                                            #model=model_name,       # e.g., "gemini-2.0-flash"
                                            #contents=prompt         # your formatted prompt)
                                if not hasattr(response, "text"):
                                    logger.warning("Gemini response missing 'text' attribute.")
                                    continue

                                raw_text = response.text.strip()
                                logger.info(f"Raw Gemini response (step {step+1}):\n{raw_text}")

                                try:
                                    parsed = extract_json_from_response(raw_text)
                                except json.JSONDecodeError:
                                    logger.warning(f"❌ Could not parse Gemini response as JSON:\n{raw_text}")
                                    return raw_text
                                # Store answer for next iteration context
                                if "answer" in parsed and parsed["answer"]:
                                    answers_history.append(f"Answer (step {step+1}): {parsed['answer']}")
                                # Check if API call is needed
                                need_api = parsed.get("need_api", False)
                                if isinstance(need_api, str) and need_api.strip().lower() == "false":
                                    need_api = False
                                # If API call is needed, do it and feed result into next loop
                                if need_api and isinstance(need_api, dict):
                                    api_result = await self.handle_need_api(need_api)
                                    intermediate_results.append(f"API Result (step {step+1}):\n{api_result}")
                                else:
                                    # No more API needed — return final answer
                                    return parsed.get("answer", "No answer returned.")
                        except Exception as e:
                            logger.warning(f"Model {model_name} failed with key {api_key}: {e}")
                logger.warning("Retrying Gemini after failures...")
            raise RuntimeError("All attempts to call Gemini API failed.")

        except Exception as e:
            logger.error(f"Error in process_query_with_gemini: {e}")
            return f"Error processing query: {e}"



# Global instances
document_cache = DocumentCache(cache_dir=CACHE_DIR, logger=logger)
system = TextExtractionSystem()

def verify_token(authorization: str = None):
    """Simple token verification"""
    if authorization != BEARER_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    return True


# API Router
api_router = APIRouter(prefix="/api/v1")

@api_router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file and return its local URL"""
    try:
        logger.info(f"Uploading file: {file.filename}")
        
        # Create upload directory if it doesn't exist
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Generate a unique filename
        import uuid
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ""
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save the file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Return the local file URL
        file_url = f"file:/{os.path.abspath(file_path)}"
        
        logger.info(f"File uploaded successfully: {file_url}")
        return {
            "url": file_url,
            "filename": file.filename,
            "size": len(content),
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Upload error: {e}")

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("=== HEALTH CHECK ENDPOINT CALLED ===")
    response_data = {"status": "healthy", "timestamp": time.time()}
    logger.info(f"Health check response: {response_data}")
    logger.info("=== HEALTH CHECK ENDPOINT COMPLETED ===\n")
    return response_data

@api_router.post("/parse")
async def parse_single_document(request: ParseRequest):
    """Parse a single document from URL using direct extraction"""
    try:
        logger.info(f"Parsing document from URL: {request.url}")
        
        # Use direct extraction
        text, metadata = extract_text_from_url(
            request.url,
            enable_ocr=True,
            cleaning_options=CleaningOptions(
                normalize_unicode=True,
                remove_urls=False,
                remove_emails=False,
                clean_whitespace=True,
                preserve_structure=True,
                max_length=100000,
                enable_ocr=True,
                aggressive_cleaning=True
            )
        )
        
        return {
            "text": text,
            "metadata": metadata,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Error parsing document: {e}")
        raise HTTPException(status_code=500, detail=f"Parsing error: {e}")

@api_router.post("/parse/batch")
async def parse_batch_documents(request: ParseBatchRequest):
    """Parse multiple documents from URLs using direct batch extraction"""
    try:
        logger.info(f"Parsing {len(request.urls)} documents from URLs")
        
        # Use direct batch extraction
        results = extract_text_from_multiple_urls(
            request.urls,
            enable_ocr=True,
            cleaning_options=CleaningOptions(
                normalize_unicode=True,
                remove_urls=False,
                remove_emails=False,
                clean_whitespace=True,
                preserve_structure=True,
                max_length=100000,
                enable_ocr=True,
                aggressive_cleaning=True
            ),
            max_workers=MAX_WORKERS
        )
        
        # Format results
        formatted_results = []
        for i, (text, metadata) in enumerate(results):
            formatted_results.append({
                "url": request.urls[i] if i < len(request.urls) else "unknown",
                "text": text,
                "metadata": metadata,
                "status": "success" if text.strip() else "failed"
            })
        
        return {
            "results": formatted_results,
            "total_files": len(request.urls),
            "successful": len([r for r in formatted_results if r["status"] == "success"])
        }
    
    except Exception as e:
        logger.error(f"Error in batch parsing: {e}")
        raise HTTPException(status_code=500, detail=f"Batch parsing error: {e}")

@api_router.post("/search")
async def search_documents(request: SearchRequest):
    """Advanced search endpoint with configurable strategies"""
    try:
        if not system.chunks:
            raise HTTPException(status_code=400, detail="No documents indexed. Please ingest documents first.")
        
        results = system.advanced_search(request.query, request.strategy, request.top_k)
        return results
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

@api_router.post("/search/multi")
async def multi_search_documents(request: MultiSearchRequest):
    """Multi-query search endpoint"""
    try:
        if not system.chunks:
            raise HTTPException(status_code=400, detail="No documents indexed. Please ingest documents first.")
        
        all_results = []
        for query in request.queries:
            result = system.advanced_search(query, request.strategy, request.top_k)
            all_results.append({
                "query": query,
                "results": result
            })
        
        return {
            "queries": request.queries,
            "strategy": request.strategy,
            "results": all_results,
            "total_queries": len(request.queries)
        }
    except Exception as e:
        logger.error(f"Error in multi-search: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-search error: {e}")

@api_router.get("/search/strategies")
async def get_search_strategies():
    """Get available search strategies"""
    return {
        "strategies": {
            "semantic": "Pure semantic search using FAISS embeddings",
            "lexical": "Pure lexical search using BM25",
            "hybrid": "Combined semantic and lexical search with reranking",
            "ensemble": "Ensemble search using multiple embedding models",
            "ensemble_parallel": "Parallel ensemble search using concurrent execution"
        },
        "default": "hybrid",
        "recommendations": {
            "semantic": "Best for conceptual queries and paraphrasing",
            "lexical": "Best for exact keyword matching",
            "hybrid": "Best overall performance for most queries",
            "ensemble": "Best for complex queries requiring multiple perspectives",
            "ensemble_parallel": "Fastest ensemble search with parallel execution"
        }
    }

@api_router.get("/search/stats")
async def get_search_stats():
    """Get search system statistics"""
    if not system.chunks:
        return {"error": "No documents indexed"}
    
    return {
        "total_chunks": len(system.chunks),
        "total_documents": len(set(chunk.metadata.get("source", "unknown") for chunk in system.chunks)),
        "chunk_types": {
            chunk_type: len([c for c in system.chunks if c.metadata.get("chunk_type") == chunk_type])
            for chunk_type in set(c.metadata.get("chunk_type", "unknown") for c in system.chunks)
        },
        "indices_built": {
            "faiss": system.faiss_index is not None,
            "bm25": system.bm25 is not None
        },
        "embedding_models": {
            "BAAI/bge-small-en-v1.5": {
                "available": system.bge_embeddings is not None,
                "dimension": system.bge_embeddings.shape[1] if system.bge_embeddings is not None else None
            },
            "all-MiniLM-L6-v2": {
                "available": system.all_mini_embeddings is not None,
                "dimension": system.all_mini_embeddings.shape[1] if system.all_mini_embeddings is not None else None
            }
        },
        "optimization_settings": {
            "embedding_batch_size": EMBEDDING_BATCH_SIZE,
            "embedding_workers": EMBEDDING_WORKERS
        }
    }

@api_router.get("/cache/stats")
async def get_cache_stats():
    """Get document cache statistics"""
    return document_cache.get_stats()

@api_router.post("/cache/clear")
async def clear_cache():
    """Clear all cached documents"""
    success = document_cache.clear_all()
    return {
        "success": success,
        "message": "All cache entries cleared" if success else "Failed to clear cache"
    }

@api_router.post("/cache/clear-expired")
async def clear_expired_cache():
    """Clear expired cache entries (no-op since no expiry)"""
    return {
        "success": True,
        "cleared_entries": 0,
        "message": "No cache expiry configured - no entries to clear"
    }



# Main endpoint
@api_router.post("/hackrx/run")
async def hackrx_run(req: HackRxRunRequest):
    """
    Main endpoint: processes documents and answers questions using FAISS+BM25+BGE+Gemini
    """
    logger.info("=== HACKRX RUN ENDPOINT CALLED ===")
    logger.info(f"Documents: {req.documents}")
    logger.info(f"Questions: {req.questions}")
    logger.info(f"Search Strategy: {req.search_strategy}")
    
    time_start = time.time()
    
    # Prepare document inputs
    doc_inputs = [req.documents] if isinstance(req.documents, str) else req.documents
    all_answers = [None] * len(req.questions)
    
    try:
        # Process all documents using the text extraction service
        logger.info(f"Processing {len(doc_inputs)} documents using text extraction service...")
        
        # Use async ingestion that calls the text extraction service
        ingested_results = await system.ingest_documents_async(doc_inputs)
        # Check for ingestion failure
        if all(not res.success for res in ingested_results):
            logger.warning("All documents failed to process (e.g., binary files).")
            return Response(
                content=json.dumps({
                    "answers": ["The document appears to be empty, so I couldn't provide any useful information."]
                }, indent=2, ensure_ascii=False),
                media_type="application/json"
            )
        # Process questions in parallel with proper ordering
        async def process_question_async(idx_q):
            idx, q = idx_q
            try:
                logger.info(f"Processing question {idx + 1}/{len(req.questions)}: {q}")
                result = await system.process_query_with_gemini(q, req.search_strategy)
                logger.info(f"Question {idx + 1} processed successfully")
                #logger.info(f"Answer: {result}")
                return idx, result
            except Exception as e:
                logger.error(f"Error processing question {idx}: {e}")
                return idx, f"Error: {e}"
        
        # Create tasks for parallel processing
        logger.info(f"Processing {len(req.questions)} questions in parallel...")
        tasks = [process_question_async((idx, q)) for idx, q in enumerate(req.questions)]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Populate answers in correct order
        """for result in results:
            if isinstance(result, tuple):
                idx, answer = result
                if isinstance(answer, str):
                    answer = answer.strip()
                    if answer:
                        answer = answer[0].upper() + answer[1:]
                final_answer = (extract_json_from_response(answer)).get("answer", "No answer found.")
                final_answer = clean_string_post(extract_final_answer(final_answer))
                all_answers[idx] = final_answer
                logger.info(f"Answer {idx + 1}: {all_answers[idx]}...")
            else:
                # Handle exceptions
                logger.error(f"Exception in parallel processing: {result}")"""
        def postprocess_answer(result):
            if isinstance(result, tuple):
                idx, answer = result
                try:
                    if isinstance(answer, str):
                        answer = answer.strip()
                        if answer:
                            answer = answer[0].upper() + answer[1:]
                    final_answer = (extract_json_from_response(answer)).get("answer", "No answer found.")
                    final_answer = clean_string_post(extract_final_answer(final_answer))
                    return idx, final_answer
                except Exception as e:
                    logger.warning(f"Post-processing error on answer {idx}: {e}")
                    return idx, "Post-processing error"
            return None

        # Run post-processing in parallel threads
        postprocessing_tasks = [asyncio.to_thread(postprocess_answer, result) for result in results]
        processed_results = await asyncio.gather(*postprocessing_tasks)
        for item in processed_results:
            if item:
                idx, ans = item
                all_answers[idx] = ans
                logger.info(f"Answer {idx + 1}: {ans}...")
        # Fill any remaining None values
        for idx in range(len(all_answers)):
            if all_answers[idx] is None:
                all_answers[idx] = "No answer found."
        
    except Exception as e:
        logger.error(f"Error in hackrx_run: {e}")
        # Fill all answers with error message
        for idx in range(len(req.questions)):
            all_answers[idx] = f"Processing error: {e}"
    
    time_end = time.time()
    processing_time = time_end - time_start
    logger.info(f"Total processing time: {processing_time:.2f} seconds")
    logger.info(f"Total questions: {len(req.questions)}")
    logger.info(f"Total documents: {len(doc_inputs)}")
    logger.info("=== HACKRX RUN ENDPOINT COMPLETED ===\n")
    
    response_data = {
        "answers": all_answers,
    }
    #logger.info(f"Endpoint Response Time: {time.time() - time_start:.2f} seconds")
    return Response(
        content=json.dumps(response_data, indent=2, ensure_ascii=False),
        media_type="application/json"
    )

# Application setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("=== APPLICATION STARTUP ===")
    logger.info("Starting up FAISS BM25 BGE system with document caching...")
    
    # Cache is permanent - no expiry cleanup needed
    logger.info("Document cache is permanent - no expiry cleanup needed")
    logger.info("=== APPLICATION STARTUP COMPLETED ===\n")
    
    yield
    
    # Shutdown
    logger.info("=== APPLICATION SHUTDOWN ===")
    logger.info("Shutting down...")
    logger.info("=== APPLICATION SHUTDOWN COMPLETED ===\n")

app = FastAPI(
    title="FAISS BM25 BGE Text Extraction API",
    description="High-performance document processing with FAISS, BM25, BGE embeddings, and Gemini AI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)

@app.middleware("http")
async def log_request_data(request: Request, call_next):
    # Log incoming request
    logger.info(f"=== INCOMING REQUEST ===")
    logger.info(f"Method: {request.method}")
    logger.info(f"URL: {request.url}")
    logger.info(f"Path: {request.url.path}")
    logger.info(f"Query params: {request.query_params}")
    
    # Log headers (excluding sensitive ones)
    headers_dict = dict(request.headers)
    safe_headers = {k: v for k, v in headers_dict.items() 
                   if k.lower() not in ['authorization', 'cookie', 'x-api-key']}
    logger.info(f"Headers: {safe_headers}")
    
    # Log request body if present
    try:
        body = await request.body()
        if body:
            body_str = body.decode('utf-8')
            # Truncate very long bodies
            if len(body_str) > 1000:
                body_str = body_str[:1000] + "... [truncated]"
            logger.info(f"Request Body: {body_str}")
    except Exception as e:
        logger.warning(f"Could not read request body: {e}")
    
    # Process the request
    start_time = time.time()
    response = await call_next(request)
    processing_time = time.time() - start_time
    
    # Log response
    logger.info(f"=== RESPONSE ===")
    logger.info(f"Status Code: {response.status_code}")
    logger.info(f"Processing Time: {processing_time:.3f}s")
    
    # Log response headers
    response_headers = dict(response.headers)
    safe_response_headers = {k: v for k, v in response_headers.items() 
                           if k.lower() not in ['set-cookie']}
    logger.info(f"Response Headers: {safe_response_headers}")
    
    # Try to log response body for JSON responses
    try:
        if hasattr(response, 'body') and response.body:
            response_body = response.body.decode('utf-8')
            if len(response_body) > 2000:
                response_body = response_body[:2000] + "... [truncated]"
            logger.info(f"Response Body: {response_body}")
    except Exception as e:
        logger.debug(f"Could not log response body: {e}")
    
    logger.info(f"=== END REQUEST/RESPONSE ===\n")
    return response
# Root endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    data = {
        "message": "Enhanced FAISS BM25 BGE Text Extraction API with Advanced Search",
        "version": "2.0.0",
        "features": {
            "text_extraction": "Advanced document processing with multiple chunking strategies",
            "search_strategies": ["semantic", "lexical", "hybrid", "ensemble"],
            "embedding_models": ["BAAI/bge-small-en-v1.5", "all-MiniLM-L6-v2"],
            "reranking": "Sophisticated reranking with multiple factors",
            "file_type_detection": "Automatic file type detection from URLs and blob URLs",
            "parallel_processing": "Optimized parallel embedding generation and search",
            "document_caching": "Smart caching to avoid reprocessing identical documents",
            "cache_management": "Cache statistics and management endpoints"
        },
        "endpoints": {
            "health": "/api/v1/health",
            "parse_single": "/api/v1/parse",
            "parse_batch": "/api/v1/parse/batch",
            "hackrx_run": "/api/v1/hackrx/run",
            "search": "/api/v1/search",
            "multi_search": "/api/v1/search/multi",
            "search_strategies": "/api/v1/search/strategies",
            "search_stats": "/api/v1/search/stats",
            "cache_stats": "/api/v1/cache/stats",
            "cache_clear": "/api/v1/cache/clear",
            "cache_clear_expired": "/api/v1/cache/clear-expired"
        },
        "search_capabilities": {
            "semantic": "Pure semantic search using FAISS embeddings",
            "lexical": "Pure lexical search using BM25",
            "hybrid": "Combined semantic and lexical search with reranking",
            "ensemble": "Ensemble search using multiple embedding models",
            "ensemble_parallel": "Fastest ensemble search with parallel execution"
        },
        "supported_file_types": [
            "PDF (.pdf)", "Text (.txt)", "CSV (.csv)", "Excel (.xls, .xlsx)",
            "PowerPoint (.ppt, .pptx)", "Word (.doc, .docx)", "Email (.eml)",
            "HTML (.html)", "XML (.xml)", "JSON (.json)", "Markdown (.md)",
            "RTF (.rtf)", "Images (.jpg, .png, .gif, .svg)", "Archives (.zip, .rar)"
        ],
        "url_types": {
            "document": "PDF, DOCX, XLSX, TXT, CSV files",
            "web": "HTML web pages with enhanced parsing",
            "api": "JSON API responses",
            "binary": "Binary files (not supported)"
        }
    }
    return JSONResponse(content=json.loads(json.dumps(data, indent=4, ensure_ascii=False)))

@app.get("/health")
async def root_health():
    """Root health check"""
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    logger.info("=== STARTING UVICORN SERVER ===")
    logger.info("Server will start on http://0.0.0.0:8000")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_config=None,  # Use our custom logging config
        access_log=True
    )