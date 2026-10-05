"""Chunking and retrieval for immutable document-set indexes."""
import hashlib
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import faiss
import numpy as np
import torch
from fastapi import HTTPException
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 512
OVERLAP_SIZE = 50
SEMANTIC_THRESHOLD_CHUNK_SCORE = 0.25
ENSEMBLE_THRESHOLD_SCORE = 0.0
JACCARD_SIMILARITY_THRESHOLD = 0.85
CONTAINED_RATIO = 0.85
EMBEDDING_BATCH_SIZE = 64 if DEVICE == "cuda" else 32
EMBEDDING_WORKERS = 2 if DEVICE == "cuda" else 4

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


def find_overlap(a: str, b: str, min_overlap: int = 5) -> int:
    """
    Find the length of the maximum suffix of 'a' that matches a prefix of 'b'.
    Returns the length of the overlap (0 if none).
    """
    max_len = min(len(a), len(b))
    for i in range(max_len, min_overlap - 1, -1):
        if a.endswith(b[:i]):
            return i
    return 0


def _tokenize_for_similarity(text: str) -> set:
    tokens = re.findall(r"\w+", text.lower())
    return set(tokens)


def _jaccard_similarity(a: str, b: str) -> float:
    set_a = _tokenize_for_similarity(a)
    set_b = _tokenize_for_similarity(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def _is_contained_with_ratio(a: str, b: str, min_ratio: float = CONTAINED_RATIO) -> bool:
    # True if the shorter string is contained within the longer string
    if not a or not b:
        return False
    if len(a) < len(b):
        shorter, longer = a, b
    else:
        shorter, longer = b, a
    if shorter in longer:
        return (len(shorter) / max(1, len(longer))) >= min_ratio
    return False


def merge_chunks(chunks: List[DocumentChunk], min_overlap: int = 5) -> List[DocumentChunk]:
    """Merge overlapping chunks while retaining their metadata."""
    merged: List[DocumentChunk] = []
    for chunk in chunks:
        if not merged:
            merged.append(chunk)
            continue
        previous = merged[-1]
        if _is_contained_with_ratio(previous.text, chunk.text) or _jaccard_similarity(previous.text, chunk.text) >= JACCARD_SIMILARITY_THRESHOLD:
            if len(chunk.text) > len(previous.text):
                chunk.metadata = {**previous.metadata, **chunk.metadata}
                merged[-1] = chunk
            continue
        overlap = find_overlap(previous.text, chunk.text, min_overlap)
        if overlap:
            previous.text += chunk.text[overlap:]
            previous.metadata["end_idx"] = chunk.metadata.get("end_idx", previous.metadata.get("end_idx"))
        else:
            merged.append(chunk)
    return merged


def merge_chunks_search(results: List[SearchResult], min_overlap: int = 5) -> List[SearchResult]:
    """
    Merge SearchResult chunks if they overlap by at least `min_overlap` characters.
    Keeps the highest combined_score when merging.
    """
    texts = [(r.chunk.text, r.combined_score, r) for r in results]
    merged = True

    while merged:
        merged = False
        new_texts = []
        skip = set()

        for i in range(len(texts)):
            if i in skip:
                continue
            merged_text, merged_score, orig_result = texts[i]

            for j in range(i + 1, len(texts)):
                if j in skip:
                    continue

                t2, score2, other_result = texts[j]
                if orig_result.chunk.metadata.get("source") != other_result.chunk.metadata.get("source"):
                    continue

                # Containment check (keep longer, best score)
                if _is_contained_with_ratio(merged_text, t2, min_ratio=CONTAINED_RATIO):
                    if len(t2) > len(merged_text):
                        merged_text = t2
                    merged_score = max(merged_score, score2)
                    skip.add(j)
                    merged = True
                    continue

                # High-similarity check via Jaccard (near-duplicates)
                jacc = _jaccard_similarity(merged_text, t2)
                if jacc >= JACCARD_SIMILARITY_THRESHOLD:
                    if len(t2) > len(merged_text):
                        merged_text = t2
                    merged_score = max(merged_score, score2)
                    skip.add(j)
                    merged = True
                    continue

                # Forward overlap
                overlap = find_overlap(merged_text, t2, min_overlap)
                if overlap > 0:
                    logger.debug("Merged overlapping search chunks")
                    merged_text = merged_text + t2[overlap:]
                    merged_score = max(merged_score, score2)  # keep best score
                    skip.add(j)
                    merged = True
                    continue

                # Reverse overlap
                overlap = find_overlap(t2, merged_text, min_overlap)
                if overlap > 0:
                    logger.debug("Merged overlapping search chunks")
                    merged_text = t2 + merged_text[overlap:]
                    merged_score = max(merged_score, score2)
                    skip.add(j)
                    merged = True

            # Create new SearchResult with merged text
            new_result = SearchResult(
                chunk=DocumentChunk(merged_text, metadata=orig_result.chunk.metadata.copy()),
                combined_score=merged_score,
                search_strategy=orig_result.search_strategy
            )
            # Assign a stable chunk_id based on text to improve downstream deduplication if needed
            new_result.chunk.chunk_id = f"merged_{hashlib.md5(merged_text.encode('utf-8')).hexdigest()[:8]}"
            new_texts.append((merged_text, merged_score, new_result))

        texts = new_texts

    return [t[2] for t in texts]


class RetrievalIndex:
    def __init__(self, bge_model=None, all_mini_model=None):
        # Multiple embedding models for ensemble search
        self.bge_model = bge_model if bge_model is not None else SentenceTransformer('BAAI/bge-small-en-v1.5')
        self.all_mini_model = all_mini_model if all_mini_model is not None else SentenceTransformer('all-MiniLM-L6-v2')
    
        # Move models to appropriate device
        if DEVICE == "cuda" and (bge_model is None or all_mini_model is None):
            self.bge_model = self.bge_model.to(DEVICE)
            self.all_mini_model = self.all_mini_model.to(DEVICE)
            logger.info("Models moved to GPU for faster inference")
        
        # Search indices
        self.faiss_index = None
        self.bm25 = None
        self.chunks = []
        self.bge_embeddings = None  # Separate BGE embeddings
        self.all_mini_embeddings = None  # Separate All-MiniLM embeddings
        # Reranking configuration
        self.rerank_top_k = 20
        
        logger.info("Retrieval index initialized")


    def _dynamic_resize(self, chunks: List[DocumentChunk],
                        min_chunk_size: int = 150,
                        max_chunk_size: int = CHUNK_SIZE) -> List[DocumentChunk]:
        """
        Dynamically resize chunks between min and max token size.
        """
        adjusted = []
        for chunk in chunks:
            tokens = chunk.text.split()
            if len(tokens) > max_chunk_size:
                # break into sub-chunks
                for i in range(0, len(tokens), max_chunk_size):
                    sub_text = " ".join(tokens[i:i+max_chunk_size])
                    new_chunk = DocumentChunk(text=sub_text,
                                            metadata={**chunk.metadata, "resized": True})
                    #new_chunk.semantic_score = chunk.semantic_score
                    adjusted.append(new_chunk)
            elif len(tokens) < min_chunk_size:
                # keep small ones for now, filtering happens later
                adjusted.append(chunk)
            else:
                adjusted.append(chunk)
        return adjusted


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

        # 🔹 Apply dynamic chunk resizing
        resized_chunks = self._dynamic_resize(unique_chunks)
        """for i, c in enumerate(resized_chunks, 1):
            print(f"Chunk {i} | Score: {c.semantic_score:.3f} | Text preview: {c.text[:60]}...")"""
        # ✅ Merge overlapping/similar chunks
        merged_chunks = merge_chunks(resized_chunks, min_overlap=5)
        logger.info(f"📌 Merged chunks: input={len(resized_chunks)} → output={len(merged_chunks)}")
        return merged_chunks


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
        unique_chunks: List[DocumentChunk] = []
        
        for chunk in chunks:
            candidate_text = chunk.text.strip()
            if len(candidate_text) <= 10:
                continue
            skip_add = False
            replace_index = None
            
            for idx, existing in enumerate(unique_chunks):
                existing_text = existing.text.strip()
                # Exact match
                if candidate_text.lower() == existing_text.lower():
                    skip_add = True
                    break
                # Containment deduplication (prefer longer)
                if _is_contained_with_ratio(candidate_text, existing_text, min_ratio=CONTAINED_RATIO):
                    if len(candidate_text) > len(existing_text):
                        replace_index = idx
                    else:
                        skip_add = True
                    break
                # High Jaccard similarity (near-duplicate)
                if _jaccard_similarity(candidate_text, existing_text) >= JACCARD_SIMILARITY_THRESHOLD:
                    if len(candidate_text) > len(existing_text):
                        replace_index = idx
                    else:
                        skip_add = True
                    break
            
            if replace_index is not None:
                unique_chunks[replace_index] = chunk
            elif not skip_add:
                unique_chunks.append(chunk)
        
        return unique_chunks


    def _create_chunks(self, content: str, file_path: str) -> List[DocumentChunk]:
        """Create document chunks from content with enhanced metadata"""
        if not content.strip():
            logger.warning("No content to chunk")
            return []
        
        # Use enhanced chunking strategies
        chunks = self.chunk_text(content)
        
        # Add source metadata to all chunks
        for chunk in chunks:
            chunk.metadata["source"] = file_path
            chunk.metadata["file_type"] = os.path.splitext(file_path)[1] if "." in file_path else "unknown"
            chunk.metadata["processing_timestamp"] = time.time()
        
        logger.info("Created %s chunks", len(chunks))
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
            # Wait for both to complete
            self.bge_embeddings = bge_future.result()
            self.all_mini_embeddings = all_mini_future.result()
        
        # === Concatenate both embeddings ===
        logger.info("Concatenating BGE and AllMiniLM embeddings...")
        # Concatenate BGE + AllMiniLM (both 384-dim) → 768-dim
        self.combined_embeddings = np.concatenate(
            [self.bge_embeddings, self.all_mini_embeddings], axis=1
        )


        # Store embeddings for ensemble search
        for i, chunk in enumerate(chunks):
            chunk.embedding = self.combined_embeddings[i]  # Store combined embedding
            chunk.chunk_id = f"chunk_{i}"

        logger.info("Building unified FAISS index with combined embeddings...")
        dimension = self.combined_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for normalized embeddings
        self.faiss_index.add(self.combined_embeddings.astype('float32'))
            
        
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


    def semantic_search(self, query: str, top_k: int = 10, score_threshold: float = SEMANTIC_THRESHOLD_CHUNK_SCORE) -> List[SearchResult]:
        """Semantic search using the combined BGE and MiniLM embeddings."""
        if not self.faiss_index:
            raise HTTPException(status_code=500, detail="FAISS index not built")

        embedding_bge = self.bge_model.encode([query], normalize_embeddings=True)
        embedding_all_mini = self.all_mini_model.encode([query], normalize_embeddings=True)

        # Concatenate to match the FAISS index format
        query_embedding = np.concatenate([embedding_bge, embedding_all_mini], axis=1)

        # Search
        scores, indices = self.faiss_index.search(
            query_embedding.astype('float32'), min(top_k*3, len(self.chunks))
        )

        # Prepare results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if score < score_threshold:
                logger.info(f"Dropping chunk {idx} | Score: {score:.3f}")
            if idx < len(self.chunks) and score >= score_threshold:
                chunk = self.chunks[idx]
                result = SearchResult(
                    chunk=chunk,
                    semantic_score=float(score),
                    search_strategy="semantic"
                )
                results.append(result)
                
        return results[:top_k]


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


    def ensemble_search(self, query: str, top_k: int = 10, score_threshold: float = ENSEMBLE_THRESHOLD_SCORE) -> List[SearchResult]:
        """Ensemble search using multiple embedding models"""
        start_time = time.time()
        if not self.faiss_index or not self.bm25:
            raise HTTPException(status_code=500, detail="Indices not built")
        
        # Get results from different models
        bge_results = self.semantic_search(query, top_k * 3) # Semantic (BGE+AllMini) weight
        bm25_results = self.lexical_search(query, top_k * 3)
        
        # Combine results using reciprocal rank fusion
        combined_scores = {}
        
        for i, result in enumerate(bge_results):
            chunk_id = result.chunk.chunk_id
            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = {"chunk": result.chunk, "scores": []}
            combined_scores[chunk_id]["scores"].append(1.0 / (i + 20))
        
        for i, result in enumerate(bm25_results):
            chunk_id = result.chunk.chunk_id
            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = {"chunk": result.chunk, "scores": []}
            combined_scores[chunk_id]["scores"].append(1.0 / (i + 40))  # BM25 weight
        
        # Calculate final scores
        results = []
        for chunk_id, data in combined_scores.items():
            final_score = sum(data["scores"])
            logger.debug("Chunk %s scored %.3f", chunk_id, final_score)
            if final_score >= score_threshold:  # <-- filter low-scoring chunks
                result = SearchResult(
                        chunk=data["chunk"],
                        combined_score=final_score,
                        search_strategy="ensemble"
                    )
                results.append(result)

        # 🔹 Merge duplicates / overlapping chunks
        results = merge_chunks_search(results)
        # Sort by final score and return top-k
        results.sort(key=lambda x: x.combined_score, reverse=True)
        logger.info(f"Ensemble search took {time.time() - start_time:.2f} seconds")
        return results[:top_k]


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
