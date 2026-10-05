import os
import re
import unicodedata
import json
import csv
import email
import logging
import tempfile
import fitz  # PyMuPDF
import hashlib
import zipfile
import subprocess
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
from PyPDF2 import PdfReader
from docx import Document
from pptx import Presentation
import pytesseract
import cv2
from bs4 import BeautifulSoup
import pandas as pd
import xlrd
import numpy as np
from tempfile import NamedTemporaryFile
from io import BytesIO
from utils import clean_ocr_text
from source_security import MAX_SOURCE_BYTES, MAX_ARCHIVE_BYTES, MAX_ARCHIVE_MEMBERS, read_bounded
from extraction_models import ExtractionResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('text_extraction.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CleaningOptions:
    """Enhanced cleaning options with better defaults and validation"""
    normalize_unicode: bool = True
    remove_urls: bool = False
    remove_emails: bool = False
    clean_whitespace: bool = True
    preserve_structure: bool = True
    max_length: int = 100000
    enable_ocr: bool = True
    aggressive_cleaning: bool = False
    hybrid_mode: bool = True
    confidence_threshold: float = 0.6
    language: str = 'eng'  # OCR language
    preserve_formatting: bool = True
    remove_metadata: bool = False
    custom_patterns: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate options"""
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")

class EnhancedRAGTextCleaner:
    """Improved text cleaner with caching and better performance"""
    
    def __init__(self, options: CleaningOptions):
        self.options = options
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance"""
        self.url_pattern = re.compile(r'http[s]?://\S+', re.IGNORECASE)
        self.email_pattern = re.compile(r'\S+@\S+', re.IGNORECASE)
        self.whitespace_pattern = re.compile(r'[ \t\r\f]+')
        self.newline_pattern = re.compile(r'\n+')
        self.space_pattern = re.compile(r' +')
        
        # OCR-specific patterns
        self.ocr_ligatures = {
            'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl'
        }
        self.ocr_pattern = re.compile('|'.join(re.escape(k) for k in self.ocr_ligatures.keys()))
    
    @lru_cache(maxsize=1000)
    def _fix_ocr_ligatures(self, text: str) -> str:
        """Fix common OCR ligatures with caching"""
        return self.ocr_pattern.sub(lambda m: self.ocr_ligatures[m.group()], text)
    
    def clean_text(self, text: str, is_ocr: bool = False) -> str:
        """Enhanced text cleaning with better performance"""
        if not text:
            return ""
        
        original_length = len(text)
        
        # Unicode normalization
        if self.options.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # OCR-specific cleaning
        if is_ocr and self.options.aggressive_cleaning:
            text = self._fix_ocr_ligatures(text)
            text = re.sub(r'[|]', 'l', text)  # Common OCR error
            text = re.sub(r'(?<=[a-z])[0O](?=[a-z])', 'o', text)  # 0/O confusion
            text = ''.join(c for c in text if c.isprintable() or c.isspace())
        
        # URL and email removal
        if self.options.remove_urls:
            text = self.url_pattern.sub('', text)
        if self.options.remove_emails:
            text = self.email_pattern.sub('', text)
        
        # Apply custom patterns
        for pattern, replacement in self.options.custom_patterns.items():
            text = re.sub(pattern, replacement, text)
        
        # Whitespace cleaning
        if self.options.clean_whitespace:
            if self.options.preserve_structure:
                text = self.whitespace_pattern.sub(' ', text)
                text = self.newline_pattern.sub('\n', text)
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                text = '\n'.join(lines)
            else:
                text = re.sub(r'\s+', ' ', text)
        
        # Final cleanup
        text = self.space_pattern.sub(' ', text)
        
        # Length limiting with smart truncation
        if len(text) > self.options.max_length:
            truncated = text[:self.options.max_length]
            # Try to truncate at sentence boundary
            last_period = truncated.rfind('.')
            last_newline = truncated.rfind('\n')
            cut_point = max(last_period, last_newline)
            
            if cut_point > self.options.max_length * 0.8:  # If we can save 20% by smart truncation
                text = truncated[:cut_point + 1]
            else:
                text = truncated.rsplit(' ', 1)[0] + "..."
        
        logger.debug(f"Text cleaned: {original_length} -> {len(text)} chars")
        return text.strip()

class FastPPTXOCRExtractor:
    def __init__(self, dpi: int = 300, max_workers: int = 4):
        self.dpi = dpi
        self.max_workers = max_workers

    def _extract_text_native(self, pptx_path: str) -> str:
        prs = Presentation(pptx_path)
        slides_text = []
        for idx, slide in enumerate(prs.slides, 1):
            slide_parts = [shape.text.strip() for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()]
            if slide_parts:
                slides_text.append(f"Slide {idx}:\n" + "\n".join(slide_parts))
        return "\n\n".join(slides_text)

    def _pptx_to_pdf(self, pptx_path: str, out_dir: str) -> str:
        subprocess.run([
            "soffice", "--headless", "--convert-to", "pdf", "--outdir", out_dir, pptx_path
        ], check=True)
        base_name = os.path.splitext(os.path.basename(pptx_path))[0] + ".pdf"
        return os.path.join(out_dir, base_name)

    def _pdf_to_images(self, pdf_path: str, out_dir: str) -> list[str]:
        doc = fitz.open(pdf_path)
        paths = []
        for i, page in enumerate(doc, 1):
            pix = page.get_pixmap(dpi=self.dpi)
            img_path = os.path.join(out_dir, f"slide_{i}.png")
            pix.save(img_path)
            paths.append(img_path)
        return paths

    def _ocr_image(self, image_path: str) -> str:
        with Image.open(image_path) as image:
            return clean_ocr_text(pytesseract.image_to_string(image))

    def _parallel_ocr_images(self, image_paths: list[str]) -> list[str]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(self._ocr_image, image_paths))

    def extract_text_from_pptx_slides(self, pptx_path: str) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            native_text = self._extract_text_native(pptx_path)
            try:
                pdf_path = self._pptx_to_pdf(pptx_path, tmp)
                images = self._pdf_to_images(pdf_path, tmp)
                return [native_text] + self._parallel_ocr_images(images)
            except Exception as e:
                if native_text:
                    logger.warning("Presentation OCR unavailable: %s", type(e).__name__)
                    return [native_text]
                raise

class PDFExtractor:
    """Optimized PDF extractor with better error handling"""
    
    def _needs_ocr_enhancement(self, text: str, page_area: float = 0) -> bool:
        """Improved OCR need detection"""
        if not text.strip():
            return True
        
        text_length = len(text.strip())
        word_count = len(text.split())
        
        # Multiple heuristics
        indicators = [
            text_length < 50 and page_area > 1000000,  # Little text, large page
            word_count > 0 and text_length / word_count > 50,  # Very long "words"
            len(re.findall(r'[|]{2,}', text)) > 0,  # Multiple vertical bars
            len(re.findall(r'[0O]{3,}', text)) > 0,  # OCR confusion patterns
            text_length > 0 and word_count < text_length * 0.05,  # Very few words
        ]
        
        return sum(indicators) >= 2  # Need at least 2 indicators
    
    def _fuse_hybrid_text(self, digital_text: str, ocr_text: str) -> str:
        """Improved text fusion algorithm"""
        if not digital_text.strip() and not ocr_text.strip():
            return ""
        
        if not digital_text.strip():
            return ocr_text.strip()
        
        if not ocr_text.strip():
            return digital_text.strip()
        
        # Extract important patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        
        digital_emails = set(re.findall(email_pattern, digital_text, re.IGNORECASE))
        digital_urls = set(re.findall(url_pattern, digital_text))
        ocr_emails = set(re.findall(email_pattern, ocr_text, re.IGNORECASE))
        ocr_urls = set(re.findall(url_pattern, ocr_text))
        
        all_emails = digital_emails.union(ocr_emails)
        all_urls = digital_urls.union(ocr_urls)
        
        # Choose primary text based on quality metrics
        digital_score = self._calculate_text_quality(digital_text)
        ocr_score = self._calculate_text_quality(ocr_text)
        
        if digital_score > ocr_score * 1.2:  # Digital significantly better
            base_text = digital_text
            supplement_text = ocr_text
        elif ocr_score > digital_score * 1.2:  # OCR significantly better
            base_text = ocr_text
            supplement_text = digital_text
        else:  # Similar quality, prefer digital
            base_text = digital_text
            supplement_text = ocr_text
        
        # Add unique valuable content from supplement
        supplement_lines = supplement_text.split('\n')
        unique_content = []
        
        for line in supplement_lines:
            line = line.strip()
            if line and line not in base_text and len(line) > 5:
                # Check if line contains valuable information
                if (any(email in line for email in all_emails) or 
                    any(url in line for url in all_urls) or
                    len(line.split()) > 2):  # Multi-word lines
                    unique_content.append(line)
        
        # Combine texts
        if unique_content:
            fused_text = base_text + "\n\n" + "\n".join(unique_content)
        else:
            fused_text = base_text
        
        # Ensure important data is preserved
        missing_emails = all_emails - set(re.findall(email_pattern, fused_text, re.IGNORECASE))
        missing_urls = all_urls - set(re.findall(url_pattern, fused_text))
        
        if missing_emails or missing_urls:
            appendix = []
            if missing_emails:
                appendix.append("Contact Information: " + ", ".join(missing_emails))
            if missing_urls:
                appendix.append("Web Links: " + ", ".join(missing_urls))
            
            if appendix:
                fused_text += "\n\n" + "\n".join(appendix)
        
        return fused_text.strip()
    
    def _calculate_text_quality(self, text: str) -> float:
        """Calculate text quality score"""
        if not text.strip():
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        # Various quality metrics
        avg_word_length = sum(len(word) for word in words) / len(words)
        alpha_ratio = sum(1 for char in text if char.isalpha()) / len(text)
        space_ratio = text.count(' ') / len(text)
        sentence_count = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
        
        # Penalize OCR artifacts
        artifact_penalty = (
            text.count('|') * 0.1 +
            len(re.findall(r'[0O]{2,}', text)) * 0.2 +
            len(re.findall(r'[l1]{3,}', text)) * 0.2
        )
        
        quality_score = (
            min(avg_word_length / 5, 1.0) * 0.3 +  # Reasonable word length
            alpha_ratio * 0.3 +  # Alphabetic content
            min(space_ratio * 10, 1.0) * 0.2 +  # Proper spacing
            min(sentence_count / 10, 1.0) * 0.2  # Sentence structure
        ) - artifact_penalty
        
        return max(0.0, quality_score)
    
    def _process_page_optimized(self, doc, page_num: int, enable_ocr: bool, 
                               custom_config: str, hybrid_mode: bool = True) -> Tuple[int, str, bool]:
        """Enhanced page processing with better error handling"""
        try:
            page = doc.load_page(page_num)
            digital_text = page.get_text() or ""
            ocr_used = False
            ocr_text = ""
            
            # Determine if OCR is needed
            if hybrid_mode and enable_ocr:
                page_rect = page.rect
                page_area = page_rect.width * page_rect.height
                needs_ocr = self._needs_ocr_enhancement(digital_text, page_area)
                
                if needs_ocr or not digital_text.strip():
                    try:
                        # Enhanced OCR with multiple strategies
                        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))  # Higher resolution
                        img_data = pix.tobytes("png")
                        img = Image.open(BytesIO(img_data))
                        
                        # Try multiple OCR approaches
                        ocr_results = self._try_multiple_ocr(img, custom_config)
                        
                        if ocr_results:
                            ocr_text = max(ocr_results, key=len)
                            ocr_used = True
                            logger.debug(f"OCR used for page {page_num + 1}")
                        
                    except Exception as e:
                        logger.warning(f"OCR failed for page {page_num + 1}: {e}")
            
            # Fuse texts intelligently
            if hybrid_mode and ocr_used and ocr_text.strip():
                final_text = self._fuse_hybrid_text(digital_text, ocr_text)
            else:
                final_text = digital_text
            
            return (page_num, final_text, ocr_used)
            
        except Exception as e:
            logger.error(f"Failed to process page {page_num + 1}: {e}")
            return (page_num, "", False)
    
    def _try_multiple_ocr(self, img: Image.Image, base_config: str) -> List[str]:
        """Try multiple OCR strategies"""
        results = []
        
        # Strategy 1: Basic OCR
        try:
            text = pytesseract.image_to_string(img, config=base_config)
            if text.strip():
                results.append(text)
        except Exception:
            pass
        
        # Strategy 2: Preprocessed OCR
        try:
            img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            
            # Multiple preprocessing approaches
            processed_images = [
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
                cv2.medianBlur(gray, 3)
            ]
            
            for processed in processed_images:
                try:
                    text = pytesseract.image_to_string(processed, config=base_config)
                    if text.strip() and len(text.strip()) > 10:
                        results.append(text)
                        break  # Use first successful result
                except Exception:
                    continue
                    
        except Exception:
            pass
        
        # Strategy 3: Different PSM modes
        if not results or len(results[0].strip()) < 20:
            for psm in [3, 4, 6, 8, 13]:
                try:
                    config = f'--oem 3 --psm {psm}'
                    text = pytesseract.image_to_string(img, config=config)
                    if text.strip() and len(text.strip()) > 10:
                        results.append(text)
                        break
                except Exception:
                    continue
        
        return results
    
    def extract(self, file_bytes: bytes, enable_ocr: bool = True, 
                hybrid_mode: bool = True) -> ExtractionResult:
        """Extract text from PDF with comprehensive error handling"""
        start_time = time.time()
        
        try:
            # Try PyMuPDF first
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            metadata = {
                "pages": doc.page_count,
                "method": "PyMuPDF_hybrid" if hybrid_mode else "PyMuPDF",
                "ocr_used": False,
                "ocr_pages": [],
                "hybrid_mode": hybrid_mode
            }
            
            # Process pages with controlled concurrency
            max_workers = min(8, os.cpu_count() or 1)  # Limit workers
            results = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._process_page_optimized, 
                        doc, page_num, enable_ocr, "--oem 1 --psm 4", hybrid_mode
                    ): page_num
                    for page_num in range(doc.page_count)
                }
                
                for future in as_completed(futures, timeout=600):  # 5 minute total timeout
                    try:
                        result = future.result(timeout=120)  # 1 minute per page
                        results.append(result)
                    except Exception as e:
                        page_num = futures[future]
                        logger.warning(f"Page {page_num + 1} processing failed: {e}")
                        results.append((page_num, "", False))
            
            # Sort and assemble results
            results.sort(key=lambda x: x[0])
            page_texts = []
            
            for page_num, page_text, ocr_used in results:
                if page_text.strip():
                    page_texts.append(page_text)
                    if ocr_used:
                        metadata["ocr_pages"].append(page_num)
            
            text = "\n\n".join(page_texts)
            metadata["ocr_used"] = len(metadata["ocr_pages"]) > 0
            doc.close()
            
            processing_time = time.time() - start_time
            
            return ExtractionResult(
                text=text.strip(),
                metadata=metadata,
                success=True,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"PyMuPDF failed, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            try:
                reader = PdfReader(BytesIO(file_bytes))
                page_texts = []
                
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        page_texts.append(page_text)
                
                text = "\n\n".join(page_texts)
                processing_time = time.time() - start_time
                
                return ExtractionResult(
                    text=text.strip(),
                    metadata={
                        "pages": len(reader.pages),
                        "method": "PyPDF2_fallback",
                        "ocr_used": False,
                        "fallback_used": True
                    },
                    success=True,
                    processing_time=processing_time
                )
                
            except Exception as e2:
                processing_time = time.time() - start_time
                return ExtractionResult(
                    text="",
                    metadata={"method": "failed"},
                    success=False,
                    error=f"Both PyMuPDF and PyPDF2 failed: {str(e2)}",
                    processing_time=processing_time
                )

class EnhancedTextExtractionService:
    """Dispatch byte content to format specific extractors."""

    def __init__(self):
        
        # Initialize extractors
        self.pdf_extractor = PDFExtractor()
        
        # Register extractors
        self.extractors = {
            ".pdf": self._extract_pdf,
            ".docx": self._extract_docx,
            ".pptx": self._extract_pptx,
            ".png": self._extract_image,
            ".jpg": self._extract_image,
            ".jpeg": self._extract_image,
            ".tiff": self._extract_image,
            ".bmp": self._extract_image,
            ".txt": self._extract_text,
            ".eml": self._extract_email,
            ".html": self._extract_html,
            ".csv": self._extract_csv,
            ".json": self._extract_json,
            ".xlsx": self._extract_excel,
            ".xls": self._extract_excel,
            ".zip": self._extract_zip,
            ".bin": self._extract_binary,
        }
    
    def extract_text_from_bytes(self, file_bytes: bytes, filename: str, 
                               cleaning_options: Optional[CleaningOptions] = None) -> ExtractionResult:
        """Extract text from file bytes; DocumentService owns the cache."""
        start_time = time.time()
        if len(file_bytes) > MAX_SOURCE_BYTES:
            return ExtractionResult(text="", metadata={"filename": filename}, success=False,
                                    error="Document exceeds the size limit")
        
        # Set default options
        if cleaning_options is None:
            cleaning_options = CleaningOptions()
        
        try:
            ext = os.path.splitext(filename)[-1].lower()
            if ext in {".zip", ".docx", ".pptx", ".xlsx"}:
                with zipfile.ZipFile(BytesIO(file_bytes)) as archive:
                    members = archive.infolist()
                    max_members = MAX_ARCHIVE_MEMBERS if ext == ".zip" else 1000
                    if len(members) > max_members or sum(member.file_size for member in members) > MAX_ARCHIVE_BYTES:
                        return ExtractionResult(text="", metadata={"filename": filename}, success=False,
                                                error="Archive exceeds the extraction limit")
            
            if ext not in self.extractors:
                return ExtractionResult(
                    text="",
                    metadata={"error": f"Unsupported file type: {ext}"},
                    success=False,
                    error=f"Unsupported file type: {ext}. Supported: {list(self.extractors.keys())}"
                )
            
            # Extract text
            result = self.extractors[ext](file_bytes, cleaning_options)
            
            # Apply cleaning
            if result.success and cleaning_options:
                cleaner = EnhancedRAGTextCleaner(cleaning_options)
                is_ocr = result.metadata.get("ocr_used", False)
                result.text = cleaner.clean_text(result.text, is_ocr=is_ocr)
            
            # Add processing metadata
            result.processing_time = time.time() - start_time
            result.file_hash = hashlib.sha256(file_bytes).hexdigest()
            result.metadata.update({
                "filename": filename,
                "file_type": ext,
                "file_size": len(file_bytes),
                "cleaning_applied": cleaning_options is not None
            })
            
            return result
            
        except Exception as e:
            logger.error("Text extraction failed: %s", type(e).__name__)
            return ExtractionResult(
                text="",
                metadata={"filename": filename},
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    # Extractor methods
    def _extract_pdf(self, file_bytes: bytes, options: CleaningOptions) -> ExtractionResult:
        return self.pdf_extractor.extract(file_bytes, options.enable_ocr, options.hybrid_mode)
    
    def _extract_docx(self, file_bytes: bytes, options: CleaningOptions) -> ExtractionResult:
        try:
            doc = Document(BytesIO(file_bytes))
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            
            # Extract table content
            table_text = []
            for table in doc.tables:
                for row in table.rows:
                    row_text = [cell.text.strip() for cell in row.cells]
                    table_text.append(" | ".join(row_text))
            
            text = "\n".join(paragraphs)
            if table_text:
                text += "\n\nTables:\n" + "\n".join(table_text)
            
            return ExtractionResult(
                text=text.strip(),
                metadata={
                    "paragraphs": len(paragraphs),
                    "tables": len(doc.tables),
                    "method": "python-docx"
                },
                success=True
            )
        except Exception as e:
            return ExtractionResult(
                text="",
                metadata={"method": "python-docx"},
                success=False,
                error=str(e)
            )

    def _extract_pptx(self, file_bytes: bytes, options: CleaningOptions) -> ExtractionResult:
        import tempfile
        import os
        
        # Create a temporary file for the uploaded pptx bytes
        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as tmp_pptx:
            tmp_pptx.write(file_bytes)
            tmp_pptx_path = tmp_pptx.name

        try:
            # Initialize your extractor
            extractor = FastPPTXOCRExtractor(dpi=400, max_workers=6)

            # Extract text from slides
            slide_texts = extractor.extract_text_from_pptx_slides(tmp_pptx_path)

            # Join all slide texts with two newlines between slides
            full_text = "\n\n".join(slide_texts)

            return ExtractionResult(
                text=full_text,
                metadata={"slides": len(slide_texts), "method": "FastPPTXOCRExtractor"},
                success=True
            )
        except Exception as e:
            return ExtractionResult(
                text="",
                metadata={"method": "FastPPTXOCRExtractor"},
                success=False,
                error=str(e)
            )
        finally:
            # Clean up the temporary pptx file
            if os.path.exists(tmp_pptx_path):
                os.remove(tmp_pptx_path)

    def _extract_image(self, file_bytes: bytes, options: CleaningOptions) -> ExtractionResult:
        try:
            img = Image.open(BytesIO(file_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            results = []
            methods_used = []
            
            # Method 1: Basic Tesseract
            try:
                config = f'--oem 3 --psm 6 -l {options.language}'
                text_basic = pytesseract.image_to_string(img, config=config)
                if text_basic.strip():
                    results.append(text_basic)
                    methods_used.append("Tesseract-basic")
            except Exception as e:
                logger.warning(f"Basic Tesseract failed: {e}")
            
            # Method 2: Preprocessed if basic failed or insufficient
            if not results or len(results[0].strip()) < 50:
                try:
                    img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                    
                    # Multiple preprocessing strategies
                    processed_variants = [
                        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
                        cv2.medianBlur(gray, 3)
                    ]
                    
                    for i, processed in enumerate(processed_variants):
                        try:
                            text_preproc = pytesseract.image_to_string(processed, config=config)
                            if text_preproc.strip() and len(text_preproc.strip()) > (len(results[0].strip()) if results else 0):
                                results = [text_preproc]  # Replace with better result
                                methods_used = [f"Tesseract-preprocessed-{i}"]
                                break
                        except Exception:
                            continue
                            
                except Exception as e:
                    logger.warning(f"Preprocessed OCR failed: {e}")
            
            # Method 3: Different PSM modes if still insufficient
            if not results or len(results[0].strip()) < 20:
                for psm in [3, 4, 8, 13]:
                    try:
                        psm_config = f'--oem 3 --psm {psm} -l {options.language}'
                        text_psm = pytesseract.image_to_string(img, config=psm_config)
                        if text_psm.strip() and len(text_psm.strip()) > (len(results[0].strip()) if results else 0):
                            results = [text_psm]
                            methods_used = [f"Tesseract-psm{psm}"]
                            break
                    except Exception:
                        continue
            
            # Get confidence scores if available
            confidence_scores = []
            if results:
                try:
                    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)
                    confidence_scores = [int(conf) for conf in data['conf'] if int(conf) > 0]
                except Exception:
                    pass
            
            final_text = results[0] if results else ""
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            return ExtractionResult(
                text=final_text.strip(),
                metadata={
                    "methods": methods_used,
                    "confidence_scores": confidence_scores,
                    "avg_confidence": avg_confidence,
                    "image_size": img.size,
                    "image_mode": img.mode
                },
                success=bool(final_text.strip())
            )
            
        except Exception as e:
            return ExtractionResult(
                text="",
                metadata={"method": "OCR"},
                success=False,
                error=str(e)
            )
    
    def _extract_text(self, file_bytes: bytes, options: CleaningOptions) -> ExtractionResult:
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            text = ""
            encoding_used = ""
            
            for encoding in encodings:
                try:
                    text = file_bytes.decode(encoding)
                    encoding_used = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if not text:
                # Fallback with error handling
                text = file_bytes.decode('utf-8', errors='ignore')
                encoding_used = 'utf-8-ignore'
            
            return ExtractionResult(
                text=text.strip(),
                metadata={"encoding": encoding_used, "method": "direct"},
                success=True
            )
        except Exception as e:
            return ExtractionResult(
                text="",
                metadata={"method": "direct"},
                success=False,
                error=str(e)
            )
    
    def _extract_email(self, file_bytes: bytes, options: CleaningOptions) -> ExtractionResult:
        try:
            content_str = file_bytes.decode('utf-8', errors='ignore')
            msg = email.message_from_string(content_str)
            
            text_content = []
            text_content.append(f"Subject: {msg.get('Subject', '')}")
            text_content.append(f"From: {msg.get('From', '')}")
            text_content.append(f"To: {msg.get('To', '')}")
            text_content.append(f"Date: {msg.get('Date', '')}")
            text_content.append("")  # Empty line
            
            body_parts = []
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body_parts.append(payload.decode('utf-8', errors='ignore'))
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body_parts.append(payload.decode('utf-8', errors='ignore'))
            
            if body_parts:
                text_content.extend(body_parts)
            
            return ExtractionResult(
                text="\n".join(text_content).strip(),
                metadata={
                    "subject": msg.get('Subject', ''),
                    "from": msg.get('From', ''),
                    "to": msg.get('To', ''),
                    "multipart": msg.is_multipart(),
                    "method": "email"
                },
                success=True
            )
        except Exception as e:
            return ExtractionResult(
                text="",
                metadata={"method": "email"},
                success=False,
                error=str(e)
            )
    
    def _extract_html(self, file_bytes: bytes, options: CleaningOptions) -> ExtractionResult:
        try:
            content = file_bytes.decode('utf-8', errors='ignore')
            soup = BeautifulSoup(content, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text with structure preservation
            if options.preserve_structure:
                text = soup.get_text(separator='\n')
            else:
                text = soup.get_text(separator=' ')
            
            # Extract metadata
            title = soup.find('title')
            meta_description = soup.find('meta', attrs={'name': 'description'})
            
            return ExtractionResult(
                text=text.strip(),
                metadata={
                    "title": title.string if title else "",
                    "description": meta_description.get('content', '') if meta_description else "",
                    "parser": "html.parser",
                    "method": "BeautifulSoup"
                },
                success=True
            )
        except Exception as e:
            return ExtractionResult(
                text="",
                metadata={"method": "BeautifulSoup"},
                success=False,
                error=str(e)
            )
    
    def _extract_csv(self, file_bytes: bytes, options: CleaningOptions) -> ExtractionResult:
        try:
            # Try different encodings and delimiters
            encodings = ['utf-8', 'latin-1', 'cp1252']
            delimiters = [',', ';', '\t', '|']
            
            for encoding in encodings:
                try:
                    content = file_bytes.decode(encoding)
                    # Auto-detect delimiter
                    sample = content[:1024]
                    sniffer = csv.Sniffer()
                    
                    try:
                        delimiter = sniffer.sniff(sample).delimiter
                    except:
                        delimiter = ','  # Default
                    
                    # Read CSV
                    df = pd.read_csv(BytesIO(file_bytes), encoding=encoding, delimiter=delimiter)
                    
                    return ExtractionResult(
                        text=df.to_string(index=False),
                        metadata={
                            "rows": len(df),
                            "columns": len(df.columns),
                            "encoding": encoding,
                            "delimiter": delimiter,
                            "method": "pandas"
                        },
                        success=True
                    )
                except Exception:
                    continue
            
            # Fallback
            raise Exception("Could not parse CSV with any encoding/delimiter combination")
            
        except Exception as e:
            return ExtractionResult(
                text="",
                metadata={"method": "pandas"},
                success=False,
                error=str(e)
            )
    
    def _extract_json(self, file_bytes: bytes, options: CleaningOptions) -> ExtractionResult:
        try:
            content = file_bytes.decode('utf-8', errors='ignore')
            data = json.loads(content)
            
            if options.preserve_formatting:
                formatted_json = json.dumps(data, indent=2, ensure_ascii=False)
            else:
                formatted_json = json.dumps(data, ensure_ascii=False)
            
            return ExtractionResult(
                text=formatted_json,
                metadata={
                    "valid": True,
                    "type": type(data).__name__,
                    "method": "json"
                },
                success=True
            )
        except json.JSONDecodeError as e:
            return ExtractionResult(
                text="",
                metadata={"method": "json", "valid": False},
                success=False,
                error=f"Invalid JSON: {str(e)}"
            )
        except Exception as e:
            return ExtractionResult(
                text="",
                metadata={"method": "json"},
                success=False,
                error=str(e)
            )
    
    def _extract_excel(self, file_bytes: bytes, options: CleaningOptions) -> ExtractionResult:
        try:
            with NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            
            try:
                # Try with pandas (modern Excel files)
                try:
                    df_dict = pd.read_excel(tmp_path, sheet_name=None, engine='openpyxl')
                    sheets_text = []
                    
                    for name, df in df_dict.items():
                        sheet_content = f"Sheet: {name}\n{df.to_string(index=False)}"
                        sheets_text.append(sheet_content)
                    
                    return ExtractionResult(
                        text="\n\n".join(sheets_text),
                        metadata={
                            "sheets": len(df_dict),
                            "total_rows": sum(len(df) for df in df_dict.values()),
                            "method": "pandas-openpyxl"
                        },
                        success=True
                    )
                except Exception:
                    # Fallback to xlrd for older files
                    book = xlrd.open_workbook(tmp_path)
                    sheets_text = []
                    
                    for sheet in book.sheets():
                        sheet_data = []
                        for row_idx in range(sheet.nrows):
                            row = [str(sheet.cell_value(row_idx, col)) for col in range(sheet.ncols)]
                            sheet_data.append(", ".join(row))
                        
                        sheet_content = f"Sheet: {sheet.name}\n" + "\n".join(sheet_data)
                        sheets_text.append(sheet_content)
                    
                    return ExtractionResult(
                        text="\n\n".join(sheets_text),
                        metadata={
                            "sheets": len(book.sheets()),
                            "method": "xlrd"
                        },
                        success=True
                    )
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            return ExtractionResult(
                text="",
                metadata={"method": "excel"},
                success=False,
                error=str(e)
            )

    def _extract_zip(self, file_bytes: bytes, options: CleaningOptions) -> ExtractionResult:
        """Extract text from ZIP archives by processing all supported files within"""
        try:
            with zipfile.ZipFile(BytesIO(file_bytes), 'r') as zip_file:
                members = zip_file.infolist()
                if len(members) > MAX_ARCHIVE_MEMBERS or sum(member.file_size for member in members) > MAX_ARCHIVE_BYTES:
                    return ExtractionResult(text="", metadata={"method": "zip"}, success=False,
                                            error="Archive exceeds the extraction limit")
                # Get list of files in the archive
                file_list = zip_file.namelist()
                
                # Filter for supported file types
                supported_extensions = {
                    '.pdf', '.docx', '.pptx', '.txt', '.html', '.csv', 
                    '.json', '.xlsx', '.xls', '.eml'
                }
                
                supported_files = [
                    f for f in file_list 
                    if os.path.splitext(f)[1].lower() in supported_extensions
                    and not f.startswith('__MACOSX/')  # Skip macOS metadata
                    and not f.startswith('.')  # Skip hidden files
                ]
                
                if not supported_files:
                    return ExtractionResult(
                        text="",
                        metadata={
                            "method": "zip",
                            "total_files": len(file_list),
                            "supported_files": 0,
                            "file_list": file_list[:10]  # First 10 files for reference
                        },
                        success=False,
                        error="No supported text files found in ZIP archive"
                    )
                
                # Process supported files
                extracted_texts = []
                processed_files = []
                failed_files = []
                total_uncompressed = 0
                
                for filename in supported_files:
                    try:
                        # Read file from ZIP
                        with zip_file.open(filename) as file_in_zip:
                            file_bytes_in_zip = read_bounded(
                                file_in_zip,
                                min(MAX_SOURCE_BYTES, MAX_ARCHIVE_BYTES - total_uncompressed)
                            )
                        total_uncompressed += len(file_bytes_in_zip)
                        
                        # Extract text using the main service
                        result = self.extract_text_from_bytes(file_bytes_in_zip, filename, options)
                        
                        if result.success and result.text.strip():
                            extracted_texts.append(f"=== {filename} ===\n{result.text}")
                            processed_files.append(filename)
                        else:
                            failed_files.append(filename)
                            
                    except ValueError:
                        return ExtractionResult(text="", metadata={"method": "zip"}, success=False,
                                                error="Archive exceeds the extraction limit")
                    except Exception as e:
                        logger.warning("ZIP member extraction failed: %s", type(e).__name__)
                        failed_files.append(filename)
                
                if not extracted_texts:
                    return ExtractionResult(
                        text="",
                        metadata={
                            "method": "zip",
                            "total_files": len(file_list),
                            "supported_files": len(supported_files),
                            "processed_files": processed_files,
                            "failed_files": failed_files
                        },
                        success=False,
                        error="Failed to extract text from any files in ZIP archive"
                    )
                
                # Combine all extracted texts
                combined_text = "\n\n".join(extracted_texts)
                
                return ExtractionResult(
                    text=combined_text.strip(),
                    metadata={
                        "method": "zip",
                        "total_files": len(file_list),
                        "supported_files": len(supported_files),
                        "processed_files": processed_files,
                        "failed_files": failed_files,
                        "archive_contents": file_list[:20]  # First 20 files for reference
                    },
                    success=True
                )
                
        except zipfile.BadZipFile:
            return ExtractionResult(
                text="",
                metadata={"method": "zip"},
                success=False,
                error="Invalid or corrupted ZIP file"
            )
        except Exception as e:
            return ExtractionResult(
                text="",
                metadata={"method": "zip"},
                success=False,
                error=f"Error processing ZIP file: {str(e)}"
            )

    def _extract_binary(self, file_bytes: bytes, options: CleaningOptions) -> ExtractionResult:
        """Handle binary files that cannot be parsed as text"""
        try:
            # Try to detect if it's actually a text file with wrong extension
            try:
                # Attempt to decode as text with various encodings
                for encoding in ['utf-8', 'utf-16', 'latin-1', 'cp1252']:
                    try:
                        text_content = file_bytes.decode(encoding)
                        # If we can decode it and it looks like text, treat it as text
                        if len(text_content) > 0 and any(c.isprintable() for c in text_content[:100]):
                            return ExtractionResult(
                                text=text_content.strip(),
                                metadata={
                                    "method": "binary_as_text",
                                    "encoding": encoding,
                                    "file_size": len(file_bytes),
                                    "note": "Binary file contained readable text content"
                                },
                                success=True
                            )
                    except UnicodeDecodeError:
                        continue
                
                # If we can't decode as text, it's truly binary
                return ExtractionResult(
                    text="",
                    metadata={
                        "method": "binary",
                        "file_size": len(file_bytes),
                        "file_type": "binary_data",
                        "note": "This is a binary file that cannot be parsed as text. Binary files contain non-text data (images, executables, compressed data, etc.) and cannot be processed for text extraction."
                    },
                    success=False,
                    error="Binary file cannot be parsed as text. This file contains non-text data (images, executables, compressed data, etc.) and cannot be processed for text extraction."
                )
                
            except Exception as e:
                return ExtractionResult(
                    text="",
                    metadata={
                        "method": "binary",
                        "file_size": len(file_bytes),
                        "error": str(e)
                    },
                    success=False,
                    error=f"Error analyzing binary file: {str(e)}"
                )
                
        except Exception as e:
            return ExtractionResult(
                text="",
                metadata={"method": "binary"},
                success=False,
                error=f"Unexpected error processing binary file: {str(e)}"
            )
