#!/usr/bin/env python3
"""
Wikipedia Knowledge Base Builder

This script builds a knowledge base from Wikipedia articles based on topics
extracted from presentation files or provided directly. It uses the Wikipedia API
to fetch content and creates a structured knowledge base for use in RAG systems.
Supports both English and Chinese languages.
"""

import os
import json
import re
import time
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import requests
from tqdm import tqdm

# Import project modules
from config import BASE_DIR
from models.factory import ModelFactory
from models.base import Message, MessageRole, MessageContent
from utils import get_temp_dir, DEFAULT_EMBEDDING_MODEL, build_vector_index
from utils.rag import search_knowledge_base

# LLM settings
LLM_KEYWORD_MODEL_PROVIDER = "gpt"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build Wikipedia knowledge base from presentations or topic list.")
    parser.add_argument(
        "--test-set-dir", 
        type=Path,
        default=BASE_DIR / "data" / "test_set",
        help="Path to the directory containing the test set presentation files (pptx/pdf)."
    )
    parser.add_argument(
        "--output-dir", 
        type=Path,
        default=BASE_DIR / "data" / "wiki_knowledge_base",
        help="Path to the directory where Wikipedia knowledge base will be saved."
    )
    parser.add_argument(
        "--topics-file", 
        type=Path,
        default=BASE_DIR / "data" / "test_set" / "topics.json",
        help="Path to save or load the extracted topics in JSON format."
    )
    parser.add_argument(
        "--topics",
        type=str,
        nargs="*",
        help="Direct list of topics to use instead of extracting from presentations."
    )
    parser.add_argument(
        "--max-articles", 
        type=int,
        default=0,
        help="Maximum number of Wikipedia articles to include in the knowledge base. Set to 0 for no limit."
    )
    parser.add_argument(
        "--extract-topics-only",
        action="store_true",
        help="Only extract topics from the test set without building the knowledge base."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Wikipedia language code (e.g., 'en' for English, 'zh' for Chinese)."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Size of chunks (in characters) for splitting Wikipedia articles."
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=300,
        help="Minimum size of chunks to keep."
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model to use for vectorizing chunks (default: BGE-M3)."
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build Milvus vector index for search after creating the knowledge base."
    )
    parser.add_argument(
        "--test-query",
        type=str,
        help="Optional test query to evaluate retrieval after building the knowledge base."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to return when testing retrieval."
    )
    parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Enable multilingual search (English and Chinese)."
    )
    parser.add_argument(
        "--clean-content",
        action="store_true",
        default=True,
        help="Apply data cleaning to Wikipedia content (default: True)."
    )
    parser.add_argument(
        "--no-clean-content",
        action="store_false",
        dest="clean_content",
        help="Disable data cleaning for Wikipedia content."
    )
    return parser.parse_args()

def get_presentation_files(test_set_dir: Path) -> List[Path]:
    """Gets a list of presentation files from the test set directory."""
    files = []
    if not test_set_dir.is_dir():
        logger.error(f"Test set directory not found: {test_set_dir}")
        return files
    
    for item in test_set_dir.iterdir():
        # Add more extensions if needed
        if item.is_file() and item.suffix.lower() in ['.pptx', '.pdf']:
            files.append(item)
            
    logger.info(f"Found {len(files)} presentation files in {test_set_dir}")
    return files

def generate_topics_from_captions(ppt_file_path: Path, captions_file: Path, multilingual: bool = False) -> Optional[Dict]:
    """Generates topics using an LLM based on captions file."""
    base_filename = ppt_file_path.stem
    
    if not captions_file.is_file():
        logger.warning(f"Captions file not found for {base_filename}: {captions_file}")
        return None
        
    try:
        with open(captions_file, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
            
        all_captions_text = "\n".join(f"Slide {k}: {v}" for k, v in captions_data.items())
        
        if not all_captions_text.strip():
             logger.warning(f"Captions file is empty for {base_filename}: {captions_file}")
             return None
             
        logger.info(f"Generating topics for {base_filename} using LLM.")

        # Prepare prompt for LLM
        language_prompt = "English and Chinese" if multilingual else "English"
        prompt = (
            f"Based on the following captions extracted from the slides of a presentation named '{base_filename}', "
            f"identify 3 to 5 main topics that would make good Wikipedia article searches. "
            f"Format the topics as a comma-separated list. "
            f"Focus on clear, concise, general topics that are likely to exist as Wikipedia articles. "
            f"Prioritize general concepts over specific technical terms. "
            f"Return topics in {language_prompt} if the content contains characters in those languages. "
            f"\n\nCaptions:\n---\n{all_captions_text[:2000]}..." # Limit context size
        )

        # Instantiate the LLM
        llm = ModelFactory.get_model(LLM_KEYWORD_MODEL_PROVIDER)

        system_message = Message(
            role=MessageRole.SYSTEM, 
            content=MessageContent.from_text(f"You are an assistant that identifies key topics for Wikipedia research from educational content. You can work with {language_prompt} content.")
        )
        user_message = Message(
            role=MessageRole.USER, 
            content=MessageContent.from_text(prompt)
        )

        # Call the LLM
        response = llm.call(
            messages=[system_message, user_message],
            temperature=0.2,
            max_tokens=100
        )

        topics = response.content.strip()
        # Basic cleaning
        topics = re.sub(r'["\'`]', '', topics)
        topics = re.sub(r'\s+', ' ', topics).strip()

        if not topics:
            logger.warning(f"LLM generated empty topics for {base_filename}.")
            return None

        # Split by commas and clean up
        topic_list = [topic.strip() for topic in topics.split(',')]
        
        logger.info(f"LLM identified topics for {base_filename}: {topic_list}")
        
        return {
            "presentation": base_filename,
            "topics": topic_list,
            "file_path": str(ppt_file_path),
        }

    except Exception as e:
        logger.error(f"Error generating topics for {base_filename}: {e}", exc_info=True)
        return None

def extract_topics_from_test_set(test_set_dir: Path, output_file: Path, multilingual: bool = False) -> List[Dict]:
    """Extract topics from all presentations in the test set directory."""
    presentation_files = get_presentation_files(test_set_dir)
    if not presentation_files:
        logger.error(f"No presentation files found in {test_set_dir}")
        return []
    
    all_topics = []
    
    for ppt_file in tqdm(presentation_files, desc="Extracting topics", unit="file"):
        base_filename = ppt_file.stem
        logger.info(f"Processing presentation: {base_filename}")
        
        # Find captions file
        try:
            temp_dir = get_temp_dir(str(ppt_file))
            captions_file_path = temp_dir / "captions.json"
            
            if not captions_file_path.is_file():
                logger.warning(f"Captions file not found for {base_filename}. Skipping.")
                continue
                
            # Generate topics from captions
            topics_data = generate_topics_from_captions(ppt_file, captions_file_path, multilingual)
            if topics_data:
                all_topics.append(topics_data)
                
        except Exception as e:
            logger.error(f"Error processing {ppt_file.name}: {e}", exc_info=True)
            
    # Save all topics to output file
    if all_topics:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_topics, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved topics for {len(all_topics)} presentations to {output_file}")
    
    return all_topics

def clean_wikipedia_content(content: str, language: str = "en") -> str:
    """Clean Wikipedia content by removing noise and formatting artifacts.
    
    Args:
        content: The raw Wikipedia content to clean
        language: The language of the content ('en' for English, 'zh' for Chinese)
        
    Returns:
        str: Cleaned content
    """
    if not content:
        return ""
    
    # Remove citation references (e.g., [1], [2], [3])
    content = re.sub(r'\[\d+\]', '', content)
    
    # Remove edit section markers that may have been included
    content = re.sub(r'\[edit\]', '', content)
    
    # Clean up multiple newlines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Remove URLs
    content = re.sub(r'https?://\S+', '', content)
    
    # Clean up excessive whitespace
    content = re.sub(r'\s+', ' ', content).strip()
    content = re.sub(r' +', ' ', content)
    
    # Restore paragraph breaks
    content = re.sub(r'\. ([A-Z0-9])', r'.\n\n\1', content)
    
    # Language-specific cleaning
    if language == "zh":
        # For Chinese content, handle specific patterns
        # Remove English text in parentheses often found in Chinese Wikipedia
        content = re.sub(r'\([^)]*[a-zA-Z][^)]*\)', '', content)
    else:
        # English-specific cleaning
        # Remove phrases like "Main article: X"
        content = re.sub(r'Main article: [^\n]+', '', content)
        
        # Remove "See also:" sections
        content = re.sub(r'See also:[^\n]+', '', content)
    
    # Fix broken sentences due to reference removal
    content = re.sub(r'\.+', '.', content)
    content = re.sub(r'\s+\.', '.', content)
    
    # Fix spacing after punctuation
    content = re.sub(r'(\.)([A-Za-z0-9])', r'\1 \2', content)
    content = re.sub(r'(\,)([A-Za-z0-9])', r'\1 \2', content)
    
    # Restore newlines after section header pattern
    if language == "en":
        content = re.sub(r'([A-Z][a-z]+ [A-Za-z ]+)\.', r'\1.\n\n', content)
    
    # Final newline normalization
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content.strip()

def detect_and_remove_noise_sections(content: str, language: str = "en") -> str:
    """Detect and remove noisy sections typically found at the end of Wikipedia articles.
    
    Args:
        content: The Wikipedia content
        language: The language of the content
        
    Returns:
        str: Content with noisy sections removed
    """
    noise_section_patterns = {
        "en": [
            r"See also[\s\n]+",
            r"References[\s\n]+",
            r"External links[\s\n]+",
            r"Further reading[\s\n]+",
            r"Bibliography[\s\n]+",
            r"Notes[\s\n]+"
        ],
        "zh": [
            r"参见[\s\n]+",
            r"参考资料[\s\n]+",
            r"参考文献[\s\n]+",
            r"外部链接[\s\n]+",
            r"扩展阅读[\s\n]+",
            r"注释[\s\n]+"
        ]
    }
    
    # Get the appropriate patterns for the language
    patterns = noise_section_patterns.get(language, noise_section_patterns["en"])
    
    # Split content into sections
    sections = re.split(r'\n\n+', content)
    
    # Find the first noise section
    first_noise_index = len(sections)
    for i, section in enumerate(sections):
        for pattern in patterns:
            if re.match(pattern, section):
                first_noise_index = min(first_noise_index, i)
                break
    
    # Join the clean sections
    clean_content = "\n\n".join(sections[:first_noise_index])
    
    return clean_content

class WikipediaKB:
    """Wikipedia Knowledge Base builder class."""
    
    def __init__(self, output_dir: Path, language: str = "en", chunk_size: int = 1000, min_chunk_size: int = 300, clean_content: bool = True):
        """Initialize the Wikipedia knowledge base builder.
        
        Args:
            output_dir: Directory to save the knowledge base
            language: Wikipedia language code (default: "en" for English)
            chunk_size: Size of chunks for splitting articles (in characters)
            min_chunk_size: Minimum size of chunks to keep
            clean_content: Whether to clean Wikipedia content (default: True)
        """
        self.output_dir = output_dir
        self.language = language
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.clean_content = clean_content
        self.user_agent = "WikiKnowledgeBaseBuilder/1.0"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Language-specific settings
        if language == "zh":
            logger.info("Using Chinese Wikipedia settings")
            # For Chinese, we might need different chunking parameters
            self.chunk_size = max(chunk_size // 2, 500)  # Chinese characters convey more information per character
        
    def _request_with_retry(self, params, max_retries=3):
        """Make API request with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(self.api_url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to retrieve data after {max_retries} attempts")
                    return None
                time.sleep(1 + random.uniform(0, 2))  # Randomized backoff
    
    def search_topic(self, topic, limit=5):
        """Search Wikipedia for a topic and return potential article matches."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": topic,
            "format": "json",
            "srlimit": limit
        }
        
        logger.info(f"Searching Wikipedia for: {topic}")
        result = self._request_with_retry(params)
        
        if result and "query" in result and "search" in result["query"]:
            search_results = result["query"]["search"]
            return [(item["title"], item.get("snippet", "")) for item in search_results]
        
        return []
    
    def get_article_content(self, title):
        """Retrieve full article content for a given title."""
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts|categories|links|pageimages|info|redirects",
            "inprop": "url",
            "explaintext": True,  # Get plain text content
            "exsectionformat": "plain",
            "format": "json"
        }
        
        logger.info(f"Fetching article: {title}")
        result = self._request_with_retry(params)
        
        if not result or "query" not in result or "pages" not in result["query"]:
            logger.warning(f"Failed to retrieve article: {title}")
            return None
        
        # Extract the first page (there should only be one)
        pages = result["query"]["pages"]
        if not pages:
            return None
        
        page_id = next(iter(pages))
        if page_id == "-1":  # Page doesn't exist
            logger.warning(f"Article doesn't exist: {title}")
            return None
        
        page_data = pages[page_id]
        
        # Check if content exists
        if "extract" not in page_data:
            logger.warning(f"No content found for article: {title}")
            return None
        
        # Get categories
        categories = []
        if "categories" in page_data:
            categories = [cat["title"].replace("Category:", "") for cat in page_data["categories"]]
        
        # Get links
        links = []
        if "links" in page_data:
            links = [link["title"] for link in page_data["links"]]
        
        # Get URL
        url = page_data.get("fullurl", f"https://{self.language}.wikipedia.org/wiki/{title.replace(' ', '_')}")
        
        # Get thumbnail if available
        thumbnail = None
        if "thumbnail" in page_data:
            thumbnail = page_data["thumbnail"].get("source")
        
        # Clean the content if enabled
        content = page_data["extract"]
        original_length = len(content)
        
        if self.clean_content:
            # First remove noisy sections
            content = detect_and_remove_noise_sections(content, self.language)
            
            # Then clean the remaining content
            content = clean_wikipedia_content(content, self.language)
            
            cleaned_length = len(content)
            if original_length > 0:
                reduction_percent = 100 * (original_length - cleaned_length) / original_length
                logger.info(f"Cleaned article '{title}': {original_length} → {cleaned_length} chars ({reduction_percent:.1f}% reduction)")
        
        return {
            "title": page_data["title"],
            "pageid": page_data["pageid"],
            "content": content,
            "url": url,
            "categories": categories,
            "links": links,
            "thumbnail": thumbnail,
            "last_modified": page_data.get("touched"),
            "language": self.language,  # Add language to track content source language
            "cleaned": self.clean_content  # Track if content was cleaned
        }
    
    def is_valid_chunk(self, chunk: str) -> bool:
        """Check if a chunk is valid and contains meaningful content.
        
        Args:
            chunk: Text chunk to validate
            
        Returns:
            bool: True if chunk is valid, False otherwise
        """
        # Skip chunks that are too short
        if len(chunk) < self.min_chunk_size:
            return False
            
        # Skip chunks that are mostly whitespace
        if len(chunk.strip()) / len(chunk) < 0.5:
            return False
            
        # Skip chunks with too few sentences
        sentences = re.split(r'[.!?]', chunk)
        if len([s for s in sentences if len(s.strip()) > 0]) < 2:
            return False
            
        # Skip chunks with very high special character ratio
        alpha_ratio = sum(1 for c in chunk if c.isalnum()) / len(chunk)
        if alpha_ratio < 0.5:  # More than half non-alphanumeric
            return False
            
        return True
    
    def chunk_article(self, article_data):
        """Split article content into manageable chunks."""
        content = article_data["content"]
        chunks = []
        
        # Find section headings (capitalized text followed by newlines)
        section_pattern = r'([A-Z][A-Za-z0-9 ]+)\n\n' if article_data["language"] == "en" else r'([\u4e00-\u9fff]+)\n\n'
        sections = re.split(section_pattern, content)
        
        # If we found sections, use them as chunk boundaries
        if len(sections) > 1:
            current_chunk = ""
            current_heading = ""
            
            for i, section in enumerate(sections):
                # Even indices are headings, odd are content
                if i % 2 == 0:
                    current_heading = section
                else:
                    # Add the section heading to the content
                    section_content = f"{current_heading}\n\n{section}" if current_heading else section
                    
                    # Check if adding this section would exceed the chunk size
                    if len(current_chunk) + len(section_content) > self.chunk_size and len(current_chunk) >= self.min_chunk_size:
                        if self.is_valid_chunk(current_chunk):
                            chunks.append(current_chunk.strip())
                        current_chunk = section_content
                    else:
                        if current_chunk:
                            current_chunk += "\n\n" + section_content
                        else:
                            current_chunk = section_content
            
            # Add the last chunk if it's valid
            if current_chunk and self.is_valid_chunk(current_chunk):
                chunks.append(current_chunk.strip())
        else:
            # If no sections were found, fall back to splitting by paragraphs
            paragraphs = re.split(r'\n\n+', content)
            
            current_chunk = ""
            for paragraph in paragraphs:
                # If adding this paragraph would exceed the chunk size, save the current chunk and start a new one
                if len(current_chunk) + len(paragraph) > self.chunk_size and len(current_chunk) >= self.min_chunk_size:
                    if self.is_valid_chunk(current_chunk):
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
            
            # Add the last chunk if it's valid
            if current_chunk and self.is_valid_chunk(current_chunk):
                chunks.append(current_chunk.strip())
        
        # If we still couldn't create good chunks, fall back to simpler approach
        if not chunks:
            # Simple chunking by character count
            for i in range(0, len(content), self.chunk_size):
                chunk = content[i:i + self.chunk_size].strip()
                if self.is_valid_chunk(chunk):
                    chunks.append(chunk)
        
        logger.info(f"Article '{article_data['title']}' split into {len(chunks)} chunks")
        return chunks
    
    def save_article(self, article_data):
        """Save article content and metadata to the knowledge base."""
        if not article_data:
            return None
        
        title = article_data["title"]
        # Use more permissive safe title generation for Chinese content
        if self.language == "zh":
            # For Chinese, we'll just replace problematic characters
            safe_title = re.sub(r'[/\\:*?"<>|]', '_', title)
            safe_title = safe_title.replace(' ', '_')
        else:
            safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        
        # Add language code prefix to directory to avoid name collisions between languages
        lang_prefix = f"{self.language}_" if self.language != "en" else ""
        
        # Create directory for this article
        article_dir = self.output_dir / f"{lang_prefix}{safe_title}"
        article_dir.mkdir(exist_ok=True)
        
        # Create chunks
        chunks = self.chunk_article(article_data)
        
        # Save each chunk as a separate file
        chunk_files = []
        for i, chunk in enumerate(chunks):
            chunk_file = article_dir / f"chunk_{i+1:03d}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(chunk)
            chunk_files.append({
                "filename": chunk_file.name,
                "size": len(chunk),
                "path": str(chunk_file.relative_to(self.output_dir))
            })
        
        # Save full content for reference
        full_content_file = article_dir / "full_content.txt"
        with open(full_content_file, 'w', encoding='utf-8') as f:
            f.write(article_data["content"])
        
        # Create metadata without the full content (to keep it smaller)
        metadata = {
            "title": article_data["title"],
            "pageid": article_data["pageid"],
            "url": article_data["url"],
            "categories": article_data["categories"],
            "links": article_data["links"][:100] if len(article_data["links"]) > 100 else article_data["links"],  # Limit links to avoid huge files
            "thumbnail": article_data["thumbnail"],
            "last_modified": article_data["last_modified"],
            "chunks": chunk_files,
            "chunk_count": len(chunks),
            "total_size": len(article_data["content"]),
            "language": article_data["language"],  # Preserve language information
            "cleaned": article_data.get("cleaned", False)  # Track if content was cleaned
        }
        
        # Save metadata
        metadata_file = article_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved article '{title}' ({self.language}) with {len(chunks)} chunks")
        return metadata
    
    def build_knowledge_base(self, topics, max_articles=0):
        """Build knowledge base from list of topics.
        
        Args:
            topics: List of topics to search
            max_articles: Maximum number of articles (0 means no limit)
        """
        all_metadata = []
        articles_processed = 0
        article_titles_saved = set()  # Track already saved articles
        total_chunks = 0
        total_size = 0
        start_time = time.time()
        articles_by_lang = {}
        chunks_by_lang = {}
        
        # Process topics and search for Wikipedia articles
        for topic in tqdm(topics, desc="Processing topics"):
            if max_articles > 0 and articles_processed >= max_articles:
                break
                
            logger.info(f"Processing topic: {topic}")
            
            # Search for matching articles
            search_results = self.search_topic(topic)
            
            if not search_results:
                logger.warning(f"No search results found for topic: {topic}")
                continue
            
            # Try each search result until we find a good article
            for title, snippet in search_results:
                if max_articles > 0 and articles_processed >= max_articles:
                    break
                
                # Skip if we've already saved this article
                if title in article_titles_saved:
                    logger.info(f"Skipping already saved article: {title}")
                    continue
                
                # Get full article content
                article_data = self.get_article_content(title)
                
                if not article_data:
                    continue
                
                # Save article and get metadata
                metadata = self.save_article(article_data)
                
                if metadata:
                    metadata["original_topic"] = topic
                    all_metadata.append(metadata)
                    article_titles_saved.add(title)
                    articles_processed += 1
                    
                    # Track statistics
                    total_chunks += metadata["chunk_count"]
                    total_size += metadata["total_size"]
                    
                    # Track language statistics
                    lang = metadata["language"]
                    articles_by_lang[lang] = articles_by_lang.get(lang, 0) + 1
                    chunks_by_lang[lang] = chunks_by_lang.get(lang, 0) + metadata["chunk_count"]
                
                # Rate limiting to be nice to Wikipedia
                time.sleep(1 + random.uniform(0, 1))
        
        # Calculate build statistics
        end_time = time.time()
        build_time = end_time - start_time
        hours, remainder = divmod(build_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        build_time_formatted = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        # Create build statistics
        build_stats = {
            "total_articles": articles_processed,
            "total_chunks": total_chunks,
            "total_size_chars": total_size,
            "total_topics_searched": len(topics),
            "articles_by_language": articles_by_lang,
            "chunks_by_language": chunks_by_lang,
            "build_time_seconds": build_time,
            "build_time_formatted": build_time_formatted,
            "average_chunks_per_article": total_chunks / max(1, articles_processed),
            "average_chars_per_chunk": total_size / max(1, total_chunks),
            "average_chars_per_article": total_size / max(1, articles_processed),
            "content_cleaned": self.clean_content,
            "build_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Create index file
        index_path = self.output_dir / "index.json"
        index_data = {
            "articles": all_metadata,
            "count": len(all_metadata),
            "topics": topics,
            "language": self.language,
            "cleaned": self.clean_content,
            "build_stats": build_stats
        }
        
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        
        # Save build statistics separately for easier access
        stats_path = self.output_dir / "build_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(build_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Knowledge base built with {len(all_metadata)} articles containing {total_chunks} chunks")
        logger.info(f"Total build time: {build_time_formatted}")
        logger.info(f"Build statistics saved to {stats_path}")
        
        return all_metadata, build_stats

def main():
    """Main function to build Wikipedia knowledge base."""
    args = parse_args()
    
    # Create direct list of topics if provided
    if args.topics:
        topics = args.topics
        logger.info(f"Using {len(topics)} provided topics: {topics}")
    else:
        # Extract topics from test set
        logger.info(f"Extracting topics from presentations in {args.test_set_dir}")
        topics_data = extract_topics_from_test_set(args.test_set_dir, args.topics_file, args.multilingual)
        
        # If only extracting topics, exit now
        if args.extract_topics_only:
            logger.info("Topic extraction completed. Exiting as requested.")
            return
            
        # Get topic strings for Wikipedia searching
        topics = []
        
        # For multilingual support, we need to handle non-English characters differently
        if args.multilingual:
            # No need to filter out non-ASCII characters
            for topic_data in topics_data:
                for topic in topic_data.get("topics", []):
                    # Clean up topic for use as a search term
                    clean_topic = re.sub(r'\s+', ' ', topic.strip())
                    
                    # Skip very short topics
                    if clean_topic and len(clean_topic) > 3:
                        topics.append(clean_topic)
        else:
            # Define pattern to detect non-English characters
            non_english_pattern = re.compile(r'[^\x00-\x7F]+')
            
            for topic_data in topics_data:
                for topic in topic_data.get("topics", []):
                    # Clean up topic for use as a search term
                    clean_topic = re.sub(r'\s+', ' ', topic.strip())
                    
                    # Skip very short topics and non-English topics if language is English
                    if clean_topic and len(clean_topic) > 3:
                        if non_english_pattern.search(clean_topic):
                            # Skip topics with non-English characters if language is English
                            if args.language == "en":
                                logger.debug(f"Skipping non-English topic: {clean_topic}")
                            else:
                                topics.append(clean_topic)
                        else:
                            topics.append(clean_topic)
        
        # If no topics extracted, use default topics based on language
        if not topics:
            if args.language == "zh":
                topics = [
                    "计算机科学", 
                    "数学", 
                    "物理学",
                    "生物学", 
                    "化学", 
                    "经济学",
                    "历史",
                    "工程",
                    "人工智能",
                    "机器学习"
                ]
            else:  # Default to English
                topics = [
                    "Computer Science", 
                    "Mathematics", 
                    "Physics",
                    "Biology", 
                    "Chemistry", 
                    "Economics",
                    "History",
                    "Engineering",
                    "Artificial Intelligence",
                    "Machine Learning"
                ]
            logger.info(f"Using {len(topics)} default topics for {args.language}: {topics}")
        else:
            # Remove duplicates
            topics = list(set(topics))
            logger.info(f"Using {len(topics)} extracted topics")
    
    # Build knowledge base
    logger.info(f"Building Wikipedia knowledge base in {args.output_dir} for language {args.language}")
    logger.info(f"Content cleaning is {'enabled' if args.clean_content else 'disabled'}")
    if args.max_articles > 0:
        logger.info(f"Limiting to maximum {args.max_articles} articles")
    else:
        logger.info("No article limit set - will process all available articles")
    
    kb = WikipediaKB(
        output_dir=args.output_dir,
        language=args.language,
        chunk_size=args.chunk_size,
        min_chunk_size=args.min_chunk_size,
        clean_content=args.clean_content
    )
    
    articles, build_stats = kb.build_knowledge_base(topics, max_articles=args.max_articles)
    
    # Print summary of build statistics
    logger.info("=" * 50)
    logger.info("KNOWLEDGE BASE BUILD SUMMARY:")
    logger.info(f"Total articles: {build_stats['total_articles']}")
    logger.info(f"Total chunks: {build_stats['total_chunks']}")
    logger.info(f"Total size: {build_stats['total_size_chars']:,} characters")
    logger.info(f"Build time: {build_stats['build_time_formatted']}")
    if len(build_stats['articles_by_language']) > 1:
        for lang, count in build_stats['articles_by_language'].items():
            logger.info(f"  {lang} articles: {count}")
    logger.info("=" * 50)
    
    # Build vector index if requested
    if args.build_index:
        logger.info("Building Milvus vector index")
        build_vector_index(args.output_dir, args.embedding_model)
    
    # Test search if a query is provided
    if args.test_query and args.build_index:
        logger.info(f"Testing search with query: {args.test_query}")
        results = search_knowledge_base(args.output_dir, args.test_query, args.top_k, args.embedding_model)
        
        if results:
            logger.info(f"Found {len(results)} results:")
            for i, result in enumerate(results):
                logger.info(f"Result {i+1}: {result['metadata']['title']} (score: {result['score']:.4f})")
                logger.info(f"  URL: {result['metadata']['url']}")
                logger.info(f"  Language: {result['metadata'].get('language', 'en')}")
                logger.info(f"  Content: {result['content'][:100]}...")
        else:
            logger.warning("No results found")
    
    logger.info(f"Knowledge base building completed. Output saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 