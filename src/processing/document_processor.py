import asyncio
import json
import re
from pathlib import Path
from typing import Dict, Any, Set
from datetime import datetime, timezone
from src.core.services.embedding import EmbeddingService
from src.utils.logging import logger
from src.core.services.db_service import DatabaseService
from .markdown_converter import MarkdownConverter
from src.config.settings import settings


class DocumentProcessor:
    def __init__(
        self,
        db_service: DatabaseService,
        embedding_service: EmbeddingService
    ):
        self.db_service = db_service
        self.embedding_service = embedding_service
        self.markdown_converter = MarkdownConverter()
        self.progress_file = Path("processing_progress.json")
    
    def _load_progress(self) -> Dict[str, Set[str]]:
        """Load processing progress from file."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
                # Convert lists back to sets
                return {k: set(v) for k, v in progress.items()}
        return {}

    def _save_progress(self, progress: Dict[str, Set[str]]):
        """Save processing progress to file."""
        # Convert sets to lists for JSON serialization
        progress_json = {k: list(v) for k, v in progress.items()}
        with open(self.progress_file, 'w') as f:
            json.dump(progress_json, f)

    async def process_chunk(
        self,
        chunk: Dict[str, Any],
        chunk_number: int,
        file_path: str,
        version: int
    ):
        try:
            # Get the header path from metadata
            header_path = chunk["metadata"].get("header_path", "")
            
            # Get document URL - only use the URL part, not the version
            documentation_url, _ = self.markdown_converter.convert_path_to_url(
                file_path,
                header_path
            )
            
            # Extract title
            title = self.extract_title_from_chunk(chunk)
            
            # Get embedding
            embedding = await self.embedding_service.get_embedding(chunk["content"])
            
            # Prepare metadata
            metadata = {
                "source": "markdown_file",
                "chunk_size": len(chunk["content"]),
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "filename": Path(file_path).name,
                "version_str": f"{version/10:.1f}",
                **chunk["metadata"]
            }
            
            # Insert into database
            return await self._insert_chunk({
                "url": documentation_url,  # Now only contains the URL string
                "chunk_number": chunk_number,
                "title": title,
                "content": chunk["content"],
                "metadata": metadata,
                "embedding": embedding,
                "version": version
            })
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            raise

    async def process_file(self, file_path: str, version: int):
        """Process individual file with chunk tracking."""
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Read and chunk the markdown file
            chunks = self.markdown_converter.chunk_markdown(file_path)
            logger.info(f"Split into {len(chunks)} chunks")
            
            # Process chunks with retries
            for i, chunk in enumerate(chunks):
                max_retries = 3
                retry_delay = 1
                
                for attempt in range(max_retries):
                    try:
                        await self.process_chunk(chunk, i, file_path, version)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        logger.warning(f"Retry {attempt + 1}/{max_retries} for chunk {i} due to: {e}")
                        await asyncio.sleep(retry_delay * (attempt + 1))
            
            logger.info(f"Successfully processed {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise

    async def process_directory(self, base_directory: str):
        """Process directory with progress tracking."""
        progress = self._load_progress()
        
        try:
            version_dirs = settings.odoo_versions_list
            for version_str in version_dirs:
                version = int(float(version_str) * 10)
                version_path = Path(base_directory) / "versions" / version_str
                
                if not version_path.exists():
                    logger.warning(f"Version directory {version_path} does not exist")
                    continue
                
                # Initialize progress tracking for this version if not exists
                if version_str not in progress:
                    progress[version_str] = set()
                
                logger.info(f"Processing version {version_str}")
                
                # Get all markdown files
                markdown_files = list(version_path.rglob("*.md"))
                logger.info(f"Found {len(markdown_files)} markdown files")
                
                # Process unprocessed files
                for file_path in markdown_files:
                    file_str = str(file_path)
                    if file_str in progress[version_str]:
                        logger.info(f"Skipping already processed file: {file_str}")
                        continue
                    
                    try:
                        await self.process_file(file_str, version)
                        progress[version_str].add(file_str)
                        self._save_progress(progress)
                        logger.info(f"Successfully processed and saved progress for {file_str}")
                    except Exception as e:
                        logger.error(f"Error processing file {file_str}: {e}")
                        # Don't save progress for failed file
                        raise
                        
        except Exception as e:
            logger.error(f"Error processing directory {base_directory}: {e}")
            raise
        finally:
            # Ensure progress is saved even if there's an error
            self._save_progress(progress)

    async def _insert_chunk(self, chunk_data: Dict[str, Any]):
        try:
            result = await self.db_service.insert_document(chunk_data)
            logger.info(
                f"Inserted chunk {chunk_data['chunk_number']} "
                f"(version {chunk_data['metadata']['version_str']}): "
                f"{chunk_data['title']}"
            )
            return result
        except Exception as e:
            logger.error(f"Error inserting chunk: {e}")
            raise
    
    def extract_title_from_chunk(self, chunk: Dict[str, Any]) -> str:
        """Extract a title from a chunk of text.

        Args:
            chunk (Dict[str, Any]): Dictionary containing content and metadata for a chunk

        Returns:
            str: Extracted title from the chunk
        """
        # First try to use the header path if available
        if "header_path" in chunk["metadata"] and chunk["metadata"]["header_path"]:
            return chunk["metadata"]["header_path"]
        
        # Then try individual headers from metadata
        metadata = chunk["metadata"]
        for header_level in range(1, 5):
            header_key = f"Header {header_level}"
            if header_key in metadata and metadata[header_key]:
                return metadata[header_key]
        
        # Remove header path from content if present
        content = chunk["content"]
        content_lines = content.split("\n")
        if len(content_lines) > 0 and "[#" in content_lines[0] and " > " in content_lines[0]:
            content = "\n".join(content_lines[1:])
        
        # Try to find headers in remaining content
        header_match = re.search(r'^#+\s+(.+)$', content, re.MULTILINE)
        if header_match:
            return header_match.group(1)
        
        # Final fallback to first line of actual content
        first_line = content.split('\n')[0].strip()
        if len(first_line) > 100:
            return first_line[:97] + "..."
        return first_line
    
    async def process_chunk_with_update(
        self,
        chunk: Dict[str, Any],
        chunk_number: int,
        file_path: str,
        version: int
    ):
        """Process a chunk and update if it exists, otherwise insert."""
        try:
            # Get document URL - only use the URL part, not the version
            documentation_url, _ = self.markdown_converter.convert_path_to_url(
                file_path, 
                chunk["metadata"].get("header_path", "")
            )
            
            # Extract filename for matching
            filename = Path(file_path).name
            version_str = f"{version/10:.1f}"
            
            # Extract title
            title = self.extract_title_from_chunk(chunk)
            
            # Get embedding
            embedding = await self.embedding_service.get_embedding(chunk["content"])
            
            # Prepare metadata
            metadata = {
                "source": "markdown_file",
                "chunk_size": len(chunk["content"]),
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "filename": filename,
                "version_str": version_str,
                **chunk["metadata"]
            }
            
            try:
                # Delete existing records based on metadata
                await self.db_service.delete_document_by_metadata(filename, version_str)
                
                # Prepare record data
                document = {
                    "url": documentation_url,
                    "chunk_number": chunk_number,
                    "title": title,
                    "content": chunk["content"],
                    "metadata": metadata,
                    "embedding": embedding,
                    "version": version
                }
                
                # Insert new record
                result = await self.db_service.insert_document(document)
                
                logger.info(
                    f"Processed chunk {chunk_number} "
                    f"(version {metadata['version_str']}): "
                    f"{title}"
                )
                
                return result
                
            except Exception as e:
                logger.warning(f"Operation failed: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            raise

    async def _delete_existing_record(
        self,
        url: str,
        chunk_number: int,
        version: int
    ) -> None:
        """Delete an existing record if it exists."""
        try:
            await self.db_service.delete_document(url, chunk_number, version)
            await asyncio.sleep(0.5)  # Keep the delay for safety
            logger.debug(f"Deleted existing record for URL: {url}, chunk: {chunk_number}, version: {version}")
        except Exception as e:
            raise Exception(f"Error in delete operation: {e}")

    async def process_file_with_update(self, file_path: str, version: int):
        """Process a markdown file and update existing records if they exist."""
        try:
            logger.info(f"Processing file with update: {file_path}")
            
            # Read and chunk the markdown file
            chunks = self.markdown_converter.chunk_markdown(file_path)
            logger.info(f"Split into {len(chunks)} chunks")
            
            # Process chunks sequentially to avoid race conditions
            for i, chunk in enumerate(chunks):
                await self.process_chunk_with_update(chunk, i, file_path, version)
            
            logger.info(f"Successfully processed {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise