import os
import hashlib
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Tuple
import json
from src.utils.logging import logger
from src.processing.markdown_converter import MarkdownConverter
from src.processing.document_processor import DocumentProcessor
from src.config.settings import settings

class FileUpdateHandler:
    def __init__(
        self,
        document_processor: DocumentProcessor,
        markdown_converter: MarkdownConverter,
        cache_file: str = None
    ):
        # Use a persistent location for the cache file
        if cache_file is None:
            # Store in the project root directory
            project_root = Path(__file__).parent.parent.parent
            self.cache_file = str(project_root / '.file_cache.json')
        else:
            self.cache_file = cache_file
            
        self.document_processor = document_processor
        self.markdown_converter = markdown_converter
        self.file_cache = self._load_cache()
        logger.info(f"Using cache file: {self.cache_file}")
        logger.info(f"Current cache has {len(self.file_cache)} files")

    def _load_cache(self) -> Dict[str, str]:
        """Load the file cache from disk."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                    logger.info(f"Loaded existing cache with {len(cache)} entries")
                    return cache
            logger.info("No existing cache found")
            return {}
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return {}

    def _save_cache(self):
        """Save the file cache to disk."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.file_cache, f)
            logger.info(f"Saved cache with {len(self.file_cache)} entries")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def _get_file_hash(self, filepath: str) -> str:
        """Calculate MD5 hash of a file."""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {filepath}: {e}")
            return ""

    def _get_version_from_path(self, filepath: str) -> int:
        """Extract version number from file path."""
        path = Path(filepath)
        version_str = path.parts[path.parts.index('versions') + 1]
        return int(float(version_str) * 10)

    async def check_and_process_updates(
        self,
        raw_dir: str,
        markdown_dir: str
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """Check for file updates and process changed files."""
        current_files = {}
        added_files = set()
        modified_files = set()
        removed_files = set()
        total_files = 0
        unchanged_files = 0
        processed_successfully = True  # Track if all processing succeeded

        # Scan current files
        logger.info("Starting file scan...")
        for version in settings.odoo_versions_list:
            version_path = Path(raw_dir) / 'versions' / version / 'content'
            if not version_path.exists():
                continue

            for rst_file in version_path.rglob('*.rst'):
                total_files += 1
                file_path = str(rst_file)
                current_hash = self._get_file_hash(file_path)
                current_files[file_path] = current_hash

                # Only track changes if we have an existing cache
                if self.file_cache:
                    if file_path not in self.file_cache:
                        logger.info(f"New file detected: {file_path}")
                        added_files.add(file_path)
                    elif self.file_cache[file_path] != current_hash:
                        logger.info(f"Modified file detected: {file_path}")
                        modified_files.add(file_path)
                    else:
                        unchanged_files += 1

        # Only check for removed files if we have an existing cache
        if self.file_cache:
            removed_files = set(self.file_cache.keys()) - set(current_files.keys())
            for file_path in removed_files:
                logger.info(f"Removed file detected: {file_path}")

        # Log summary
        logger.info(f"Scan complete:")
        logger.info(f"Total files scanned: {total_files}")

        if not self.file_cache:
            logger.info("Creating initial cache without processing files")
            self.file_cache = current_files
            self._save_cache()
            return set(), set(), set()

        logger.info(f"Files unchanged: {unchanged_files}")
        logger.info(f"New files: {len(added_files)}")
        logger.info(f"Modified files: {len(modified_files)}")
        logger.info(f"Removed files: {len(removed_files)}")

        # Store the original cache in case we need to rollback
        original_cache = self.file_cache.copy()

        # Process changes only if there are any
        files_to_process = added_files | modified_files
        if not files_to_process:
            logger.info("No files need to be updated")
        else:
            logger.info(f"Processing {len(files_to_process)} files...")
            for idx, file_path in enumerate(files_to_process, 1):
                try:
                    logger.info(f"Processing file {idx}/{len(files_to_process)}: {file_path}")
                    
                    # Convert RST to markdown
                    version = self._get_version_from_path(file_path)
                    rel_path = Path(file_path).relative_to(Path(raw_dir) / 'versions' / f"{version/10:.1f}" / 'content')
                    md_path = Path(markdown_dir) / 'versions' / f"{version/10:.1f}" / 'content' / rel_path.with_suffix('.md')
                    
                    # Ensure directory exists
                    md_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Convert content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    md_content = self.markdown_converter.convert_rst_to_markdown(content)
                    
                    # Write markdown file
                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write(md_content)
                    
                    # Process markdown for database
                    await self.document_processor.process_file_with_update(str(md_path), version)
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    processed_successfully = False
                    # Restore original cache
                    self.file_cache = original_cache
                    self._save_cache()
                    logger.info("Restored original cache due to processing error")
                    break

        # Only update cache if all processing was successful
        if processed_successfully:
            self.file_cache = current_files
            self._save_cache()
            logger.info("Cache updated successfully")
        else:
            logger.warning("Cache not updated due to processing errors")

        return added_files, modified_files, removed_files