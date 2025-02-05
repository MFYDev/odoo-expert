# src/processing/markdown.py
import re
import os
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Dict, Any
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)
from src.config.settings import settings
from src.utils.logging import logger

class MarkdownConverter:
    def __init__(self):
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]

    def process_directory(self, base_dir: str, output_dir: str = None):
        """Process all RST files in the given directory and its subdirectories.
        
        Args:
            base_dir (str): Source directory containing RST files
            output_dir (str, optional): Target directory for markdown files.
                If not provided, defaults to base_dir/markdown
        """
        base_path = Path(base_dir)
        # If output_dir is not provided, use the default path
        output_path = Path(output_dir if output_dir is not None else base_path / 'markdown')
        versions = settings.odoo_versions_list
        
        for version in versions:
            source_dir = base_path / 'versions' / version / 'content'
            target_dir = output_path / 'versions' / version / 'content'
            
            if not source_dir.exists():
                logger.warning(f"Source directory {source_dir} does not exist")
                continue
                
            # Walk through all files in the source directory
            for rst_file in source_dir.rglob('*.rst'):
                # Calculate the relative path from the source_dir
                rel_path = rst_file.relative_to(source_dir)
                
                # Create the corresponding markdown file path
                md_file = target_dir / rel_path.with_suffix('.md')
                
                # Create target directory if it doesn't exist
                md_file.parent.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Processing: {rst_file} -> {md_file}")
                try:
                    # Read RST content
                    with open(rst_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Convert the content
                    md_content = self.convert_rst_to_markdown(content)
                    
                    # Write to markdown file
                    with open(md_file, 'w', encoding='utf-8') as f:
                        f.write(md_content)
                        
                except Exception as e:
                    logger.error(f"Error processing file {rst_file}: {e}")

    def convert_rst_to_markdown(self, content: str) -> str:
        """Convert RST content to markdown."""
        try:
            # Create a temporary file for the RST content
            with NamedTemporaryFile(mode='w', suffix='.rst', encoding='utf-8', delete=False) as temp_rst:
                temp_rst.write(content)
                temp_rst_path = temp_rst.name
                
            # Create a temporary file for the intermediate markdown
            with NamedTemporaryFile(mode='w', suffix='.md', encoding='utf-8', delete=False) as temp_md:
                temp_md_path = temp_md.name
                
            try:
                # Run pandoc conversion
                subprocess.run(
                    ['pandoc', temp_rst_path, '-f', 'rst', '-t', 'markdown', '-o', temp_md_path], 
                    check=True,
                    capture_output=True
                )
                
                # Read the converted content
                with open(temp_md_path, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                    
                # Clean up the markdown content
                return self.clean_markdown(md_content)
                
            finally:
                # Clean up temporary files
                os.unlink(temp_rst_path)
                os.unlink(temp_md_path)
                    
        except subprocess.CalledProcessError as e:
            logger.error(f"Pandoc conversion failed: {e.stderr.decode()}")
            raise
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            raise
    
    def clean_markdown(self, content: str) -> str:
        """Clean up the markdown content.
        
        Args:
            content (str): Raw markdown content to clean
            
        Returns:
            str: Cleaned markdown content
        """
        # Remove initial metadata before first heading while preserving structure
        lines = content.split('\n')
        first_content_line = 0
        in_metadata = True
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Stop looking for metadata if we hit a heading, table, or other structured content
            if (stripped.startswith('#') or
                stripped.startswith('+--') or
                stripped.startswith('|') or
                (stripped and not stripped == ':' and 
                not any(marker in stripped.lower() for marker in 
                        ['show-content', 'hide-page-toc', 'show-toc', 'nosearch', 'orphan']))):
                in_metadata = False
                first_content_line = i
                break
                
        # Keep content from first non-metadata line onwards
        content = '\n'.join(lines[first_content_line:])
        
        # First fix line breaks (but preserve tables and other formatted content)
        content = self.fix_line_breaks(content)
        
        # Clean up directive blocks
        content = re.sub(r'::: seealso\n(.*?)\n:::', r'::: seealso\n\1\n:::', content, flags=re.DOTALL)
        content = re.sub(r':::: tip\n::: title\nTip\n:::\n\n(.*?)\n::::', r'Tip: \1', content, flags=re.DOTALL)
        content = re.sub(r':::: note\n::: title\nNote\n:::\n\n(.*?)\n::::', r'Note: \1', content, flags=re.DOTALL)
        content = re.sub(r':::: important\n::: title\nImportant\n:::\n\n(.*?)\n::::', r'Important: \1', content, flags=re.DOTALL)
        
        # Clean up all RST-style roles
        content = re.sub(r'\{\.interpreted-text\s+role="[^"]+"\}', '', content, flags=re.DOTALL)
        
        # Convert related content block to a list
        def format_related_content(match):
            items = match.group(1).split()
            formatted_items = "\n".join(f"- {item.strip()}" for item in items if item.strip())
            return f"## Related content:\n\n{formatted_items}"
        
        content = re.sub(
            r'::: \{\.toctree titlesonly=""\}\n(.*?)\n:::',
            format_related_content,
            content,
            flags=re.DOTALL,
        )
        
        # Remove extra blank lines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()

    def fix_line_breaks(self, content: str) -> str:
        """Fix unnecessary line breaks while preserving formatting.
        
        Args:
            content (str): Content to fix line breaks in
            
        Returns:
            str: Content with fixed line breaks
        """
        lines = content.split('\n')
        result = []
        current_line = ''
        in_code_block = False
        in_table = False
        
        def should_preserve_line_break(line):
            return (line.strip().startswith('#') or
                    line.strip().startswith(':::') or
                    line.strip().startswith('- ') or
                    line.strip().startswith('* ') or
                    line.strip().startswith('[') or
                    line.strip().startswith('+') or  # Table markers
                    line.strip().startswith('|') or  # Table content
                    not line.strip())  # Empty lines

        for line in lines:
            stripped_line = line.strip()
            
            # Check for table markers
            if stripped_line.startswith('+') and '-' in stripped_line:
                in_table = True
                result.append(line)
                continue
                
            # If in table, preserve formatting
            if in_table:
                if stripped_line.startswith('+'):  # End of table section
                    in_table = False
                result.append(line)
                continue
            
            # Handle code blocks
            if stripped_line.startswith('```'):
                if current_line:
                    result.append(current_line)
                    current_line = ''
                result.append(line)
                in_code_block = not in_code_block
                continue
            
            # Preserve code block content
            if in_code_block:
                result.append(line)
                continue
            
            # Handle preserved lines
            if should_preserve_line_break(line):
                if current_line:
                    result.append(current_line)
                    current_line = ''
                result.append(line)
                continue
            
            # Handle regular content
            if current_line:
                current_line += ' ' + stripped_line
            else:
                current_line = stripped_line
        
        # Add any remaining content
        if current_line:
            result.append(current_line)
        
        return '\n'.join(result)
    
    def chunk_markdown(self, file_path: str, chunk_size: int = 5000, chunk_overlap: int = 500) -> List[Dict[str, Any]]:
        """Split a markdown file into chunks based on headers and size.
        
        Args:
            file_path (str): Path to the markdown file
            chunk_size (int): Maximum chunk size in characters
            chunk_overlap (int): Overlap between chunks in characters
            
        Returns:
            List[Dict[str, Any]]: List of chunks with content and metadata
        """
        try:
            # Read the markdown file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split by headers first
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=self.headers_to_split_on,
                strip_headers=False
            )
            md_header_splits = markdown_splitter.split_text(text)
            
            # Then split by size if needed
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            final_splits = text_splitter.split_documents(md_header_splits)
            
            # Convert to list of dicts with content and metadata
            chunks = []
            for split in final_splits:
                # Create header path
                header_path = self.create_header_path(split.metadata)
                
                # Combine header path with content
                full_content = f"{header_path}\n{split.page_content}" if header_path else split.page_content
                
                chunks.append({
                    "content": full_content,
                    "metadata": {
                        **split.metadata,
                        "header_path": header_path
                    }
                })
            
            return chunks
        except Exception as e:
            logger.error(f"Error chunking markdown file {file_path}: {e}")
            raise

    def create_header_path(self, metadata: Dict[str, str]) -> str:
        """Create a hierarchical header path from metadata.
        
        Args:
            metadata (Dict[str, str]): Metadata dictionary containing headers
            
        Returns:
            str: String representing the header hierarchy
        """
        headers = []
        for i in range(1, 5):
            key = f"Header {i}"
            if key in metadata and metadata[key]:
                header_level = "#" * i
                headers.append(f"[{header_level}] {metadata[key]}")
        
        return " > ".join(headers) if headers else ""
    
    def convert_path_to_url(self, file_path: str, header_path: str = "") -> tuple[str, int]:
        """Convert a local file path to a full URL for the Odoo documentation and extract version.

        Args:
            file_path (str): Local file path to convert
            header_path (str, optional): Header path for section anchors. Defaults to "".

        Returns:
            tuple[str, int]: Full URL for the documentation page and version number
        """
        # Extract version from path
        version_match = re.search(r'/versions/(\d+\.\d+)/', file_path)
        if not version_match:
            raise ValueError(f"Could not extract version from path: {file_path}")
        
        version_str = version_match.group(1)
        version = int(float(version_str) * 10)  # Convert "16.0" to 160, "17.0" to 170, etc.
        
        # Extract the path after the version number
        path_match = re.search(r'/versions/\d+\.\d+/(.+?)\.md$', file_path)
        if not path_match:
            raise ValueError(f"Could not extract content path from: {file_path}")
        
        content_path = path_match.group(1)
        # Remove 'content/' from the path if it exists
        content_path = re.sub(r'^content/', '', content_path)
        
        base_url = f"https://www.odoo.com/documentation/{version_str}"
        url = f"{base_url}/{content_path}.html"
        
        # Add section anchor if header path is provided
        section_anchor = self.extract_section_anchor(header_path)
        if section_anchor:
            url = f"{url}#{section_anchor}"
        
        return url, version
    
    def extract_section_anchor(self, header_path: str) -> str:
        """Extract the last section from a header path to create an anchor.
        
        Args:
            header_path (str): Full header path (e.g., "[#] Database management > [##] Installation")
            
        Returns:
            str: Section anchor or empty string if no valid section found
        """
        if not header_path:
            return ""
            
        # Get the last section from the header path
        sections = header_path.split(" > ")
        if sections:
            last_section = sections[-1]
            # Remove the header level indicator (e.g., "[##]")
            last_section = re.sub(r'\[#+\]\s*', '', last_section)
            # Clean the section title to create the anchor
            return self.clean_section_name(last_section)
        return ""
    
    def clean_section_name(self, title: str) -> str:
        """Convert a section title to a URL-friendly anchor.
        
        Args:
            title (str): The section title to convert
            
        Returns:
            str: URL-friendly anchor name
            
        Examples:
            "Installation" -> "installation"
            "Invite / remove users" -> "invite-remove-users"
            "Database Management" -> "database-management"
        """
        # Remove markdown header markers and any {#...} custom anchors
        title = re.sub(r'\[#+\]\s*', '', title)
        title = re.sub(r'\{#.*?\}', '', title)
        
        # Remove special characters and extra spaces
        title = re.sub(r'[^a-zA-Z0-9\s-]', '', title)
        
        # Convert to lowercase and replace spaces with dashes
        title = title.lower().strip()
        title = re.sub(r'\s+', '-', title)
        
        return title
    