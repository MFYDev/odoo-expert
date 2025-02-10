import uvicorn
import argparse
import asyncio
import subprocess
from pathlib import Path
from src.api.app import app
from src.processing.document_processor import DocumentProcessor
from src.processing.markdown_converter import MarkdownConverter
from src.processing.file_update_handler import FileUpdateHandler
from src.core.services.embedding import EmbeddingService
from src.core.services.db_service import DatabaseService
from src.config.settings import settings
from src.config.model_provider import ModelProvider
from src.utils.logging import logger

async def process_documents(base_dir: str):
    """Process markdown documents to embeddings"""
    model_provider = ModelProvider()
    db_service = DatabaseService()
    embedding_service = EmbeddingService(model_provider.embed_model)
    processor = DocumentProcessor(db_service, embedding_service)
    await processor.process_directory(base_dir)

async def process_raw_data(raw_dir: str, output_dir: str, process_docs: bool = False):
    """Process raw RST files to markdown and optionally process documents
    
    Args:
        raw_dir (str): Directory containing raw RST files
        output_dir (str): Output directory for markdown files
        process_docs (bool): Whether to process documents after conversion
    """
    # Step 1: Convert RST to Markdown
    converter = MarkdownConverter()
    converter.process_directory(raw_dir, output_dir)
    
    # Step 2: Process markdown files to documents (optional)
    if process_docs:
        await process_documents(output_dir)

async def check_updates(raw_dir: str, markdown_dir: str):
    """Check for updates in raw data and process changed files.

    Args:
        raw_dir (str): Raw data directory
        markdown_dir (str): Markdown data directory

    Returns:
        Added, modified, removed files
    """
    model_provider = ModelProvider()
    
    db_service = DatabaseService()
    embedding_service = EmbeddingService(model_provider.embed_model)
    document_processor = DocumentProcessor(db_service, embedding_service)
    markdown_converter = MarkdownConverter()
    
    update_handler = FileUpdateHandler(
        document_processor=document_processor,
        markdown_converter=markdown_converter
    )
    
    added, modified, removed = await update_handler.check_and_process_updates(
        raw_dir=raw_dir,
        markdown_dir=markdown_dir
    )
    
    logger.info(f"Added files: {len(added)}")
    logger.info(f"Modified files: {len(modified)}")
    logger.info(f"Removed files: {len(removed)}")
    
    return added, modified, removed

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Odoo Documentation Assistant')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Server command
    server_parser = subparsers.add_parser('serve', help='Run the server')
    server_parser.add_argument('--mode', choices=['api', 'ui'], required=True,
                             help='Run mode: api for FastAPI server or ui for Streamlit interface')
    server_parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
    server_parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    
    # Process commands
    process_raw_parser = subparsers.add_parser('process-raw', help='Process raw RST files')
    process_raw_parser.add_argument('--process-docs', action='store_true',
                                  help='Process documents after conversion')
    
    process_docs_parser = subparsers.add_parser('process-docs', help='Process markdown documents')

    # Add check-updates command
    check_updates_parser = subparsers.add_parser('check-updates', 
                                                help='Check and process updated files')
    
    args = parser.parse_args()
    
    if args.command == 'serve':
        if args.mode == 'api':
            uvicorn.run(app, host=args.host, port=args.port)
        elif args.mode == 'ui':
            subprocess.run(["streamlit", "run", "src/ui/streamlit_app.py"])
    elif args.command == 'process-raw':
        asyncio.run(process_raw_data(settings.RAW_DATA_DIR, settings.MARKDOWN_DATA_DIR, args.process_docs))
    elif args.command == 'process-docs':
        async def run_sequential():
            # First run check-updates to generate file cache
            await check_updates(settings.RAW_DATA_DIR, settings.MARKDOWN_DATA_DIR)
            # Then process the documents
            await process_documents(settings.MARKDOWN_DATA_DIR)
        
        asyncio.run(run_sequential())
    elif args.command == 'check-updates':
        asyncio.run(check_updates(settings.RAW_DATA_DIR, settings.MARKDOWN_DATA_DIR))
    else:
        parser.print_help()
