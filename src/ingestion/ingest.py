"""
Main ingestion script for processing documents into the RAG vector store (Chroma).

Preserves document processing logic (Docling, chunking, embeddings); persistence
is handled by Chroma (see pipeline implementation).
"""

import os
import asyncio
import logging
import glob
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse
from dataclasses import dataclass

import chromadb
from dotenv import load_dotenv

from src.ingestion.chunker import ChunkingConfig, create_chunker, DocumentChunk
from src.ingestion.embedder import create_embedder
from src.projects import resolve_project, PROJECTS
from src.settings import load_settings

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for document ingestion."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 2000
    max_tokens: int = 512


@dataclass
class IngestionResult:
    """Result of document ingestion."""
    document_id: str
    title: str
    chunks_created: int
    processing_time_ms: float
    errors: List[str]


class DocumentIngestionPipeline:
    """Pipeline for ingesting documents into the RAG vector store (Chroma)."""

    def __init__(
        self,
        config: IngestionConfig,
        documents_folder: str = "documents",
        clean_before_ingest: bool = True
    ):
        """
        Initialize ingestion pipeline.

        Args:
            config: Ingestion configuration
            documents_folder: Folder containing documents (subfolders = project keys)
            clean_before_ingest: Whether to clean existing data before ingestion
        """
        self.config = config
        self.documents_folder = documents_folder
        self.clean_before_ingest = clean_before_ingest

        # Load settings
        self.settings = load_settings()

        # Initialize components
        self.chunker_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_chunk_size=config.max_chunk_size,
            max_tokens=config.max_tokens
        )

        self.chunker = create_chunker(self.chunker_config)
        self.embedder = create_embedder()

        self.chroma_client: Optional[Any] = None
        self.chroma_collection: Optional[Any] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize pipeline and Chroma client/collection."""
        if self._initialized:
            return
        logger.info("Inicializando pipeline de ingestão...")
        self.chroma_client = chromadb.PersistentClient(path=self.settings.chroma_path)
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name="rag_chunks",
            metadata={"description": "Chunks de todos os projetos"},
        )
        self._initialized = True
        logger.info("Pipeline de ingestão inicializado")

    async def close(self) -> None:
        """Close pipeline resources."""
        if self._initialized:
            self.chroma_client = None
            self.chroma_collection = None
            self._initialized = False
            logger.info("Pipeline fechado")

    async def _clean_chroma_collection(self) -> None:
        """Clear all chunks from Chroma (drop and recreate collection)."""
        if not self.chroma_collection:
            return
        logger.warning("Limpando collection Chroma rag_chunks...")
        self.chroma_client.delete_collection("rag_chunks")
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name="rag_chunks",
            metadata={"description": "Chunks de todos os projetos"},
        )
        logger.info("Collection rag_chunks limpa e recriada")

    def _document_slug(self, source_path: str) -> str:
        """Slug from document path (e.g. termos_de_uso from master_detox/termos_de_uso.pdf)."""
        name = Path(source_path).stem
        return re.sub(r"[^a-zA-Z0-9_-]", "_", name)

    async def _save_to_chroma(
        self,
        project_id: str,
        project_name: str,
        document_title: str,
        document_path: str,
        embedded_chunks: List[DocumentChunk],
    ) -> None:
        """
        Save chunks to Chroma with deterministic ids and metadata.

        Chroma API: documents (text), embeddings, metadatas. Batch size 100.
        """
        if not embedded_chunks or not self.chroma_collection:
            return
        doc_slug = self._document_slug(document_path)
        batch_size = 100
        for i in range(0, len(embedded_chunks), batch_size):
            batch = embedded_chunks[i : i + batch_size]
            ids = [
                f"{project_id}:{doc_slug}:{chunk.index}"
                for chunk in batch
            ]
            documents = [chunk.content for chunk in batch]
            embeddings = [chunk.embedding for chunk in batch if chunk.embedding is not None]
            if len(embeddings) != len(batch):
                continue
            metadatas = [
                {
                    "project_id": project_id,
                    "project_name": project_name,
                    "document_title": document_title,
                    "document_path": document_path,
                    "chunk_index": chunk.index,
                    "token_count": chunk.token_count or 0,
                }
                for chunk in batch
            ]
            await asyncio.to_thread(
                self.chroma_collection.add,
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        logger.info(f"Inseridos {len(embedded_chunks)} chunks no Chroma")

    def _find_document_files(self) -> List[str]:
        """
        Find all supported document files in the documents folder.

        Returns:
            List of file paths
        """
        if not os.path.exists(self.documents_folder):
            logger.error(f"Pasta de documentos não encontrada: {self.documents_folder}")
            return []

        # Supported file patterns - Docling + text formats + audio
        patterns = [
            "*.md", "*.markdown", "*.txt",  # Text formats
            "*.pdf",  # PDF
            "*.docx", "*.doc",  # Word
            "*.pptx", "*.ppt",  # PowerPoint
            "*.xlsx", "*.xls",  # Excel
            "*.html", "*.htm",  # HTML
            "*.mp3", "*.wav", "*.m4a", "*.flac",  # Audio formats
        ]
        files = []

        for pattern in patterns:
            files.extend(
                glob.glob(
                    os.path.join(self.documents_folder, "**", pattern),
                    recursive=True
                )
            )

        # Only include files under a project subfolder (first segment = project_key in projects.json)
        def under_project(path: str) -> bool:
            rel = os.path.relpath(path, self.documents_folder)
            parts = Path(rel).parts
            if len(parts) < 2:
                return False
            return parts[0] in PROJECTS

        return sorted(f for f in files if under_project(f))

    def _read_document(self, file_path: str) -> tuple[str, Optional[Any]]:
        """
        Read document content from file - supports multiple formats via Docling.

        Args:
            file_path: Path to the document file

        Returns:
            Tuple of (markdown_content, docling_document).
            docling_document is None only for text files.
        """
        file_ext = os.path.splitext(file_path)[1].lower()

        # Audio formats - transcribe with Whisper ASR
        audio_formats = ['.mp3', '.wav', '.m4a', '.flac']
        if file_ext in audio_formats:
            # Returns tuple: (markdown_content, docling_document)
            return self._transcribe_audio(file_path)

        # Docling-supported formats (convert to markdown)
        docling_formats = [
            '.pdf', '.docx', '.doc', '.pptx', '.ppt',
            '.xlsx', '.xls', '.html', '.htm',
            '.md', '.markdown'  # Markdown files for HybridChunker
        ]

        if file_ext in docling_formats:
            try:
                from docling.document_converter import DocumentConverter

                logger.info(
                    f"Convertendo arquivo {file_ext} usando Docling: "
                    f"{os.path.basename(file_path)}"
                )

                converter = DocumentConverter()
                result = converter.convert(file_path)

                # Export to markdown for consistent processing
                markdown_content = result.document.export_to_markdown()
                logger.info(
                    f"Arquivo {os.path.basename(file_path)} convertido com sucesso "
                    f"para markdown"
                )

                # Return both markdown and DoclingDocument for HybridChunker
                return (markdown_content, result.document)

            except Exception as e:
                logger.error(f"Falha ao converter {file_path} com Docling: {e}")
                # Fall back to raw text if Docling fails
                logger.warning(f"Usando extração de texto simples para {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return (f.read(), None)
                except Exception:
                    return (
                        f"[Erro: Não foi possível ler o arquivo {os.path.basename(file_path)}]",
                        None
                    )

        # Text-based formats (read directly)
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return (f.read(), None)
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    return (f.read(), None)

    def _transcribe_audio(self, file_path: str) -> tuple[str, Optional[Any]]:
        """
        Transcribe audio file using Whisper ASR via Docling.

        Args:
            file_path: Path to the audio file

        Returns:
            Tuple of (markdown_content, docling_document)
        """
        try:
            from pathlib import Path
            from docling.document_converter import (
                DocumentConverter,
                AudioFormatOption
            )
            from docling.datamodel.pipeline_options import AsrPipelineOptions
            from docling.datamodel import asr_model_specs
            from docling.datamodel.base_models import InputFormat
            from docling.pipeline.asr_pipeline import AsrPipeline

            # Use Path object - Docling expects this
            audio_path = Path(file_path).resolve()
            logger.info(
                f"Transcrevendo arquivo de áudio usando Whisper Turbo: {audio_path.name}"
            )

            # Verify file exists
            if not audio_path.exists():
                raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")

            # Configure ASR pipeline with Whisper Turbo model
            pipeline_options = AsrPipelineOptions()
            pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

            converter = DocumentConverter(
                format_options={
                    InputFormat.AUDIO: AudioFormatOption(
                        pipeline_cls=AsrPipeline,
                        pipeline_options=pipeline_options,
                    )
                }
            )

            # Transcribe the audio file
            result = converter.convert(audio_path)

            # Export to markdown with timestamps
            markdown_content = result.document.export_to_markdown()
            logger.info(f"Arquivo {os.path.basename(file_path)} transcrito com sucesso")

            # Return both markdown and DoclingDocument for HybridChunker
            return (markdown_content, result.document)

        except Exception as e:
            logger.error(f"Falha ao transcrever {file_path} com Whisper ASR: {e}")
            return (
                f"[Erro: Não foi possível transcrever o arquivo de áudio "
                f"{os.path.basename(file_path)}]",
                None
            )

    def _extract_title(self, content: str, file_path: str) -> str:
        """
        Extract title from document content or filename.

        Args:
            content: Document content
            file_path: Path to the document file

        Returns:
            Document title
        """
        # Try to find markdown title
        lines = content.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()

        # Fallback to filename
        return os.path.splitext(os.path.basename(file_path))[0]

    def _extract_document_metadata(
        self,
        content: str,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Extract metadata from document content.

        Args:
            content: Document content
            file_path: Path to the document file

        Returns:
            Document metadata dictionary
        """
        metadata = {
            "file_path": file_path,
            "file_size": len(content),
            "ingestion_date": datetime.now().isoformat()
        }

        # Try to extract YAML frontmatter
        if content.startswith('---'):
            try:
                import yaml
                end_marker = content.find('\n---\n', 4)
                if end_marker != -1:
                    frontmatter = content[4:end_marker]
                    yaml_metadata = yaml.safe_load(frontmatter)
                    if isinstance(yaml_metadata, dict):
                        metadata.update(yaml_metadata)
            except ImportError:
                logger.warning(
                    "PyYAML não instalado, pulando extração de frontmatter"
                )
            except Exception as e:
                logger.warning(f"Falha ao analisar frontmatter: {e}")

        # Extract some basic metadata from content
        lines = content.split('\n')
        metadata['line_count'] = len(lines)
        metadata['word_count'] = len(content.split())

        return metadata

    async def _ingest_single_document(self, file_path: str) -> IngestionResult:
        """
        Ingest a single document. Project must be resolved before embedding.

        Args:
            file_path: Path to the document file

        Returns:
            Ingestion result
        """
        start_time = datetime.now()

        # Resolve project from path (first segment of path relative to documents_folder)
        document_source = os.path.relpath(file_path, self.documents_folder)
        project_key = Path(document_source).parts[0]
        project = resolve_project(project_key)
        project_id = project["project_id"]
        project_name = project["name"]

        # Read document (returns tuple: content, docling_doc)
        document_content, docling_doc = self._read_document(file_path)
        document_title = self._extract_title(document_content, file_path)

        # Extract metadata from content
        document_metadata = self._extract_document_metadata(
            document_content,
            file_path
        )

        logger.info(f"Processando documento: {document_title} (projeto={project_id})")

        # Chunk the document - pass DoclingDocument for HybridChunker
        chunks = await self.chunker.chunk_document(
            content=document_content,
            title=document_title,
            source=document_source,
            metadata=document_metadata,
            docling_doc=docling_doc  # Pass DoclingDocument for HybridChunker
        )

        if not chunks:
            logger.warning(f"Nenhum chunk criado para {document_title}")
            return IngestionResult(
                document_id="",
                title=document_title,
                chunks_created=0,
                processing_time_ms=(
                    datetime.now() - start_time
                ).total_seconds() * 1000,
                errors=["Nenhum chunk criado"]
            )

        logger.info(f"Criados {len(chunks)} chunks")

        # Generate embeddings
        embedded_chunks = await self.embedder.embed_chunks(chunks)
        logger.info(f"Embeddings gerados para {len(embedded_chunks)} chunks")

        # Save to Chroma (deterministic ids: project_id:doc_slug:chunk_index)
        await self._save_to_chroma(
            project_id=project_id,
            project_name=project_name,
            document_title=document_title,
            document_path=document_source,
            embedded_chunks=embedded_chunks,
        )

        processing_time = (
            datetime.now() - start_time
        ).total_seconds() * 1000

        return IngestionResult(
            document_id=f"{project_id}:{self._document_slug(document_source)}",
            title=document_title,
            chunks_created=len(chunks),
            processing_time_ms=processing_time,
            errors=[]
        )

    async def ingest_documents(
        self,
        progress_callback: Optional[callable] = None
    ) -> List[IngestionResult]:
        """
        Ingest all documents from the documents folder.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            List of ingestion results
        """
        if not self._initialized:
            await self.initialize()

        if self.clean_before_ingest:
            await self._clean_chroma_collection()

        # Find all supported document files
        document_files = self._find_document_files()

        if not document_files:
            logger.warning(
                f"Nenhum arquivo de documento suportado encontrado em {self.documents_folder}"
            )
            return []

        logger.info(f"Encontrados {len(document_files)} arquivos de documento para processar")

        results = []

        for i, file_path in enumerate(document_files):
            try:
                logger.info(
                    f"Processando arquivo {i+1}/{len(document_files)}: {file_path}"
                )

                result = await self._ingest_single_document(file_path)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(document_files))

            except Exception as e:
                logger.exception(f"Falha ao processar {file_path}: {e}")
                results.append(IngestionResult(
                    document_id="",
                    title=os.path.basename(file_path),
                    chunks_created=0,
                    processing_time_ms=0,
                    errors=[str(e)]
                ))

        # Log summary
        total_chunks = sum(r.chunks_created for r in results)
        total_errors = sum(len(r.errors) for r in results)

        logger.info(
            f"Ingestão concluída: {len(results)} documentos, "
            f"{total_chunks} chunks, {total_errors} erros"
        )

        return results


async def main() -> None:
    """Main function for running ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into RAG vector store (Chroma)"
    )
    parser.add_argument(
        "--documents", "-d",
        default="documents",
        help="Documents folder path"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip cleaning existing data before ingestion"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for splitting documents"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap size"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per chunk for embeddings"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create ingestion configuration
    config = IngestionConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_chunk_size=args.chunk_size * 2,
        max_tokens=args.max_tokens
    )

    # Create and run pipeline - clean by default unless --no-clean is specified
    pipeline = DocumentIngestionPipeline(
        config=config,
        documents_folder=args.documents,
        clean_before_ingest=not args.no_clean  # Clean by default
    )

    def progress_callback(current: int, total: int) -> None:
        print(f"Progresso: {current}/{total} documentos processados")

    try:
        start_time = datetime.now()

        results = await pipeline.ingest_documents(progress_callback)

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Print summary
        print("\n" + "="*50)
        print("RESUMO DA INGESTÃO")
        print("="*50)
        print(f"Documentos processados: {len(results)}")
        print(f"Total de chunks criados: {sum(r.chunks_created for r in results)}")
        print(f"Total de erros: {sum(len(r.errors) for r in results)}")
        print(f"Tempo total de processamento: {total_time:.2f} segundos")
        print()

        # Print individual results
        for result in results:
            status = "[OK]" if not result.errors else "[FAILED]"
            print(f"{status} {result.title}: {result.chunks_created} chunks")

            if result.errors:
                for error in result.errors:
                    print(f"  Erro: {error}")

        # Print next steps
        print("\n" + "="*50)
        print("PRÓXIMOS PASSOS")
        print("="*50)
        print("Chunks foram processados. Execute o CLI para consultar a base.")

    except KeyboardInterrupt:
        print("\nIngestão interrompida pelo usuário")
    except Exception as e:
        logger.exception(f"Ingestão falhou: {e}")
        raise
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
