#!/usr/bin/env python3
"""
Semantic Man-Page Agent - Production Version
Semantic search over Linux man pages with reliable scoring and UX
"""

import os
import sys
import re
import gzip
import hashlib
import subprocess
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import argparse
import logging
import tempfile
from functools import lru_cache
import threading
import time
import platform

# Version
__version__ = "0.1.0"

# Third-party imports
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
    from rich.panel import Panel
    from rich import box
    from rich.text import Text
    from rich.markdown import Markdown
    try:
        import readline
    except ImportError:
        readline = None
except ImportError as e:
    print(f"❌ Missing required dependency: {e}", file=sys.stderr)
    print("\n📦 To install dependencies:", file=sys.stderr)
    print("  pip install sentence-transformers chromadb numpy rich", file=sys.stderr)
    print("\n💡 Or create a virtual environment:", file=sys.stderr)
    print("  python3 -m venv venv", file=sys.stderr)
    print("  source venv/bin/activate", file=sys.stderr)
    print("  pip install sentence-transformers chromadb numpy rich", file=sys.stderr)
    sys.exit(1)

# Setup logging
logger = logging.getLogger(__name__)

# Rich console
console = Console()

# ============================================================================
# ERROR CLASSES
# ============================================================================

class SemanticManError(Exception):
    """Base exception for Semantic Man-Page Agent."""
    pass

class DependencyError(SemanticManError):
    """Missing dependency."""
    pass

class IndexNotFoundError(SemanticManError):
    """Index not found."""
    pass

class ModelLoadError(SemanticManError):
    """Failed to load model."""
    pass

class ParseError(SemanticManError):
    """Parsing failed."""
    pass

class CacheCorruptedError(SemanticManError):
    """Cache is corrupted."""
    pass

# Exit codes
EXIT_SUCCESS = 0
EXIT_DEPENDENCY_ERROR = 1
EXIT_INDEX_NOT_FOUND = 2
EXIT_MODEL_LOAD_ERROR = 3
EXIT_PARSE_ERROR = 4
EXIT_CACHE_ERROR = 5
EXIT_INTERRUPTED = 130

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration for Semantic Man-Page Agent."""
    # Paths
    man_dir: Path = Path("/usr/share/man")
    cache_dir: Path = Path.home() / ".cache" / "semantic-man"
    chroma_dir: Path = Path.home() / ".cache" / "semantic-man" / "chroma"
    log_dir: Path = Path.home() / ".cache" / "semantic-man" / "logs"
    
    # Model
    dense_model: str = "all-MiniLM-L6-v2"
    
    # Search
    default_top_k: int = 10
    max_top_k: int = 30
    retrieval_k: int = 30
    
    # Sections
    sections: List[int] = field(default_factory=lambda: [1, 8])
    
    # Weights
    section_weights: Dict[str, float] = field(default_factory=lambda: {
        'name': 1.0,
        'synopsis': 1.0,
        'description': 1.0,
        'title': 0.8,
        'options': 0.7,
        'examples': 0.7,
        'see_also': 0.5,
    })
    
    # Performance
    embed_batch_size: int = 32
    use_gpu: bool = False
    parse_timeout: int = 10
    
    # Indexing
    max_parse_failures: int = 100  # Abort if too many failures
    
    def ensure_dirs(self):
        """Create necessary directories."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

# Global config
CONFIG = Config()

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ManPage:
    """Represents a man page."""
    name: str
    section: int
    title: str
    name_description: str
    synopsis: str
    description: str
    options: str
    examples: str
    see_also: str
    file_path: Path
    full_text: str
    
    @property
    def doc_id(self) -> str:
        return f"{self.name}.{self.section}"

@dataclass
class Chunk:
    """A chunk for embedding."""
    doc_id: str
    chunk_type: str
    content: str
    weight: float
    metadata: Dict[str, Any]

@dataclass
class SearchResult:
    """Search result with normalized score."""
    command: str
    section: int
    description: str
    score: float  # 0-1 normalized score
    chunk_type: str
    content: str
    
    def format_score(self) -> str:
        """Format score as percentage."""
        return f"{self.score * 100:.0f}%"

@dataclass
class IndexStats:
    """Index statistics."""
    total_chunks: int
    total_pages: int
    parsed_pages: int
    failed_pages: int
    index_size_mb: float
    model_name: str
    indexed_at: datetime
    is_complete: bool

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(debug: bool = False, log_file: Optional[Path] = None):
    """Setup logging with file output."""
    log_level = logging.DEBUG if debug else logging.INFO
    
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level if debug else logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    handlers.append(console_handler)
    
    # File handler
    if log_file is None:
        log_file = CONFIG.log_dir / f"semantic-man-{datetime.now().strftime('%Y%m%d')}.log"
    CONFIG.log_dir.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    handlers.append(file_handler)
    
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        handlers=handlers
    )

# ============================================================================
# HEALTH CHECK
# ============================================================================

def doctor() -> Dict[str, Any]:
    """Run health check and return results."""
    results = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'ok': True,
        'issues': [],
        'info': []
    }
    
    # Check Python version
    if sys.version_info < (3, 8):
        results['ok'] = False
        results['issues'].append(f"Python 3.8+ required (found {platform.python_version()})")
    
    # Check dependencies
    deps = ['sentence_transformers', 'chromadb', 'numpy', 'rich']
    for dep in deps:
        try:
            __import__(dep)
        except ImportError:
            results['ok'] = False
            results['issues'].append(f"Missing dependency: {dep}")
    
    # Check directories
    try:
        CONFIG.ensure_dirs()
        # Test write permission
        test_file = CONFIG.cache_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        results['info'].append(f"Cache directory writable: {CONFIG.cache_dir}")
    except Exception as e:
        results['ok'] = False
        results['issues'].append(f"Cache directory not writable: {e}")
    
    # Check man pages
    man_dir = Path("/usr/share/man")
    if not man_dir.exists():
        results['ok'] = False
        results['issues'].append(f"Man directory not found: {man_dir}")
    else:
        results['info'].append(f"Man directory found: {man_dir}")
    
    # Check groff
    groff = shutil.which('groff')
    if groff:
        results['info'].append(f"groff found: {groff}")
    else:
        results['info'].append("groff not found (will use fallback parsing)")
    
    # Check index
    store = NormalizedVectorStore(CONFIG)
    count = store.count()
    if count > 0:
        results['info'].append(f"Index found: {count} chunks")
    else:
        results['info'].append("No index found. Run 'semantic-man index'")
    
    return results

def cmd_doctor(args):
    """Run health check."""
    results = doctor()
    
    console.print(Panel.fit(
        "[bold cyan]🔬 Semantic Man-Page Agent Health Check[/bold cyan]",
        border_style="cyan"
    ))
    
    if results['ok']:
        console.print("[bold green]✓ System is healthy[/bold green]")
    else:
        console.print("[bold red]✗ Issues detected[/bold red]")
    
    console.print()
    
    if results['issues']:
        console.print("[bold red]Issues:[/bold red]")
        for issue in results['issues']:
            console.print(f"  [red]•[/red] {issue}")
        console.print()
    
    if results['info']:
        console.print("[bold blue]Info:[/bold blue]")
        for info in results['info']:
            console.print(f"  [blue]•[/blue] {info}")
        console.print()
    
    sys.exit(0 if results['ok'] else 1)

# ============================================================================
# ROBUST MAN PAGE PARSER
# ============================================================================

class RobustManPageParser:
    """Robust parser with multiple fallbacks."""
    
    @staticmethod
    def render_man_page(file_path: Path) -> Optional[str]:
        """Render man page using multiple methods."""
        methods = [
            RobustManPageParser._render_with_man,
            RobustManPageParser._render_with_col,
            RobustManPageParser._render_raw,
        ]
        
        for method in methods:
            try:
                result = method(file_path)
                if result and len(result.strip()) > 50:
                    return result
            except Exception:
                continue
        return None
    
    @staticmethod
    def _render_with_man(file_path: Path) -> Optional[str]:
        """Use man command."""
        try:
            name = file_path.stem
            if file_path.suffix == '.gz':
                name = file_path.stem
            name = re.sub(r'\.\d+$', '', name)
            
            cmd = ['man', '--pager=cat', name]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout:
                return result.stdout
        except:
            pass
        return None
    
    @staticmethod
    def _render_with_col(file_path: Path) -> Optional[str]:
        """Use col -b."""
        try:
            name = file_path.stem
            name = re.sub(r'\.\d+$', '', name)
            cmd = f"man {name} 2>/dev/null | col -b"
            result = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout:
                return result.stdout
        except:
            pass
        return None
    
    @staticmethod
    def _render_raw(file_path: Path) -> Optional[str]:
        """Read raw content."""
        try:
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            
            # Clean groff markup
            content = re.sub(r'\\f[BI]', '', content)
            content = re.sub(r'\\[()]', '', content)
            content = re.sub(r'\.[A-Z][a-z]+.*$', '', content, flags=re.MULTILINE)
            content = re.sub(r'^\.\s*$', '', content, flags=re.MULTILINE)
            content = re.sub(r'\s+', ' ', content)
            return content
        except:
            pass
        return None
    
    @staticmethod
    def extract_sections(text: str) -> Dict[str, str]:
        """Extract sections from rendered text."""
        sections = {
            'NAME': '',
            'SYNOPSIS': '',
            'DESCRIPTION': '',
            'OPTIONS': '',
            'EXAMPLES': '',
            'SEE ALSO': '',
        }
        
        lines = text.split('\n')
        current_section = None
        section_content = []
        
        section_patterns = {
            'NAME': r'^NAME$',
            'SYNOPSIS': r'^SYNOPSIS$',
            'DESCRIPTION': r'^DESCRIPTION$',
            'OPTIONS': r'^OPTIONS$',
            'EXAMPLES': r'^EXAMPLES$',
            'SEE ALSO': r'^SEE ALSO$',
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            is_section = False
            section_name = None
            
            for name, pattern in section_patterns.items():
                if re.match(pattern, line, re.IGNORECASE):
                    is_section = True
                    section_name = name
                    break
            
            if is_section and section_name in sections:
                if current_section and section_content:
                    sections[current_section] = '\n'.join(section_content).strip()
                    section_content = []
                current_section = section_name
                continue
            
            if current_section and line:
                if not re.match(r'^\s*Page \d+', line) and not re.match(r'^User Commands', line):
                    section_content.append(line)
        
        if current_section and section_content:
            sections[current_section] = '\n'.join(section_content).strip()
        
        return sections
    
    @staticmethod
    def parse_name_section(name_text: str) -> Tuple[str, str]:
        """Parse NAME section."""
        if not name_text:
            return '', ''
        
        match = re.match(r'^([^\-]+)\s*[-–]\s*(.+)$', name_text, re.DOTALL)
        if match:
            names = match.group(1).strip()
            description = match.group(2).strip()
            command_name = names.split(',')[0].strip()
            return command_name, description
        
        lines = name_text.split('\n')
        if lines:
            return lines[0].strip(), ''
        
        return '', ''
    
    @staticmethod
    def parse_man_page(file_path: Path) -> Optional[ManPage]:
        """Parse a man page with robust fallbacks."""
        try:
            rendered = RobustManPageParser.render_man_page(file_path)
            if not rendered:
                return None
            
            sections = RobustManPageParser.extract_sections(rendered)
            
            name_text = sections.get('NAME', '')
            command_name, description = RobustManPageParser.parse_name_section(name_text)
            
            if not command_name:
                command_name = file_path.stem
                if file_path.suffix == '.gz':
                    command_name = file_path.stem
                command_name = re.sub(r'\.\d+$', '', command_name)
            
            section = 1
            section_match = re.search(r'\.(\d+)$', str(file_path))
            if section_match:
                section = int(section_match.group(1))
            
            return ManPage(
                name=command_name,
                section=section,
                title=command_name,
                name_description=description or command_name,
                synopsis=sections.get('SYNOPSIS', ''),
                description=sections.get('DESCRIPTION', ''),
                options=sections.get('OPTIONS', ''),
                examples=sections.get('EXAMPLES', ''),
                see_also=sections.get('SEE ALSO', ''),
                file_path=file_path,
                full_text=rendered
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            return None

# ============================================================================
# SMART CHUNKER
# ============================================================================

class SmartChunker:
    """Smart chunking with section-based splitting."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def chunk_man_page(self, man_page: ManPage) -> List[Chunk]:
        """Create intelligent chunks from a man page."""
        chunks = []
        weights = self.config.section_weights
        
        # NAME chunk
        if man_page.name_description:
            content = f"{man_page.name} - {man_page.name_description}"
            chunks.append(Chunk(
                doc_id=man_page.doc_id,
                chunk_type='name',
                content=content,
                weight=weights.get('name', 1.0),
                metadata={'command': man_page.name, 'section': man_page.section, 'chunk_type': 'name'}
            ))
        
        # Title chunk
        if man_page.name:
            chunks.append(Chunk(
                doc_id=man_page.doc_id,
                chunk_type='title',
                content=f"{man_page.name} command",
                weight=weights.get('title', 0.8),
                metadata={'command': man_page.name, 'section': man_page.section, 'chunk_type': 'title'}
            ))
        
        # SYNOPSIS
        if man_page.synopsis:
            chunks.append(Chunk(
                doc_id=man_page.doc_id,
                chunk_type='synopsis',
                content=man_page.synopsis[:600],
                weight=weights.get('synopsis', 1.0),
                metadata={'command': man_page.name, 'section': man_page.section, 'chunk_type': 'synopsis'}
            ))
        
        # DESCRIPTION
        if man_page.description:
            descs = SmartChunker._split_text(man_page.description, 500)
            for i, desc in enumerate(descs):
                chunks.append(Chunk(
                    doc_id=man_page.doc_id,
                    chunk_type='description',
                    content=desc,
                    weight=weights.get('description', 1.0),
                    metadata={'command': man_page.name, 'section': man_page.section, 'chunk_type': 'description', 'part': i + 1}
                ))
        
        return chunks
    
    @staticmethod
    def _split_text(text: str, max_len: int = 500) -> List[str]:
        """Split text into chunks."""
        if len(text) <= max_len:
            return [text]
        
        sentences = text.split('. ')
        chunks = []
        current = []
        current_len = 0
        
        for sentence in sentences:
            sentence_len = len(sentence) + 2
            if current_len + sentence_len > max_len and current:
                chunks.append('. '.join(current) + '.')
                current = []
                current_len = 0
            current.append(sentence)
            current_len += sentence_len
        
        if current:
            chunks.append('. '.join(current) + '.')
        
        return chunks

# ============================================================================
# EFFICIENT EMBEDDER
# ============================================================================

class EfficientEmbedder:
    """Efficient embedder with model caching."""
    
    _dense_model = None
    _model_lock = threading.Lock()
    
    def __init__(self, config: Config):
        self.config = config
        self._load_models()
    
    def _load_models(self):
        """Load models with caching."""
        with self._model_lock:
            if EfficientEmbedder._dense_model is None:
                try:
                    device = 'cuda' if self.config.use_gpu else 'cpu'
                    console.print("[dim]Loading embedding model (first time may download ~80MB)...[/dim]")
                    with console.status("[bold green]Loading embedding model..."):
                        EfficientEmbedder._dense_model = SentenceTransformer(
                            self.config.dense_model,
                            device=device
                        )
                    logger.info(f"Loaded model: {self.config.dense_model}")
                except Exception as e:
                    raise ModelLoadError(f"Failed to load model: {e}")
    
    @staticmethod
    def get_dense_model():
        return EfficientEmbedder._dense_model
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings in batches."""
        model = self.get_dense_model()
        if not model:
            raise RuntimeError("Model not loaded")
        
        try:
            embeddings = model.encode(
                texts,
                batch_size=self.config.embed_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        return self.embed_texts([text])[0]

# ============================================================================
# NORMALIZED VECTOR STORE
# ============================================================================

class NormalizedVectorStore:
    """Vector store with proper score normalization."""
    
    def __init__(self, config: Config):
        self.config = config
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB."""
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.config.chroma_dir),
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
            
            collection_name = "man_pages_production"
            existing = self.client.list_collections()
            
            if collection_name in [c.name for c in existing]:
                self.collection = self.client.get_collection(collection_name)
                logger.info(f"Loaded collection: {self.collection.count()} chunks")
            else:
                self.collection = self.client.create_collection(collection_name)
                logger.info("Created new collection")
                
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")
            raise CacheCorruptedError(f"Failed to initialize vector store: {e}")
    
    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray, progress_callback=None):
        """Add chunks in batches with progress."""
        if not chunks:
            return
        
        BATCH_SIZE = 2000
        total = len(chunks)
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in range(0, total, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, total)
            batch = chunks[i:batch_end]
            batch_embs = embeddings[i:batch_end]
            
            ids = []
            documents = []
            metadatas = []
            
            for idx, chunk in enumerate(batch):
                chunk_id = hashlib.md5(
                    f"{chunk.doc_id}_{chunk.chunk_type}_{i+idx}".encode()
                ).hexdigest()[:16]
                
                ids.append(chunk_id)
                documents.append(chunk.content)
                metadatas.append({
                    'doc_id': chunk.doc_id,
                    'command': chunk.metadata.get('command', ''),
                    'section': chunk.metadata.get('section', 1),
                    'chunk_type': chunk.chunk_type,
                    'weight': chunk.weight
                })
            
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=batch_embs.tolist(),
                metadatas=metadatas
            )
            
            if progress_callback:
                progress_callback(i // BATCH_SIZE + 1, total_batches)
            
            logger.info(f"Added batch {i//BATCH_SIZE + 1}/{total_batches}")
    
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Search with proper distance normalization."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted = []
            if results['documents'] and results['documents'][0]:
                distances = results['distances'][0]
                
                # Convert distances to similarity scores (0-1)
                similarities = [max(0, 1 - d) for d in distances]
                
                # Normalize to 0-1 range
                min_sim = min(similarities) if similarities else 0
                max_sim = max(similarities) if similarities else 1
                range_sim = max_sim - min_sim if max_sim > min_sim else 1
                
                for i in range(len(results['documents'][0])):
                    raw_score = similarities[i]
                    normalized_score = (raw_score - min_sim) / range_sim if range_sim > 0 else 0.5
                    
                    # Apply chunk weight boost (0.8-1.2)
                    weight = results['metadatas'][0][i].get('weight', 1.0)
                    final_score = min(1.0, normalized_score * (0.8 + 0.2 * weight))
                    
                    formatted.append({
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': final_score,
                        'raw_distance': distances[i]
                    })
            
            formatted.sort(key=lambda x: x['score'], reverse=True)
            return formatted
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def count(self) -> int:
        return self.collection.count() if self.collection else 0
    
    def clear(self):
        if self.collection:
            self.client.delete_collection(self.collection.name)
            self._initialize()
    
    def get_stats(self) -> Dict:
        """Get detailed stats."""
        return {
            'total_chunks': self.count(),
            'collection_name': self.collection.name if self.collection else None
        }

# ============================================================================
# MAIN INDEXER WITH PROGRESS
# ============================================================================

class MainIndexer:
    """Main indexer with robust parsing and progress tracking."""
    
    def __init__(self, config: Config):
        self.config = config
        self.parser = RobustManPageParser()
        self.chunker = SmartChunker(config)
        self.embedder = EfficientEmbedder(config)
        self.store = NormalizedVectorStore(config)
        self.start_time = None
        self.stats = {
            'total_pages': 0,
            'parsed_pages': 0,
            'failed_pages': 0,
            'total_chunks': 0,
            'failed_files': []
        }
    
    def index_all(self, force_rebuild: bool = False, resume: bool = False):
        """Index all man pages with progress tracking."""
        self.start_time = time.time()
        
        if force_rebuild:
            logger.info("Force rebuild: clearing data")
            self.store.clear()
        
        existing = self.store.count()
        if existing > 0 and not resume:
            console.print(f"[dim]Found existing index with {existing} chunks[/dim]")
            response = input("Re-index? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping indexing")
                return
        
        # Find man pages
        console.print("[bold blue]📚 Finding man pages...[/bold blue]")
        man_files = self._find_man_pages()
        
        if not man_files:
            console.print("[bold red]❌ No man pages found![/bold red]")
            raise ParseError("No man pages found")
        
        self.stats['total_pages'] = len(man_files)
        console.print(f"[bold green]✓ Found {len(man_files)} man pages[/bold green]")
        console.print(f"[dim]Parsing sections: {self.config.sections}[/dim]")
        console.print()
        
        # Parse and chunk
        all_chunks = []
        
        console.print("[bold blue]🔍 Parsing man pages...[/bold blue]")
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Parsing...", total=len(man_files))
            
            for file_path in man_files:
                man_page = self.parser.parse_man_page(file_path)
                if man_page:
                    chunks = self.chunker.chunk_man_page(man_page)
                    all_chunks.extend(chunks)
                    self.stats['parsed_pages'] += 1
                else:
                    self.stats['failed_pages'] += 1
                    self.stats['failed_files'].append(str(file_path))
                    
                    # Check failure threshold
                    if self.stats['failed_pages'] > self.config.max_parse_failures:
                        console.print(f"[bold red]❌ Too many parse failures ({self.stats['failed_pages']})[/bold red]")
                        console.print("[dim]Check logs for details[/dim]")
                        raise ParseError(f"Failed to parse {self.stats['failed_pages']} pages")
                
                progress.update(task, advance=1)
        
        if self.stats['failed_pages'] > 0:
            console.print(f"[dim]⚠️ Failed to parse {self.stats['failed_pages']} pages ({self.stats['failed_pages']/len(man_files)*100:.1f}%)[/dim]")
        
        if not all_chunks:
            console.print("[bold red]❌ No chunks created![/bold red]")
            raise ParseError("No chunks created")
        
        self.stats['total_chunks'] = len(all_chunks)
        console.print(f"[bold green]✓ Created {len(all_chunks)} chunks[/bold green]")
        console.print()
        
        # Generate embeddings
        console.print("[bold blue]🧠 Generating embeddings...[/bold blue]")
        texts = [chunk.content for chunk in all_chunks]
        
        all_embeddings = []
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Embedding...", total=len(texts))
            
            for i in range(0, len(texts), 64):
                batch = texts[i:i+64]
                embs = self.embedder.embed_texts(batch)
                all_embeddings.append(embs)
                progress.update(task, advance=len(batch))
        
        embeddings = np.vstack(all_embeddings)
        console.print(f"[bold green]✓ Generated {len(embeddings)} embeddings[/bold green]")
        console.print()
        
        # Store
        console.print("[bold blue]💾 Storing in vector database...[/bold blue]")
        
        def progress_callback(current, total):
            console.print(f"[dim]Batch {current}/{total}[/dim]", end="\r")
        
        self.store.add_chunks(all_chunks, embeddings, progress_callback)
        
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        console.print()
        console.print(Panel(
            f"""[bold green]✓ Indexing complete![/bold green]

[bold]Summary:[/bold]
  • Pages found: {self.stats['total_pages']:,}
  • Pages parsed: {self.stats['parsed_pages']:,}
  • Pages failed: {self.stats['failed_pages']:,}
  • Chunks created: {self.stats['total_chunks']:,}
  • Time elapsed: {minutes}m {seconds}s
  • Model: {self.config.dense_model}

[dim]Index location: {self.config.chroma_dir}[/dim]""",
            border_style="green",
            width=60
        ))
        
        if self.stats['failed_pages'] > 0:
            console.print()
            console.print("[dim]Failed pages logged to: {}/parse_errors.log[/dim]".format(self.config.log_dir))
            self._log_failures()
    
    def _find_man_pages(self) -> List[Path]:
        """Find all man pages."""
        man_files = []
        for section in self.config.sections:
            section_dir = self.config.man_dir / f"man{section}"
            if section_dir.exists():
                for f in section_dir.glob(f"*.{section}*"):
                    if f.suffix == '.gz' or f.suffix == f'.{section}':
                        man_files.append(f)
        return man_files
    
    def _log_failures(self):
        """Log failed pages."""
        if self.stats['failed_files']:
            log_file = self.config.log_dir / "parse_errors.log"
            with open(log_file, 'w') as f:
                f.write(f"Parse failures from {datetime.now().isoformat()}\n")
                f.write("=" * 50 + "\n\n")
                for path in self.stats['failed_files']:
                    f.write(f"{path}\n")
            logger.info(f"Wrote parse errors to {log_file}")

# ============================================================================
# MAIN SEARCH ENGINE
# ============================================================================

class MainSearchEngine:
    """Main search engine with normalized scores."""
    
    def __init__(self, config: Config):
        self.config = config
        self.embedder = EfficientEmbedder(config)
        self.store = NormalizedVectorStore(config)
    
    def search(self, query: str, top_k: int = None) -> List[SearchResult]:
        """Search with normalized scores."""
        if top_k is None:
            top_k = self.config.default_top_k
        
        if top_k > self.config.max_top_k:
            top_k = self.config.max_top_k
        
        if self.store.count() == 0:
            raise IndexNotFoundError("No index found. Run 'semantic-man index' first.")
        
        # Get query embedding
        query_embedding = self.embedder.embed_single(query)
        
        # Search
        results = self.store.search(query_embedding, top_k=top_k * 2)
        
        if not results:
            return []
        
        # Deduplicate by command
        unique = {}
        for result in results:
            cmd = result['metadata'].get('command', '')
            if cmd not in unique or result['score'] > unique[cmd]['score']:
                unique[cmd] = result
        
        # Format results
        formatted = []
        for cmd, result in list(unique.items())[:top_k]:
            metadata = result['metadata']
            
            # Get description
            desc = cmd
            if metadata.get('chunk_type') == 'name':
                # Try to extract description from content
                content = result['document']
                if ' - ' in content:
                    desc = content.split(' - ', 1)[1][:60]
            
            formatted.append(SearchResult(
                command=cmd,
                section=metadata.get('section', 0),
                description=desc,
                score=result['score'],
                chunk_type=metadata.get('chunk_type', ''),
                content=result['document']
            ))
        
        formatted.sort(key=lambda x: x.score, reverse=True)
        return formatted
    
    def get_stats(self) -> Dict:
        """Get comprehensive stats."""
        store_stats = self.store.get_stats()
        return {
            'total_chunks': store_stats['total_chunks'],
            'model': self.config.dense_model,
            'index_dir': str(self.config.chroma_dir),
            'collection': store_stats['collection_name']
        }

# ============================================================================
# INTERACTIVE AGENT
# ============================================================================

class InteractiveAgent:
    """Interactive CLI agent with helpful prompts."""
    
    def __init__(self, config: Config):
        self.config = config
        self.engine = MainSearchEngine(config)
        self.running = True
        self.history = []
        
        if readline:
            hist_file = config.cache_dir / "cli_history"
            try:
                readline.read_history_file(str(hist_file))
            except:
                pass
            import atexit
            atexit.register(lambda: readline.write_history_file(str(hist_file)))
    
    def run(self):
        """Run interactive agent."""
        console.print(Panel.fit(
            "[bold cyan]🐧 Semantic Man-Page Agent[/bold cyan]\n"
            "[dim]Ask about any Linux command - type /help for commands[/dim]",
            border_style="cyan"
        ))
        
        # Show index status
        try:
            stats = self.engine.get_stats()
            console.print(f"[dim]Index: {stats['total_chunks']:,} chunks loaded[/dim]")
        except:
            console.print("[dim]⚠️ No index found. Run 'semantic-man index' first.[/dim]")
        
        console.print()
        console.print("[bold green]Commands:[/bold green]")
        console.print("  [yellow]/help[/yellow]      - Show help")
        console.print("  [yellow]/stats[/yellow]     - Show statistics")
        console.print("  [yellow]/clear[/yellow]     - Clear screen")
        console.print("  [yellow]/quit[/yellow]      - Exit")
        console.print("  [dim]Or type your question[/dim]")
        console.print()
        
        while self.running:
            try:
                query = input(Text("❯ ", style="bold cyan").plain)
                
                if not query:
                    continue
                
                if query.startswith('/'):
                    self._handle_command(query)
                else:
                    self._handle_query(query)
                    
            except KeyboardInterrupt:
                console.print("\n[dim]Type /quit to exit[/dim]")
            except EOFError:
                break
            except Exception as e:
                console.print(f"[bold red]Error: {e}[/bold red]")
    
    def _handle_command(self, cmd: str):
        cmd = cmd.lower().strip()
        
        if cmd in ['/quit', '/exit']:
            self.running = False
            console.print("[bold green]Goodbye! 👋[/bold green]")
        elif cmd == '/help':
            self._show_help()
        elif cmd == '/stats':
            self._show_stats()
        elif cmd == '/clear':
            os.system('clear')
        else:
            console.print(f"[bold red]Unknown command: {cmd}[/bold red]")
            console.print("  Type /help for available commands")
    
    def _handle_query(self, query: str):
        try:
            with console.status(f"[bold blue]Searching...[/bold blue]"):
                results = self.engine.search(query)
            
            if not results:
                console.print("[bold yellow]No results found. Try rephrasing your question.[/bold yellow]")
                return
            
            self._display_results(query, results)
            
        except IndexNotFoundError as e:
            console.print(f"[bold yellow]⚠️ {e}[/bold yellow]")
            console.print("  Run [bold]semantic-man index[/bold] to build the index")
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
    
    def _display_results(self, query: str, results: List[SearchResult]):
        console.print()
        console.print(Panel(f"[bold green]Results for '{query}'[/bold green]", border_style="green"))
        
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("#", style="dim", width=4)
        table.add_column("Command", style="bold cyan", no_wrap=True)
        table.add_column("Section", style="green", width=8)
        table.add_column("Relevance", style="yellow", width=10)
        table.add_column("Description", width=50)
        
        for i, result in enumerate(results[:10], 1):
            # Color based on score
            if result.score > 0.8:
                score_color = "green"
            elif result.score > 0.6:
                score_color = "yellow"
            else:
                score_color = "red"
            
            desc = result.description[:60]
            if len(result.description) > 60:
                desc += "..."
            
            table.add_row(
                str(i),
                result.command,
                str(result.section),
                f"[{score_color}]{result.format_score()}[/{score_color}]",
                desc
            )
        
        console.print(table)
        
        # Show command suggestions with explanations
        console.print("\n[bold blue]💡 Try these commands:[/bold blue]")
        for result in results[:3]:
            console.print(f"  [cyan]•[/cyan] [bold]{result.command}[/bold]")
            if result.score > 0.7:
                console.print(f"    [dim]✓ Highly relevant match[/dim]")
        
        console.print()
    
    def _show_help(self):
        help_text = """
# 🐧 Semantic Man-Page Agent Help

## Basic Usage
Type your question in natural language and the agent will find relevant Linux commands.

## Examples
- `how do I compress a folder?`
- `show network connections`
- `find large files on disk`
- `check disk usage`
- `install software`
- `monitor system resources`

## Commands
- `/help` - Show this help
- `/stats` - Show index statistics
- `/clear` - Clear the screen
- `/quit` - Exit the agent

## Tips
- Be descriptive: "list running processes" works better than "ps"
- Use natural language like you're asking a colleague
- The agent understands intent, not just keywords

## Scoring
Results show a relevance score (0-100%) indicating how well the command matches your query.
- **80-100%**: Highly relevant
- **60-79%**: Moderately relevant
- **Below 60%**: Somewhat relevant, consider rephrasing
        """
        console.print(Markdown(help_text))
    
    def _show_stats(self):
        try:
            stats = self.engine.get_stats()
            console.print(Panel(
                f"""[bold green]📊 Index Statistics[/bold green]

[bold]Total chunks:[/bold] {stats['total_chunks']:,}
[bold]Model:[/bold] {stats['model']}
[bold]Index location:[/bold] {stats['index_dir']}
[bold]Collection:[/bold] {stats['collection']}""",
                border_style="green",
                width=60
            ))
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")

# ============================================================================
# CLI COMMANDS
# ============================================================================

def cmd_index(args):
    """Index man pages."""
    CONFIG.ensure_dirs()
    
    # Setup logging
    setup_logging(debug=args.debug)
    
    try:
        indexer = MainIndexer(CONFIG)
        indexer.index_all(force_rebuild=args.force, resume=args.resume)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]⚠️ Indexing interrupted[/bold yellow]")
        console.print("[dim]Index may be incomplete. Run with --resume to continue.[/dim]")
        sys.exit(EXIT_INTERRUPTED)
    except Exception as e:
        console.print(f"[bold red]❌ Indexing failed: {e}[/bold red]")
        sys.exit(EXIT_PARSE_ERROR)

def cmd_search(args):
    """Search man pages."""
    if not args.query:
        console.print("[bold red]Error: Please provide a search query[/bold red]")
        console.print("  Example: semantic-man search 'compress folder'")
        sys.exit(1)
    
    CONFIG.ensure_dirs()
    setup_logging(debug=args.debug)
    
    try:
        engine = MainSearchEngine(CONFIG)
        results = engine.search(args.query, top_k=args.top_k)
        
        if not results:
            console.print("[bold yellow]No results found. Try rephrasing your query.[/bold yellow]")
            console.print("  Example: 'compress folder' instead of 'how to compress'")
            sys.exit(0)
        
        console.print(f"\n[bold green]Results for '{args.query}':[/bold green]\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim", width=4)
        table.add_column("Command", style="bold cyan")
        table.add_column("Section", style="green", width=8)
        table.add_column("Relevance", style="yellow", width=10)
        table.add_column("Description")
        
        for i, result in enumerate(results[:args.top_k], 1):
            if result.score > 0.8:
                score_color = "green"
            elif result.score > 0.6:
                score_color = "yellow"
            else:
                score_color = "red"
            
            desc = result.description[:60]
            if len(result.description) > 60:
                desc += "..."
            
            table.add_row(
                str(i),
                result.command,
                str(result.section),
                f"[{score_color}]{result.format_score()}[/{score_color}]",
                desc
            )
        
        console.print(table)
        
    except IndexNotFoundError as e:
        console.print(f"[bold yellow]⚠️ {e}[/bold yellow]")
        console.print("  Run [bold]semantic-man index[/bold] to build the index")
        sys.exit(EXIT_INDEX_NOT_FOUND)
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)

def cmd_agent(args):
    """Start interactive agent."""
    CONFIG.ensure_dirs()
    setup_logging(debug=args.debug)
    
    try:
        agent = InteractiveAgent(CONFIG)
        agent.run()
    except KeyboardInterrupt:
        console.print("\n[bold green]Goodbye! 👋[/bold green]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)

def cmd_stats(args):
    """Show statistics."""
    CONFIG.ensure_dirs()
    setup_logging(debug=args.debug)
    
    try:
        engine = MainSearchEngine(CONFIG)
        stats = engine.get_stats()
        
        console.print(Panel(
            f"""[bold green]📊 Semantic Man-Page Agent Statistics[/bold green]

[bold]Total chunks:[/bold] {stats['total_chunks']:,}
[bold]Model:[/bold] {stats['model']}
[bold]Index location:[/bold] {stats['index_dir']}
[bold]Collection:[/bold] {stats['collection']}""",
            border_style="green",
            width=60
        ))
        
        if stats['total_chunks'] == 0:
            console.print("\n[yellow]⚠️ Index is empty. Run 'semantic-man index' to build it.[/yellow]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)

def cmd_clear(args):
    """Clear indexed data."""
    response = input("Delete all indexed data? (y/N): ")
    if response.lower() == 'y':
        CONFIG.ensure_dirs()
        try:
            store = NormalizedVectorStore(CONFIG)
            store.clear()
            console.print("[bold green]✓ Index cleared![/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            sys.exit(1)
    else:
        console.print("[dim]Operation cancelled[/dim]")

def cmd_version(args):
    """Show version."""
    console.print(f"semantic-man {__version__}")
    sys.exit(0)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="🐧 Semantic Man-Page Agent - Find Linux commands by describing what you want",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  semantic-man index                    # Index man pages
  semantic-man agent                    # Interactive mode
  semantic-man search "compress folder" # Search
  semantic-man stats                    # Show statistics
  semantic-man doctor                   # Health check
        """
    )
    
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Index
    p = subparsers.add_parser('index', help='Index man pages')
    p.add_argument('--force', action='store_true', help='Force rebuild')
    p.add_argument('--resume', action='store_true', help='Resume interrupted indexing')
    p.set_defaults(func=cmd_index)
    
    # Search
    p = subparsers.add_parser('search', help='Search man pages')
    p.add_argument('query', nargs='?', help='Search query')
    p.add_argument('--top-k', type=int, default=CONFIG.default_top_k,
                  help=f'Number of results (default: {CONFIG.default_top_k})')
    p.set_defaults(func=cmd_search)
    
    # Agent
    p = subparsers.add_parser('agent', help='Start interactive agent')
    p.set_defaults(func=cmd_agent)
    
    # Stats
    p = subparsers.add_parser('stats', help='Show statistics')
    p.set_defaults(func=cmd_stats)
    
    # Clear
    p = subparsers.add_parser('clear', help='Clear indexed data')
    p.set_defaults(func=cmd_clear)
    
    # Doctor
    p = subparsers.add_parser('doctor', help='Run health check')
    p.set_defaults(func=cmd_doctor)
    
    # Version
    p = subparsers.add_parser('version', help='Show version')
    p.set_defaults(func=cmd_version)
    
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(0)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted[/bold yellow]")
        sys.exit(EXIT_INTERRUPTED)
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()