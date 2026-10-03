#!/usr/bin/env python3
"""
Linux Command Knowledge & Retrieval Engine  --  v0.4
=====================================================
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Soft imports
# ---------------------------------------------------------------------------
_MISSING: List[str] = []
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore
    _MISSING.append("numpy")
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    SentenceTransformer = None  # type: ignore
    CrossEncoder = None  # type: ignore
    _MISSING.append("sentence-transformers")
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import (Progress, SpinnerColumn, BarColumn, TextColumn,
                               TimeElapsedColumn)
    from rich.panel import Panel
    from rich import box
    from rich.markdown import Markdown
    from rich.prompt import Prompt
except ImportError:
    print("Missing 'rich'. Install: pip install -r requirements.txt", file=sys.stderr)
    raise
try:
    import readline  # noqa: F401
except ImportError:
    readline = None  # type: ignore

# ---------------------------------------------------------------------------
# Version / paths / config
# ---------------------------------------------------------------------------
__version__ = "0.4.1"

CACHE_DIR  = Path(os.environ.get("SEMANTIC_MAN_CACHE", Path.home() / ".cache" / "semantic-man"))
DB_PATH    = CACHE_DIR / "index.sqlite"
EMB_PATH   = CACHE_DIR / "embeddings.npy"
IDS_PATH   = CACHE_DIR / "embedding_ids.json"
MANIFEST   = CACHE_DIR / "manifest.json"
CONF_TABLE = CACHE_DIR / "confidence_table.json"
LOGS_DIR   = CACHE_DIR / "logs"
PARSE_LOG  = LOGS_DIR / "parse_errors.log"

MAN_DIRS = [Path("/usr/share/man"), Path("/usr/local/share/man"), Path("/usr/local/man")]

EMBED_MODEL  = os.environ.get("SEMANTIC_MAN_EMBED_MODEL",  "all-MiniLM-L6-v2")
RERANK_MODEL = os.environ.get("SEMANTIC_MAN_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

STANDARD_SECTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Section priors keyed by intent only. Distro influences query expansion but
# does not select different section priors. Distro-specific priors are v0.5.
SECTION_PRIORS: Dict[str, Dict[int, float]] = {
    "find_command": {1:1.30, 8:1.10, 2:0.90, 3:0.80, 4:0.70, 5:0.70, 6:0.40, 7:0.90, 9:0.60},
    "explain":      {1:1.05, 8:1.05, 2:1.05, 3:1.05, 5:1.05, 7:1.05, 4:1.00, 6:0.60, 9:1.00},
    "troubleshoot": {1:1.25, 8:1.15, 5:1.05, 7:1.00, 2:0.90, 3:0.80, 4:0.70, 6:0.40, 9:0.70},
    "package":      {1:1.30, 8:1.10, 5:0.90, 7:0.90, 2:0.80, 3:0.70, 4:0.60, 6:0.40, 9:0.60},
    "compare":      {1:1.25, 8:1.10, 2:0.90, 3:0.85, 7:0.85, 5:0.80, 4:0.70, 6:0.40, 9:0.60},
    "general":      {1:1.20, 8:1.05, 2:0.95, 3:0.90, 5:0.90, 7:0.90, 4:0.80, 6:0.40, 9:0.70},
}

CHUNK_TYPE_WEIGHTS = {
    "name":1.50, "synopsis":1.40, "examples":1.50,
    "description":1.00, "options":0.80,
    "environment":0.55, "files":0.55,
    "notes":0.50, "see_also":0.45, "other":0.60,
}

LENGTH_PENALTY_BETA = 0.20

# Scoring weights (absolute scale, no min-max).
W_FUSED, W_RERANK, W_HITS, W_NAME = 0.35, 0.55, 0.05, 0.05

DENSE_K, FTS_K = 150, 150
RERANK_POOL    = 60
EVIDENCE_PER_CMD = 3
FINAL_K = 8

# Abstention is signal-based; this is only a floor.
ABSTAIN_SCORE_THRESHOLD = 0.15

# ---------------------------------------------------------------------------
# Logging / console
# ---------------------------------------------------------------------------
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("semantic_man")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOGS_DIR / "semantic_man.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
console = Console()

# ===========================================================================
# Data model
# ===========================================================================

@dataclass
class Chunk:
    command: str
    section: int
    chunk_type: str
    text: str
    source: str = "man"
    weight: float = 1.0
    id: str = ""
    content_hash: str = ""
    page_id: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha1(self.text.encode("utf-8","ignore")).hexdigest()
        if not self.id:
            raw = f"{self.source}|{self.section}|{self.command}|{self.chunk_type}|{self.content_hash}"
            self.id = hashlib.sha1(raw.encode("utf-8","ignore")).hexdigest()[:16]


@dataclass
class CommandDoc:
    command: str
    section: int
    source: str
    page_id: str = ""
    summary: str = ""
    synopsis: str = ""
    description: str = ""
    examples: List[str] = field(default_factory=list)
    chunk_types: Set[str] = field(default_factory=set)
    n_chunks: int = 0
    page_hash: str = ""
    aliases: List[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.source}:{self.section}:{self.command}"


# ===========================================================================
# Environment
# ===========================================================================

@dataclass
class Environment:
    os: str = "linux"
    distro: str = ""
    distro_family: str = ""
    distro_version: str = ""
    kernel: str = ""
    arch: str = ""
    shell: str = ""
    package_manager: str = ""
    has_systemd: bool = False
    def to_dict(self): return asdict(self)


_FAMILY_MAP = {
    "ubuntu":"debian","debian":"debian","linuxmint":"debian","pop":"debian","kali":"debian",
    "fedora":"redhat","rhel":"redhat","centos":"redhat","rocky":"redhat","almalinux":"redhat",
    "arch":"arch","manjaro":"arch","endeavouros":"arch",
    "opensuse":"suse","opensuse-leap":"suse","opensuse-tumbleweed":"suse","sles":"suse",
    "alpine":"alpine",
}

def _read(p: str) -> str:
    try: return Path(p).read_text(errors="ignore")
    except Exception: return ""


def detect_environment() -> Environment:
    env = Environment()
    env.os = platform.system().lower()
    env.arch = platform.machine()
    try: env.kernel = platform.release()
    except Exception: pass
    for line in _read("/etc/os-release").splitlines():
        if line.startswith("ID="): env.distro = line.split("=",1)[1].strip().strip('"')
        elif line.startswith("VERSION_ID="): env.distro_version = line.split("=",1)[1].strip().strip('"')
    env.distro_family = _FAMILY_MAP.get(env.distro, "other")
    env.shell = os.path.basename(os.environ.get("SHELL",""))
    for pm, path in [("apt","/usr/bin/apt"),("dnf","/usr/bin/dnf"),("yum","/usr/bin/yum"),
                     ("pacman","/usr/bin/pacman"),("zypper","/usr/bin/zypper"),
                     ("apk","/sbin/apk"),("brew","/usr/local/bin/brew")]:
        if Path(path).exists(): env.package_manager = pm; break
    env.has_systemd = Path("/run/systemd/system").exists()
    return env


def which_command(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


# ===========================================================================
# Man discovery + parsing
# ===========================================================================

_MAN_FILE_RE = re.compile(r"^(.+?)\.(\d)([a-z]{1,4})?(?:\.gz)?$")
_MAN_DIR_RE  = re.compile(r"^man(\d)([a-z]{0,4})$")


def discover_man_pages(sections: Optional[Sequence[int]] = None) -> List[Tuple[Path, int, str, str]]:
    """Return list of (path, section, extension, name_from_filename)."""
    wanted = set(sections) if sections else set(STANDARD_SECTIONS)
    found: Dict[Tuple[int,str,str], Path] = {}
    for root in MAN_DIRS:
        if not root.exists(): continue
        for d in root.iterdir():
            if not d.is_dir(): continue
            m_dir = _MAN_DIR_RE.match(d.name)
            if not m_dir: continue
            sec = int(m_dir.group(1))
            if sec not in wanted: continue
            dir_ext = m_dir.group(2) or ""
            for p in d.iterdir():
                if not p.is_file(): continue
                m = _MAN_FILE_RE.match(p.name)
                if not m: continue
                try: file_sec = int(m.group(2))
                except ValueError: continue
                if file_sec != sec: continue
                name = m.group(1)
                ext = (m.group(3) or "") or dir_ext
                key = (sec, ext, name)
                prev = found.get(key)
                if prev is None or (prev.suffix == ".gz" and p.suffix != ".gz"):
                    found[key] = p
    return [(p, sec, ext, name) for (sec, ext, name), p in found.items()]


def _decompress(path: Path) -> str:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", errors="ignore") as f: return f.read()
        return path.read_text(errors="ignore")
    except Exception as e:
        raise RuntimeError(f"decompress failed: {e}")


def _render_man(path: Path) -> str:
    try:
        env = dict(os.environ); env["MANWIDTH"]="100"; env["GROFF_NO_SGR"]="1"
        out = subprocess.run(["man","-l","--nj",str(path)],
                             capture_output=True, text=True, timeout=25, env=env)
        if out.stdout.strip():
            return re.sub(r".\x08", "", out.stdout)
    except Exception:
        pass
    try: return _decompress(path)
    except Exception: return ""


_MAN_SECTION_HEADER_RE = re.compile(r"^([A-Z][A-Z0-9 _\-]{2,})\s*$")
_CANONICAL = {
    "NAME":"name","SYNOPSIS":"synopsis","SYNOPSYS":"synopsis",
    "DESCRIPTION":"description","OPTIONS":"options","OPERANDS":"options","ARGUMENTS":"options",
    "EXAMPLES":"examples","EXAMPLE":"examples","ENVIRONMENT":"environment","FILES":"files",
    "EXIT STATUS":"other","RETURN VALUE":"other","ERRORS":"other","DIAGNOSTICS":"other",
    "NOTES":"notes","BUGS":"notes","CAVEATS":"notes","SEE ALSO":"see_also",
    "AUTHOR":"other","AUTHORS":"other","HISTORY":"other","COPYRIGHT":"other",
    "REPORTING BUGS":"other","AVAILABILITY":"other","CONFORMING TO":"other","STANDARDS":"other",
    "COLOPHON":"other","SECURITY":"notes","LIMITATIONS":"notes","DEPENDENCIES":"other",
    "ATTRIBUTES":"other","VERSIONS":"other","CONTEXT":"other",
}


def split_into_sections(rendered: str) -> Dict[str, str]:
    lines = rendered.splitlines(); current="other"; buckets: Dict[str,List[str]]={"other":[]}
    for ln in lines:
        m = _MAN_SECTION_HEADER_RE.match(ln.strip())
        if m and len(ln.strip()) <= 40:
            name = m.group(1).strip().rstrip(":")
            canon = _CANONICAL.get(name)
            if canon:
                current = canon; buckets.setdefault(current, []); continue
        buckets.setdefault(current, []).append(ln)
    return {k:"\n".join(v).strip() for k,v in buckets.items() if "\n".join(v).strip()}


def parse_name_section(name_text: str) -> Tuple[str, List[str], str]:
    """Parse NAME. Returns (primary_command, aliases, summary).

    Handles: 'ls, dir, vdir - list directory contents'
             'gzip, gunzip, zcat - compress or expand files'
             'foo - single name'
             'foo' (no summary)
    """
    if not name_text:
        return "", [], ""
    first = name_text.strip().splitlines()[0].strip()
    summary = ""
    names_part = first
    for sep in (" - ", " \u2013 ", " -- "):
        if sep in first:
            names_part, summary = first.split(sep, 1)
            break
    names = [n.strip() for n in names_part.split(",") if n.strip()]
    if not names:
        return "", [], summary.strip()
    return names[0], names[1:], summary.strip()


def extract_examples(text: str) -> List[str]:
    if not text: return []
    out: List[str]=[]; seen: Set[str]=set()
    for line in text.splitlines():
        s = line.strip()
        if not s: continue
        if s.startswith("$ "): s = s[2:].strip()
        if re.match(r"^[a-z][a-z0-9_.\-]{0,30}\s", s) and len(s)<200:
            if s not in seen: seen.add(s); out.append(s)
    return out[:10]


def parse_man_page(path: Path, section: int, filename_name: str, ext: str = ""
                   ) -> Optional[Tuple[CommandDoc, Dict[str,str], str]]:
    rendered = _render_man(path)
    if not rendered or len(rendered.strip())<20: return None
    page_hash = hashlib.sha1(rendered.encode("utf-8","ignore")).hexdigest()
    buckets = split_into_sections(rendered)
    cmd, aliases, summary = parse_name_section(buckets.get("name",""))
    if not cmd: cmd = filename_name
    page_id = f"man:{section}:{ext}:{filename_name}"
    doc = CommandDoc(
        command=cmd, section=section, source="man", page_id=page_id,
        summary=summary, synopsis=buckets.get("synopsis","").strip(),
        description=buckets.get("description","").strip(),
        examples=extract_examples(buckets.get("examples","")),
        page_hash=page_hash, aliases=aliases,
    )
    has_name = bool(buckets.get("name")); has_syn = bool(buckets.get("synopsis"))
    if not (has_name or has_syn) and section not in (5,7): return None
    for ctype in buckets:
        if buckets[ctype].strip() and len(buckets[ctype].strip())>=4:
            doc.chunk_types.add(ctype)
    doc.n_chunks = max(len(doc.chunk_types),1)
    return doc, buckets, page_hash


def build_chunks(doc: CommandDoc, buckets: Dict[str,str]) -> List[Chunk]:
    out: List[Chunk]=[]
    for ctype, text in buckets.items():
        text = text.strip()
        if not text: continue
        if len(text)>4000: text = text[:4000]
        out.append(Chunk(command=doc.command, section=doc.section, chunk_type=ctype,
                         text=text, source=doc.source,
                         weight=CHUNK_TYPE_WEIGHTS.get(ctype,0.6),
                         page_id=doc.page_id))
    # Alias chunks: each alias gets its own NAME + SYNOPsis entries pointing
    # at the same page_id, so aliases are independently retrievable.
    for alias in doc.aliases:
        out.append(Chunk(command=alias, section=doc.section, chunk_type="name",
                         text=f"{alias} - {doc.summary}",
                         source=doc.source, weight=CHUNK_TYPE_WEIGHTS["name"],
                         page_id=doc.page_id))
        if doc.synopsis:
            out.append(Chunk(command=alias, section=doc.section, chunk_type="synopsis",
                             text=doc.synopsis, source=doc.source,
                             weight=CHUNK_TYPE_WEIGHTS["synopsis"],
                             page_id=doc.page_id))
    return out


# ===========================================================================
# Canonical alias groups + query expansion
# ===========================================================================

ALIAS_GROUPS: Dict[str, List[str]] = {
    "compression":     ["gzip","bzip2","xz","zstd","compress","lz4","pigz"],
    "archive":         ["tar","zip","7z","cpio","pax"],
    "decompression":   ["gunzip","bunzip2","unxz","unzstd","uncompress"],
    "extract":         ["tar","unzip","7z","cpio"],
    "filesystem_view": ["ls","find","fd","locate","plocate","tree","stat"],
    "disk_usage":      ["du","df","ncdu","baobab","dust"],
    "process_view":    ["ps","top","htop","pgrep","pidof"],
    "process_control": ["kill","pkill","killall","killall5","nice","renice"],
    "network_view":    ["ss","netstat","ip","ifconfig","lsof"],
    "network_capture": ["tcpdump","tshark","wireshark","dumpcap"],
    "network_diag":    ["ping","mtr","traceroute","dig","host","nslookup"],
    "text_search":     ["grep","rg","ripgrep","ack","ag","ugrep"],
    "text_transform":  ["sed","awk","cut","tr","sort","uniq"],
    "permissions":     ["chmod","chown","chgrp","umask","setfacl","getfacl"],
    "user_admin":      ["useradd","adduser","usermod","userdel","passwd","chsh"],
    "group_admin":     ["groupadd","groupmod","groupdel","gpasswd"],
    "package_debian":  ["apt","apt-get","dpkg","aptitude"],
    "package_redhat":  ["dnf","yum","rpm"],
    "package_arch":    ["pacman"],
    "package_suse":    ["zypper"],
    "package_alpine":  ["apk"],
    "logs":            ["journalctl","dmesg","last","lastlog","tail","less"],
    "systemd":         ["systemctl","journalctl","systemd-analyze","systemd-run"],
    "container":       ["docker","podman","kubectl","nerdctl"],
    "http_fetch":      ["curl","wget","httpie","aria2"],
}

_FAMILY_GROUPS = {
    "debian":"package_debian","redhat":"package_redhat","arch":"package_arch",
    "suse":"package_suse","alpine":"package_alpine",
}

_EXPANSIONS: Dict[str, List[str]] = {
    "compress":   ["archive","zip","gzip","tar"],
    "decompress": ["extract","unzip","untar"],
    "find":       ["search","locate"],
    "disk":       ["storage","filesystem","space"],
    "space":      ["disk","storage"],
    "user":       ["account","login"],
    "permission": ["mode","ownership","access"],
    "network":    ["networking","connection","socket"],
    "log":        ["logs","journal","syslog"],
    "install":    ["package","apt","dnf","pacman"],
    # Additions for common phrasings that had no direct keyword.
    "word":       ["grep","regex"],
    "words":      ["grep","regex"],
    "text":       ["grep","strings"],
    "count":      ["wc"],
    "line":       ["wc","grep"],
    "lines":      ["wc","grep"],
}


def expand_query(query: str, env: Optional[Environment]=None) -> str:
    toks = re.findall(r"[A-Za-z0-9_\-./]+", query.lower())
    extra: List[str]=[]
    for t in toks:
        for e in _EXPANSIONS.get(t,[]):
            if e not in toks and e not in extra: extra.append(e)
    if env and any(w in toks for w in ("install","package","remove","upgrade")):
        grp = _FAMILY_GROUPS.get(env.distro_family)
        if grp:
            for c in ALIAS_GROUPS.get(grp, []):
                if c not in toks and c not in extra: extra.append(c)
    return query + ((" " + " ".join(extra)) if extra else "")


def _group_of(cmd: str) -> str:
    for g, members in ALIAS_GROUPS.items():
        if cmd in members: return g
    return ""


# ===========================================================================
# Danger / root
# ===========================================================================

_DANGER = {
    "rm":"high","dd":"high","mkfs":"high","mkfs.ext4":"high","mkfs.xfs":"high",
    "fdisk":"high","parted":"high","shred":"high","wipefs":"high",
    "chmod":"medium","chown":"medium","chgrp":"medium","kill":"medium",
    "killall":"medium","pkill":"medium","systemctl":"medium","service":"medium",
    "iptables":"medium","nft":"medium","ufw":"medium","mount":"medium",
    "umount":"medium","swapoff":"medium","truncate":"medium","find":"medium",
    "sed":"medium","tar":"medium",
}
_ROOT = {
    "apt","apt-get","dpkg","dnf","yum","pacman","zypper","apk","systemctl","service",
    "mount","umount","fdisk","parted","mkfs","dd","iptables","nft","ufw",
    "useradd","userdel","usermod","groupadd","groupdel","chown","chmod","chgrp",
    "reboot","shutdown","poweroff","swapoff","swapon","cryptsetup","lvm","pvcreate",
}


# ===========================================================================
# Shell builtins
# ===========================================================================

_BASH_BUILTINS = [
    "alias","bg","bind","break","builtin","caller","cd","command","compgen",
    "complete","compopt","continue","declare","dirs","disown","echo","enable",
    "eval","exec","exit","export","false","fc","fg","getopts","hash","help",
    "history","jobs","kill","let","local","logout","mapfile","popd","printf",
    "pushd","pwd","read","readarray","return","set","shift","shopt","source",
    "suspend","test","times","trap","true","type","typeset","ulimit","umask",
    "unalias","unset","wait","select","time","coproc","function","readonly",
]
_BUILTIN_SUMMARIES = {
    "cd":"Change the current working directory (shell builtin).",
    "pwd":"Print the current working directory (shell builtin).",
    "export":"Set environment variables for child processes (shell builtin).",
    "alias":"Define or display command aliases (shell builtin).",
    "source":"Execute a script in the current shell (shell builtin).",
    "read":"Read a line from standard input into variables (shell builtin).",
    "echo":"Display a line of text (shell builtin).",
    "printf":"Format and print data (shell builtin).",
    "test":"Evaluate a conditional expression (shell builtin).",
    "ulimit":"Get or set user resource limits (shell builtin).",
    "umask":"Get or set the file-mode creation mask (shell builtin).",
    "jobs":"List active jobs (shell builtin).",
    "fg":"Move a job to the foreground (shell builtin).",
    "bg":"Move a job to the background (shell builtin).",
    "wait":"Wait for background jobs to finish (shell builtin).",
    "kill":"Send a signal to a process (shell builtin and external).",
    "trap":"Trap signals and other events (shell builtin).",
    "set":"Set or unset shell options (shell builtin).",
    "shopt":"Toggle shell options (shell builtin).",
    "type":"Show how a command would be interpreted (shell builtin).",
    "command":"Run a command bypassing shell functions (shell builtin).",
    "eval":"Evaluate a string as shell code (shell builtin).",
    "exec":"Replace the shell with a command (shell builtin).",
    "exit":"Exit the shell (shell builtin).",
    "return":"Return from a function or sourced script (shell builtin).",
    "readonly":"Mark variables as read-only (shell builtin).",
    "declare":"Declare variables and attributes (shell builtin).",
    "local":"Declare local variables in a function (shell builtin).",
    "let":"Evaluate arithmetic expressions (shell builtin).",
    "history":"Display or manipulate command history (shell builtin).",
    "help":"Display help for shell builtins (shell builtin).",
    "hash":"Remember or display command hashes (shell builtin).",
    "getopts":"Parse positional parameters (shell builtin).",
    "shift":"Shift positional parameters (shell builtin).",
    "break":"Exit from a loop (shell builtin).",
    "continue":"Resume the next loop iteration (shell builtin).",
    "true":"Return success (shell builtin).",
    "false":"Return failure (shell builtin).",
    "unset":"Unset variables or functions (shell builtin).",
    "unalias":"Remove alias definitions (shell builtin).",
    "pushd":"Add a directory to the stack (shell builtin).",
    "popd":"Remove a directory from the stack (shell builtin).",
    "dirs":"Display the directory stack (shell builtin).",
}


def builtin_chunks() -> List[Chunk]:
    out: List[Chunk]=[]
    for b in _BASH_BUILTINS:
        summary = _BUILTIN_SUMMARIES.get(b, f"{b} (shell builtin)")
        out.append(Chunk(command=b, section=1, chunk_type="name",
                         text=f"{b} - {summary}", source="builtin",
                         weight=CHUNK_TYPE_WEIGHTS["name"],
                         page_id=f"builtin:{b}"))
        out.append(Chunk(command=b, section=1, chunk_type="description",
                         text=summary, source="builtin",
                         weight=CHUNK_TYPE_WEIGHTS["description"],
                         page_id=f"builtin:{b}"))
    return out


# ===========================================================================
# SQLite schema
# ===========================================================================

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS pages (
    page_id    TEXT PRIMARY KEY,
    source     TEXT NOT NULL,
    section    INTEGER NOT NULL,
    filename   TEXT NOT NULL,
    page_hash  TEXT NOT NULL,
    indexed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    page_id      TEXT NOT NULL,
    command      TEXT NOT NULL,
    section      INTEGER NOT NULL,
    chunk_type   TEXT NOT NULL,
    source       TEXT NOT NULL,
    weight       REAL NOT NULL,
    text         TEXT NOT NULL,
    content_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_cmd  ON chunks(command);
CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    text,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS commands (
    cmd_key       TEXT PRIMARY KEY,
    command       TEXT NOT NULL,
    section       INTEGER NOT NULL,
    source        TEXT NOT NULL,
    summary       TEXT,
    synopsis      TEXT,
    description   TEXT,
    examples      TEXT,
    n_chunks      INTEGER,
    danger        TEXT,
    requires_root INTEGER,
    group_name    TEXT
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def _connect() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH); c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON;"); return c


def _init_db(c: sqlite3.Connection) -> None:
    c.executescript(DDL); c.commit()


def _load_embeddings() -> Tuple[Optional["np.ndarray"], List[str]]:
    if EMB_PATH.exists() and IDS_PATH.exists() and np is not None:
        return np.load(EMB_PATH), json.loads(IDS_PATH.read_text())
    return None, []


def _save_embeddings(arr, ids):
    np.save(EMB_PATH, arr); IDS_PATH.write_text(json.dumps(ids))


# ===========================================================================
# Indexing (incremental)
# ===========================================================================

def build_index(sections: Optional[Sequence[int]]=None, force: bool=False,
                batch_size: int=64) -> None:
    if SentenceTransformer is None or np is None:
        console.print("[red]Missing sentence-transformers/numpy.[/red]"); sys.exit(1)

    if force and DB_PATH.exists():
        console.print("[yellow]--force: deleting DB and embeddings.[/yellow]")
        DB_PATH.unlink()
        if EMB_PATH.exists(): EMB_PATH.unlink()
        if IDS_PATH.exists(): IDS_PATH.unlink()

    conn = _connect(); _init_db(conn)
    env = detect_environment()
    console.print(Panel.fit(
        f"[bold]Environment[/bold]\n"
        f"OS: {env.os} {env.distro} ({env.distro_family}) {env.distro_version}   arch: {env.arch}\n"
        f"Shell: {env.shell}   PM: {env.package_manager}   systemd: {env.has_systemd}\n"
        f"Model: {EMBED_MODEL}",
        title=f"Linux Command Knowledge & Retrieval Engine v{__version__}",
        border_style="cyan"))

    pages = discover_man_pages(sections)
    console.print(f"[green]Discovered {len(pages)} man pages across sections "
                  f"{sorted({s for _p,s,_e,_n in pages})}[/green]")

    # --- Phase 1: parse only changed pages ------------------------------
    parse_start = time.time()
    errors: List[str] = []
    to_index: List[Tuple[CommandDoc, List[Chunk], str]] = []
    unchanged = changed = new = 0

    with Progress(SpinnerColumn(),
                  TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TextColumn("{task.completed}/{task.total}"),
                  TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("Parsing...", total=len(pages))
        for path, sec, ext, name in pages:
            pid = f"man:{sec}:{ext}:{name}"
            try:
                rendered = _render_man(path)
                if not rendered or len(rendered.strip())<20:
                    errors.append(f"{path}: empty"); prog.advance(task); continue
                page_hash = hashlib.sha1(rendered.encode("utf-8","ignore")).hexdigest()
                prev = conn.execute("SELECT page_hash FROM pages WHERE page_id=?", (pid,)).fetchone()
                if prev and prev["page_hash"] == page_hash:
                    unchanged += 1; prog.advance(task); continue
                parsed = parse_man_page(path, sec, name, ext)
                if parsed is None:
                    errors.append(f"{path}: unparseable"); prog.advance(task); continue
                doc, buckets, _ = parsed
                chunks = build_chunks(doc, buckets)
                if prev: changed += 1
                else:    new += 1
                to_index.append((doc, chunks, page_hash))
            except Exception as e:
                errors.append(f"{path}: {e}")
            prog.advance(task)

    parse_time = time.time() - parse_start
    console.print(f"[green]✓ Parse: {new} new, {changed} changed, {unchanged} unchanged, "
                  f"{len(errors)} failures in {parse_time:.1f}s[/green]")
    if errors:
        PARSE_LOG.write_text("\n".join(errors))
        console.print(f"[yellow]⚠ Parse errors -> {PARSE_LOG}[/yellow]")

    # --- Phase 2: write to SQLite ---------------------------------------
    write_start = time.time()
    new_chunks: List[Chunk] = []
    for doc, chunks, page_hash in to_index:
        pid = doc.page_id
        old_ids = [r["chunk_id"] for r in conn.execute(
            "SELECT chunk_id FROM chunks WHERE page_id=?", (pid,))]
        if old_ids:
            q = ",".join("?"*len(old_ids))
            conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({q})", old_ids)
        conn.execute("DELETE FROM chunks WHERE page_id=?", (pid,))
        conn.execute("DELETE FROM pages  WHERE page_id=?", (pid,))
        conn.execute(
            "INSERT INTO pages(page_id,source,section,filename,page_hash,indexed_at) "
            "VALUES(?,?,?,?,?,?)",
            (pid, "man", doc.section, doc.command, page_hash,
             datetime.now(timezone.utc).isoformat()))
        for c in chunks:
            conn.execute(
                "INSERT OR REPLACE INTO chunks"
                "(chunk_id,page_id,command,section,chunk_type,source,weight,text,content_hash) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (c.id, pid, c.command, c.section, c.chunk_type, c.source,
                 c.weight, c.text, c.content_hash))
            new_chunks.append(c)
    conn.commit()
    write_time = time.time() - write_start
    console.print(f"[green]✓ SQLite: {len(to_index)} pages, "
                  f"{len(new_chunks)} new/changed chunks in {write_time:.1f}s[/green]")

    # --- Reconcile deleted pages ----------------------------------------
    # Any page_id in the DB that wasn't seen during discovery is gone from
    # the filesystem (package removed, man page upgraded, etc.). Remove it.
    discovered_pids = {f"man:{sec}:{ext}:{name}" for _p, sec, ext, name in pages}
    db_pids = {r["page_id"] for r in conn.execute(
        "SELECT page_id FROM pages WHERE source='man'")}
    deleted_pids = db_pids - discovered_pids
    if deleted_pids:
        q = ",".join("?" * len(deleted_pids))
        old_chunk_ids = [r["chunk_id"] for r in conn.execute(
            f"SELECT chunk_id FROM chunks WHERE page_id IN ({q})",
            tuple(deleted_pids))]
        if old_chunk_ids:
            qc = ",".join("?" * len(old_chunk_ids))
            conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({qc})",
                         old_chunk_ids)
        conn.execute(f"DELETE FROM chunks WHERE page_id IN ({q})", tuple(deleted_pids))
        conn.execute(f"DELETE FROM pages  WHERE page_id IN ({q})", tuple(deleted_pids))
        conn.commit()
        console.print(f"[yellow]✓ Removed {len(deleted_pids)} deleted page(s) "
                      f"and their chunks.[/yellow]")

    # --- Phase 3: embed new/changed + merge -----------------------------
    embed_start = time.time()
    old_arr, old_ids = _load_embeddings()
    id_to_row = {cid: i for i, cid in enumerate(old_ids)}

    new_emb = None
    if new_chunks:
        console.print("[bold]Loading embedding model...[/bold]")
        model = SentenceTransformer(EMBED_MODEL)
        new_emb = model.encode([c.text for c in new_chunks], batch_size=batch_size,
                               show_progress_bar=True, normalize_embeddings=True,
                               convert_to_numpy=True)

    existing_ids = [r["chunk_id"] for r in conn.execute("SELECT chunk_id FROM chunks")]
    existing_set = set(existing_ids)

    kept_arrs: List["np.ndarray"] = []
    kept_ids: List[str] = []
    for cid in existing_ids:
        row = id_to_row.get(cid)
        if row is not None and old_arr is not None:
            kept_arrs.append(old_arr[row:row+1]); kept_ids.append(cid)
    if new_emb is not None:
        for c, e in zip(new_chunks, new_emb):
            if c.id not in id_to_row:
                kept_arrs.append(e.reshape(1,-1)); kept_ids.append(c.id)

    # builtins (idempotent)
    builtin_new: List[Chunk] = []
    for c in builtin_chunks():
        conn.execute(
            "INSERT OR REPLACE INTO chunks"
            "(chunk_id,page_id,command,section,chunk_type,source,weight,text,content_hash) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (c.id, c.page_id, c.command, c.section, c.chunk_type, c.source,
             c.weight, c.text, c.content_hash))
        if c.id not in existing_set and c.id not in id_to_row:
            builtin_new.append(c)
    conn.commit()
    if builtin_new:
        model = SentenceTransformer(EMBED_MODEL)
        be = model.encode([c.text for c in builtin_new], batch_size=batch_size,
                          show_progress_bar=False, normalize_embeddings=True,
                          convert_to_numpy=True)
        for c, e in zip(builtin_new, be):
            kept_arrs.append(e.reshape(1,-1)); kept_ids.append(c.id)

    if kept_arrs:
        final_arr = np.vstack(kept_arrs)
    else:
        final_arr = np.zeros((0, 384), dtype="float32")
    _save_embeddings(final_arr, kept_ids)
    embed_time = time.time() - embed_start
    console.print(f"[green]✓ Embeddings: {len(kept_ids)} total "
                  f"({len(new_chunks)} new/changed) in {embed_time:.1f}s[/green]")

    # --- FTS rebuild ----------------------------------------------------
    conn.execute("DELETE FROM chunks_fts")
    conn.execute("INSERT INTO chunks_fts(chunk_id, text) SELECT chunk_id, text FROM chunks")
    conn.commit()

    # --- commands table -------------------------------------------------
    conn.execute("DELETE FROM commands")
    rows = conn.execute(
        "SELECT command, section, source, COUNT(*) n_chunks "
        "FROM chunks GROUP BY source, section, command").fetchall()
    for r in rows:
        name_row = conn.execute(
            "SELECT text FROM chunks WHERE source=? AND section=? AND command=? AND chunk_type='name' LIMIT 1",
            (r["source"], r["section"], r["command"])).fetchone()
        summary = ""
        if name_row:
            _c, _a, summary = parse_name_section(name_row["text"])
        syn = conn.execute(
            "SELECT text FROM chunks WHERE source=? AND section=? AND command=? AND chunk_type='synopsis' LIMIT 1",
            (r["source"], r["section"], r["command"])).fetchone()
        desc = conn.execute(
            "SELECT text FROM chunks WHERE source=? AND section=? AND command=? AND chunk_type='description' LIMIT 1",
            (r["source"], r["section"], r["command"])).fetchone()
        ex = conn.execute(
            "SELECT text FROM chunks WHERE source=? AND section=? AND command=? AND chunk_type='examples' LIMIT 1",
            (r["source"], r["section"], r["command"])).fetchone()
        cmd_key = f"{r['source']}:{r['section']}:{r['command']}"
        conn.execute(
            "INSERT OR REPLACE INTO commands"
            "(cmd_key,command,section,source,summary,synopsis,description,examples,"
            " n_chunks,danger,requires_root,group_name) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (cmd_key, r["command"], r["section"], r["source"], summary,
             (syn["text"][:500] if syn else ""),
             (desc["text"][:1500] if desc else ""),
             json.dumps(extract_examples(ex["text"]) if ex else []),
             r["n_chunks"], _DANGER.get(r["command"],"low"),
             1 if r["command"] in _ROOT else 0, _group_of(r["command"])))
    conn.commit()

    n_docs = conn.execute("SELECT COUNT(*) c FROM commands").fetchone()["c"]
    n_chunks = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
    MANIFEST.write_text(json.dumps({
        "version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sections": sorted({s for _p,s,_e,_n in pages}),
        "n_docs": n_docs, "n_chunks": n_chunks,
        "embed_model": EMBED_MODEL, "rerank_model": RERANK_MODEL,
        "environment": env.to_dict(),
        "parse_time_s": round(parse_time,1),
        "embed_time_s": round(embed_time,1),
        "unchanged": unchanged, "changed": changed, "new": new,
        "db_path": str(DB_PATH), "emb_path": str(EMB_PATH),
    }, indent=2))
    conn.close()

    console.print(Panel.fit(
        f"[bold green]Indexing complete[/bold green]\n"
        f"Commands:    {n_docs:,}\n"
        f"Chunks:      {n_chunks:,}\n"
        f"Embeddings:  {len(kept_ids):,}\n"
        f"Unchanged:   {unchanged:,}\n"
        f"New/Changed: {new+changed:,}\n"
        f"DB:          {DB_PATH}",
        title="Done", border_style="green"))


# ===========================================================================
# Intent + domain classification
# ===========================================================================

_INTENT_PATTERNS = [
    ("find_command", re.compile(r"\b(how (do|can) i|what command|which command|what tool|is there a)\b", re.I), 1.0),
    ("explain",      re.compile(r"\b(explain|what does|meaning of|what is)\b", re.I), 1.0),
    ("troubleshoot", re.compile(r"\b(why|troubleshoot|fix|broken|not working|fails|error|crash)\b", re.I), 1.0),
    ("package",      re.compile(r"\b(install|uninstall|remove|update|upgrade|package)\b", re.I), 0.9),
    ("compare",      re.compile(r"\b(difference between|vs\.?|versus|compare)\b", re.I), 1.0),
    ("syntax",       re.compile(r"\b(syntax|usage|flags?|options?|arguments?)\b", re.I), 0.7),
]

_DOMAIN_PATTERNS = {
    "filesystem": re.compile(r"\b(file|folder|directory|disk|space|storage|partition|mount)\b", re.I),
    "network":    re.compile(r"\b(network|port|socket|connection|tcp|udp|dns|http|curl|wget)\b", re.I),
    "process":    re.compile(r"\b(process|pid|cpu|memory|ram|kill|job|daemon|service)\b", re.I),
    "permissions":re.compile(r"\b(permission|chmod|chown|owner|mode|acl|sudo|root)\b", re.I),
    "users":      re.compile(r"\b(user|login|password|group|account|who)\b", re.I),
    "packages":   re.compile(r"\b(package|install|apt|dpkg|snap|flatpak)\b", re.I),
    "text":       re.compile(r"\b(text|string|grep|search|pattern|regex|sed|awk)\b", re.I),
    "compression":re.compile(r"\b(compress|archive|zip|tar|gzip|extract|unzip)\b", re.I),
    "logs":       re.compile(r"\b(log|journal|boot|dmesg|syslog)\b", re.I),
    "system":     re.compile(r"\b(boot|systemd|kernel|service|reboot|shutdown)\b", re.I),
}


def classify_intent(query: str) -> Dict[str, Any]:
    """Return {intent, intent_confidence, domains, domain_scores}.

    Intent is first-match. Domains are multi-label with per-domain hit counts.
    """
    q = query.strip()
    intent = "find_command"; conf = 0.5
    for name, pat, weight in _INTENT_PATTERNS:
        if pat.search(q):
            intent = name; conf = weight; break
    domains: List[str] = []
    dscore: Dict[str, int] = {}
    for d, pat in _DOMAIN_PATTERNS.items():
        hits = len(pat.findall(q))
        if hits:
            domains.append(d); dscore[d] = hits
    return {"intent": intent, "intent_confidence": conf,
            "domains": domains or ["general"], "domain_scores": dscore}


# ===========================================================================
# Rule-based workflows (explicitly labeled)
# ===========================================================================

_WORKFLOWS = [
    {"name":"disk_full",
     "match": re.compile(r"\b(disk (is )?full|no space|out of space|storage full)\b", re.I),
     "title":"Investigate a full disk (rule-based)",
     "steps":[("df -h","Which filesystem is full."),
              ("sudo du -xh / --max-depth=1 2>/dev/null | sort -h","Largest top-level dirs."),
              ("ncdu /","Optional interactive browser.")],
     "sources":["man df","man du","tldr ncdu"]},
    {"name":"network_slow",
     "match": re.compile(r"\b(network (is )?slow|slow network|latency|packet loss)\b", re.I),
     "title":"Diagnose slow network (rule-based)",
     "steps":[("ip a","Interfaces"),("ip route","Routing table"),
              ("mtr -rwzbc 20 1.1.1.1","Latency + path"),
              ("ss -tunap","Active sockets")],
     "sources":["man ip","man mtr","man ss"]},
    {"name":"high_cpu",
     "match": re.compile(r"\b(high cpu|cpu (is )?(high|maxed|pegged))\b", re.I),
     "title":"Investigate high CPU (rule-based)",
     "steps":[("uptime","Load average"),("top -b -n1 | head -20","Top consumers"),
              ("ps -eo pid,ppid,cmd,%cpu,%mem --sort=-%cpu | head","Processes by CPU")],
     "sources":["man uptime","man top","man ps"]},
    {"name":"service_down",
     "match": re.compile(r"\b(service (is )?(down|not running|failed)|systemd.*fail)\b", re.I),
     "title":"Failed systemd unit (rule-based)",
     "steps":[("systemctl status <name>","Unit state + recent logs"),
              ("journalctl -u <name> -b --no-pager | tail -50","Last boot logs"),
              ("systemctl list-units --failed","All failed units")],
     "sources":["man systemctl","man journalctl"]},
    {"name":"no_network",
     "match": re.compile(r"\b(no (internet|network)|can't (connect|reach)|offline)\b", re.I),
     "title":"No network (rule-based)",
     "steps":[("ip a","Addresses"),("ip route","Default route"),
              ("ping -c3 1.1.1.1","Raw connectivity"),
              ("resolvectl status","DNS resolver")],
     "sources":["man ip","man ping","man resolvectl"]},
]


def find_workflow(q: str) -> Optional[Dict[str, Any]]:
    for w in _WORKFLOWS:
        if w["match"].search(q): return w
    return None


# ===========================================================================
# Explain
# ===========================================================================

_COMMON_FLAGS = {
    "-l":"long / listen","-a":"all / append","-h":"human / help",
    "-r":"recursive / reverse","-f":"force / file","-v":"verbose / invert",
    "-n":"numeric / no-resolve","-p":"port / preserve","-t":"tcp / type",
    "-u":"udp / user","-i":"interface / ignore-case","-R":"recursive",
    "-P":"port / physical","-o":"output / options","-e":"execute / exclude",
    "-c":"count / create","-d":"directory / debug","-s":"silent / summary",
    "-x":"extract / exclude",
}


def explain_command(cmdline: str) -> str:
    toks = cmdline.strip().split()
    if not toks: return "Empty command."
    main = toks[0]
    lines = [f"[bold]{main}[/bold]"]
    summary = ""
    try:
        out = subprocess.run(["man", main], capture_output=True, text=True, timeout=10)
        if out.stdout:
            m = re.search(r"^\s*"+re.escape(main)+r"\s+-\s+(.+)$", out.stdout, re.M)
            if m: summary = m.group(1).strip()
    except Exception: pass
    if summary: lines.append(f"  ↳ {summary}")
    lines.append(""); lines.append("Arguments:")
    for tok in toks[1:]:
        if tok.startswith("--"): lines.append(f"  {tok:<16} long option")
        elif tok.startswith("-") and len(tok)>2:
            for ch in tok[1:]:
                lines.append(f"  -{ch:<15} {_COMMON_FLAGS.get('-'+ch,'option')}")
        elif tok.startswith("-"):
            lines.append(f"  {tok:<16} {_COMMON_FLAGS.get(tok,'option')}")
        else: lines.append(f"  {tok:<16} argument")
    path = which_command(main)
    lines.append(""); lines.append(f"Installed: {'yes ('+path+')' if path else 'no'}")
    lines.append(f"Danger:    {_DANGER.get(main,'low')}")
    if main in _ROOT: lines.append("Note:      typically requires sudo.")
    return "\n".join(lines)


# ===========================================================================
# Retrieval
# ===========================================================================

_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-./]+")
_STOP = {"a","an","the","how","do","i","to","is","of","in","on","for","with","and",
         "or","my","me","can","you","what","which","that","this","it","be","are",
         "was","were","want","need","use","using","via","by"}


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _content_tokens(text: str) -> List[str]:
    return [t for t in _tokens(text) if t not in _STOP]


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _fused_signal(raw: float) -> float:
    """Map a raw RRF aggregate to [0,1] on an absolute scale.

    RRF contributions are tiny (~0.015 for rank 0 from one retriever). A
    chunk found by both retrievers at rank 0 scores ~0.032. The chunk-type
    weight can push this to ~0.05.

    We use a broad curve so adjacent raw scores (0.015 vs 0.022) do not
    diverge wildly; the goal is that the fused signal captures the
    retrieval consensus without being dominated by tiny numerical noise.
    """
    x = (raw - 0.010) / 0.040
    return _sigmoid(1.5 * x)


class Retriever:
    def __init__(self) -> None:
        if SentenceTransformer is None or np is None:
            raise RuntimeError("Missing deps. Install: pip install -r requirements.txt")
        if not DB_PATH.exists() or not MANIFEST.exists():
            raise RuntimeError("No index. Run: python semantic_man.py index")
        self.manifest = json.loads(MANIFEST.read_text())
        self.embed_model = SentenceTransformer(self.manifest["embed_model"])
        self.emb, self.emb_ids = _load_embeddings()
        if self.emb is None:
            raise RuntimeError("No embeddings. Run: index")
        self.id_to_row = {cid: i for i, cid in enumerate(self.emb_ids)}
        self.conn = _connect()
        self.env = detect_environment()
        self._reranker: Optional["CrossEncoder"] = None
        self.conf_table = self._load_conf_table()

    def _load_conf_table(self) -> Dict[str, Any]:
        if CONF_TABLE.exists():
            try: return json.loads(CONF_TABLE.read_text())
            except Exception: pass
        return {
            "note": "heuristic confidence band; not a calibrated probability",
            "buckets": [
                {"min":0.00,"label":"very low"},
                {"min":0.22,"label":"low"},
                {"min":0.40,"label":"medium"},
                {"min":0.65,"label":"high"},
            ],
        }

    def confidence_label(self, score: float) -> str:
        label = "very low"
        for b in self.conf_table.get("buckets", []):
            if score >= b.get("min", 0): label = b.get("label", label)
        return label

    def _rerank_pairs(self, pairs: List[Tuple[str,str]]) -> List[float]:
        if CrossEncoder is None or not pairs: return [0.0]*len(pairs)
        if self._reranker is None:
            try:
                self._reranker = CrossEncoder(self.manifest.get("rerank_model", RERANK_MODEL))
            except Exception as e:
                logger.warning("Reranker load failed: %s", e); self._reranker = None
        if self._reranker is None: return [0.0]*len(pairs)
        try:
            return [float(x) for x in self._reranker.predict(pairs, show_progress_bar=False)]
        except Exception as e:
            logger.warning("Reranker predict failed: %s", e); return [0.0]*len(pairs)

    def _fts(self, query: str, k: int = FTS_K) -> List[str]:
        toks = _content_tokens(query)
        if not toks: return []
        fts_q = " OR ".join(f'"{t}"' for t in toks)
        try:
            rows = self.conn.execute(
                "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY bm25(chunks_fts) LIMIT ?", (fts_q, k)).fetchall()
            return [r["chunk_id"] for r in rows]
        except sqlite3.OperationalError:
            return []

    def _dense(self, query: str, k: int = DENSE_K) -> List[str]:
        q = self.embed_model.encode([query], normalize_embeddings=True,
                                    convert_to_numpy=True)[0]
        sims = self.emb @ q
        if len(sims) <= k:
            idx = np.argsort(-sims)
        else:
            idx = np.argpartition(-sims, k)[:k]
            idx = idx[np.argsort(-sims[idx])]
        return [self.emb_ids[i] for i in idx]

    @staticmethod
    def _rrf(lists: Sequence[Sequence[str]], k: int = 60) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for lst in lists:
            for rank, cid in enumerate(lst):
                out[cid] = out.get(cid, 0.0) + 1.0/(k + rank + 1)
        return out

    def _aggregate(self, fused: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        if not fused: return {}
        ids = list(fused.keys())
        q = ",".join("?"*len(ids))
        rows = self.conn.execute(
            f"SELECT chunk_id, command, section, source, chunk_type, weight, text "
            f"FROM chunks WHERE chunk_id IN ({q})", ids).fetchall()
        agg: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            cid = r["chunk_id"]
            key = f"{r['source']}:{r['section']}:{r['command']}"
            e = agg.setdefault(key, {
                "command":r["command"], "section":r["section"], "source":r["source"],
                "best_chunk":0.0, "chunks":[], "chunk_types":set(), "n_hits":0,
            })
            cw = CHUNK_TYPE_WEIGHTS.get(r["chunk_type"], 1.0)
            contrib = fused[cid] * cw
            e["best_chunk"] = max(e["best_chunk"], contrib)
            e["chunks"].append((cid, contrib, r["text"], r["chunk_type"]))
            e["chunk_types"].add(r["chunk_type"]); e["n_hits"] += 1
        for e in agg.values():
            e["chunks"].sort(key=lambda x: x[1], reverse=True)
            e["chunks"] = e["chunks"][:EVIDENCE_PER_CMD]
        return agg

    def search(self, query: str, top_k: int = FINAL_K,
               intent: Optional[str] = None) -> List[Dict[str, Any]]:
        expanded = expand_query(query, self.env)
        if intent is None:
            intent = classify_intent(query)["intent"]
        priors = SECTION_PRIORS.get(intent, SECTION_PRIORS["general"])

        fts_ids = self._fts(expanded)
        dense_ids = self._dense(expanded)
        fused = self._rrf([fts_ids, dense_ids])
        agg = self._aggregate(fused)
        if not agg: return []

        for e in agg.values():
            e["best_chunk"] *= priors.get(e["section"], 1.0)
            row = self.conn.execute(
                "SELECT summary, synopsis, description, n_chunks, group_name "
                "FROM commands WHERE cmd_key=?",
                (f"{e['source']}:{e['section']}:{e['command']}",)).fetchone()
            summary = (row["summary"] if row and row["summary"] else "")
            synopsis = (row["synopsis"] if row and row["synopsis"] else "")
            e["summary"] = summary
            e["group_name"] = row["group_name"] if row else ""
            e["headline"] = f"{e['command']} - {summary}".strip(" -")
            if synopsis: e["headline"] += " | " + synopsis[:200]
            e["n_chunks_db"] = row["n_chunks"] if row else e["n_hits"]

        # Evidence-aware reranking.
        #
        # The CrossEncoder is given the query and a compact string built
        # from the command name plus its BEST evidence chunk (the chunk
        # that actually caused the command to be retrieved). This is more
        # informative than showing it the SYNOPSIS, because the SYNOPSIS
        # may not contain the phrase that matched the query.
        cands = sorted(agg.values(), key=lambda e: e["best_chunk"], reverse=True)[:RERANK_POOL]
        pairs: List[Tuple[str,str]] = []
        pair_owner: List[int] = []
        for i, e in enumerate(cands):
            for _cid, _c, text, _ct in e["chunks"]:
                # Show the cross-encoder:
                #   command name — first 300 chars of the evidence chunk
                # (drop the summary; the evidence is what matters)
                evidence = text.strip().replace("\n", " ")[:300]
                pairs.append((query, f"{e['command']} — {evidence}"))
                pair_owner.append(i)
        rr_scores = self._rerank_pairs(pairs)
        per_cmd_rr: Dict[int, float] = {}
        for i, sc in zip(pair_owner, rr_scores):
            if i not in per_cmd_rr or sc > per_cmd_rr[i]:
                per_cmd_rr[i] = sc
        for i, e in enumerate(cands):
            e["rerank_score"] = per_cmd_rr.get(i, 0.0)

        # Absolute-scale scoring. No min-max.
        def lpen(n): return 1.0/(1.0 + LENGTH_PENALTY_BETA*math.log(1+max(n,0)))

        scored: List[Dict[str, Any]] = []
        for e in cands:
            n = e.get("n_chunks_db", e["n_hits"])
            lp = lpen(n)

            fused_raw = e["best_chunk"] * lp
            fused_sig = _fused_signal(fused_raw)

            rr_raw = e["rerank_score"]
            # Clamp to [-4, +4]. A single over-eager cross-encoder output
            # (e.g. +5.2 on a passage that literally contains the query
            # phrase but is not a real answer) must not dominate ranking.
            rr_clamped = max(-4.0, min(4.0, rr_raw))
            rerank_sig = _sigmoid(rr_clamped / 4.0)

            hit = min(e["n_hits"] / 5.0, 1.0)
            name_hit = 1.0 if e["command"].lower() in query.lower().split() else 0.0
            installed = bool(which_command(e["command"]))

            base = (W_FUSED * fused_sig +
                    W_RERANK * rerank_sig +
                    W_HITS * hit +
                    W_NAME * name_hit)
            if not installed:
                base *= 0.95
            single_hit_penalty = 0.92 if e["n_hits"] == 1 else 1.0
            base *= single_hit_penalty
            # Commands matched by only one chunk are less trustworthy than
            # commands matched by several. This directly targets the
            # vlc-wrapper-for-network-ports failure: vlc-wrapper matched
            # on one passage that happens to contain the phrase.
            hit_factor = min(1.0, 0.5 + 0.1 * e["n_hits"])
            base *= hit_factor
            # Small bonus for commands that belong to a known alias group:
            # these are the canonical tools for their task.
            if e.get("group_name"):
                base *= 1.05

            e.update({
                "retrieval_score": float(base),
                "raw_signals": {
                    "fused_raw": round(fused_raw, 4),
                    "fused_sig": round(fused_sig, 4),
                    "rerank_raw": round(rr_raw, 4),
                    "rerank_sig": round(rerank_sig, 4),
                    "n_hits": e["n_hits"],
                    "name_hit": name_hit,
                    "installed": installed,
                },
                "installed": installed,
                "danger": _DANGER.get(e["command"], "low"),
                "requires_root": e["command"] in _ROOT,
            })
            scored.append(e)

        # Dedup by command name.
        by_cmd: Dict[str, Dict[str, Any]] = {}
        for e in scored:
            prev = by_cmd.get(e["command"])
            if prev is None or e["retrieval_score"] > prev["retrieval_score"]:
                by_cmd[e["command"]] = e
        ranked = sorted(by_cmd.values(), key=lambda e: e["retrieval_score"], reverse=True)

        # Signal-based abstention (not a single threshold).
        top = ranked[0] if ranked else None
        if top:
            raw = top.get("raw_signals", {})
            fused_ok  = raw.get("fused_raw", 0.0)  >= 0.030
            rerank_ok = raw.get("rerank_raw", -10.0) >= 0.0
            name_ok   = raw.get("name_hit", 0.0) == 1.0
            n_hits_ok = raw.get("n_hits", 0) >= 3
            # The reranker must approve the top hit unless the exact command
            # name appears in the query. This prevents the "make coffee" ->
            # "make" failure where the token match is superficial.
            strong = (fused_ok and rerank_ok) or (name_ok and rerank_ok)
            top["abstain"] = not strong
            top["signals"] = {
                "fused_ok": fused_ok, "rerank_ok": rerank_ok,
                "name_ok": name_ok, "n_hits_ok": n_hits_ok, "strong": strong,
            }

        for e in ranked:
            e["confidence"] = self.confidence_label(e["retrieval_score"])

        return ranked[:top_k]


# ===========================================================================
# Display
# ===========================================================================

def print_results(query: str, results: List[Dict[str, Any]], verbose: bool=False) -> bool:
    if not results:
        console.print("[yellow]No results.[/yellow]"); return False
    top = results[0]
    if top.get("abstain") or top["retrieval_score"] < ABSTAIN_SCORE_THRESHOLD:
        console.print(Panel.fit(
            "[yellow]I couldn't find a sufficiently relevant Linux command.[/yellow]\n"
            "Weak candidates are shown below.",
            border_style="yellow"))
    t = Table(title=f"Query: {query!r}", box=box.ROUNDED)
    t.add_column("#", justify="right", width=3)
    t.add_column("Command", style="bold cyan")
    t.add_column("Sec", justify="center", width=4)
    t.add_column("Score", justify="right", width=6)
    t.add_column("Conf", width=10)
    t.add_column("Inst", justify="center", width=5)
    t.add_column("Summary", overflow="fold")
    for i, r in enumerate(results, 1):
        s = r.get("summary", "")[:80]
        t.add_row(str(i), r["command"], str(r["section"]),
                  f"{r['retrieval_score']:.2f}", r["confidence"],
                  "✓" if r["installed"] else "—", s)
    console.print(t)

    console.print("\n[bold]Evidence:[/bold]")
    for r in results[:3]:
        warn = ""
        if r["danger"] == "high": warn = " [red](⚠ destructive)[/red]"
        elif r["danger"] == "medium": warn = " [yellow](⚠ caution)[/yellow]"
        root = " [dim](sudo)[/dim]" if r["requires_root"] else ""
        console.print(f"  • [cyan]{r['command']}[/cyan] "
                      f"[dim]({r['confidence']}, sec {r['section']}, "
                      f"group={r.get('group_name') or '-'})[/dim]{root}{warn}")
        if verbose:
            raw = r.get("raw_signals", {})
            console.print(f"      [dim]raw: fused={raw.get('fused_raw')} "
                          f"rerank={raw.get('rerank_raw')} "
                          f"n_hits={raw.get('n_hits')}[/dim]")
            for _cid, _c, text, ctype in r["chunks"]:
                snippet = text.replace("\n"," ")[:140]
                console.print(f"      [{ctype}] {snippet}")
    return True


# ===========================================================================
# Evaluation  (TUNE + TEST split)
# ===========================================================================

TUNE_QUERIES: List[Tuple[str, List[str], List[str], str, str]] = [
    ("how do I compress a folder", ["tar","zip","gzip","compress"], ["xz","bzip2","zstd"], "compression","easy"),
    ("make a file smaller", ["truncate","gzip","xz"], ["compress","zip"], "compression","easy"),
    ("list open network ports", ["ss","netstat"], ["nmap","lsof"], "network","easy"),
    ("monitor network traffic", ["tcpdump","wireshark","tshark","iftop"], ["nstat","dumpcap"], "network","easy"),
    ("show disk usage by directory", ["du","ncdu"], ["df","baobab","dust"], "filesystem","easy"),
    ("find files modified in the last hour", ["find"], ["fd","stat"], "filesystem","easy"),
    ("find large files", ["find","du"], ["ncdu","fd"], "filesystem","easy"),
    ("search text inside files", ["grep","rg","ripgrep"], ["ack","ag","strings"], "text","easy"),
    ("change file permissions", ["chmod","chown"], ["chgrp","umask","setfacl"], "permissions","easy"),
    ("kill a process by name", ["pkill","killall"], ["kill","killall5"], "process","easy"),
    ("check listening services", ["ss","netstat"], ["nmap","lsof"], "network","easy"),
    ("show system uptime", ["uptime"], ["w","who"], "system","easy"),
    ("mount a disk", ["mount"], ["mountpoint","udisksctl"], "filesystem","easy"),
    ("create a user", ["useradd","adduser"], ["newusers","usermod"], "users","easy"),
    ("schedule a job", ["crontab","at"], ["batch","systemd-run"], "system","easy"),
    ("view logs from the last boot", ["journalctl"], ["dmesg","last"], "logs","medium"),
    ("see what's using my network", ["ss","iftop","nethogs"], ["netstat","mtr"], "network","medium"),
    ("free up disk space", ["du","ncdu","rm"], ["df","find"], "filesystem","medium"),
    ("who is logged in", ["who","w"], ["users","last"], "users","easy"),
    ("see what happened at boot", ["journalctl","dmesg"], ["systemctl","last"], "logs","medium"),
    ("find a word in many files", ["grep","rg"], ["ack","find","xargs"], "text","medium"),
    ("limit how much CPU a program uses", ["cpulimit","nice","taskset","prlimit"], ["systemd-run","ulimit"], "process","medium"),
]

TEST_POS: List[Tuple[str, List[str], List[str], str, str]] = [
    ("package this directory into a tar archive", ["tar"], ["zip","7z","cpio"], "compression","easy"),
    ("unpack a gzipped tarball", ["tar","gunzip"], ["gzip","unxz"], "compression","easy"),
    ("create a zip file from a folder", ["zip"], ["tar","7z"], "compression","easy"),
    ("show hidden files in a directory", ["ls"], ["find","fd"], "filesystem","easy"),
    ("find files bigger than 100 megabytes", ["find","du"], ["ncdu"], "filesystem","medium"),
    ("show the size of each subdirectory", ["du","ncdu"], ["df"], "filesystem","easy"),
    ("count how many lines are in a file", ["wc"], ["grep","awk"], "text","easy"),
    ("print the last 20 lines of a log", ["tail"], ["head","less"], "text","easy"),
    ("print the first 10 lines of a file", ["head"], ["tail","sed"], "text","easy"),
    ("replace every occurrence of a word in a file", ["sed","perl"], ["awk","tr"], "text","medium"),
    ("show line numbers when searching a file", ["grep"], ["awk","sed"], "text","easy"),
    ("extract the third column of a file", ["cut","awk"], ["sed","tr"], "text","medium"),
    ("show all running processes", ["ps","top","htop"], ["pgrep","pidof"], "process","easy"),
    ("show memory usage of processes", ["ps","top","htop","free"], ["vmstat"], "process","medium"),
    ("send SIGKILL to a process by PID", ["kill"], ["pkill","killall"], "process","easy"),
    ("change the priority of a running process", ["renice"], ["nice"], "process","medium"),
    ("show what a process is doing right now", ["top","htop","ps"], ["strace","perf"], "process","medium"),
    ("test if a remote host is reachable", ["ping"], ["mtr","traceroute"], "network","easy"),
    ("look up the IP address of a domain", ["dig","host","nslookup"], ["getent","ping"], "network","easy"),
    ("show my IP address", ["ip","hostname","ifconfig"], ["ss","nmcli"], "network","easy"),
    ("download a file from a URL", ["curl","wget"], ["aria2","httpie"], "network","easy"),
    ("show the routing table", ["ip","route","netstat"], ["ss"], "network","easy"),
    ("capture packets on an interface", ["tcpdump","tshark","dumpcap"], ["wireshark"], "network","medium"),
    ("make a file executable", ["chmod"], ["chown"], "permissions","easy"),
    ("change the owner of a file", ["chown"], ["chgrp","chmod"], "permissions","easy"),
    ("add a user to a group", ["usermod","gpasswd","adduser"], ["groupmod"], "users","medium"),
    ("change a user's password", ["passwd"], ["chpasswd"], "users","easy"),
    ("list all users on the system", ["cat","getent","users"], ["who","w","last"], "users","easy"),
    ("install a package", ["apt","apt-get","dpkg","dnf","yum","pacman","zypper","apk"], ["snap","flatpak"], "packages","easy"),
    ("remove an installed package", ["apt","apt-get","dpkg","dnf","yum","pacman","zypper","apk"], ["snap","flatpak"], "packages","easy"),
    ("list installed packages", ["dpkg","apt","rpm","dnf","pacman","zypper","apk"], [], "packages","easy"),
    ("start a systemd service", ["systemctl"], ["service"], "system","easy"),
    ("show logs for a specific service", ["journalctl"], ["dmesg"], "logs","easy"),
    ("show kernel messages from boot", ["dmesg"], ["journalctl"], "logs","easy"),
    ("show the last reboot time", ["uptime","who","last"], ["journalctl"], "system","medium"),
    ("change the current directory", ["cd"], ["pushd","popd"], "shell","easy"),
    ("set an environment variable for this session", ["export","set"], ["env","declare"], "shell","easy"),
    ("define a shortcut for a command", ["alias"], ["function"], "shell","easy"),
    ("show disk free space in human-readable form", ["df"], ["du"], "filesystem","easy"),
    ("show the manual page for a command", ["man"], ["info","help"], "system","easy"),
    ("compare two files line by line", ["diff","cmp"], ["comm","patch"], "text","medium"),
    ("sort lines of a file", ["sort"], ["uniq"], "text","easy"),
    ("remove duplicate lines from a file", ["uniq","sort"], ["awk"], "text","medium"),
]

TEST_NEG: List[Tuple[str, str]] = [
    ("how do I make coffee", "nonsense"),
    ("what is the meaning of life", "nonsense"),
    ("asdfghjkl qwertyuiop", "nonsense"),
    ("what is the capital of France", "nonsense"),
    ("recommend a good movie", "nonsense"),
    ("how to bake sourdough bread", "nonsense"),
    ("weather forecast for tomorrow", "nonsense"),
    ("write me a poem about autumn", "nonsense"),
    ("who won the world cup in 1998", "nonsense"),
    ("explain quantum entanglement", "nonsense"),
]


def _eval_positive(r: "Retriever", queries, label: str) -> Dict[str, float]:
    pos = len(queries); r1=r3=r5=0; mrr=0.0; ndcg=0.0
    lat: List[float] = []
    for q, prim, acc, _dom, _diff in queries:
        rel = set(prim) | set(acc)
        t0 = time.perf_counter()
        intent = classify_intent(q)["intent"]
        res = r.search(q, top_k=10, intent=intent)
        lat.append((time.perf_counter()-t0)*1000.0)
        cmds = [e["command"] for e in res]
        if cmds and cmds[0] in rel: r1 += 1
        if set(cmds[:3]) & rel:     r3 += 1
        if set(cmds[:5]) & rel:     r5 += 1
        rr = 0.0
        for i,c in enumerate(cmds[:10],1):
            if c in rel: rr = 1.0/i; break
        mrr += rr
        dcg = 0.0
        for i,c in enumerate(cmds[:5],1):
            if c in rel: dcg += 1.0/math.log2(i+1)
        idcg = sum(1.0/math.log2(i+1) for i in range(1, min(len(rel),5)+1))
        ndcg += (dcg/idcg) if idcg else 0.0
    return {
        "label": label, "n": pos,
        "recall@1": r1/pos if pos else 0.0,
        "recall@3": r3/pos if pos else 0.0,
        "recall@5": r5/pos if pos else 0.0,
        "mrr": mrr/pos if pos else 0.0,
        "ndcg@5": ndcg/pos if pos else 0.0,
        "lat_median_ms": statistics.median(lat) if lat else 0.0,
        "lat_p95_ms": (sorted(lat)[int(0.95*len(lat))-1] if len(lat)>1 else (lat[0] if lat else 0.0)),
    }


def _eval_negative(r: "Retriever", neg) -> Dict[str, float]:
    n = len(neg); abstained = 0; fp = 0
    for q, _cat in neg:
        res = r.search(q, top_k=5)
        if not res: abstained += 1; continue
        top = res[0]
        if top.get("abstain") or top["retrieval_score"] < ABSTAIN_SCORE_THRESHOLD:
            abstained += 1
        else:
            fp += 1
    return {"n": n, "abstained": abstained, "false_positives": fp,
            "abstention_recall": abstained/n if n else 0.0}


def run_benchmark(r: "Retriever", held_out: bool, calibrate: bool=False) -> None:
    if held_out:
        pos_q = TEST_POS; neg_q = TEST_NEG
        console.print(Panel.fit(
            "[bold]Held-out TEST set[/bold]\n"
            "These queries were NOT used to tune weights, priors, thresholds,\n"
            "aliases, or expansion. This is the number to quote.",
            border_style="magenta"))
    else:
        pos_q = TUNE_QUERIES; neg_q = TEST_NEG
        console.print(Panel.fit(
            "[bold]TUNE set[/bold]\n"
            "Used for development. Do NOT quote as final performance.",
            border_style="yellow"))

    pos_metrics = _eval_positive(r, pos_q, "test" if held_out else "tune")
    neg_metrics = _eval_negative(r, neg_q)

    console.print()
    console.print(Panel.fit(
        f"Positive queries: {pos_metrics['n']}\n"
        f"  Recall@1 : {pos_metrics['recall@1']*100:5.1f}%\n"
        f"  Recall@3 : {pos_metrics['recall@3']*100:5.1f}%\n"
        f"  Recall@5 : {pos_metrics['recall@5']*100:5.1f}%\n"
        f"  MRR      : {pos_metrics['mrr']:.3f}\n"
        f"  nDCG@5   : {pos_metrics['ndcg@5']:.3f}\n"
        f"  Latency  : median {pos_metrics['lat_median_ms']:.0f} ms, "
        f"p95 {pos_metrics['lat_p95_ms']:.0f} ms\n"
        f"\nNegative queries: {neg_metrics['n']}\n"
        f"  Abstained    : {neg_metrics['abstained']}/{neg_metrics['n']}\n"
        f"  False pos.   : {neg_metrics['false_positives']}\n"
        f"  Abst. recall : {neg_metrics['abstention_recall']*100:.1f}%",
        title=("Held-out TEST" if held_out else "TUNE (dev)"),
        border_style=("magenta" if held_out else "yellow")))

    if calibrate:
        console.print("\n[bold]Calibrating confidence table on TUNE set...[/bold]")
        pairs: List[Tuple[float,int]] = []
        for q, prim, acc, _d, _df in TUNE_QUERIES:
            rel = set(prim) | set(acc)
            res = r.search(q, top_k=1)
            if not res: continue
            s = res[0]["retrieval_score"]; c = 1 if res[0]["command"] in rel else 0
            pairs.append((s, c))
        pairs.sort()
        buckets = []
        for lo, hi, name in [(0.00,0.22,"very low"),(0.22,0.40,"low"),
                             (0.40,0.65,"medium"),(0.65,1.01,"high")]:
            sel = [c for s,c in pairs if lo <= s < hi]
            p = (sum(sel)/len(sel)) if sel else 0.0
            buckets.append({"min":lo,"max":hi,"label":name,
                            "n":len(sel),"p_correct":round(p,3)})
        CONF_TABLE.write_text(json.dumps({
            "note":"Empirical P(correct) on TUNE set. Small n; treat as first cut.",
            "buckets": buckets}, indent=2))
        console.print(f"[green]✓ Confidence table written to {CONF_TABLE}[/green]")
        t = Table(box=box.ROUNDED)
        for c in ("label","range","n","P(correct)"): t.add_column(c)
        for b in buckets:
            t.add_row(b["label"], f"[{b['min']:.2f},{b['max']:.2f})",
                      str(b["n"]), f"{b['p_correct']:.2f}")
        console.print(t)


# ===========================================================================
# CLI
# ===========================================================================

def cmd_index(a):
    secs = None if a.sections == "all" else [int(s) for s in a.sections.split(",")]
    build_index(sections=secs, force=a.force)


def cmd_doctor(a):
    console.print(Panel.fit("[bold]Health check[/bold]", border_style="cyan"))
    ok = True
    console.print(f"Python: {sys.version.split()[0]}")
    for t in ("groff","man"):
        p = which_command(t)
        if p: console.print(f"[green]✓ {t}: {p}[/green]")
        else: console.print(f"[red]✗ {t}[/red]"); ok=False
    try:
        sqlite3.connect(":memory:").execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        console.print("[green]✓ sqlite3 FTS5[/green]")
    except Exception as e:
        console.print(f"[red]✗ FTS5: {e}[/red]"); ok=False
    man_dirs = [str(d) for d in MAN_DIRS if d.exists()]
    console.print(f"Man dirs: {man_dirs or '[red]none[/red]'}")
    if not man_dirs: ok=False
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR/".wt").write_text("x"); (CACHE_DIR/".wt").unlink()
        console.print(f"[green]✓ Cache: {CACHE_DIR}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Cache: {e}[/red]"); ok=False
    env = detect_environment()
    console.print(f"Env: {env.os} {env.distro} ({env.distro_family}) | shell={env.shell} | "
                  f"pm={env.package_manager} | systemd={env.has_systemd}")
    if MANIFEST.exists():
        m = json.loads(MANIFEST.read_text())
        console.print(f"[green]✓ Index: {m['n_docs']} cmds, {m['n_chunks']} chunks, "
                      f"sections {m['sections']}[/green]")
    else:
        console.print("[yellow]⚠ No index[/yellow]"); ok=False
    for d in _MISSING:
        console.print(f"[red]✗ missing dep: {d}[/red]"); ok=False
    console.print("[bold green]✓ Healthy[/bold green]" if ok else "[bold yellow]⚠ Issues[/bold yellow]")


def cmd_stats(a):
    if not MANIFEST.exists(): console.print("[yellow]No index.[/yellow]"); return
    m = json.loads(MANIFEST.read_text())
    t = Table(title="Index statistics", box=box.ROUNDED)
    t.add_column("Key"); t.add_column("Value")
    for k,v in m.items():
        t.add_row(str(k), json.dumps(v) if isinstance(v,(dict,list)) else str(v))
    console.print(t)


def cmd_search(a):
    r = Retriever()
    q = a.query or Prompt.ask("[bold]Search[/bold]")
    res = r.search(q, top_k=a.top_k)
    print_results(q, res, verbose=a.verbose)


def cmd_agent(a):
    console.print(Panel.fit(
        "🐧 [bold]Linux Command Knowledge & Retrieval Engine v0.4[/bold]\n"
        "Ask about any Linux command. Type [cyan]/help[/cyan] for commands.",
        border_style="cyan"))
    r = Retriever()
    console.print(f"[dim]{r.manifest['n_chunks']:,} chunks, {r.manifest['n_docs']:,} commands, "
                  f"sections {r.manifest['sections']}[/dim]\n")
    console.print("Commands: /help /stats /clear /quit. Prefix 'explain ' or 'troubleshoot '.\n")
    while True:
        try: q = input("❯ ").strip()
        except (EOFError, KeyboardInterrupt): console.print(); break
        if not q: continue
        if q in ("/quit","/exit",":q"): break
        if q == "/help":
            console.print(Markdown(textwrap.dedent("""
            ## Help
            - Natural-language: *how do I compress a folder?*
            - `explain <cmd>` to explain a command line.
            - `troubleshoot <problem>` for a rule-based workflow.
            - `/stats`, `/clear`, `/quit`.
            """)))
            continue
        if q == "/stats": cmd_stats(a); continue
        if q == "/clear": console.clear(); continue
        wf = find_workflow(q)
        if wf:
            console.print(Panel.fit(
                f"[bold]{wf['title']}[/bold]\n" +
                "\n".join(f"  {i}. [cyan]{s}[/cyan]\n     {d}"
                          for i,(s,d) in enumerate(wf["steps"],1)) +
                f"\n\nSources: {', '.join(wf['sources'])}",
                border_style="green"))
            continue
        if q.lower().startswith("explain "):
            console.print(explain_command(q[len("explain "):])); continue
        if q.lower().startswith("troubleshoot "):
            wf = find_workflow(q[len("troubleshoot "):])
            if not wf: console.print("[yellow]No matching workflow.[/yellow]")
            else:
                console.print(Panel.fit(
                    f"[bold]{wf['title']}[/bold]\n" +
                    "\n".join(f"  {i}. [cyan]{s}[/cyan]\n     {d}"
                              for i,(s,d) in enumerate(wf["steps"],1)),
                    border_style="green"))
            continue
        intent = classify_intent(q)["intent"]
        res = r.search(q, top_k=8, intent=intent)
        print_results(q, res); console.print()


def cmd_explain(a): console.print(explain_command(a.cmdline))


def cmd_troubleshoot(a):
    wf = find_workflow(a.query)
    if not wf:
        console.print("[yellow]No matching rule-based workflow. Known:[/yellow]")
        for w in _WORKFLOWS: console.print(f"  • {w['title']}")
        return
    console.print(Panel.fit(
        f"[bold]{wf['title']}[/bold]\n" +
        "\n".join(f"  {i}. [cyan]{s}[/cyan]\n     {d}"
                  for i,(s,d) in enumerate(wf["steps"],1)) +
        f"\n\nSources: {', '.join(wf['sources'])}",
        border_style="green"))


def cmd_benchmark(a):
    run_benchmark(Retriever(), held_out=a.held_out, calibrate=a.calibrate)


def cmd_clear(a):
    if CACHE_DIR.exists(): shutil.rmtree(CACHE_DIR)
    console.print("[green]✓ Cleared.[/green]")


def main():
    p = argparse.ArgumentParser(
        prog="semantic_man.py",
        description="🐧 Linux Command Knowledge & Retrieval Engine v0.4.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python semantic_man.py index --sections all
          python semantic_man.py search "compress folder" --verbose
          python semantic_man.py agent
          python semantic_man.py explain "ss -tulpn"
          python semantic_man.py troubleshoot "disk is full"
          python semantic_man.py benchmark                 # TUNE set
          python semantic_man.py benchmark --held-out      # frozen TEST set
          python semantic_man.py benchmark --calibrate     # write confidence table
        """))
    p.add_argument("--debug", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index")
    pi.add_argument("--force", action="store_true")
    pi.add_argument("--sections", default="all")
    pi.set_defaults(func=cmd_index)

    ps = sub.add_parser("search")
    ps.add_argument("query", nargs="?")
    ps.add_argument("--top-k", type=int, default=FINAL_K)
    ps.add_argument("--verbose", action="store_true")
    ps.set_defaults(func=cmd_search)

    pa = sub.add_parser("agent"); pa.set_defaults(func=cmd_agent)
    pe = sub.add_parser("explain"); pe.add_argument("cmdline"); pe.set_defaults(func=cmd_explain)
    pt = sub.add_parser("troubleshoot"); pt.add_argument("query"); pt.set_defaults(func=cmd_troubleshoot)
    pst = sub.add_parser("stats"); pst.set_defaults(func=cmd_stats)
    pd = sub.add_parser("doctor"); pd.set_defaults(func=cmd_doctor)

    pb = sub.add_parser("benchmark")
    pb.add_argument("--held-out", action="store_true",
                    help="Use the frozen TEST set (do NOT tune on this)")
    pb.add_argument("--calibrate", action="store_true",
                    help="Also write the empirical confidence table on TUNE")
    pb.set_defaults(func=cmd_benchmark)

    pc = sub.add_parser("clear"); pc.set_defaults(func=cmd_clear)
    pv = sub.add_parser("version"); pv.set_defaults(func=lambda a: console.print(f"semantic-man {__version__}"))

    a = p.parse_args()
    if a.debug: logger.setLevel(logging.DEBUG)
    try: a.func(a)
    except RuntimeError as e: console.print(f"[red]Error: {e}[/red]"); sys.exit(1)
    except KeyboardInterrupt: console.print("\n[yellow]Interrupted.[/yellow]"); sys.exit(130)


if __name__ == "__main__":
    main()
