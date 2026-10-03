# Linux Command Knowledge & Retrieval Engine

**semantic-man-agent v0.4.1**

A local semantic search and troubleshooting assistant for Linux command and
manual-page knowledge. It retrieves relevant commands and documentation using
lexical search, dense embeddings, reciprocal-rank fusion, command-level
aggregation, and optional cross-encoder reranking.

## Status

- Release: **v0.4.1**
- Runtime: local/offline after required models and data are present
- Primary corpus: Linux manual pages, sections 1–8
- Dense model: `all-MiniLM-L6-v2`
- Lexical retrieval: SQLite FTS5
- Dense retrieval: NumPy embeddings
- Fusion: reciprocal-rank fusion (RRF)
- Optional reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- CLI includes search, agent, explain, troubleshoot, stats, doctor, and benchmark commands

## Architecture

```text
Linux man pages / local command help
                │
                ▼
        parser + chunker
                │
                ├──────────────► SQLite + FTS5
                │
                └──────────────► sentence embeddings
                                      │
                                      ▼
                              NumPy embedding store
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │       query            │
                         └───────────┬────────────┘
                                     │
                         lexical + dense retrieval
                                     │
                                     ▼
                                   RRF
                                     │
                                     ▼
                         command-level aggregation
                                     │
                                     ▼
                         optional cross-encoder
                                     │
                                     ▼
                              final ranked hits
```

## Installation

```bash
git clone git@github.com:amirradnia99/semantic-man-agent.git
cd semantic-man-agent

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For a local development index, run:

```bash
python semantic_man.py index
```

The index is stored under:

```text
~/.cache/semantic-man/
```

The cache contains the SQLite database, embeddings, embedding IDs, manifest,
confidence metadata, and logs.

## Basic usage

Search for a command:

```bash
python semantic_man.py search "list open network ports"
```

Other examples:

```bash
python semantic_man.py search "find a word in many files"
python semantic_man.py search "show disk usage by directory"
python semantic_man.py search "how do I compress a folder"
```

Inspect the environment and index:

```bash
python semantic_man.py doctor
python semantic_man.py stats
```

Use the assistant-style interface:

```bash
python semantic_man.py agent
# Then enter the query at the interactive prompt, for example:
#   disk is full

python semantic_man.py troubleshoot "the network is slow"
```

Show command/document evidence:

```bash
python semantic_man.py explain "ss -tulpn"
```

Run the held-out benchmark:

```bash
python semantic_man.py benchmark --held-out
```

## Final v0.4.1 evaluation

The release README records the benchmark values from the final benchmark run
used for this release:

| Metric | Result |
|---|---:|
| Recall@1 | 51.2% |
| Recall@3 | 65.1% |
| Recall@5 | 81.4% |
| MRR | 0.612 |
| nDCG@5 | 0.375 |
| Median latency | 1.28 s |
| p95 latency | 1.74 s |
| Negative-query abstention | 100% |
| False positives on negative set | 0/10 |

Benchmark scope:

- 43 positive held-out queries
- 10 negative queries
- Results are for the local v0.4.1 code/index and benchmark set used for the release
- Latency depends on hardware, Python environment, model availability, and index state
- Retrieval scores are ranking signals, not probabilities

## Index contents

The final local build used:

- 10,254 discovered man pages
- 10,180 successfully parsed pages
- 74 parsing failures
- sections 1–8
- 9,187 command entries
- 46,973 chunks
- 73,992 embeddings
- `all-MiniLM-L6-v2`

The corpus is generated from the Linux manual pages available on the indexing
machine. A shipped/frozen release corpus should therefore be treated as a
specific knowledge-base snapshot rather than a universal representation of
every Linux distribution.

## Search design

The search pipeline combines several signals:

1. Query expansion and intent classification.
2. SQLite FTS5 lexical retrieval.
3. Dense embedding retrieval.
4. Reciprocal-rank fusion.
5. Command-level aggregation and deduplication.
6. Section and domain priors.
7. Optional cross-encoder reranking.
8. Exact-name and environment-aware signals.
9. Abstention for low-confidence results.
10. Evidence display for explainability.

The confidence bands are heuristic. They are not statistically calibrated
probabilities.

## Environment awareness

The tool can inspect local environment metadata such as:

- Linux distribution family
- shell
- package manager
- systemd availability
- installed commands

Environment information can influence retrieval metadata and ranking, but the
knowledge base remains a local corpus snapshot.

## Safety

The assistant does not execute shell commands as part of normal search or
troubleshooting. It returns documentation, command suggestions, and evidence
for the user to inspect.

Users should review commands before running them, especially commands that
modify filesystems, permissions, networking, packages, services, or system
configuration.

## Portable/fixed knowledge base

For a reproducible release, keep the generated knowledge base separate from
source code:

```text
semantic-man
semantic-man.data/
    index.sqlite
    embeddings.npy
    embedding_ids.json
    manifest.json
```

The current development cache is approximately 200 MB. A future standalone
single-file executable will also need to package the Python runtime and ML
dependencies, so its final size will be larger than the database alone.

Large generated binary indexes should generally be distributed as release
assets or through Git LFS rather than committed directly to the normal Git
history.

## Project layout

```text
semantic-man-agent/
├── semantic_man.py
├── install.sh
├── requirements.txt
├── README.md
├── .gitignore
└── venv/                 # local development only; not committed
```

## Development checks

```bash
python3 -m py_compile semantic_man.py
bash -n install.sh
python semantic_man.py doctor
python semantic_man.py benchmark --held-out
```

## Known limitations

- Dense retrieval currently uses a NumPy-backed search path rather than a
  specialized approximate-nearest-neighbor index.
- Cross-encoder reranking can require downloading its model the first time.
- The benchmark is a held-out regression set, not a universal measure of Linux
  command-search quality.
- The corpus is based on the manual pages available during indexing.
- Section coverage is currently sections 1–8.
- Incremental indexing depends on stable page/chunk identity and manifest
  metadata.
- Confidence bands are heuristic rather than statistically calibrated.
- Search ranking can still place a related command above the most canonical
  command for some natural-language queries.

## License

See the repository's license file for the applicable project license.
