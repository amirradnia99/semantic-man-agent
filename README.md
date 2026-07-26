cat > README.md << 'EOF'
# Semantic search over Linux man pages using natural language

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

## Overview

The Semantic Man-Page Agent is a production-grade semantic search engine for Linux manual pages. Instead of relying on keyword matching (like man -k), it uses machine learning embeddings to understand the meaning of your queries and find the most relevant Linux commands.

### Key Features

- Natural Language Search - Ask "how do I compress a folder?" instead of remembering command names
- Semantic Understanding - Finds commands based on intent, not just keywords
- Interactive Agent - Chat-like interface for exploring commands
- Relevance Scoring - Shows how well each command matches your query (0-100%)
- Robust Parser - Handles man pages with inconsistent formatting
- Persistent Index - Build once, search instantly
- Health Checks - semantic-man doctor verifies your system is ready

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Linux system with man pages installed (typically under /usr/share/man)
- ~500 MB disk space for the vector index

### Installation

```bash
# Clone the repository
git clone https://github.com/amirradnia99/semantic-man-agent.git
cd semantic-man-agent

# Make the installer executable and run it
chmod +x install.sh
./install.sh

## License

- Code: MIT License
- Source corpus: external dataset under original license
- Derived inventories: Silver-standard research artifact

## Contact

Repository:
https://github.com/amirradnia99/semantic-man-agent

