# ALP — Automated Legal Policy Analyzer

A Retrieval-Augmented Generation (RAG) pipeline that leverages multiple large language models (LLMs) to extract structured data from U.S. state renewable energy legislation, then evaluates model accuracy against a curated ground-truth database.

## Overview

Renewable energy policy analysis requires reading through hundreds of state bills and manually extracting key data points — commitment percentages, target dates, eligible technologies, and more. **ALP** automates this process by:

1. **Ingesting** PDF legal documents and splitting them into semantically meaningful chunks.
2. **Embedding** each chunk using OpenAI's `text-embedding-ada-002` model.
3. **Indexing** embeddings in a FAISS vector store for fast similarity search.
4. **Querying** multiple LLMs (GPT-4o, Claude 3.5 Sonnet, Llama 3.1, Mixtral, and others) with retrieved context to answer structured questions about each law.
5. **Evaluating** model responses against ground-truth data using question-type–specific accuracy metrics.

## Architecture

```
Laws/ (PDF documents)
  │
  ▼
┌──────────────┐     ┌────────────────────┐     ┌──────────────┐
│  PDF Loader  │────▶│  Text Splitter     │────▶│  Embeddings  │
│  (PyPDF)     │     │  (Recursive Char)  │     │  (Ada-002)   │
└──────────────┘     └────────────────────┘     └──────┬───────┘
                                                       │
                                                       ▼
                                                ┌──────────────┐
                                                │  FAISS Index │
                                                └──────┬───────┘
                                                       │
                     ┌─────────────────────────────────┤
                     │  Retrieval (top-k similarity)   │
                     ▼                                 ▼
              ┌─────────────┐                   ┌─────────────┐
              │  Questions  │                   │  Retrieved  │
              │  (Excel)    │                   │  Context    │
              └──────┬──────┘                   └──────┬──────┘
                     │                                 │
                     └──────────┬───────────────────────┘
                                ▼
                     ┌─────────────────────┐
                     │   LLM Completion    │
                     │  (Multi-Model)      │
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │  Accuracy Check     │
                     │  (vs Ground Truth)  │
                     └──────────┬──────────┘
                                │
                                ▼
                        results/ (Excel)
```

## Models Supported

| Model | Provider |
|---|---|
| GPT-4o, GPT-4-turbo, GPT-4o-mini, GPT-3.5-turbo | OpenAI |
| Claude 3.5 Sonnet | Anthropic |
| Meta-Llama-3.1-8B-Instruct | Meta (via Hugging Face) |
| Mixtral-8x7B-Instruct | Mistral AI |
| Gemma-1.1-7B-IT | Google |
| Mistral-7B-Instruct | Mistral AI |

## Question Types & Evaluation Metrics

The pipeline asks structured questions about each law and evaluates answers using type-specific metrics:

| Question Type | Example | Metric |
|---|---|---|
| **Numerical** | RPS commitment %, initial commitment | Exact match |
| **Dates** | Year commitment passed into law | Exact year match |
| **Binary** | Voluntary components present? (Yes/No) | Exact match |
| **Categorical** | Eligible energy sources, credit multipliers | Levenshtein similarity with omission/addition penalties |

## Project Structure

```
ALP/
├── main.py                     # Primary RAG pipeline (multi-model, multi-document)
├── accuracy_check.py           # Evaluation module — compares model outputs to ground truth
├── jupyter.py                  # Alternative pipeline script (iterates over Laws/ directory)
├── testing.py                  # Single-document pipeline variant for rapid prototyping
├── requirements.txt            # Python dependencies
├── .env                        # API keys (not committed)
├── Laws/                       # Input PDF documents (86 state energy laws)
├── Questions.xlsx              # Structured questions and prompt engineering instructions
├── Database June 20 2022.xlsx  # Ground-truth database for accuracy evaluation
├── reencoded_col_BD_from_database.xlsx  # Credit multiplier reference data
├── technology_mapping.xlsx     # Technology-to-category mappings
├── faiss_index/                # Pre-built FAISS vector index
├── chroma_db/                  # ChromaDB vector store (alternative backend)
├── results/                    # Model output spreadsheets
└── va-code.pdf                 # Sample Virginia legal code PDF
```

## Getting Started

### Prerequisites

- Python 3.10+
- API keys for one or more LLM providers (see below)

### Installation

```bash
# Clone the repository
git clone https://github.com/dexterfire861/ALP.git
cd ALP

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root with your API keys:

```env
UF_API_KEY=your_uf_api_key
OPENAI_API_KEY=your_openai_api_key
CLAUDE_API_KEY=your_claude_api_key
HUGGING_FACE_API_KEY=your_hugging_face_api_key
```

### Usage

**Run the full pipeline** (processes all PDFs in `Laws/` across all models):

```bash
python main.py
```

Results are saved as Excel files in the `results/` directory, one per law–model combination (e.g., `VA HB 1451_gpt-4o.xlsx`).

**Evaluate accuracy** against ground truth:

```bash
python accuracy_check.py
```

This reads files from `results/` and appends an `Accuracy` column to each spreadsheet.

## Technology Stack

- **LangChain** — Document loading, text splitting, prompt management
- **FAISS** — Vector similarity search for document retrieval
- **LiteLLM** — Unified API for multi-provider LLM completions
- **OpenAI** — Embeddings (`text-embedding-ada-002`) and chat completions
- **Anthropic** — Claude model integration
- **Hugging Face** — Llama model inference
- **Pandas** — Data manipulation and Excel I/O
- **python-dotenv** — Environment variable management

## License

This project is part of academic research at the University of Florida.
