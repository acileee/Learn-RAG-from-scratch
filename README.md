# Learn RAG From Scratch

## Overview

This project is a learning-focused local Retrieval-Augmented Generation system built from scratch.

Each major RAG component is implemented manually first so the internals are visible before comparing the system with tools like Chroma, Ollama, LangChain, Graph-RAG, and Self-RAG.

The project takes documents, splits them into chunks, embeds those chunks, retrieves relevant context for a user question, checks whether the retrieved context is strong enough, and then uses a local Ollama model to generate an answer grounded in the retrieved text.

It also includes retrieval evaluation, Chroma vector storage, query rewriting, multi-query retrieval, reranking, Graph-RAG expansion, a simple Self-RAG retry loop, comments for learners, and a learning guide.

## Learning Guide

This repo includes a [`LEARNING_GUIDE.md`](LEARNING_GUIDE.md) file for people who want to understand the project step by step.

The guide explains:

* what RAG is
* how the pipeline works
* what each major file does
* how to read the code comments
* debugging tips
* examples of common issues
* suggested learning paths through the codebase

If you are learning RAG from the ground up, start with the learning guide first.

## Features

* Document chunking with overlap
* Local embeddings
* Manual semantic search
* Retriever evaluation
* Chunk size and top-k experiments
* JSON storage utilities
* Chroma vector database support
* RAG prompt construction
* Query rewriting
* Multi-query retrieval
* Token-overlap reranking
* Answerability gate
* Local Ollama answering
* End-to-end local RAG pipeline
* RAG evaluation
* LangChain rebuild
* Entity extraction
* Knowledge graph construction with NetworkX
* Graph-RAG expansion
* Self-RAG retry loop
* Automated tests
* Learning-focused comments and examples

## Project Structure

```text
src/
  chunk_documents.py        # Splits raw documents into chunks
  local_embeddings.py       # Embeds text and chunks locally
  semantic_search.py        # Manual vector similarity search
  evaluate_retrieval.py     # Retrieval evaluation utilities
  rag_pipeline.py           # Formats context and builds RAG prompts
  storage.py                # Save/load JSON helpers
  chroma_store.py           # Chroma client, collection, upsert, and search
  query_rewriting.py        # Query rewriting and query variants
  multi_query.py            # Multi-query retrieval
  reranking.py              # Token-overlap reranking
  answerability.py          # Checks whether retrieved chunks are strong enough
  local_llm.py              # Calls Ollama and answers from RAG input
  end_to_end_rag.py         # Full documents-to-answer pipeline
  rag_evaluation.py         # End-to-end RAG evaluation
  langchain_basic_rag.py    # LangChain version of the RAG pipeline
  kg_extraction.py          # Rule-based entity extraction
  kg_graph.py               # NetworkX knowledge graph helpers
  graph_rag.py              # Graph-RAG retrieval expansion
  self_rag.py               # Self-RAG retry loop
```

## Pipeline

The core pipeline is:

```text
documents
→ chunks
→ embeddings
→ retrieval
→ context formatting
→ prompt
→ local LLM answer
```

The full pipeline adds stronger retrieval and safety checks:

```text
documents
→ chunk documents
→ embed chunks
→ store/search with Chroma
→ rewrite query if needed
→ retrieve chunks
→ rerank chunks
→ check answerability
→ build prompt
→ call Ollama
→ return grounded answer or fallback
```

## How It Works

### 1. Chunking

Documents are split into smaller chunks so retrieval can return focused pieces of context instead of entire documents.

Chunk overlap is used so important information is not lost at chunk boundaries.

### 2. Embeddings

Each chunk is converted into an embedding vector.

The query is also embedded, and semantic search compares the query vector to chunk vectors.

This makes retrieval semantic instead of only keyword-based.

### 3. Retrieval

The retriever returns the top-k chunks most relevant to the question.

Manual semantic search was implemented first, then Chroma was added as a persistent vector database.

### 4. Prompting

Retrieved chunks are formatted into context and inserted into a RAG prompt.

The prompt tells the model to answer only using the provided context and say:

```text
I don't know based on the provided context.
```

if the context does not contain the answer.

### 5. Answerability Gate

Before calling the LLM, the system checks whether retrieved chunks are strong enough.

It uses:

* retrieval score
* token overlap between question and chunk text

If the evidence is weak, the system returns the fallback answer instead of forcing the model to answer.

### 6. Local LLM

The project uses Ollama to run a local model such as:

```bash
ollama pull llama3.2:1b
```

The local LLM receives the RAG prompt and generates the final answer.

## Advanced Retrieval

### Query Rewriting

The system rewrites queries by expanding abbreviations:

```text
RAG → retrieval augmented generation
LLM → large language model
AI → artificial intelligence
KG → knowledge graph
```

This improves retrieval when the documents use expanded terms instead of abbreviations.

### Multi-Query Retrieval

Instead of searching once, the system searches with multiple query versions:

```text
original question
rewritten query
simple keyword query
```

Results are merged and deduplicated by `chunk_id`.

### Reranking

After retrieval, chunks are reranked using a simple token-overlap score.

The final rerank score combines:

```text
original retrieval score + overlap score
```

This rewards chunks that are both semantically similar and contain concrete query terms.

### Graph-RAG

The system extracts entities from chunks and builds a NetworkX knowledge graph.

The graph connects:

```text
chunk nodes → entity nodes
entity nodes → co-occurring entity nodes
```

Graph-RAG expands retrieval by:

```text
retrieved chunk
→ connected entities
→ other chunks connected to those entities
```

This can add useful related context that vector search alone may miss.

### Self-RAG

The Self-RAG loop retries retrieval if the first attempt is not answerable.

Flow:

```text
try original question
check answerability
if weak, rewrite query
try again
answer or return fallback
```

This adds a simple agentic control loop around retrieval.

## Evaluation

The project includes retrieval and full RAG evaluation.

Retrieval evaluation checks:

* whether the expected document was retrieved
* retrieval accuracy
* false retrievals
* failures

RAG evaluation checks:

* retrieval success
* answerability prediction
* whether an answer was produced
* whether unanswerable questions correctly return the fallback answer

Example summary metrics:

```text
overall_accuracy
retrieval_accuracy
answerability_accuracy
```

## LangChain Rebuild

After building the system manually, the project rebuilds the basic RAG pipeline using LangChain.

LangChain components used:

* `Document`
* `RecursiveCharacterTextSplitter`
* `Chroma`
* `OllamaEmbeddings`
* `OllamaLLM`
* retriever wrapper

This shows how the manual pipeline maps to framework abstractions.

## Setup

Clone the repository:

```bash
git clone https://github.com/acileee/learn-rag-from-scratch.git
cd learn-rag-from-scratch
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install Ollama models:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

Start Ollama:

```bash
ollama serve
```

In another terminal, run the project:

```bash
python src/end_to_end_rag.py
```

## Example Usage

```python
from end_to_end_rag import answer_from_documents

documents = [
    {
        "doc_id": "rag",
        "filename": "rag.txt",
        "text": "RAG retrieves relevant chunks before generation. It helps language models answer using external context.",
    },
    {
        "doc_id": "kg",
        "filename": "kg.txt",
        "text": "Knowledge graphs store entities and relationships.",
    },
]

result = answer_from_documents(
    question="What does RAG retrieve before generation?",
    documents=documents,
    collection_name="demo_collection",
    chunk_size=120,
    overlap=20,
    top_k=3,
    model="llama3.2:1b",
)

print(result["answer"])
```

## Debugging Tips

For detailed debugging tips and step-by-step checks, please see the [`LEARNING_GUIDE.md`](LEARNING_GUIDE.md) file.

It includes guidance for:

* checking Ollama
* checking installed models
* debugging embeddings
* diagnosing retrieval issues
* understanding Chroma behavior
* debugging answerability checks
* inspecting prompts
* reading evaluation results
* understanding common setup problems

## Limitations

* Entity extraction is rule-based and simple.
* Relationship extraction is based on co-occurrence, not true semantic relations.
* Answerability checking is heuristic.
* Reranking uses token overlap, not a trained cross-encoder.
* Graph-RAG expansion can add noisy chunks.
* Evaluation data is small and manually defined.
* Local answer quality depends on the Ollama model used.
* Chroma scores and manual semantic scores may not be perfectly comparable.

## Future Improvements

* Add hybrid BM25 + vector search.
* Use a stronger embedding model.
* Add a cross-encoder reranker.
* Use LLM-based entity and relation extraction.
* Add citations to generated answers.
* Add a FastAPI endpoint.
* Add a CLI for indexing and querying documents.
* Add better evaluation datasets.
* Expand the test suite.
* Add streaming responses from Ollama.
* Add document upload support.

## Learning Process

This project was built as a hands-on learning project, not from a copied template.

I implemented each major RAG component manually first so I could understand how the pipeline works under the hood before comparing it with tools like Chroma, Ollama, and LangChain.

AI tools were used as a learning assistant during the process: to clarify concepts, support debugging, add explanatory comments, help create tests, and refine documentation.

The core implementation was built, reviewed, and tested step by step as part of the learning process.

## Using This Repo

If this repo helps you learn RAG or if you use it as a reference for your own project, please consider linking back to it so more people can find it and learn from it too.

## To Summarize

This project shows the full RAG stack from first principles:

```text
chunking
→ embeddings
→ retrieval
→ prompt construction
→ answer generation
→ evaluation
→ advanced retrieval
```

It also compares manual RAG implementation with LangChain and extends the system with graph-based retrieval and a Self-RAG retry loop.
