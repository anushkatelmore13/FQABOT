# Document FAQ Assistant

A Streamlit-based Retrieval-Augmented Generation (RAG) FAQ bot that answers questions from a single uploaded document using real Gemini or OpenAI API keys.

This project was completed for the AIML/DA intern challenge by fixing the broken RAG pipeline, adding extra file-format support, and improving the frontend.

## Features

- Upload and query real documents.
- Supports PDF, TXT, Markdown, and CSV files.
- Splits documents into overlapping chunks for better context coverage.
- Retrieves the most relevant chunks using cosine similarity.
- Generates grounded answers using Gemini or OpenAI.
- Shows retrieved source chunks for traceability.
- Includes unit tests for extraction, chunking, and retrieval.

## Pipeline Fixes

The original repository had three intentional bugs:

1. Chunking skipped text instead of creating overlapping windows.
2. Retrieval returned the least relevant chunks first.
3. The LLM prompt was sent without retrieved context.

All three issues are fixed.

## Added Feature

Markdown and CSV upload support was added.

- Markdown files are indexed as readable document text.
- CSV files are converted into row-based text so tabular FAQ/policy data can be queried.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Add a real API key:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
```

Or use OpenAI:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
```

Do not commit real API keys to GitHub.

## Run The App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Run Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Expected result:

```text
Ran 10 tests
OK
```

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── .env.example
├── data/
│   └── sample_docs/
├── src/
│   ├── chunk.py
│   ├── embed.py
│   ├── extract.py
│   ├── generate.py
│   └── retrieve.py
└── tests/
    ├── test_chunk.py
    ├── test_extract.py
    └── test_retrieve.py
```

## GitHub Push Commands

If this folder is not already connected to GitHub, create an empty GitHub repository first. Then run:

```bash
git init
git add .
git commit -m "Fix RAG pipeline and add document FAQ frontend"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

If the repository already has a remote:

```bash
git add .
git commit -m "Fix RAG pipeline and add document FAQ frontend"
git push
```

## Notes

- The app requires a real Gemini or OpenAI API key for answer generation.
- Mock data or mock answers are not used in the main app flow.
- Retrieved chunks are displayed in the UI so answers can be checked against the uploaded document.
