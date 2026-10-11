# langchain-pdf-chatbot
PDF question answering chatbot built with LangChain, Ollama, FAISS and Streamlit.
# PDF Question-Answering Chatbot

A conversational PDF chatbot built with **LangChain, Streamlit, Ollama, HuggingFace embeddings and FAISS**. Upload a PDF and ask questions about its content using a locally running language model.

## Features

- Upload a PDF and extract its text.
- Split documents into chunks for retrieval.
- Generate text embeddings using HuggingFace.
- Store and retrieve relevant chunks using FAISS.
- Generate answers grounded in the uploaded PDF using Retrieval-Augmented Generation (RAG).
- Handle conversational follow-up questions using question rewriting.
- Continue general conversation when no PDF is uploaded.
- Clear the chat history.

## Tech Stack

- **Python** — application logic
- **Streamlit** — user interface
- **LangChain** — LLM orchestration and RAG pipeline
- **Ollama** — local Llama 3.2 language model
- **HuggingFace Sentence Transformers** — text embeddings
- **FAISS** — vector similarity search
- **PyPDF** — PDF loading and text extraction

## How It Works

1. Upload a PDF.
2. Extract the document text.
3. Split the text into overlapping chunks.
4. Generate embeddings and store them in FAISS.
5. Rewrite follow-up questions into standalone questions.
6. Retrieve relevant chunks and pass them to the language model.
7. Generate an answer using the retrieved context.

## Installation

### Prerequisites

- Python installed
- Ollama installed
- Git (optional, for cloning the repository)

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Atharvaweb/langchain-pdf-chatbot.git
   cd langchain-pdf-chatbot
   ```

2. Create and activate a virtual environment.

   **Windows:**

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the language model:

   ```bash
   ollama pull llama3.2:3b
   ```

5. Start the chatbot:

   ```bash
   streamlit run app.py
   ```

## Current Limitations

- PDF answers depend on the quality of extracted text and retrieved chunks.
- Scanned PDFs may require OCR support.
- Running the language model locally requires sufficient system resources.
- The project has not yet been configured for public cloud deployment.

## Future Improvements

- Evaluate retrieval and answer quality across different documents.
- Improve source attribution and error handling.
- Add automated tests.
- Deploy a hosted version with a suitable model backend.
