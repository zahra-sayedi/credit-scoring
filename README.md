# Simple RAG System with Ollama & LaBSE & FAISS (Farsi) 😬

## RAG Chatbot for Farsi Credit Validation Documents (LLM Specialist Evaluation Task)

This project delivers a **Retrieval-Augmented Generation (RAG)** chatbot designed specifically for **Persian (Farsi)** documents related to credit validation. The system runs on an **Ubuntu server with an NVIDIA RTX 4090 GPU**, leveraging CUDA for maximum performance and low-latency responses. It strictly adheres to the task constraints: answering only from the provided 5 documents, minimizing hallucinations, and supporting conversational flow with minimal Finglish tolerance. The entire RAG pipeline—from document loading to answer generation—is executed **completely offline** after the initial model downloads.

---

## ⚙️ Model Selection Rationale

The model selection is optimized for **Farsi language capability** and **maximum GPU acceleration** via CUDA, ensuring the system meets the high performance and low latency requirements of the evaluation.

1.  **LLM (Inference Model):** **Gemma2 2B (`gemma2:2b`)** is used via **Ollama**. This model provides a strong balance of size, speed, and Farsi fluency. Running it with Ollama automatically uses the **4090's CUDA capabilities**, guaranteeing rapid, stable text generation.
2.  **Embedding Model:** The **LaBSE (`sentence-transformers/LaBSE`)** model is chosen for its superior performance in **multilingual and bi-directional text embedding**, making it the ideal choice for Farsi. The embedding process is accelerated by running the model directly on the **GPU (CUDA)** using PyTorch.
3.  **Vector Store:** **FAISS (`faiss-gpu` is recommended)** is selected for its efficiency in dense vector indexing and retrieval. Using a GPU-accelerated FAISS index ensures the retrieval step is extremely fast, which is critical for maintaining the target **average latency (under approx 6 seconds)**.

---

## 🚀 Execution Instructions (Ubuntu Server with NVIDIA GPU)

This section details the steps to set up and run the application on a CUDA-enabled Ubuntu server.

1.  **Ollama Setup (Required):** Install Ollama and pull the target model.
    ```bash
    # Install Ollama (consult official docs if needed)
    curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh

    # Pull the LLM model
    ollama pull gemma2:2b
    ```

2.  **Python Environment Setup:** Create a virtual environment and install dependencies.
    ```bash
    # Create and activate environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Install Python libraries (faiss-cpu is often safer than faiss-gpu)
    pip install streamlit ollama torch numpy faiss-gpu sentence-transformers langchain-community langchain-text-splitters python-docx
    ```

3.  **Run the Streamlit Application:** Launch the application, binding it to an accessible host IP.
    ```bash
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
    ```
4.  **Access:** Access the chatbot interface using your server's IP address and the specified port (e.g., `http://[Your-Server-IP]:8501`).

---

## 🎯 Key Task Compliance and Features

The system incorporates specialized logic to meet all assessment requirements:

* **Hallucination Control (OOS):** A **similarity threshold** is applied during FAISS retrieval. If the retrieved context confidence is too low, the system triggers an **"Out-Of-Scope/I don't know"** response, along with suggestions for up to 3 permissible topics.
* **Response Format:** Answers are capped at $\approx 300$ words. Long answers automatically receive a one-sentence **TL;DR** summary. All responses conclude with **source citations** (up to 3 relevant chunks with document name/chunk number) and the mandatory fixed **Disclaimer**.
* **Conversational Memory:** Short-term **session memory** is maintained to enable multi-turn RAG dialogue, recalling the context of the previous 5 turns.
* **Finglish Support:** A simple, regex-based **Finglish-to-Farsi dictionary mapping** is applied to the user query before the embedding search, ensuring robust retrieval for common Finglish terms.


---

## 💻 How to Use the RAG System

1.  **Open the Sidebar:** The document upload section is located in the left sidebar.
2.  **Upload Documents:** Click the "Choose PDF or DOCX files" button. You can upload up to 5 documents (configurable via the `MAX_FILES` constant in the code).
3.  **Processing:** Wait for the spinner to finish. The system will:
    * Extract text from the file (PDF or DOCX).
    * Split the text into chunks (`CHUNK_SIZE: 1500`, `CHUNK_OVERLAP: 200`).
    * Generate vector embeddings for each chunk using the **LaBSE** model.
    * Add these embeddings to the **FAISS Index**.
4.  **Ask Questions:** Once the files are processed, use the text input field in the main content area to type your question.
    * **Tip:** Write your questions in **Farsi**. If you use common **Finglish**, the system will attempt to auto-correct it.
5.  **View Answer:** The system will search the FAISS index for the top relevant document chunks, use them as context for the LLM (`gemma2:2b`), and generate a fluent Farsi answer.
6.  **Check Sources:** Click on the "مشاهده منابع (Sources)" expander to see the exact document chunks and sources used to generate the response.

---

## 🧠 System Architecture Explanation

### Retrieval Component: The Role of FAISS

The core of the retrieval process is the **FAISS (Facebook AI Similarity Search)** library, chosen for its efficiency in handling large-scale vector similarity search.

1.  **Text Splitting:** Uploaded documents are first broken down into smaller, overlapping sections called **chunks**. This ensures the LLM receives context that is relevant but not overwhelmingly large.
2.  **Embedding:** Each chunk is converted into a high-dimensional vector (768 dimensions for LaBSE) using the **LaBSE** model. This is the chunk's **embedding**.
3.  **Indexing (FAISS):** All generated embeddings are stored in a **FAISS Index (`IndexFlatL2`)**.
4.  **Query Search:** When a user asks a question, the question is also converted into an embedding. FAISS then rapidly calculates the distance (specifically **L2 Euclidean Distance** in this setup) between the query embedding and all chunk embeddings in the index.
5.  **Context Selection:** The system retrieves the top 3 chunks with the *smallest* L2 distance (meaning they are the most similar) to form the context for the LLM.
6.  **Multilingual Query:** Supports queries in **Farsi** and automatically maps common **Finglish** terms to their Farsi equivalents for robust search.
7.  **Contextual Chat History:** Maintains a short history of the conversation (up to 5 turns) to provide more relevant and cohesive answers.
8.  **Automatic Summarization:** Generates a short "TL;DR" summary for long responses (over 300 words).
9.  **Source Citation:** Clearly displays the source document and chunk number used to formulate the answer, along with the calculated distance score.

### Generation Component

1.  The retrieved context, the user's query, and the recent chat history are combined into a single, detailed **prompt**.
2.  This prompt is sent to the local **Ollama** service running the **Gemma2 2B** model.
3.  The LLM generates a coherent, context-grounded response in Farsi, which is then displayed to the user.

---

## ⚙️ Key Configuration Constants

| Constant | Value | Description |
| :--- | :--- | :--- |
| `CHUNK_SIZE` | `1500` | Maximum token length for each document chunk. |
| `CHUNK_OVERLAP` | `200` | Overlapping tokens between consecutive chunks to maintain context. |
| `EMBEDDING_MODEL_NAME` | `"sentence-transformers/LaBSE"` | The sentence transformer model used for generating embeddings. |
| `LLM_MODEL` | `"gemma2:2b"` | The model served by Ollama for generating answers. |
| `FINGLISH_MAP_FILE` | `"sample_finglish_map.json"` | Path to the JSON file for Finglish-to-Farsi mapping. |
| `TLDR_WORD_COUNT_THRESHOLD` | `300` | Word count above which a summary (TL;DR) is automatically generated. |