# 🧠 Attentionist

**Attentionist** is a Retrieval-Augmented Generation (RAG) assistant designed to answer questions about transformer architectures and attention mechanisms. It retrieves relevant passages from a curated collection of nine foundational research papers and uses **Google Gemini** to generate grounded, source-backed responses, helping reduce hallucinations and improve factual accuracy.

This project was developed as the final project for the **DataTalksClub LLM Zoomcamp 2026**.

---

# ✨ Features

* 📚 Knowledge base built from **9 influential transformer and attention research papers**
* 🔍 Semantic retrieval using vector embeddings
* 🤖 Grounded answer generation with **Google Gemini**
* 💬 Interactive chat interface built with **Streamlit**
* 📖 Source citations for every generated response
* 👍👎 User feedback collection for response quality
* 📊 SQLite-based interaction monitoring
* 📈 Retrieval and generation evaluation pipeline
* 🐳 Dockerized deployment for reproducible execution

---

# 📄 Knowledge Base

Attentionist retrieves information from the following research papers:

* Attention Is All You Need
* BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
* Language Models are Few-Shot Learners (GPT-3)
* An Image is Worth 16×16 Words (Vision Transformer)
* Reformer
* Longformer
* Performer
* RoFormer
* Mamba

---

# 🏗️ Project Architecture

```text
                   User
                     │
                     ▼
            Streamlit Chat Interface
                     │
                     ▼
               User Question
                     │
                     ▼
          Query Embedding Generation
                     │
                     ▼
          Vector Database Retrieval
                     │
                     ▼
         Top-k Relevant Document Chunks
                     │
                     ▼
      Prompt Construction + Retrieved Context
                     │
                     ▼
               Google Gemini
                     │
                     ▼
        Grounded Answer + Source Chunks
                     │
                     ▼
      Interaction Logging & User Feedback
```

---

# ⚙️ Tech Stack

* Python
* Streamlit
* Google Gemini API
* ChromaDB (Vector Database)
* SQLite
* Docker
* Pandas
* NumPy

---

# 📁 Repository Structure

```text
Attentionist/
│
├── app/                     # Streamlit application
├── src/                     # Core RAG pipeline
│   ├── ingest.py
│   ├── embedder.py
│   ├── retrieval.py
│   ├── llm.py
│   └── rag.py
│
├── eval/                    # Evaluation pipeline
│   ├── generate_ground_truth.py
│   ├── evaluate_retrieval.py
│   ├── generate_rag_answers.py
│   ├── generate_norag_answer.py
│   ├── judge_answer.py
│   ├── select_judge_sample.py
│   └── results/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── Dockerfile
├── .dockerignore
├── .env.example
├── monitoring.py
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/malakhishams/Attentionist.git
cd Attentionist
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Or, on Windows, manually create a `.env` file based on `.env.example`.

Then add your Gemini API key:

```text
GEMINI_API_KEY=your_api_key_here
```

---

# ▶️ Running the Application

Launch the Streamlit interface:

```bash
streamlit run app/app.py
```

Then open the local URL displayed in your terminal (typically `http://localhost:8501`).

---

# 🐳 Running with Docker

Build the Docker image:

```bash
docker build -t attentionist .
```

Run the container:

```bash
docker run -p 8501:8501 -e GEMINI_API_KEY=your_api_key_here attentionist
```

Then open:

```text
http://localhost:8501
```

Docker packages the application and all required dependencies into a portable container, ensuring the project runs consistently across different environments without requiring manual dependency installation.

---

# 📊 Evaluation Pipeline

The repository includes a complete evaluation workflow for both retrieval quality and answer quality.

### Generate Ground Truth

```bash
python eval/generate_ground_truth.py
```

### Evaluate Retrieval

Measures retrieval performance using metrics such as **Hit Rate** and **Mean Reciprocal Rank (MRR)**.

```bash
python eval/evaluate_retrieval.py
```

### Generate RAG Answers

```bash
python eval/generate_rag_answers.py
```

### Generate Baseline (Without RAG)

```bash
python eval/generate_norag_answer.py
```

### LLM-as-a-Judge Evaluation

```bash
python eval/judge_answer.py
```

Evaluation outputs are stored in:

```text
eval/results/
```

---

# 📈 Monitoring

Every interaction is automatically logged into a lightweight SQLite database.

For each conversation, the system stores:

* Timestamp
* User question
* Generated answer
* Retrieved source documents
* Number of retrieved chunks
* Answer length
* User feedback (👍 / 👎)

These logs can be used to monitor system usage and evaluate response quality over time.

---

# 💬 Example Questions

* What is self-attention?
* Explain multi-head attention.
* Why does BERT use masked language modeling?
* Compare BERT and GPT-3.
* How does Longformer reduce the complexity of self-attention?
* What are the key innovations introduced by Mamba?
* Which papers address the quadratic complexity of transformers?

---

# 📌 Future Improvements

* Hybrid keyword + semantic retrieval
* Retrieval reranking
* Conversation memory
* Streaming responses
* Support for additional transformer research papers
* Cloud deployment
* CI/CD pipeline with GitHub Actions

---

# 👩‍💻 Author

**Malak Hisham**

Computer Science Student at Ain Shams University

AI & Machine Learning Enthusiast

GitHub: https://github.com/malakhishams

---

# 🙏 Acknowledgements

* DataTalksClub LLM Zoomcamp
* Google Gemini
* Hugging Face
* ChromaDB
* Streamlit
* The authors of the transformer and attention research papers that form the project's knowledge base.
