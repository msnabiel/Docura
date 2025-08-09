# 🔍 RAG-based Agentic FAISS Pipeline

An advanced **Retrieval-Augmented Generation (RAG)** pipeline with **agentic reasoning capabilities** for high-accuracy document search and contextual response generation.

## 📌 **Live Demo & Documentation**
🚀 **Live Website:** **https://docura-ai.vercel.app**  
_This site includes interactive UI, API documentation, and step-by-step guides._

![Main Dashboard](./images/dashboard.png)

## 🚀 Key Features

- **Multi-Embedding Hybrid Retrieval** (BGE EN v1.5 + all-MiniLM-L6-v2)
- **RFF Fusion Framework** for optimized retrieval blending
- **FAISS Vector Store** with scalable indexing
- **Agentic Reasoning** with structured JSON outputs
- **Multi-format Document Support** (PDF, DOCX, PPTX, etc.)

## 🏗 System Architecture and Flow

![System Architecture](./images/perfecto2.png)


## 📊 Retrieval Components

| Component | Model | Purpose | Weight |
|-----------|-------|---------|--------|
| Dense Embeddings | BGE EN v1.5 | Semantic similarity | 0.65 |
| Light Embeddings | all-MiniLM-L6-v2 | Fast retrieval | 0.20 |
| Keyword Search | BM25 | Exact matching | 0.15 |

## 🚀 How to Run

### Step 1: Environment Setup
```bash
# Clone repository
git clone https://github.com/msnabiel/Docura.git
cd Docura

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment
Create `.env` file:
```env
GEMINI_API_KEY_PAID=your_gemini_key
```
### Step 3: Start the Application
```bash
# Start API server
python app.py
# or
uvicorn app:app --reload --port 8000
```

### Step 4: Access the Interface
Open API at for detailed information: 
```bash 
http://localhost:8000
```

## 📈 Application Interface
![Application Interface](./images/chat_interface.png)

## 📈 Performance Metrics

![Performance Chart](./docs/images/performance_metrics.png)

| Metric | Single Dense | Hybrid (RFF) |
|--------|--------------|---------------|
| Recall@5 | 0.78 | **0.92** |
| MRR@10 | 0.64 | **0.85** |
| Response Time | 150ms | 220ms |

## 🛠 Development

### Project Structure
```
├── embeddings/        # Multi-embedding models
├── retrieval/         # FAISS + BM25 + ORFF
├── agents/            # Agentic reasoning
├── api/               # FastAPI endpoints
├── .env.example
├── requirements.txt   # Dependencies
├── images/
├── data/              # Sample documents
└── tests/             # Unit tests
```

## 📋 Supported Formats

- **Text**: PDF, DOCX, TXT, MD
- **Presentations**: PPTX, PPT
- **Spreadsheets**: XLSX, CSV
- **Web**: HTML, XML

## 🔮 Roadmap

- [ ] Multi-language embedding support
- [ ] Cross-encoder re-ranking
- [ ] Streaming responses
- [ ] GraphRAG integration

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/username/repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/msnabiel/repo/discussions)
- **Email**: msyednabiel@gmail.com