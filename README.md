# 🤖 বাংলা FAQ চ্যাটবট (Bangla FAQ Chatbot)

A Retrieval-Augmented Generation (RAG) based FAQ chatbot supporting Bangla language with automatic category detection and metadata filtering.

## ✨ Features

- **5 Bangla Topics**:  শিক্ষা (Education), স্বাস্থ্য (Health), ভ্রমণ (Travel), প্রযুক্তি (Technology), খেলাধুলা (Sports)  
- **RAG Pipeline**:  FAISS vector store with semantic search  
- **AI Category Routing**: Automatic question classification using LLM  
- **Metadata Filtering**: Filter by topic and difficulty level 
- **Fallback Handling**: Graceful responses for out-of-scope questions  
- **Chat History**: Track all conversations  
- **Streamlit UI**: Clean, interactive web interface

## ⚡ Quick Start

### 1. Setup Virtual Environment

**Linux/Mac:**
```bash
chmod +x setup_venv. sh
./setup_venv.sh
source venv/bin/activate
```

**Windows:**
```cmd
setup_venv.bat
venv\Scripts\activate.bat
```

**Manual Setup:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate.bat  # Windows
pip install -r requirements.txt
```

### 2. Configure GitHub Token

Create `.env` file in project root:
```bash
GITHUB_TOKEN=your_github_token_here
```

**Get your token:**
1. Visit https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select `repo` scope
4. Copy and paste into `.env`

### 3. Run Application

```bash
streamlit run app.py
```

App opens at:  `http://localhost:8501`

## 📝 Example Questions

| Category | Example Question |
|----------|-----------------|
| **শিক্ষা** | বিশ্ববিদ্যালয়ে ভর্তির জন্য কী প্রয়োজন? |
| **স্বাস্থ্য** | রক্তচাপ নিয়ন্ত্রণের জন্য কী করব? |
| **ভ্রমণ** | কক্সবাজারে যেতে কত খরচ হবে? |
| **প্রযুক্তি** | স্মার্টফোনের ব্যাটারি কীভাবে বাঁচাব? |
| **খেলাধুলা** | বাংলাদেশ কবে ICC ট্রফি জিতেছে? |

## 🎯 How It Works

1. **User Input**: Ask question in Bangla
2. **Category Detection**: LLM classifies question into 5 topics
3. **Metadata Filtering**: Filter FAQs by detected category
4. **Semantic Search**: FAISS finds top-3 relevant FAQs
5. **Answer Generation**: LLM generates contextual answer
6. **Display**: Show answer with retrieved context

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **UI Framework** | Streamlit |
| **Vector Store** | FAISS |
| **RAG Framework** | LangChain |
| **Embeddings** | HuggingFace (`l3cube-pune/bengali-sentence-similarity-sbert`) |
| **LLM** | OpenAI API (via GitHub Models - `gpt-4.1-nano`) |
| **Language** | Python |

## 📁 Project Structure

```
bangla-faq-chatbot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── .gitignore             # Git ignore file
└── README.md  
```

## 🎨 UI Features

- **Auto/Manual Mode**: Choose automatic category detection or manual selection
- **Live Category Display**: See detected category for each question
- **Retrieved Context**: View relevant FAQs used for answer
- **Chat History**: Track all Q&A pairs
- **Fallback Messages**: Handle invalid/out-of-scope questions
- **Example Questions**: Built-in examples in sidebar



## 👤 Author

**Nahid Muntasir**  
GitHub: [@NahidMuntasir7](https://github.com/NahidMuntasir7)

## Acknowledgments

- Bengali SBERT model by [L3Cube Pune](https://huggingface.co/l3cube-pune/bengali-sentence-similarity-sbert)
