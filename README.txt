# ClaimSense – HackRx 6.0 Submission (Team Codevedas)

## 🚀 Problem
Users struggle to get clear answers from long insurance PDFs.  
“Is knee surgery covered under a 3-month-old policy?” – Not easy to find.

## 🧠 Our Solution
ClaimSense understands natural language queries and:
- Extracts key info like age, procedure, duration
- Searches PDFs using semantic search (FAISS)
- Uses LLM to reason and return:
  - ✅ Decision (Approved/Rejected)
  - 💰 Amount
  - 📄 Justification (exact clause)

## 🔧 Tech Stack
- PyMuPDF, LangChain, FAISS
- HuggingFace Transformers (Flan-T5)
- Sentence Transformers
- Python, Google Colab / VS Code

## 📂 How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/ClaimSense.git
   cd ClaimSense
2. pip install -r requirements.txt
3. python main.py
