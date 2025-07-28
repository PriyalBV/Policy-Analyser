from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import gradio as gr

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample policy document text
document_text = """
CLAUSE 1.1: This policy provides coverage for hospitalization due to illness or accident unless specified otherwise.
CLAUSE 2.1: A minimum waiting period of 2 months is required for any surgical procedure coverage to be applicable.
CLAUSE 3.1: The insurance policy must be active for at least 3 months to be eligible for major surgery claims.
CLAUSE 4.1: Claims related to knee surgeries, including knee replacement, arthroscopy, and ACL reconstruction, are covered after a 2-month waiting period.
CLAUSE 5.1: Maximum payout for orthopedic surgeries, including knee surgery, is ₹50,000.
CLAUSE 6.1: Claims for procedures conducted outside India are not covered under this policy.
CLAUSE 7.1: Coverage is only available to individuals under the age of 65 at the time of claim.
CLAUSE 8.1: Any claim from hospitals located in Tier 1 cities (e.g., Mumbai, Pune, Delhi) will be processed on a priority basis.
CLAUSE 9.1: Pre-existing conditions are not covered until 1 year of continuous policy coverage.
"""

# Split document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(document_text)

# Generate embeddings & FAISS index
embeddings = model.encode(chunks)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Clause searcher
def search_policy(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

# Decision logic
def generate_decision(query, clauses):
    decision = "Rejected"
    amount = "₹0"
    justification = []

    for clause in clauses:
        if "knee surgery" in clause.lower() or "orthopedic" in clause.lower():
            if "covered" in clause.lower():
                decision = "Approved"
                justification.append("✔️ " + clause.strip())
            if "₹" in clause or "maximum payout" in clause.lower():
                amount = "₹50,000"
                justification.append("💰 " + clause.strip())
        if "3 months" in clause.lower() and "policy" in clause.lower():
            justification.append("📅 " + clause.strip())
        if "under the age of 65" in clause.lower():
            justification.append("👤 " + clause.strip())
        if "tier 1" in clause.lower() and "pune" in query.lower():
            justification.append("📍 " + clause.strip())

    return {
        "decision": decision,
        "amount": amount,
        "justification": justification
    }

# Gradio interface
def policy_bot(query):
    clauses = search_policy(query)
    result = generate_decision(query, clauses)
    return result

iface = gr.Interface(
    fn=policy_bot,
    inputs=gr.Textbox(lines=2, placeholder="e.g., 46-year-old male, knee surgery in Pune, 3-month-old insurance policy"),
    outputs="json",
    title="🛡️ Policy Analyzer",
    description="Ask questions related to a policy document. The model will extract clauses, make a decision, and explain it."
)

iface.launch()
