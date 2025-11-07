# -------------------- Libraries --------------------
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap
import numpy as np

# -------------------- Load CSV and Prepare Text --------------------
df = pd.read_csv(r"C:\Users\Muskan\Documents\fast-api_auth\AI_policy_data.csv")

# Combine multiple columns for NLP
df['text_for_nlp'] = (
    df['scheme_name'].astype(str) + " " +
    df['details'].astype(str) + " " +
    df['benefits'].astype(str) + " " +
    df['eligibility'].astype(str) + " " +
    df['application'].astype(str) + " " +
    df['documents'].astype(str)
).str.lower()

# -------------------- TF-IDF Vectorization --------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df['text_for_nlp'])

# Save vectorizer and matrix (optional)
joblib.dump(vectorizer, "scheme_vectorizer.pkl")
joblib.dump({"matrix": tfidf_matrix, "df": df}, "scheme_tfidf_matrix.pkl")

# -------------------- Quantum NLP Integration --------------------
try:
    QUANTUM_MODEL_PATH = r"C:\Users\Muskan\Documents\Publicpolicy\Quantum_AI_Policy\quantum_nlp_models.pkl"

    quantum_models = joblib.load(QUANTUM_MODEL_PATH)
    quantum_kernel = quantum_models.get("kernel_model")
    quantum_vectorizer = quantum_models.get("vectorizer")
    quantum_df = quantum_models.get("df")

    print("✅ Quantum NLP model loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading quantum models: {e}")
    quantum_models = None

# -------------------- FastAPI Setup --------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -------------------- Helper Function --------------------
def search_policies(query: str, top_k: int = 3):
    """Perform TF-IDF similarity search for top_k matching schemes."""
    query_vec = vectorizer.transform([query.lower()])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = sims.argsort()[::-1][:top_k]

    results = []
    for idx in top_idx:
        row = df.iloc[idx]
        results.append({
            "scheme_name": row.get("scheme_name", "Unnamed Scheme"),
            "benefits": row.get("benefits", "Not specified"),
            "eligibility": row.get("eligibility", "Not specified"),
            "summary": textwrap.shorten(str(row.get("details", "")), width=250, placeholder="..."),
            "similarity": round(float(sims[idx]), 3)
        })
    return results

# -------------------- Routes --------------------
@app.get("/", response_class=HTMLResponse)
def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/education", response_class=HTMLResponse)
def education_page(request: Request, query: str = None):
    results = search_policies(query, top_k=5) if query else None
    return templates.TemplateResponse("education.html", {"request": request, "results": results, "query": query})

@app.get("/healthcare", response_class=HTMLResponse)
def healthcare_page(request: Request, query: str = None):
    results = search_policies(query, top_k=5) if query else None
    return templates.TemplateResponse("healthcare.html", {"request": request, "results": results, "query": query})


# -------------------- Quantum Page --------------------
@app.get("/quantum", response_class=HTMLResponse)
def quantum_page(request: Request, query: str = None):
    """
    Display or search quantum-related policies using the Quantum NLP model.
    """
    if quantum_models is None:
        return templates.TemplateResponse(
            "quantum.html",
            {"request": request, "results": None, "error": "Quantum model not loaded."}
        )

    results = []
    quantum_df_local = quantum_df.copy()

    # Filter for quantum-related rows
    quantum_df_local = quantum_df_local[
        quantum_df_local['text_for_nlp'].str.contains('quantum', case=False, na=False)
    ]

    if query:
        query_vec = quantum_vectorizer.transform([query.lower()])
        sims = cosine_similarity(query_vec, quantum_vectorizer.transform(quantum_df_local['text_for_nlp'])).flatten()
        top_idx = sims.argsort()[::-1][:5]

        for idx in top_idx:
            row = quantum_df_local.iloc[idx]
            results.append({
                "scheme_name": row.get("scheme_name", "Unnamed Scheme"),
                "benefits": row.get("benefits", "Not specified"),
                "eligibility": row.get("eligibility", "Not specified"),
                "summary": textwrap.shorten(str(row.get("details", "")), width=250, placeholder="..."),
                "similarity": round(float(sims[idx]), 3)
            })
    else:
        for _, row in quantum_df_local.head(5).iterrows():
            results.append({
                "scheme_name": row.get("scheme_name", "Unnamed Scheme"),
                "benefits": row.get("benefits", "Not specified"),
                "eligibility": row.get("eligibility", "Not specified"),
                "summary": textwrap.shorten(str(row.get("details", "")), width=250, placeholder="..."),
                "similarity": 1.0
            })

    return templates.TemplateResponse("quantum.html", {"request": request, "results": results, "query": query})


# -------------------- Example Test --------------------
if __name__ == "__main__":
    question = "quantum computing research funding"
    results = search_policies(question, top_k=3)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['scheme_name']} | Similarity: {res['similarity']}")
        print(f" Benefits: {res['benefits']}")
        print(f" Eligibility: {res['eligibility']}\n")