 
# --- model.py ---
from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
import io
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap
import numpy as np
from config import templates  # Import from config.py

# -------------------- Create Router --------------------
# This is the key change:
# Instead of 'app = FastAPI()', we create a router
router = APIRouter()

# -------------------- Load CSV and Prepare Text --------------------
try:
    df = pd.read_csv(r"C:\Users\Muskan\Documents\fast-api_auth\AI_policy_data.csv")

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

    joblib.dump(vectorizer, "policy_vectorizer.pkl")
    joblib.dump({"matrix": tfidf_matrix, "df": df}, "policy__matrix.pkl")

except FileNotFoundError:
    print("⚠️ CRITICAL ERROR: policy_data.csv not found. Search will not work.")
    df = pd.DataFrame() # Create empty dataframe
    vectorizer = None
    tfidf_matrix = None

# -------------------- Quantum NLP Integration --------------------
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    quantum_model_path = os.path.join(base_dir, "Quantum_AI_Policy", "quantum_nlp_models.pkl")
    quantum_model_path = os.path.normpath(quantum_model_path)

    print(f"Attempting to load quantum model from: {quantum_model_path}")
    quantum_models = joblib.load(quantum_model_path)
    quantum_kernel = quantum_models.get("kernel_model")
    quantum_vectorizer = quantum_models.get("vectorizer")
    quantum_df = quantum_models.get("df")

    print("✅ Quantum NLP model loaded successfully.")
except Exception as e:
    print(f"⚠️ CRITICAL ERROR: Failed to load quantum model. Reason: {e}")
    quantum_models = None
    quantum_df = None
    quantum_vectorizer = None

# -------------------- Helper Functions --------------------
def search_policies(query: str, top_k: int = 3):
    if vectorizer is None: # Check if model loaded
        return []
    query_vec = vectorizer.transform([query.lower()])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    # ... (rest of your function)
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

def search_quantum_policies(query: str = None, top_k: int = 5):
    if quantum_models is None or quantum_df is None or quantum_vectorizer is None:
        return [] # Return empty list if model isn't loaded
    # ... (rest of your function)
    results = []
    quantum_df_local = quantum_df.copy()
    quantum_df_local = quantum_df_local[
        quantum_df_local['text_for_nlp'].str.contains('quantum', case=False, na=False)
    ]
    if query:
        query_vec = quantum_vectorizer.transform([query.lower()])
        sims = cosine_similarity(query_vec, quantum_vectorizer.transform(quantum_df_local['text_for_nlp'])).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
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
        for _, row in quantum_df_local.head(top_k).iterrows():
            results.append({
                "scheme_name": row.get("scheme_name", "Unnamed Scheme"),
                "benefits": row.get("benefits", "Not specified"),
                "eligibility": row.get("eligibility", "Not specified"),
                "summary": textwrap.shorten(str(row.get("details", "")), width=250, placeholder="..."),
                "similarity": 1.0
            })
    return results

# -------------------- Routes --------------------
# Note: All routes now use '@router' instead of '@app'

# DO NOT define a '/' route here. It's already in main.py
# The route for 'index.html' is now '/home' in main.py

@router.get("/education", response_class=HTMLResponse)
def education_page(request: Request, query: str = None):
    results = search_policies(query, top_k=5) if query else None
    return templates.TemplateResponse("education.html", {"request": request, "results": results, "query": query})

@router.get("/healthcare", response_class=HTMLResponse)
def healthcare_page(request: Request, query: str = None):
    results = search_policies(query, top_k=5) if query else None
    return templates.TemplateResponse("healthcare.html", {"request": request, "results": results, "query": query})

@router.get("/download_policies_csv")
async def download_policies_csv(query: str = None, category: str = None):
    data_source = pd.DataFrame()
    filename = "search_results.csv"
    
    if query and category:
        results = []
        if category == 'quantum':
            results = search_quantum_policies(query, top_k=5)
            filename = f"quantum_search_results.csv"
        elif category in ['education', 'healthcare']:
            results = search_policies(query, top_k=5)
            filename = f"{category}_search_results.csv"
        
        if results:
            data_source = pd.DataFrame(results)
            
    stream = io.StringIO()
    data_source.to_csv(stream, index=False)
    
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response

@router.get("/quantum", response_class=HTMLResponse)
def quantum_page(request: Request, query: str = None):
    if quantum_models is None:
        return templates.TemplateResponse(
            "quantum.html",
            {"request": request, "results": None, "error": "Quantum model not loaded."}
        )
    if quantum_df is None or quantum_vectorizer is None:
        return templates.TemplateResponse(
            "quantum.html",
            {"request": request, "results": None, "error": "Quantum model components are missing."}
        )
    
    results = search_quantum_policies(query, top_k=5)
    return templates.TemplateResponse("quantum.html", {"request": request, "results": results, "query": query})


# The '__main__' block is for testing this file directly
# It will not run when imported by main.py
if __name__ == "__main__":
    print("Testing model.py helper functions...")
    if vectorizer is not None:
        question = "quantum computing research funding"
        results = search_policies(question, top_k=3)
        for i, res in enumerate(results, 1):
            print(f"{i}. {res['scheme_name']} | Similarity: {res['similarity']}")
            print(f" Benefits: {res['benefits']}")
            print(f" Eligibility: {res['eligibility']}\n")
    else:
        print("Could not run tests, model data not loaded.")