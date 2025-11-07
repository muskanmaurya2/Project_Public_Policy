# policy.py
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap
import os

# --- Import from our new files ---
from auth import get_current_user  # Imports the protection dependency
from config import templates         # <-- FIXED: Import from core.py

# -------------------- Router Setup --------------------
policy_router = APIRouter()

# -------------------- Load CSV and Prepare Text --------------------
# (Copied from your app.py)
try:
    # Use a relative path to make it more portable
    # This assumes your .csv is in a specific folder path relative to policy.py
    # You may need to adjust "C:\Users\Muskan\..." if it's not in your project
    df = pd.read_csv(r"C:\Users\Muskan\Documents\fast-api_auth\AI_policy_data.csv")
    df['text_for_nlp'] = (
        df['scheme_name'].astype(str) + " " +
        df['details'].astype(str) + " " +
        df['benefits'].astype(str) + " " +
        df['eligibility'].astype(str) + " " +
        df['application'].astype(str) + " " +  # <-- FIXED: Removed typo 'D'' and added '+'
        df['documents'].astype(str)
    ).str.lower()
except Exception as e:
    print(f"⚠️ CRITICAL ERROR: Failed to load and process CSV. Reason: {e}")
    df = pd.DataFrame() # Create empty dataframe to avoid more errors

# -------------------- TF-IDF Vectorization --------------------
# (Copied from your app.py)
try:
    if not df.empty:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(df['text_for_nlp'])
        joblib.dump(vectorizer, "scheme_vectorizer.pkl")
        joblib.dump({"matrix": tfidf_matrix, "df": df}, "scheme_tfidf_matrix.pkl")
    else:
        print("⚠️ Skipping TF-IDF vectorization because dataframe is empty.")
        vectorizer = None
        tfidf_matrix = None
except Exception as e:
    print(f"⚠️ CRITICAL ERROR: Failed during TF-IDF Vectorization. Reason: {e}")
    vectorizer = None
    tfidf_matrix = None


# -------------------- Quantum NLP Integration --------------------
# (Copied from your app.py)
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming 'Quantum_AI_Policy' is a sibling to your 'fast-api_auth' project dir
    quantum_model_path = os.path.join(base_dir, "..", "Quantum_AI_Policy", "quantum_nlp_models.pkl")
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

# -------------------- Helper Function --------------------
# (Copied from your app.py)
def search_policies(query: str, top_k: int = 3):
    if vectorizer is None or tfidf_matrix is None or df.empty:
        return [] # Return empty if models failed to load
        
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

# -------------------- Protected Routes --------------------
# Note: We add 'dependencies=[Depends(get_current_user)]' to each route
# This is what secures your pages.

@policy_router.get("/education", response_class=HTMLResponse, dependencies=[Depends(get_current_user)])
def education_page(request: Request, query: str = None):
    results = search_policies(query, top_k=5) if query else None
    return templates.TemplateResponse("education.html", {"request": request, "results": results, "query": query})

@policy_router.get("/healthcare", response_class=HTMLResponse, dependencies=[Depends(get_current_user)])
def healthcare_page(request: Request, query: str = None):
    results = search_policies(query, top_k=5) if query else None
    return templates.TemplateResponse("healthcare.html", {"request": request, "results": results, "query": query})

@policy_router.get("/quantum", response_class=HTMLResponse, dependencies=[Depends(get_current_user)])
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
    
    results = []
    quantum_df_local = quantum_df.copy()
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