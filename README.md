# 🧠 AI Policy Navigator
### FastAPI-Based Project | Infosys Springboard Internship

## 📘 Overview
**AI Policy Navigator** is an AI-powered application designed to help users explore, understand, and retrieve information about various government and organizational **AI policies**.
Built using **FastAPI**, **machine learning**, and **natural language processing (NLP)**, the system allows users to enter a policy-related query and receive the most relevant policies and summaries using **cosine similarity** on TF-IDF representations.

This project was developed as part of my **Infosys Internship**, focusing on the intersection of **AI, NLP, and public policy analytics**.

---

## 🚀 Features
- 🔍 **AI-Powered Policy Search** – Uses TF-IDF vectorization and cosine similarity to retrieve relevant policies.
- ⚡ **FastAPI Backend** – Efficient API handling for queries and responses.
- 🧩 **Machine Learning Integration** – Pre-trained models for semantic text understanding.
- 📊 **Data-Driven Insights** – Supports policy comparison and contextual recommendations.
- 🌐 **Web Interface** – Clean and interactive HTML front-end for user-friendly navigation.

---

## 🧰 Tech Stack
| Component | Technology |
|------------|-------------|
| **Backend Framework** | FastAPI |
| **Language** | Python 3.x |
| **Machine Learning** | Scikit-learn, Joblib |
| **Vectorization** | TF-IDF (Term Frequency–Inverse Document Frequency) |
| **Similarity Metric** | Cosine Similarity |
| **Frontend** | HTML5, CSS3, JS (served via Jinja2) |
| **Database** | CSV Dataset / Pandas DataFrame |
| **Server** | Uvicorn |
| **IDE / Tools** | Visual Studio Code / Jupyter Notebook |
| **Quantum ML** | Qiskit, PennyLane |

---

## 🔧 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/muskanmaurya2/Project_Public_Policy.git](https://github.com/muskanmaurya2/Project_Public_Policy.git)
    cd Project_Public_Policy/AI_infosys_Policy
    ```

2.  **Create and activate a virtual environment:**
    
    First, create the environment:
    ```bash
    python -m venv venv
    ```
    
    **On Windows (in PowerShell):**
    * If you get an error about "running scripts is disabled," run this command first:
        ```powershell
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
        ```
    * Now, activate the environment:
        ```powershell
        .\venv\Scripts\activate
        ```

    **On macOS/Linux (in Bash):**
    ```bash
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    * Make sure you have a `requirements.txt` file (like the one we just built).
    * Run the installer:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Place your data:**
    * Ensure your main dataset (e.g., `AI_policy_data.csv`) is in the correct path referenced in `main.py`.
    * Ensure the `quantum_nlp_models.pkl` file is in the correct path (e.g., `Quantum_AI_Policy/quantum_nlp_models.pkl`).

---

## 🖥️ Usage

1.  **Run the FastAPI server:**
    From the `AI_infosys_Policy` directory (where `main.py` is located), run:
    ```bash
    uvicorn main:app --reload
    ```

2.  **Access the application:**
    * **Main Policy Search:** Open your browser and go to `http://127.0.0.1:8000/`
    * **Quantum Policy Search:** Navigate to `http://127.0.0.1:8000/quantum`

---




## 🔮 Quantum Integration

This project also explores the integration of **Quantum Computing** concepts to enhance text analysis and policy retrieval. The quantum components focus on improving semantic understanding and similarity detection beyond classical NLP methods.

#### 🧠 Quantum Models Used

* **Quantum Natural Language Processing (QNLP):** Enables deeper contextual understanding of policy texts.
* **Quantum Support Vector Machine (QSVM):** Improves classification accuracy for policy categorization (as seen in `quantum_nlp.ipynb`).
* **Quantum k-Means:** Clusters similar policy documents using quantum optimization techniques.
* **Quantum Neural Networks (QNNs):** Enhance semantic representation and relationship mapping (as seen in `quantum_nlp.ipynb` with PennyLane).
