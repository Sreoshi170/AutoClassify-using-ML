# 🤖 AutoClassify

**Intelligent Dataset-Aware Classification Algorithm Selection**

AutoClassify is a Streamlit application that analyzes an uploaded classification dataset, profiles its statistical characteristics, and recommends the most suitable machine learning algorithm from **Decision Tree**, **Naive Bayes**, and **Support Vector Machine (SVM)** — before backing that recommendation up with real experimental validation.

Rather than blindly training every model and picking the best score, AutoClassify first reasons about *why* a given algorithm should work well on a specific dataset (based on size, dimensionality, feature/sample ratio, correlation, and class balance), then trains and evaluates all three models to confirm or challenge that prediction.

---

## ✨ Features

- **📂 Dataset Profiling** — automatically computes instance/feature counts, class distribution, imbalance ratio, feature/sample ratio, numerical vs. categorical vs. binary feature counts, duplicate records, and missing-value percentage.
- **📊 Statistical & Correlation Analysis** — descriptive statistics table and a visual feature correlation matrix.
- **⚙️ Automated Preprocessing** — applies preprocessing decisions automatically based on the detected dataset profile.
- **🧠 Algorithm Suitability Engine** — scores Decision Tree, Naive Bayes, and SVM against the dataset's characteristics and produces an analytical prediction with human-readable reasoning.
- **🧪 Experimental Validation** — trains and evaluates all three algorithms using Accuracy, Precision, Recall, F1 Score, 5-fold cross-validation, training time, and prediction time.
- **🔍 Confusion Matrices** — per-algorithm confusion matrices for deeper error analysis.
- **🎛️ SVM Hyperparameter Search** — reports the best-performing SVM parameters and their cross-validation F1 score.
- **✅ Prediction vs. Reality Check** — compares the analytical prediction against the experimental winner and flags whether the heuristic was confirmed, with a failure-analysis explanation when it isn't.
- **📖 Explainable Output** — a final plain-language summary tying the analytical and experimental results together.
- **🖥️ Interactive Web UI** — built with Streamlit; just upload a CSV, pick a target column, and run the analysis.

---

## 🗂️ Project Structure

```
AutoClassify-using-ML/
├── app.py                 # Streamlit web application (UI layer)
├── autoclassify.py         # Core AutoClassify engine (profiling, suitability scoring, model training/evaluation)
├── autoclassify.ipynb       # Notebook version for exploration/experimentation
├── dataset/                # Sample dataset(s)
├── uploaded_dataset.csv     # Cached copy of the most recently uploaded dataset
├── requirements.txt         # Python dependencies
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Sreoshi170/AutoClassify-using-ML.git
cd AutoClassify-using-ML

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package        | Purpose                                   |
|----------------|--------------------------------------------|
| `streamlit`    | Web application UI                        |
| `pandas`       | Data loading and manipulation             |
| `numpy`        | Numerical computation                     |
| `scikit-learn` | Model training, evaluation, cross-validation |
| `matplotlib`   | Charts and visualizations                 |

### Running the App

```bash
streamlit run app.py
```

This launches a local web server (by default at `http://localhost:8501`) where you can interact with AutoClassify in your browser.

---

## 🧭 How to Use

1. **Upload a dataset** — use the sidebar file uploader to load a CSV file.
2. **Preview the data** — inspect the first rows, row/column counts, and dataset shape.
3. **Select the target column** — choose the label/class column from the sidebar dropdown.
4. **Run AutoClassify** — click **🚀 Run AutoClassify** to trigger the full pipeline.
5. **Review the results**, presented step by step:
   1. **Dataset Profile** — key statistics, class distribution, and correlation matrix.
   2. **Automated Preprocessing** — the preprocessing decisions applied to your data.
   3. **Algorithm Suitability Analysis** — suitability scores and ranking chart for each algorithm.
   4. **Analytical Prediction** — the recommended algorithm with supporting reasons.
   5. **Experimental Validation** — full metrics table, F1-score comparison chart, and confusion matrices.
   6. **SVM Hyperparameter Experiment** — best parameters found and their CV F1 score.
   7. **Final Recommendation** — analytical prediction vs. experimental winner, with confirmation status.
   8. **Explainable Recommendation** — a plain-language summary of the entire analysis.

---

## 🧠 How It Works

AutoClassify's core engine (`autoclassify.py`) follows a two-stage approach:

1. **Analytical stage** — dataset characteristics (size, dimensionality, feature-to-sample ratio, correlations, and class imbalance) are used to heuristically score how well each candidate algorithm should perform, without training any models yet.
2. **Experimental stage** — Decision Tree, Naive Bayes, and SVM are actually trained and evaluated with cross-validation to measure real-world performance.

The two stages are then compared: if the experimental winner matches the analytical prediction, the recommendation is marked as **confirmed**; if not, AutoClassify surfaces a failure analysis explaining that heuristic suitability scores can't perfectly capture every dataset's true decision boundary, and suggests refining the scoring weights using results from more datasets.

---

## 📌 Notes

- The uploaded dataset is temporarily cached as `uploaded_dataset.csv` during analysis.
- A sample dataset is included under `dataset/` for quick testing.
- An equivalent Jupyter notebook (`autoclassify.ipynb`) is provided for users who prefer exploring the pipeline interactively outside of Streamlit.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to fork the repository and submit a pull request.

## 📄 License

No license has been specified for this repository. If you intend to use or distribute this project, please contact the repository owner or add a license file.

## 👤 Author

**Sreoshi170** — [GitHub Profile](https://github.com/Sreoshi170)
