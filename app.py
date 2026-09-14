import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from autoclassify import AutoClassify


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AutoClassify",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("🤖 AutoClassify")

st.subheader(
    "Intelligent Dataset-Aware Classification Algorithm Selection"
)

st.write(
    """
    AutoClassify analyzes a classification dataset and intelligently
    recommends the most suitable algorithm from:

    - Decision Tree
    - Naive Bayes
    - Support Vector Machine (SVM)
    """
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Dataset Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)


# =========================================================
# DATASET UPLOAD
# =========================================================

if uploaded_file is not None:

    # Read CSV
    data = pd.read_csv(
        uploaded_file
    )

    st.success(
        "Dataset uploaded successfully!"
    )

    # ---------------------------------------------
    # Dataset preview
    # ---------------------------------------------

    st.header("📂 Dataset Preview")

    st.dataframe(
        data.head(10),
        use_container_width=True
    )

    # ---------------------------------------------
    # Dataset information
    # ---------------------------------------------

    st.write(
        f"**Rows:** {data.shape[0]}"
    )

    st.write(
        f"**Columns:** {data.shape[1]}"
    )

    # ---------------------------------------------
    # Target selection
    # ---------------------------------------------

    target_column = st.sidebar.selectbox(
        "Select Target Column",
        data.columns
    )

    # ---------------------------------------------
    # Run button
    # ---------------------------------------------

    run_analysis = st.sidebar.button(
        "🚀 Run AutoClassify"
    )

    if run_analysis:

        # Save temporary uploaded file
        temp_file = "uploaded_dataset.csv"

        data.to_csv(
            temp_file,
            index=False
        )

        # -----------------------------------------
        # Run framework
        # -----------------------------------------

        with st.spinner(
            "Analyzing dataset and training models..."
        ):

            result = AutoClassify(
                temp_file,
                target_column
            )

        st.success(
            "AutoClassify analysis completed!"
        )

        # =================================================
        # DATASET PROFILE
        # =================================================

        st.header(
            "1️⃣ Dataset Profile"
        )

        profile = result["profile"]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Instances",
            profile["instances"]
        )

        col2.metric(
            "Features",
            profile["features"]
        )

        col3.metric(
            "Classes",
            profile["classes"]
        )

        col4.metric(
            "Missing %",
            f"{profile['missing_percentage']:.2f}%"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Numerical Features",
            profile["numerical_features"]
        )

        col2.metric(
            "Categorical Features",
            profile["categorical_features"]
        )

        col3.metric(
            "Binary Features",
            profile["binary_features"]
        )

        col4.metric(
            "Duplicate Records",
            profile["duplicate_records"]
        )

        st.write(
            "**Class Distribution:**"
        )

        st.write(
            profile["class_distribution"]
        )

        st.write(
            "**Imbalance Ratio:**",
            profile["imbalance_ratio"]
        )

        st.write(
            "**Feature/Sample Ratio:**",
            profile["feature_sample_ratio"]
        )

        # =================================================
        # STATISTICS
        # =================================================

        st.subheader(
            "Basic Statistical Characteristics"
        )

        st.dataframe(
            profile["statistics"],
            use_container_width=True
        )

        # =================================================
        # CORRELATION
        # =================================================

        st.subheader(
            "Feature Correlation"
        )

        correlation = profile["correlation"]

        if not correlation.empty:

            fig, ax = plt.subplots(
                figsize=(10, 7)
            )

            ax.imshow(
                correlation,
                aspect="auto"
            )

            ax.set_title(
                "Feature Correlation Matrix"
            )

            st.pyplot(fig)

        # =================================================
        # PREPROCESSING
        # =================================================

        st.header(
            "2️⃣ Automated Preprocessing"
        )

        for decision in result[
            "preprocessing_decisions"
        ]:

            st.write(
                "✓",
                decision
            )

        # =================================================
        # SUITABILITY
        # =================================================

        st.header(
            "3️⃣ Algorithm Suitability Analysis"
        )

        scores = result[
            "suitability_scores"
        ]

        score_df = pd.DataFrame(
            {
                "Algorithm":
                    list(scores.keys()),

                "Suitability Score":
                    list(scores.values())
            }
        )

        st.dataframe(
            score_df,
            use_container_width=True
        )

        # Suitability chart

        fig, ax = plt.subplots()

        ax.bar(
            score_df["Algorithm"],
            score_df["Suitability Score"]
        )

        ax.set_ylabel(
            "Suitability Score"
        )

        ax.set_title(
            "Algorithm Suitability Ranking"
        )

        st.pyplot(fig)

        # =================================================
        # ANALYTICAL PREDICTION
        # =================================================

        analytical_prediction = result[
            "analytical_prediction"
        ]

        st.subheader(
            "🎯 Analytical Prediction"
        )

        st.info(
            f"The dataset-aware suitability engine predicts: "
            f"**{analytical_prediction}**"
        )

        # =================================================
        # REASONS
        # =================================================

        st.subheader(
            "Why was this algorithm predicted?"
        )

        reasons = result[
            "suitability_reasons"
        ][analytical_prediction]

        for reason in reasons:

            st.write(
                "•",
                reason
            )

        # =================================================
        # EXPERIMENTAL RESULTS
        # =================================================

        st.header(
            "4️⃣ Experimental Validation"
        )

        results_df = result[
            "experimental_results"
        ].copy()

        display_df = results_df[
            [
                "Algorithm",
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "CV F1 Mean",
                "Training Time",
                "Prediction Time"
            ]
        ].copy()

        st.dataframe(
            display_df.style.format(
                {
                    "Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1 Score": "{:.4f}",
                    "CV F1 Mean": "{:.4f}",
                    "Training Time": "{:.6f}",
                    "Prediction Time": "{:.6f}"
                }
            ),
            use_container_width=True
        )

        # =================================================
        # PERFORMANCE CHART
        # =================================================

        st.subheader(
            "Model Performance Comparison"
        )

        fig, ax = plt.subplots()

        ax.bar(
            results_df["Algorithm"],
            results_df["F1 Score"]
        )

        ax.set_ylabel(
            "F1 Score"
        )

        ax.set_title(
            "F1 Score Comparison"
        )

        plt.xticks(
            rotation=20
        )

        st.pyplot(fig)

        # =================================================
        # CONFUSION MATRICES
        # =================================================

        st.subheader(
            "Confusion Matrices"
        )

        for _, row in results_df.iterrows():

            st.write(
                f"### {row['Algorithm']}"
            )

            cm = row[
                "Confusion Matrix"
            ]

            st.dataframe(
                pd.DataFrame(cm)
            )

        # =================================================
        # SVM EXPERIMENT
        # =================================================

        st.header(
            "5️⃣ SVM Hyperparameter Experiment"
        )

        st.write(
            "**Best SVM Parameters:**"
        )

        st.json(
            result[
                "svm_best_parameters"
            ]
        )

        st.write(
            "**Best SVM Cross-Validation F1:**",
            round(
                result["svm_best_cv_score"],
                4
            )
        )

        # =================================================
        # FINAL RECOMMENDATION
        # =================================================

        st.header(
            "6️⃣ Final Recommendation"
        )

        experimental_winner = result[
            "experimental_winner"
        ]

        prediction_confirmed = result[
            "prediction_confirmed"
        ]

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Analytical Prediction",
                analytical_prediction
            )

        with col2:

            st.metric(
                "Experimental Winner",
                experimental_winner
            )

        # =================================================
        # STATUS
        # =================================================

        if prediction_confirmed:

            st.success(
                "✅ Prediction Status: CONFIRMED"
            )

            st.write(
                f"The suitability engine predicted "
                f"**{analytical_prediction}**, and experimental "
                f"validation also identified the corresponding "
                f"algorithm as the best performer."
            )

        else:

            st.warning(
                "⚠️ Prediction Status: NOT CONFIRMED"
            )

            st.write(
                f"The analytical engine predicted "
                f"**{analytical_prediction}**, but experimental "
                f"validation selected **{experimental_winner}**."
            )

            st.subheader(
                "Failure Analysis"
            )

            st.write(
                """
                The heuristic suitability engine uses dataset
                characteristics such as dataset size, dimensionality,
                feature-to-sample ratio, correlations and class
                imbalance. These characteristics cannot perfectly
                represent the actual decision boundary of every dataset.

                An improvement would be to refine the suitability
                weights using results collected from multiple
                development datasets.
                """
            )

        # =================================================
        # FINAL EXPLANATION
        # =================================================

        st.header(
            "7️⃣ Explainable Recommendation"
        )

        st.write(
            f"""
            **Analytical Prediction:** {analytical_prediction}

            **Experimental Winner:** {experimental_winner}

            The analytical recommendation was generated before
            experimental validation using automatically extracted
            dataset characteristics. The final result was then
            validated using Accuracy, Precision, Recall, F1-score,
            5-fold cross-validation and computational time.
            """
        )

else:

    st.info(
        "👈 Upload a CSV dataset from the sidebar to begin."
    )