# Interactive Machine Learning Sandbox

This project is an interactive web application built with Streamlit that serves as a playground for machine learning. It allows users to upload their own datasets, perform data cleaning and preprocessing, and train and evaluate various machine learning models without writing any code.

## Project Vision

The goal of this project is to provide a "sandbox" environment for both beginners and experts to quickly iterate on machine learning tasks. Starting from a simple CSV file or a Python script, the user can go through the entire ML pipeline, from data preparation to model evaluation. This application is designed to be a portfolio piece, showcasing a full-stack machine learning application.

## Current Status & Roadmap

This application is currently under active development. While it is not yet a production-ready tool, it serves as a strong foundation for a powerful and intuitive machine learning platform.

Here are some of the known issues and planned future enhancements:

*   **Linear Models:** The cross-validation process for Ridge and Lasso regression is currently under development and may not be fully functional.
*   **Explainable AI (XAI):** An XAI section to interpret model predictions is a high-priority future addition. 
*   **Pipeline Storage:** The ability to save and reuse preprocessing steps is a planned feature. For now, the user can export the resulting dataframe.
*   **Expanded ML Use Cases:** Support for time-series forecasting is on the roadmap.
*   **UI/UX:** The user interface will be continuously improved for a better user experience and consistency among pages.

## Features

*   **Flexible Data Loading**:
    *   Upload data directly from a CSV file.
    *   Load data using a custom Python script, with on-the-fly dependency installation.
*   **Interactive Data Cleaning**: A user-friendly interface to clean and preprocess your data.
    *   **Automatic Data Summaries**: Get a quick overview of your data with summaries for numeric, string, and datetime variables.
    *   **General Actions**: Drop variables, remove duplicate rows, and lowercase column names.
    *   **Single-Variable Actions**: A rich set of tools for individual columns, including renaming, outlier management, missing value imputation (mean, median, KNN), bucketing, and type conversions.
*   **LLM-Powered Data Transformation**:
    *   Use a Large Language Model (LLM) to perform complex data cleaning tasks using natural language.
    *   Supports local models via Ollama and API-based models like Gemini and Groq.
    *   Review the generated code and preview the transformation before applying it.
*   **Supervised Learning**:
    *   **Problem Types**: Supports binary classification, multi-class classification, and regression.
    *   **Models**: KNN, Random Forest, SVM, Logistic Regression, Naive Bayes, XGBoost, and Linear Models.
    *   **Analysis**: In-depth univariate, bivariate, and multivariate analysis tools, including distribution charts, correlation matrices, feature importance, and 3D PCA plots.
    *   **Hyperparameter Tuning**: Use GridSearchCV or HalvingGridSearchCV to find the best hyperparameters for your models.
    *   **Model Evaluation**: Get detailed performance metrics and visualizations for your models.
*   **Unsupervised Learning**:
    *   **Clustering Methods**: K-Means, DBSCAN, and Hierarchical Clustering.
    *   **Analysis**: Bivariate and multivariate analysis tools to explore the structure of your data.

## Getting Started

### Prerequisites

*   Python 3.8+
*   pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
2.  The application will open in your default web browser.

## User Journey

1.  **Load Data**: Navigate to the **Data** page and choose your input method (CSV or Python script).
2.  **Clean and Preprocess**: Use the interactive tools to handle missing values, outliers, and perform other data cleaning tasks.
3.  **Select Learning Type**: Choose between Supervised and Unsupervised learning.
4.  **Select Variables**:
    *   For **Supervised learning**, select your target variable and the features to include in the model.
    *   For **Unsupervised learning**, select the features for clustering.
5.  **Analyze Data**: Use the analysis tools to understand your data better.
6.  **Train a Model**: Select a model from the sidebar, tune its hyperparameters, and start the training process.
7.  **Evaluate Performance**: Analyze the model's performance using the provided metrics and visualizations.

## Project Structure

*   `app.py`: The main Streamlit application file that sets up the navigation.
*   `preprocessing.py`: The Streamlit page for data loading, cleaning, and preprocessing.
*   `supervised_models/`: A directory containing the Streamlit pages for each supervised learning model.
*   `unsupervised_models/`: A directory containing the Streamlit pages for each unsupervised learning model.
*   `src/`: Contains helper modules, including `preproc.py` for data processing logic.
*   `requirements.txt`: The list of Python dependencies for the project.
*   `setup.py`: The setup file for the project.
*   `.streamlit/`: Contains Streamlit configuration files.
*   `info/`: Contains informational files.
*   `honest work meme.jpg`: An inspirational image for the developers.