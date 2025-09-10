# STILL WORK-IN-PROGRESS

If you received this repository link as a portfolio indication, then consider that the application is constantly being updated. 
Some known issues are:
*   Major errors on Linear Models (regression), for the CV process that involves Ridge and Lasso as parameters to tune. Currently unusable
*   Minor bugs in the preprocessing steps
*   Absence of explainable AI section after training
*   Absence of multivariate analysis in the variable selection step
*   No storage of the preprocessing steps for future, consistent, trials
*   No handling of ML cases outside regression and classification (Time-Series and Unsupervised analysis are next upcoming improvements)
*   Absence of LLM API configuration without accessing the code:
   *   The user currently has to access the code to provide a Gemini API Key in the .env file and specify he is using Gemini
   *   If other providers apart from Google are used, the user has to change a code portion instead of automatically managing it from the settings
   *   Ollama is being treated as a requirement, while the user may not want to store a local model or install the program at all
*   UI to be deeply improved and customized


# Interactive Machine Learning Pipeline

This is an interactive web application built with Streamlit that allows you to upload your own dataset, perform data cleaning and preprocessing, and train and evaluate various machine learning models without writing any code.

## Features

*   **Interactive Data Cleaning**: A user-friendly interface to clean and preprocess your data.
*   **Multiple Model Support**: Supports a variety of models for both classification and regression tasks, including:
    *   K-Nearest Neighbors
    *   Random Forest
    *   Support Vector Machines (SVM)
    *   Logistic Regression
    *   Naive Bayes
    *   XGBoost
    *   Linear Models
*   **Hyperparameter Tuning**: Use GridSearchCV to find the best hyperparameters for your models.
*   **Model Evaluation**: Get detailed performance metrics and visualizations for your models.
*   **LLM Integration**: (Optional) Use a local Large Language Model (LLM) to perform complex data cleaning tasks using natural language.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
2.  The application will open in your web browser.

### Workflow

1.  **Upload Data**: On the **Data** page, upload your dataset as a CSV file.
2.  **Clean Data**: Use the provided tools to clean and preprocess your data. You can handle missing values, outliers, and perform feature engineering.
3.  **Select Target**: Choose your target variable for the machine learning task.
4.  **Train Model**: Navigate to one of the model pages from the sidebar to train and evaluate a model. You can tune the hyperparameters in the sidebar and see the results instantly.

## Project Structure

*   `app.py`: The main Streamlit application file that sets up the navigation.
*   `preprocessing.py`: The Streamlit page for data cleaning and preprocessing.
*   `models/`: A directory containing the Streamlit pages for each machine learning model.
*   `src/`: Contains helper modules.
    *   `util.py`: Utility functions for the Streamlit UI.
    *   `data_clean.py`: Backend logic for data cleaning operations, including LLM integration.
*   `requirements.txt`: The list of Python dependencies for the project.
