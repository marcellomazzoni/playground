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
