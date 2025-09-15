# Interactive Machine Learning Sandbox

This project is an interactive web application built with Streamlit that serves as a playground for machine learning. It allows users to upload their own datasets, perform data cleaning and preprocessing, and train and evaluate various machine learning models without writing any code.

## Project Vision

The goal of this project is to provide a "sandbox" environment for both beginners and experts to quickly iterate on machine learning tasks. Starting from a simple CSV file, the user can go through the entire ML pipeline, from data preparation to model evaluation. This application is designed to be a portfolio piece, showcasing a full-stack machine learning application.

## Current Status & Roadmap

This application is currently under active development. While it is not yet a production-ready tool, it serves as a strong foundation for a powerful and intuitive machine learning platform.

Here are some of the known issues and planned future enhancements:

*   **Linear Models:** The cross-validation process for Ridge and Lasso regression is currently under development and may not be fully functional.
*   **Preprocessing:** Minor bugs may be present in the preprocessing steps - the use of AI still covers the majority of the 
*   **Explainable AI (XAI):** An XAI section to interpret model predictions is a high-priority future addition. 
*   **Pipeline Storage:** The ability to save and reuse preprocessing steps is a planned feature. For now, the user can export the resulting dataframe.
*   **Expanded ML Use Cases:** Support for time-series forecasting and unsupervised learning is on the roadmap. Later on, dealing with text analysis.
*   **LLM Integration:** The configuration of the Large Language Model (LLM) integration will be improved to allow for easier API key management and support for multiple providers. Only supported use cases: Ollama, Gemini.
*   **UI/UX:** The user interface will be continuously improved for a better user experience and consistency among pages.

## Features

*   **Interactive Data Cleaning**: A user-friendly interface to clean and preprocess your data.
*   **Multiple Model Support**: Supports a variety of models for both classification and regression tasks.
*   **Hyperparameter Tuning**: Use GridSearchCV to find the best hyperparameters for your models.
*   **Model Evaluation**: Get detailed performance metrics and visualizations for your models.
*   **LLM Integration**: (Optional) Use a Large Language Model (LLM) to perform complex data cleaning tasks using natural language.
    *   The application supports both a local LLM via Ollama (default) and API-based models like Gemini.
    *   To switch to an API-based model, you can change the `connectivity` parameter in the `ask_llm_data_clean` function call within `src/preproc.py`. You will also need to provide your API key in a `.env` file.

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

1.  **Upload Data**: Navigate to the **Data** page and upload your dataset in CSV format.
2.  **Clean and Preprocess**: Use the interactive tools to handle missing values, outliers, and perform other data cleaning tasks.
3.  **Select Target Variable**: Choose the column you want to predict.
4.  **Train a Model**: Select a model from the sidebar, tune its hyperparameters, and start the training process.
5.  **Evaluate Performance**: Analyze the model's performance using the provided metrics and visualizations.

## Project Structure

*   `app.py`: The main Streamlit application file that sets up the navigation.
*   `preprocessing.py`: The Streamlit page for data cleaning and preprocessing.
*   `models/`: A directory containing the Streamlit pages for each machine learning model.
*   `src/`: Contains helper modules.
*   `requirements.txt`: The list of Python dependencies for the project.
