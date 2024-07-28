# Fake News Classifier
This project is a web-based application built using FastAPI for detecting fake news. 
It uses various machine learning models to classify news articles as either "Fake News" or "Real News" based on the text content. 
The best-performing model, Gradient Boosting Classifier, is used for predictions in the deployed application.

![Image about the final project](<Fake News Prediction.png>)

## Prerequisites
To run this project, you need to have the following dependencies installed:

- Python 3.x
- pandas
- matplotlib
- seaborn
- nltk
- scikit-learn
- joblib
- FastAPI
- uvicorn
- Jinja2

## Overview of the Code

1. Import Dependencies:
Necessary libraries are imported, including pandas for data manipulation, matplotlib and seaborn for visualization, nltk for text processing, and scikit-learn for machine learning.

2. Load and Inspect Data:
The dataset is loaded from a CSV file. Initial inspection includes checking the shape of the dataset and identifying any missing values.

3. Data Preprocessing:
- Handling missing values by dropping rows with missing 'title' or 'text' columns.
- Combining 'title' and 'text' into a single 'content' column.
- Dropping the 'author' column, which is considered less relevant for the prediction task.
- Visualizing label distribution to check data balance.

4. Text Processing:
- Applying stemming to the 'content' column using the PorterStemmer.
- Vectorizing the text data using TfidfVectorizer to convert text into numerical features suitable for model training.

5. Model Training and Evaluation
- **Train-Test Split**: The data is split into training and testing sets using a 90-10 split.
- **Model Training**: Several machine learning models are trained, including Logistic Regression, Random Forest Classifier, Linear SVC, Decision Tree Classifier, and Gradient Boosting Classifier.
- **Model Evaluation**: Each model is evaluated based on accuracy, confusion matrix, and classification report. The confusion matrix is visualized for better understanding of the model's performance.

6. Model Saving and Deployment
- Save the Best Model: The best-performing model (Gradient Boosting Classifier with 94% accuracy) and the TfidfVectorizer are saved using joblib.
- Deploy with FastAPI: A FastAPI application is created to provide an interactive web interface for users to input news articles and get predictions. The application loads the saved model and vectorizer, processes the input data, and returns the prediction (Fake or Real).

## Model Accuracy
The Gradient Boosting Classifier achieved the highest accuracy of 94% on the test data, making it the best model for this project.

## Contributions
Contributions are welcome! If you have any ideas, suggestions, or improvements, please feel free to open an issue or submit a pull request.
