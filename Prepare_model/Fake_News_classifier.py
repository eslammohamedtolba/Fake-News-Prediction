# Import dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import re
import os

# Download stopwords from nltk
import nltk
nltk.download('stopwords')



# Create function to stem all words in the content
def stemming(content, stemmer):
    # Work on the alphabetical words only
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    
    # Convert sentences into lower case sentences
    stemmed_content = stemmed_content.lower()
    # Split sentences into words
    stemmed_content = stemmed_content.split()
    
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    # Stem words
    stemmed_content = [stemmer.stem(word) for word in stemmed_content if word not in stop_words]
    # Join words again
    stemmed_content = ' '.join(stemmed_content)
    
    return stemmed_content


# Function to perform the model functionalities
def fit_predict(model, x_train, y_train, x_test, y_test):
    classifier = model
    
    # Make model fit data
    classifier.fit(x_train, y_train)
    
    # Get Score on train and test data
    train_score = classifier.score(x_train, y_train)
    test_score = classifier.score(x_test, y_test)
    print(f'train score is {train_score}, test score is {test_score}')
    
    # Make model predict on test data
    test_prediction = classifier.predict(x_test)
    # Get accuracy, confusion matrix and classification report
    accuracy = accuracy_score(y_test, test_prediction) 
    cf_matrix = confusion_matrix(y_test, test_prediction)
    cl_report = classification_report(y_test, test_prediction)
    
    # Plot confusion matrix
    plt.figure(figsize=(7,7))
    sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Show accuracy and classification report
    print(f'accuracy: {accuracy}\nclassification report:\n{cl_report}', '\n\n')
    
    return classifier



if __name__ == "__main__":

    # ------------------------------------------------- Load Data
    df_train = pd.read_csv('Prepare_model/train.csv')
    df_train.head()

    # Show shape of the dataset
    print(df_train.shape)
    # Show some info about dataset
    print(df_train.info())

    # ------------------------------------------------- Data Preprocessing

    # Take a copy from original training data
    df_train_proc = deepcopy(df_train)

    # Hanlding missing values

    # Check about missing values to decide if we will make handling for it or not
    print(df_train_proc.isnull().sum())

    # Drop rows that contain at least one none value based on the title and text columns
    df_train_proc = df_train_proc.dropna(subset=['title', 'text'])
    print(df_train_proc.shape)

    # Check about none values after handling missing values
    print(df_train_proc.isnull().sum())

    # Handling invalid data

    # Combine title and text columns into one column and their contents separated by colon
    df_train_proc['content'] = df_train_proc['title'] + ': ' + df_train_proc['text']
    df_train_proc.head()

    # Drop author column from dataset because it's less important to the predition process
    df_train_proc = df_train_proc.drop(columns = ['author'])
    df_train_proc.head()

    # Visualize label values to check about balancing of 0's and 1's
    counts = df_train_proc['label'].value_counts()
    print(counts)

    # Plot result of each group
    plt.bar(counts.index, counts)
    plt.show()

    # Show data input and label data after the preporcessing step has been finihsed
    df_train_proc[['content', 'label']].head()

    # -------------------------------------------- Split data into input and output data

    # Split data into input and label data
    X = df_train_proc['content']
    Y = df_train_proc['label']
    print(f'X shape {X.shape}, Y shape {Y.shape}')

    # Shwo the input data before stemming
    X.head()

    # Create porter stemmer to stemming data
    stemmer = PorterStemmer()
    # Apply stemmer on the input data
    X_stemmed = X.apply(lambda x: stemming(x, stemmer))

    # Shwo the input data after stemming
    X_stemmed.head()

    # Split data into train and test data
    x_train, x_test, y_train, y_test = train_test_split(X_stemmed, Y, train_size = 0.1, random_state = 42)
    print(f'x train shape {x_train.shape}, x test shape {x_test.shape}')
    print(f'y train shape {y_train.shape}, y test shape {y_test.shape}')

    # ---------------------------------------------------- Vectorize content into numerical values using TfidfVectorizer

    # Create vectorizer 
    vectorizer = TfidfVectorizer()

    # Fit and transform data
    X_train_vectorized = vectorizer.fit_transform(x_train)
    X_test_vectorized = vectorizer.transform(x_test)

    # Show input data after the vectorization process
    print(X_train_vectorized)

    # ------------------------------------------------- Modeling

    # Put all required models that can fit this problem
    models = {
        'LogisticRegression': LogisticRegression(),
        'RandomForestClassifier': RandomForestClassifier(),
        'LinearSVC': LinearSVC(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier()   
    }

    trained_models = {}
    for name, model in models.items():
        print(f'{name}:\n')
        model = fit_predict(model, X_train_vectorized, y_train, X_test_vectorized, y_test)
        trained_models[name] = deepcopy(model)

    # ------------------------------------------------- Save best model
    '''
    Based on the previous results, the Gradient Boosting Classifier model is the best one on the test data with accuracy 94%.
    '''

    # Access Gradient Boosting Classifier model
    GradientBoostingClassifier_model = trained_models['GradientBoostingClassifier']
    GradientBoostingClassifier_model

    # File paths
    model_path = 'Prepare model/GradientBoostingClassifier_model.sav'
    vectorizer_path = 'Prepare model/TfidfVectorizer.sav'

    # Save the Gradient Boosting Classifier model using joblib if not already saved
    if not os.path.exists(model_path):
        joblib.dump(GradientBoostingClassifier_model, model_path)
        print("Gradient Boosting Classifier model saved successfully!")
    else:
        print("Gradient Boosting Classifier model already exists.")

    # Save the TfidfVectorizer using joblib if not already saved
    if not os.path.exists(vectorizer_path):
        joblib.dump(vectorizer, vectorizer_path)
        print("TfidfVectorizer saved successfully!")
    else:
        print("TfidfVectorizer already exists.")
