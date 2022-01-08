import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pickle

def load_data(database_filepath):
    '''
    Loads data from SQL database
    INPUT 
        database_filepath - location of disaster response database
    OUTPUT
        X - dataframe with disaster response messages (predictor variable)
        Y - dataframe with disaster response categories (response variables)
        columns - disaster response categories
    '''    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('clean_disaster', engine)  
    X = df['message']
    Y = df[df.columns[4:]]
    
    return X, Y, Y.columns


def tokenize(text):
    '''
    Extract only letters, numbers and spaces, tokenize, then lower, strip and lemmatise tokens. 
    INPUT 
        text - disaster response message to be processed
    OUTPUT
        clean_tokens - tokenised and standardised disaster response messages
    '''  
    text = re.compile('[^A-Za-z0-9\s]').sub('', text)
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Extract only letters, numbers and spaces, tokenize, then lower, strip and lemmatise tokens. 
    INPUT 
        None
    OUTPUT
        cv - Machine Learning Pipeline that tokenises messages, vectorises, performs TFIDF and uses Random Forest Classification with parameters to GridSearch for optimum model
    '''  
    pipeline = Pipeline([
    ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),

    ('clf', MultiOutputClassifier(RandomForestClassifier(max_depth = 6)))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [25, 50]
    }

    cv = GridSearchCV(pipeline, param_grid = parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate trained model, comparing predictions with test data for each category on precision, recall and f1-score
    INPUT 
        model - fitted model to predict disaster response message categories
        X_test - message test data
        Y_test - category test data
        category_names - names of categories
    OUTPUT
        None
    '''  
    y_pred = model.predict(X_test)
    
    for i, col in enumerate(Y_test):
        cr = classification_report(Y_test[col], [arr[i] for arr in y_pred])
        print(col)
        print(cr) 


def save_model(model, model_filepath):
    '''
    Save model to pickle
    INPUT 
        model - trained model
        model_filepath - filepath to save model to
    OUTPUT
        None
    '''  
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()