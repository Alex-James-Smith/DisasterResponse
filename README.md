# DisasterResponse

## Description

The Disaster Response Pipelines project is an assignment for the Udacity Data Scientist course. The purpose is to classify disaster response messages according to a number of possible categories. In order to achieve this, I employ an ETL pipeline, NLP pipeline and Machine Learning Pipeline to process data, create features and classify messages.

## Project Components

1. ETL Pipeline

    1. Loads the messages and categories datasets
    2. Merges the two datasets
    3. Cleans the data
    4. Stores it in a SQLite database

2. ML Pipeline

    1. Loads data from the SQLite database
    2. Splits the dataset into training and test sets
    3. Builds a text processing and machine learning pipeline
    4. Trains and tunes a model using GridSearchCV
    5. Outputs results on the test set
    6. Exports the final model as a pickle file

3. Flask Web App

    1. Modify file paths for database and model as needed
    2. Add data visualizations using Plotly in the web app. One example is provided for you
    
    
## Instructions

1. Run the following commands in the project's root directory to set up your database and model.
    1. To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    2. To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
  
2. Run the following command in the app's directory to run your web app. python run.py

3. Go to the website that is showed in your command line, should be something like http://0.0.0.0:3001/

## Requirements

    Flask==0.12.5
    nltk==3.2.5
    numpy==1.12.1
    pandas==0.23.3
    plotly==2.0.15
    scikit_learn==1.0.2
    SQLAlchemy==1.2.19