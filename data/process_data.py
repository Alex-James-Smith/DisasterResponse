import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Merges message and categories datasets on id of message
    INPUT 
        messages_filepath - location of messages dataset
        categories_filepath - location of categories dataset
    OUTPUT
        df - dataframe with merged messages and categories datasets
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on = 'id', how = 'inner')
    
    return df


def clean_data(df):
    '''
    Cleans data: Separates categories on ;, extracts column names from row and replace column names, extracts category values, and removes duplicates
    INPUT 
        df - dataframe of merged messages and categories datasets
    OUTPUT
        df - dataframe with cleaned data
    '''
    categories  = df.categories.str.split(pat = ';', expand = True)
    
    row = categories.head(1)
    category_colnames = list(categories.head(1).apply(lambda x: x.str[:-2], axis = 1).iloc[0])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    for column in categories.columns:
        categories.loc[(categories[column]!=1)&(categories[column]!=0)] = 1
        
    df = df.drop(columns = 'categories')
    df = pd.concat([df, categories], axis=1)
    
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    '''
    Saves cleaned data to sql database table
    INPUT 
        df - location of cleaned dataset
        database_filename - location to create database
    OUTPUT
        None
        Saves data to clean_disaster table at database location
    '''
    engine = create_engine('sqlite:///' + database_filename) # DisasterResponse.db
    df.to_sql('clean_disaster', engine, index=False)     


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()