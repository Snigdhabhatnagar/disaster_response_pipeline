import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    load messages and categories dataset
    
    Args : message_filepath - path for the messages data file
           categories_filepath - path for the categories data file
         
    return : dataframe loaded with data
    """
    # read message_filepath csv file
    messages = pd.read_csv(messages_filepath) 
    
    # read categories_filepath csv file
    categories = pd.read_csv(categories_filepath) 
    
    # merge datasets
    df = pd.merge(messages,categories,on='id')
    return df
    pass


def clean_data(df):
    """
    cleans the dataset to be used in modeling
    
    Args : df - dataframe to be cleaned 
         
    return : cleaned dataframe 
    """
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df['categories'].str.split(';',expand=True))
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x:x[0:len(x)-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x.split('-')[1])    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df=df.drop(['categories'],axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # drop duplicates
    df=df.drop_duplicates(subset=None, keep='first', inplace=False)
    return df
    pass


def save_data(df, database_filename):
    """
    Save the clean dataset into an sqlite database
    
    Args : df - dataframe 
           dataase+filename - Name of the sqlite database
         
    return : none 
    """
    #create db
    engine = create_engine('sqlite:///'+database_filename)
    
    # save to db
    df.to_sql('MessageCategorised', engine, index=False)
    pass  


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