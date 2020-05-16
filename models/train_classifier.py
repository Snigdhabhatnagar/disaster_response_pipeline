# import libraries
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize 
import nltk
nltk.download(['punkt','wordnet','stopwords'])
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
stop_words = set(stopwords.words('english'))
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

def load_data(database_filepath):
    """
    Load data from database
    Args : database_filepath - file name
    return : X - message column (input variable)
             Y - categories columns (target variable)
             category_names - Names of target categories
             
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('MessageCategorised',engine)

    X=df['message']
    Y=df[df.columns.difference(['message','genre','original','id'])]
    category_names=Y.columns
    return X,Y,category_names
    pass


def tokenize(text):
    """
    Process the text data
    
    Args : text - input text data to be processed
           
    return : clean_tokens - clean tokenised text 
    """
    tokens=word_tokenize(text)
    lemmatizer=WordNetLemmatizer()
    
    clean_tokens=[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
    pass


def build_model():
    """
    Builds the model
    
    Args : none
           
    return : cv - model 
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters={'vect__max_df':[0.5,1.0],
                'clf__estimator__n_estimators':[10,20,30],
           }
    cv = GridSearchCV(pipeline,param_grid=parameters,n_jobs=1,verbose=5)
    return cv
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    """
    To test the model
    
    Args : model - model received from build model method
           X_test - Test input data
           Y_test - Test output data
           category_names - names of the categories input data is to be classified into
           
    return : none 
    """
    Y_pred=model.predict(X_test)
    acc=[]
    for i,c in enumerate(Y_test.columns):
        print(c)
        print(classification_report(Y_test[c], Y_pred[:,i]))
        acc.append(accuracy_score(Y_test[c], Y_pred[:,i]))
    print('Accuracy :',np.mean(acc))

    pass


def save_model(model, model_filepath):
    """
    Export model as a pickle file
    
    Args : model - model received from build model method
           model_filepath - path of the pickle file where model will be loaded
           
    return : none 
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, Y_test)
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train[0:700], Y_train[0:700])
        
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