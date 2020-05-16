# Disaster Response Pipeline Project

## Table Of Content 
1) [Installation](#installation)
2) [Motivation](#motivation)
3) [Files Description](#files-description)
4) [Instructions to run the project](#instructions)
5) [Licensing Authors and Acknowledgements](#licensing)

### Installation   

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.   
The code should run with no issues using Python versions 3.     

### Motivation     

The project aims to provide help to an emergency worker who can input a new message and get classification results in several categories and display visualizations of the data to help tackle the problems faced during any disaster.   

### Files Description       

Project is divided into 3 folders:    
**data** - csv files for messages and categories, process_data.py processes the raw data into more meaningfull and usable form, DisasterRersponse.db contains processed data to be usen in ML model.    
**model** - train_classifier.py trains a machine learning pipeline to fit data and evaluate the results and stores the model in classifier.pkl file(which is not present now, but will be made once you run train_classifier.py     
**app** - Contains run.py, backend of the web app which uses the trained model to predict the category of messages sent during disasters.

<a id="instructions"></a>
### Instructions to run the project:

1. Run the following commands in the project's root directory to set up your database and model.

   - To run ETL pipeline that cleans data and stores in database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
   `python run.py`

3. Go to http://0.0.0.0:3001/
    
<a id="licensing"></a>
## Licensing,Authors,and Acknowledgements
Data used in the project is provided by Figure Eight which contains labelled and categorised messages into different categories.
