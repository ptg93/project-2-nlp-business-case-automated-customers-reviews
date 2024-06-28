This is the README file in which the explanation of what you can find in this repository will be.

** Project objective & end result **

The final objective of the project is to introduce an app and reporting where we show i. review classification and ii. review summaries.

Outcome is hosted and can be approached following the link: https://romantic-charm-production.up.railway.app

** File structure **

solution >>

    >> data
    we start with the Amazon consumer review rawdata downloaded from https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products/data
    
    file renamed into data.csv for further processing

    data exploration steps included in data_exploration.ipynb file. This file is used as starting to run the models.

    >> deploy
    Here you will find the deployment files

    >> LSTM
    LSTM executable files, variables and output file are included in this file

    >> Traditional_models
    Four models i. logistic_regression, ii. Naive_bayes, iii. Random_forest and iv. svm executed with workbook for each model

    >> Transformers
    Executable files and output file are included in this file
   
    Split between 
        i. categorisation
            - distilBert
            - pegasus

        ii. summarisation task
            - distilBart
            - AmazonBart

    Project files
    - Presentation
    - Report



