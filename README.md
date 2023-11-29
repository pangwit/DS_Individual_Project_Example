# Deployed Data Science Project Sample


## Introduction
The objective of this project is to practice using the Heroku platform as a service (PaaS) for the automatic deployment of a machine learning web app. I was curious to learn if I could create and push such a project in a few hours. Your objective may be to provide a data-insightful solution to a specific problem that excites you using the tools discussed (or related to) in this course.

This project is inspired by the [the pycaret post](https://towardsdatascience.com/build-and-deploy-machine-learning-web-app-using-pycaret-and-streamlit-28883a569104) [3]. I kept their web app setup but replaced pycaret with pre-preprocessing using numpy/pandas and predictions using scikit. This allowed my free-tier Heroku pod to be significantly smaller and faster. 

Since Heroku also enables integration with GitHub, it also [automatically builds and releases](https://devcenter.heroku.com/articles/github-integration) any updates to my git repo [1]. 

I wanted to ease the web app development and [Streamlit](https://www.streamlit.io/) made this possible [2]. It is an open-source library that focuses on data science and ML web app development. 

## Selection of Data

The model processing and training are conducted using a Jupyter Notebook and is available [here](https://github.com/pangwit/DS_Individual_Project_Example/blob/main/codes/Insurance-Model_Training_Notebook.ipynb).


The data has over 1,300 samples with 6 features: age, sex, BMI, no of children, smoker, region.  
The objective is to predict hospital charges.
The dataset can found online at [git](https://github.com/stedy/Machine-Learning-with-R-datasets)[4]. Many example solutions and analysis can be found at [kaggle](https://www.kaggle.com/mirichoi0218/insurance) [5]

Data preview: 
![data screenshot](./graph/insurance_data.png)


Note that data has categorical features in 3 cols: sex, smoker, and region.
I used OneHotEncoder/ColumnTransformer on these features and kept the rest of the features as is.
When using a linear regression model, the learning accuracy shoots up from 12% to 75% with this transformation. 

We then create a pipeline for automating the process for new data. I also experimented with different imputer strategies for missing data and added second-degree polynomial features for both the numeric and categorical data. The shown values are obtained by performing a grid search over these arguments. 
Pipeline preview: 
![pipeline screenshot](./graph/pipeline.png)

I finally saved the model via joblib to be used for predictions by the web app. 

## Methods

Tools:
- NumPy, SciPy, Pandas, and Scikit-learn for data analysis and inference
- Streamlit (st) for web app design
- GitHub and Heroku for web app deployment and hosting/version control
- VS Code as IDE

Inference methods used with Scikit:
- linear regression model
- Features: OneHotEncoder/ColumnTransformer, StandardScaler, PolynomialFeatures and SimpleImputer for missing values
- Pipeline to tie it all together
- GridSearchCV for hyperparameter tuning

## Results
The app is live at https://ds-example.herokuapp.com/
It allows for online and batch processing as designed by the pycaret post:
- Online: User inputs each feature manually for predicting a single insurance cost
![online screenshot](./graph/online.png)
- Batch: It allows the user to upload a CSV file with the 6 features for predicting many instances at once. 
  - An [X_test.csv](./data/X_test.csv) is provided as a batch processing sample. Corresponding insurance prices are available at [y_test.csv](./data/y_test.csv)
![batch screenshot](./graph/batch.png)

I am not adding any visualizations to this example, though st supports it. Couple good examples are [here](https://share.streamlit.io/tylerjrichards/book_reco/books.py) and [here](https://share.streamlit.io/streamlit/demo-uber-nyc-pickups/)

## Discussion
Experimenting with various feature engineering techniques and regression algorithms, I found that linear regression with one-hot encoding provided one of the highest accuracies despite its simpler nature. Across all these trials, my training accuracy was around 75% to 77%. Thus, I decided the deploy the pipelined linear regression model. The data was split 80/20 for testing and has a test accuracy of 73%. 

I looked at some kaggle notebooks studying this problem and found this to be an acceptable level of success for this dataset. I am interested in analyzing the training data further to understand why a higher accuracy can't be easily achieved, especially with non-linear kernels. 

One unexpected challenge was the free storage capacity offered by Heroku. I experimented with various versions of the libraries listed in `requirements.txt` to achieve a reasonable memory footprint. While I couldn't include the latest pycaret library due to its size, the current setup does include TensorFlow 2.3.1 (even though not utilized by this sample project) to illustrate how much can be done in Heroku's free tier: 
```
Warning: Your slug size (326 MB) exceeds our soft limit (300 MB) which may affect boot time.
```

## Summary
This sample project deploys a supervised regression model to predict insurance costs based on 6 features. After experimenting with various feature engineering techniques, the deployed model's testing accuracy hovers around 73%. 

The web app is designed using Streamlit, and can do online and batch processing, and is deployed using Heroku and Streamlit. The Heroku app is live at https://ds-example.herokuapp.com/.
 
Streamlit is starting to offer free hosting as well. The same repo is also deployed at [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/memoatwit/dsexample/app.py)  
More info about st hosting is [here](https://docs.streamlit.io/en/stable/deploy_streamlit_app.html).


## References
[1] [GitHub Integration (Heroku GitHub Deploys)](https://devcenter.heroku.com/articles/github-integration)

[2] [Streamlit](https://www.streamlit.io/)

[3] [The pycaret post](https://towardsdatascience.com/build-and-deploy-machine-learning-web-app-using-pycaret-and-streamlit-28883a569104)

[4] [Insurance dataset: git](https://github.com/stedy/Machine-Learning-with-R-datasets)

[5] [Insurance dataset: kaggle](https://www.kaggle.com/mirichoi0218/insurance)
