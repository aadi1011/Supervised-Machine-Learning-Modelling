# Python script to deploy all the SML models in a Streamlit App.

# Importing the libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Title of the App
st.title('Supervised Machine Learning')
st.write(
    'Helping you understand the basics of Supervised Machine Learning by showcasing various models and their '
    'performance on different datasets.')
st.markdown('---')

# Choose any one model from the dropdown and if you want to see the dataset, check the checkbox.
st.sidebar.title('Choose a model')

choose_model = st.sidebar.selectbox('Select a model', (
    'Simple Linear Regression', 'Multiple Linear Regression', 'K-Nearest Neighbours', 'Random Forest Classifier',
    'Support Vector Machine', 'Logistic Regression'))

dataset = st.sidebar.checkbox('Show dataset')


# create a function to load the SalesLinearRegression model from pkl file based on the sales input
def predict_sales(sales):
    # load the model
    model = pickle.load(open('./models/SalesLinearRegression.pkl', 'rb'))
    # predict the sales
    sales_pred = model.predict([[sales]])
    return sales_pred


# create a function to load the iris-linear-prediction model from pkl file based on the sepal length, sepal width,
# petal length and petal width user input
def predict_iris_linear(sepal_length, sepal_width, petal_length, petal_width):
    # load the model
    model = pickle.load(open('./models/iris-prediction-linear-reg-model.pkl', 'rb'))
    # predict the class
    class_pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    if class_pred == 0:
        class_pred = 'Iris-setosa'
    elif class_pred == 1:
        class_pred = 'Iris-versicolor'
    elif class_pred == 2:
        class_pred = 'Iris-virginica'
    return class_pred


# create a function to load the iris-knn-prediction model from pkl file based on the sepal length, sepal width,
# petal length and petal width user input
def predict_iris_knn(sepal_length, sepal_width, petal_length, petal_width):
    # load the model
    model = pickle.load(open('./models/iris-prediction-knn-model.pkl', 'rb'))
    # predict the class
    class_pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    if class_pred == 0:
        class_pred = 'Iris-setosa'
    elif class_pred == 1:
        class_pred = 'Iris-versicolor'
    elif class_pred == 2:
        class_pred = 'Iris-virginica'
    return class_pred


# create a function to load the multiple-linear-reg bike share model from pkl file based on the user input
def predict_bikeshare_multiple(season, yr, mnth, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed):
    # load the model
    model = pickle.load(open('./models/multiple-linear-reg-days.pkl', 'rb'))
    # predict the class
    class_pred = model.predict([[season, yr, mnth, holiday, weekday, workingday, weathersit, temp, atemp, hum,
                                 windspeed]])
    return class_pred


# create a function to load the housing svm model from pkl file based on the user input
def predict_housing_svm(gender, married, dependants, self_employed, applicant_income, coapplicant_income, loan_amount,
                        loan_term, property_area):
    # load the model
    model = pickle.load(open('./models/housing_svm_model.pkl', 'rb'))
    # predict the class
    class_pred = model.predict([[gender, married, dependants, self_employed, applicant_income, coapplicant_income,
                                 loan_amount, loan_term, property_area]])
    if class_pred == 0:
        class_pred = 'Not Eligible'
    elif class_pred == 1:
        class_pred = 'Eligible'
    return class_pred


# create a function to load the social svm model from pkl file based on the user input
def predict_social_svm(age, est_salary):
    # load the model
    model = pickle.load(open('./models/social_svm_model.pkl', 'rb'))
    # predict the class
    class_pred = model.predict([[age, est_salary]])
    if class_pred == 0:
        class_pred = 'would not purchase'
    elif class_pred == 1:
        class_pred = 'would purchase'
    return class_pred


# create a function to load the random forest iris model from pkl file based on the user input
def predict_iris_random_forest(sepal_length, sepal_width, petal_length, petal_width):
    # load the model
    model = pickle.load(open('./models/randomforest_iris.pkl', 'rb'))
    # predict the class
    class_pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    if class_pred == 0:
        class_pred = 'Iris-setosa'
    elif class_pred == 1:
        class_pred = 'Iris-versicolor'
    elif class_pred == 2:
        class_pred = 'Iris-virginica'
    return class_pred


# create a function to load the random forest stroke prediction model from pkl file based on the user input
def predict_stroke_random_forest(gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status):
    # load the model
    model = pickle.load(open('./models/randomforest_stroke.pkl', 'rb'))
    # predict the class
    class_pred = model.predict([[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]])
    if class_pred == 0:
        class_pred = 'unlikely'
    elif class_pred == 1:
        class_pred = 'likely'
    return class_pred

def predict_breast_cancer_knn(radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean):
    # load the model
    model = pickle.load(open('./models/knnClass_breastcancer.pkl', 'rb'))
    # predict the class
    class_pred = model.predict([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean]])
    if class_pred == 0:
        class_pred = 'Benign'
    elif class_pred == 1:
        class_pred = 'Malignant'
    return class_pred

# ''' STREAMLIT MODEL MENU '''

# if the model selected is Simple Linear Regression, give option whether to try simple linear regression on iris dataset
# or sales dataset
if choose_model == 'Simple Linear Regression':
    st.subheader('Simple Linear Regression')
    st.write('Choose a dataset to try Simple Linear Regression on.')
    # radio button to choose the dataset
    dataset_radio = st.radio('Choose a dataset', ('Iris Dataset', 'Sales Dataset'))
    st.markdown('---')

    # if the dataset chosen is iris dataset
    if dataset_radio == 'Iris Dataset':
        st.write('Iris Dataset.')
        st.write("The Iris Dataset contains 3 classes of 50 instances each, where each class refers to a type of iris "
                 "flower")
        # get the user input
        sepal_length = st.number_input('Enter the sepal length', min_value=0.0, max_value=10.0, value=0.0)
        sepal_width = st.number_input('Enter the sepal width', min_value=0.0, max_value=10.0, value=0.0)
        petal_length = st.number_input('Enter the petal length', min_value=0.0, max_value=10.0, value=0.0)
        petal_width = st.number_input('Enter the petal width', min_value=0.0, max_value=10.0, value=0.0)
        # if the user clicks on the predict button
        if st.button('Predict'):
            # call the predict_iris_linear function to get the prediction
            prediction = predict_iris_linear(sepal_length, sepal_width, petal_length, petal_width)
            # display the prediction
            st.success('The class of the flower is {}'.format(prediction))

        if dataset:
            # load the iris dataset
            iris_df = pd.read_csv('./data/iris.csv')
            # display the dataset
            st.write(iris_df)

    # if the dataset chosen is sales dataset
    elif dataset_radio == 'Sales Dataset':
        st.write('Sales Dataset.')
        st.write("The Sales Dataset predicts the estimated sales value given the spending for advertising.")
        # get the user input
        sales = st.number_input("Enter the advertising value (0-100,000):", min_value=0.0, max_value=100000.0,
                                value=0.0)
        # if the user clicks on the predict button
        if st.button('Predict'):
            # call the predict_sales function to get the prediction
            prediction = predict_sales(sales)
            # display the prediction
            st.success('The sales prediction is {}'.format(prediction))

        if dataset:
            # load the sales dataset
            sales_df = pd.read_csv('./data/SALES.csv')
            # display the dataset
            st.write(sales_df)
            
if choose_model == 'Multiple Linear Regression':
    st.subheader('Multiple Linear Regression')
    st.write('Bike Sharing Dataset.')
    # get the user input

    season = st.radio('Enter the season', ('springer', 'summer', 'fall', 'winter'))
    if season == 'springer':
        season = 1
    elif season == 'summer':
        season = 2
    elif season == 'fall':
        season = 3
    elif season == 'winter':
        season = 4

    yr = st.radio('Enter the year', ('2011', '2012'))
    if yr == '2011':
        yr = 0
    elif yr == '2012':
        yr = 1
        
    mnth = st.selectbox('Enter the month', ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                                            'September', 'October', 'November', 'December'))
    if mnth == 'January':
        mnth = 1
    elif mnth == 'February':
        mnth = 2
    elif mnth == 'March':
        mnth = 3
    elif mnth == 'April':
        mnth = 4
    elif mnth == 'May':
        mnth = 5
    elif mnth == 'June':
        mnth = 6
    elif mnth == 'July':
        mnth = 7
    elif mnth == 'August':
        mnth = 8
    elif mnth == 'September':
        mnth = 9
    elif mnth == 'October':
        mnth = 10
    elif mnth == 'November':
        mnth = 11
    elif mnth == 'December':
        mnth = 12
        
    holiday = st.radio('Is it a holiday?', ('Yes', 'No'))
    if holiday == 'Yes':
        holiday = 1
    elif holiday == 'No':
        holiday = 0

    weekday = st.selectbox('Enter the weekday', ('Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                                                 'Saturday'))
    if weekday == 'Sunday':
        weekday = 0
    elif weekday == 'Monday':
        weekday = 1
    elif weekday == 'Tuesday':
        weekday = 2
    elif weekday == 'Wednesday':
        weekday = 3
    elif weekday == 'Thursday':
        weekday = 4
    elif weekday == 'Friday':
        weekday = 5
    elif weekday == 'Saturday':
        weekday = 6

    workingday = st.radio('Is it a working day?', ('Yes', 'No'))
    if workingday == 'Yes':
        workingday = 1
    elif workingday == 'No':
        workingday = 0

    weathersit = st.selectbox('Enter the weather situation', ('Clear, Few clouds, Partly cloudy, Partly cloudy',
                                                              'Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist',
                                                              'Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds',
                                                              'Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog'))
    if weathersit == 'Clear, Few clouds, Partly cloudy, Partly cloudy':
        weathersit = 1
    elif weathersit == 'Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist':
        weathersit = 2
    elif weathersit == 'Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds':
        weathersit = 3
    elif weathersit == 'Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog':
        weathersit = 4

    temp = st.number_input('Enter the temperature (in Celc)', min_value=0.0, max_value=50.0, value=0.0)
    
    atemp = st.number_input('Enter the feels-like temperature', min_value=0.0, max_value=50.0, value=0.0)
    
    hum = st.number_input('Enter the humidity (%)', min_value=0.0, max_value=100.0, value=0.0)

    windspeed = st.number_input('Enter the wind speed (kmph [0-67])', min_value=0.0, max_value=67.0, value=0.0)
    
    # if the user clicks on the predict button
    if st.button('Predict'):
        # call the predict_bikeshare_multiple function to get the prediction
        prediction = predict_bikeshare_multiple(season, yr, mnth, holiday, weekday, workingday, weathersit, temp, atemp,
                                                hum, windspeed)
        # display the prediction
        st.success('Predicted number of bikes to be rented: {}'.format(prediction))
    
    if dataset:
        # load the bikeshare dataset
        bikeshare_df = pd.read_csv('./data/day.csv')
        # display the dataset
        st.write(bikeshare_df)

if choose_model == 'K-Nearest Neighbours':

    knn_choice = st.sidebar.selectbox('Select the method', ('KNN Classification', 'KNN Regression'))
    if knn_choice == 'KNN Classification':
        st.subheader('K-Nearest Neighbour Classification')
        st.write('Choose a dataset to try KNN Classification on')
        # radio button to choose the dataset
        dataset_radio = st.radio('Choose a dataset', ('Iris Dataset', 'Breast Cancer Dataset'))
        st.markdown('---')
        if dataset_radio == 'Iris Dataset':
            st.markdown('##### Iris Flower Prediction')
            st.write(
                "The Iris Dataset contains 3 classes of 50 instances each, where each class refers to a type of iris "
                "flower")
            # get the user input
            sepal_length = st.number_input('Enter the sepal length', min_value=0.0, max_value=10.0, value=0.0)
            sepal_width = st.number_input('Enter the sepal width', min_value=0.0, max_value=5.0, value=0.0)
            petal_length = st.number_input('Enter the petal length', min_value=0.0, max_value=10.0, value=0.0)
            petal_width = st.number_input('Enter the petal width', min_value=0.0, max_value=5.0, value=0.0)
            # if the user clicks on the predict button
            if st.button('Predict'):
                # call the predict_knn function to get the prediction
                prediction = predict_iris_knn(sepal_length, sepal_width, petal_length, petal_width)
                # display the prediction
                st.success('The type of iris flower is: {}'.format(prediction))

            if dataset:
                # load the iris dataset
                iris = pd.read_csv('./data/iris.csv')
                # display the dataset
                st.write(iris)

        if dataset_radio == 'Breast Cancer Dataset':
            st.markdown('##### Breast Cancer Prediction')
            st.write(
                "The Breast Cancer Dataset contains 2 classes where each class refers to a type of breast cancer")
            # get the user input
            radius_mean = st.number_input('Enter the mean radius of lobes', min_value=0.0, max_value=40.0, value=0.0)
            texture_mean = st.number_input('Enter the mean surface texture', min_value=0.0, max_value=40.0, value=0.0)
            perimeter_mean = st.number_input('Enter the mean outer perimeter of lobes', min_value=0.0, max_value=300.0, value=0.0)
            area_mean = st.number_input('Enter the mean area of lobes', min_value=0.0, max_value=3000.0, value=0.0)
            smoothness_mean = st.number_input('Enter the mean smoothness level', min_value=0.0, max_value=0.3, value=0.0)
            compactness_mean = st.number_input('Enter the mean compactness', min_value=0.0, max_value=0.5, value=0.0)
            concavity_mean = st.number_input('Enter the mean concavity', min_value=0.0, max_value=0.5, value=0.0)
            concave_points_mean = st.number_input('Enter the mean concave points', min_value=0.0, max_value=0.5,
                                                    value=0.0)
            symmetry_mean = st.number_input('Enter the mean symmetry', min_value=0.0, max_value=0.5, value=0.0)
            fractal_dimension_mean = st.number_input('Enter the mean fractal dimension', min_value=0.0, max_value=0.5,
                                                        value=0.0)
