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

