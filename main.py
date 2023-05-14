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
