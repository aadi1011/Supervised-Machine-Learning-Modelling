# Python script to deploy all the SML models in a Streamlit App.

# Importing the libraries
import streamlit as st
import pandas as pd
import pickle

## ----- STREAMLIT --------

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
    'Support Vector Machine'))

dataset = st.sidebar.checkbox('Show dataset')

## -------- FUNCTIONS --------------

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
def predict_stroke_random_forest(gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type,
                                 avg_glucose_level, bmi, smoking_status):
    # load the model
    model = pickle.load(open('./models/randomforest_stroke.pkl', 'rb'))
    # predict the class
    class_pred = model.predict([[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type,
                                 avg_glucose_level, bmi, smoking_status]])
    if class_pred == 0:
        class_pred = 'unlikely'
    elif class_pred == 1:
        class_pred = 'likely'
    return class_pred


def predict_breast_cancer_knn(radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,
                              concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean):
    # load the model
    model = pickle.load(open('./models/knnClass_breastcancer.pkl', 'rb'))
    # predict the class
    class_pred = model.predict([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                                 compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
                                 fractal_dimension_mean]])
    if class_pred == 0:
        class_pred = 'Benign'
    elif class_pred == 1:
        class_pred = 'Malignant'
    return class_pred

def predict_abalone_knn(sex, length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight):
    # load the model
    model = pickle.load(open('./models/knn_regression_abalone.pkl', 'rb'))
    # predict the class
    class_pred = model.predict([[sex, length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight]])
    return class_pred


# ----------------- STREAMLIT MODEL MENU -----------------

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
            perimeter_mean = st.number_input('Enter the mean outer perimeter of lobes', min_value=0.0, max_value=300.0,
                                             value=0.0)
            area_mean = st.number_input('Enter the mean area of lobes', min_value=0.0, max_value=3000.0, value=0.0)
            smoothness_mean = st.number_input('Enter the mean smoothness level', min_value=0.0, max_value=0.3,
                                              value=0.0)
            compactness_mean = st.number_input('Enter the mean compactness', min_value=0.0, max_value=0.5, value=0.0)
            concavity_mean = st.number_input('Enter the mean concavity', min_value=0.0, max_value=0.5, value=0.0)
            concave_points_mean = st.number_input('Enter the mean concave points', min_value=0.0, max_value=0.5,
                                                  value=0.0)
            symmetry_mean = st.number_input('Enter the mean symmetry', min_value=0.0, max_value=0.5, value=0.0)
            fractal_dimension_mean = st.number_input('Enter the mean fractal dimension', min_value=0.0, max_value=0.5,
                                                     value=0.0)

            # if the user clicks on the predict button
            if st.button('Predict'):
                # call the predict_knn function to get the prediction
                prediction = predict_breast_cancer_knn(radius_mean, texture_mean, perimeter_mean, area_mean,
                                                       smoothness_mean,
                                                       compactness_mean, concavity_mean, concave_points_mean,
                                                       symmetry_mean,
                                                       fractal_dimension_mean)
                # display the prediction
                st.success('The type of breast cancer is: {}'.format(prediction))

            if dataset:
                # load the breast cancer dataset
                breast_cancer = pd.read_csv('./data/bdiag.csv')
                # display the dataset
                st.write(breast_cancer)

    if knn_choice == 'KNN Regression':
        st.subheader('K-Nearest Neighbour Regression')
        st.markdown('##### Abalone Age Prediction')
        st.write(
            "Predicting the age of abalone from physical measurements. The age of abalone is determined by cutting the "
            "shell through the cone, staining it, and counting the number of rings through a microscope -- a boring "
            "and time-consuming task. Other measurements, which are easier to obtain, are used to predict the age. "
            "Further information, such as weather patterns and location (hence food availability) may be required to "
            "solve the problem")

        # get the user input
        sex = st.radio("Choose a gender", ("Male", "Female", "Infant"))
        if sex == "Male":
            sex = 2
        if sex == "Female":
            sex = 0
        if sex == "Infant":
            sex = 1

        length = st.number_input('Enter the length of the abalone (longest shell measurement in mm)', min_value=0.0,
                         max_value=1.0, value=0.00, step=1e-3, format="%.3f")

        diameter = st.number_input('Enter the diameter of the abalone (perpendicular to length in mm)', min_value=0.0,
                         max_value=1.0, value=0.00, step=1e-3, format="%.3f")

        height = st.number_input('Enter the height of the abalone (with meat in the shell in mm)', min_value=0.0,
                         max_value=1.0, value=0.00, step=1e-3, format="%.3f")

        whole_weight = st.number_input('Enter the weight of the whole abalone (in grams)', min_value=0.0,
                            max_value=1.0, value=0.00, step=1e-3, format="%.3f")

        shucked_weight = st.number_input('Enter the weight of the meat of the abalone (in grams)', min_value=0.0,
                            max_value=1.0, value=0.00, step=1e-3, format="%.3f")

        viscera_weight = st.number_input('Enter the weight of the guts of the abalone (after bleeding in grams)', min_value=0.0,
                            max_value=1.0, value=0.00, step=1e-3, format="%.3f")

        shell_weight = st.number_input('Enter the weight of the shell of the abalone (after being dried in grams)', min_value=0.0,
                            max_value=1.0, value=0.00, step=1e-3, format="%.3f")

        # if the user clicks on the predict button
        if st.button('Predict'):
            prediction = predict_abalone_knn(sex, length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight)
            st.success('The age of the abalone is: {}'.format(prediction))

        if dataset:
            # load the abalone dataset
            abalone = pd.read_csv('./data/abalone_data.csv')
            # display the dataset
            st.write(abalone)


if choose_model == 'Support Vector Machine':
    st.write('Choose a dataset to try Support Vector Machine on')
    # radio button to choose the dataset
    dataset_radio = st.radio('Choose a dataset', ('Housing Dataset', 'Socials Dataset'))
    st.markdown('---')

    # if the user chooses the housing dataset
    if dataset_radio == 'Housing Dataset':
        st.write('The housing model predicts the load status of a house in Boston given the customer data')
        # get the user input
        gender = st.radio('Choose your gender', ('Male', 'Female'))
        if gender == 'Male':
            gender = 1
        if gender == 'Female':
            gender = 0
        married = st.radio('Are you married?', ('Yes', 'No'))
        if married == 'Yes':
            married = 1
        if married == 'No':
            married = 0
        dependents = st.slider('How many dependents do you have?', 0, 4)
        self_employed = st.radio('Are you self-employed?', ('Yes', 'No'))
        if self_employed == 'Yes':
            self_employed = 1
        if self_employed == 'No':
            self_employed = 0
        applicant_income = st.number_input('Enter your income', min_value=0.0, max_value=10000000.0, value=0.0)
        coapplicant_income = st.number_input('Enter your co-applicant\'s income', min_value=0.0, max_value=10000000.0,
                                             value=0.0)
        loan_amount = st.number_input('Enter the loan amount', min_value=0.0, max_value=100000000.0, value=0.0)
        loan_amount = loan_amount / 1000
        loan_amount_term = st.slider('Enter the loan amount term (in months)', 0, 500)
        property_area = st.radio('Choose your property area', ('Urban', 'Semi-urban', 'Rural'))
        if property_area == 'Urban':
            property_area = 2
        if property_area == 'Semi-urban':
            property_area = 1
        if property_area == 'Rural':
            property_area = 0

        # if the user clicks on the predict button
        if st.button('Predict'):
            prediction = predict_housing_svm(gender, married, dependents, self_employed, applicant_income,
                                             coapplicant_income, loan_amount, loan_amount_term, property_area)
            # display the prediction
            st.success('The loan status of the house is: {}'.format(prediction))

        if dataset:
            # load the housing dataset
            housing = pd.read_excel('./data/house_loan_train.xlsx')
            # display the dataset
            st.write(housing)

    # if the user chooses the socials dataset
    if dataset_radio == 'Socials Dataset':
        st.write(
            'The socials model predicts whether a customer would buy a product or not based on their age and salary')
        # get the user input
        age = st.number_input('Enter your age', min_value=0, max_value=100, value=0)
        estimated_salary = st.number_input('Enter your estimated salary', min_value=0.0, max_value=10000000.0,
                                           value=0.0)
        # if the user clicks on the predict button
        if st.button('Predict'):
            prediction = predict_social_svm(age, estimated_salary)
            # display the prediction
            st.success('The person {} the product'.format(prediction))

        if dataset:
            # load the socials dataset
            socials = pd.read_csv('./data/Social_Network_Ads.csv')
            # display the dataset
            st.write(socials)

if choose_model == 'Random Forest Classifier':
    st.subheader('Random Forest Classifier')
    st.write('Choose a dataset to try Random Forest Classifier on')
    # radio button to choose the dataset
    dataset_radio = st.radio('Choose a dataset', ('Iris Dataset', 'Stroke Prediction Dataset'))
    st.markdown('---')

    # if the user chooses the iris dataset
    if dataset_radio == 'Iris Dataset':
        st.write('The Iris Dataset contains 3 classes of 50 instances each, where each class refers to a type of iris '
                 'flower')
        # get the user input
        sepal_length = st.number_input('Enter the sepal length', min_value=0.0, max_value=10.0, value=0.0)
        sepal_width = st.number_input('Enter the sepal width', min_value=0.0, max_value=5.0, value=0.0)
        petal_length = st.number_input('Enter the petal length', min_value=0.0, max_value=10.0, value=0.0)
        petal_width = st.number_input('Enter the petal width', min_value=0.0, max_value=5.0, value=0.0)
        # if the user clicks on the predict button
        if st.button('Predict'):
            # call the predict_knn function to get the prediction
            prediction = predict_iris_random_forest(sepal_length, sepal_width, petal_length, petal_width)
            # display the prediction
            st.success('The type of iris flower is: {}'.format(prediction))

        if dataset:
            # load the iris dataset
            iris = pd.read_csv('./data/iris.csv')
            # display the dataset
            st.write(iris)

    if dataset_radio == 'Stroke Prediction Dataset':
        st.write('The Stroke Prediction Dataset predicts whether a person is likely to get a stroke or not based on '
                 'health data')
        # get the user input
        gender = st.radio('Select your gender:', ('Male', 'Female'))
        if gender == 'Male':
            gender = 1
        if gender == 'Female':
            gender = 0
        age = st.slider('Enter your age', min_value=0, max_value=100, value=0)
        hypertension = st.radio('Do you have hypertension?', ('Yes', 'No'))
        if hypertension == 'Yes':
            hypertension = 1
        if hypertension == 'No':
            hypertension = 0
        heart_disease = st.radio('Do you have a history heart disease?', ('Yes', 'No'))
        if heart_disease == 'Yes':
            heart_disease = 1
        if heart_disease == 'No':
            heart_disease = 0
        ever_married = st.radio('Are you married?', ('Yes', 'No'))
        if ever_married == 'Yes':
            ever_married = 1
        if ever_married == 'No':
            ever_married = 0
        work_type = st.radio('What is your work type?', ('Private', 'Self-employed', 'Govt-job'))
        if work_type == 'Private':
            work_type = 1
        if work_type == 'Self-employed':
            work_type = 2
        if work_type == 'Govt-job':
            work_type = 0
        residence_type = st.radio('What is your residence type?', ('Urban', 'Rural'))
        if residence_type == 'Urban':
            residence_type = 1
        if residence_type == 'Rural':
            residence_type = 0
        avg_glucose_level = st.number_input('Enter your average glucose level', min_value=0.0, max_value=350.0,
                                            value=0.0)
        bmi = st.number_input('Enter your BMI', min_value=0.0, max_value=60.0, value=0.0)
        smoking_status = st.selectbox('What is your smoking status?',
                                      ('Never Smoked', 'Formerly Smoked', 'Smokes', 'Unknown'))
        if smoking_status == 'Never Smoked':
            smoking_status = 2
        if smoking_status == 'Formerly Smoked':
            smoking_status = 1
        if smoking_status == 'Smokes':
            smoking_status = 3
        if smoking_status == 'Unknown':
            smoking_status = 0
        # if the user clicks on the predict button
        if st.button('Predict'):
            prediction = predict_stroke_random_forest(gender, age, hypertension, heart_disease, ever_married, work_type,
                                                      residence_type, avg_glucose_level, bmi, smoking_status)
            # display the prediction
            st.success('The person is {} to get a stroke'.format(prediction))
        if dataset:
            # load the stroke prediction dataset
            stroke = pd.read_csv('./data/stroke_prediction.csv')
            # display the dataset
            st.write(stroke)

## --------------- END OF FILE -------------------
