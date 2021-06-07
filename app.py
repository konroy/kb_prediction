import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import datetime

model = xgb.XGBRegressor()
model.load_model("model.json")

# dates = ['14/2/2021', '13/2/2021', '12/2/2021', '11/2/2021', '9/2/2021',
#        '8/2/2021', '7/2/2021', '6/2/2021', '5/2/2021', '4/2/2021',
#        '3/2/2021', '2/2/2021', '1/2/2021', '31/1/2021', '30/1/2021',
#        '29/1/2021', '28/1/2021', '27/1/2021', '26/1/2021', '25/1/2021',
#        '24/1/2021', '23/1/2021', '22/1/2021', '21/1/2021', '20/1/2021',
#        '19/1/2021', '18/1/2021', '17/1/2021', '16/1/2021', '15/1/2021',
#        '14/1/2021', '13/1/2021']



st.set_page_config(page_title ="Kapten Batik Sales Prediction",
                    initial_sidebar_state="collapsed",
                    page_icon="🔮")

st.title('📈 Kapten Batik Sales Prediction')
st.write('This app enables you to predict Kapten Batik\'s sale numbers based on their previous ad campaigns.')

with st.beta_container():
	st.subheader('Dataset 🏋️')
	st.write('The dataset used is from Kapten Batik\'s Facebook data about their ad campaigns. Here the app will try to predict the sales based on the given parameters.')
	st.write('Please note that because of the low number of usable data after cleaning there is dummy data inside the dataset.')

	with st.beta_expander('View of Dataset (Uncleaned)'):
		df = pd.read_csv('Kapten Batik Facebook Data(Ai Project).csv')
		st.write(df)

	with st.beta_expander('Parameters Used'):
		st.markdown('The dataset is cleaned and the given parameters are chosen to determine the sales.')
		st.write('Age : Age of the customer.')
		st.write('Gender : Gender of the customer')
		st.write('Day : Date of the sale.')
		st.write('Impressions : The number of times the ad is fetched.')
		st.write('Amount Spent (MYR): The amount spent in MYR for the ad.')
		st.write('Reach : Refers to the total number of people who have seen the ad.')

with st.beta_container():
	st.subheader('Input Parameters 🛠️')
	st.write('In this section you can modify the input parameters.')

	with st.beta_expander('Age'):
		age = st.selectbox(
			'Select Age',
			('13-17', '25-34', '35-44', '45-54', '55-64', '65+')
			)

	with st.beta_expander('Gender'):
		gender = st.selectbox(
			'Select Gender',
			('Male', 'Female')
			)

	with st.beta_expander('Day'):
		date = st.date_input(
			"Select Day",value = datetime.date(2021, 1, 13), min_value=datetime.date(2021, 1, 13), max_value=datetime.date(2021, 2, 14)
			)

	with st.beta_expander('Impressions'):
		impression = st.number_input(value=0, label="Input number of Impressions", min_value=0, max_value=20000)

	with st.beta_expander('Amount Spent (MYR)'):
		amountSpent = st.number_input(value=0.0, label="Input amount spent", min_value=0.0, max_value=100.0)

	with st.beta_expander('Reach'):
		reach = st.number_input(value=0, label="Input reach", min_value=0, max_value=14000)

with st.beta_container():
	st.subheader("Prediction 🔮")
	st.write('Model will try to predict sales from the input parameters given.')

	if st.checkbox('Predict Sales', key='predict'):
		try:
			with st.spinner('Predicting...'):

				male, female, age_13_17, age_25_34, age_35_44, age_45_54, age_55_64, age_65 = 0, 0, 0, 0, 0, 0, 0, 0

				if gender == 'Male':
					male = 1
				if gender == 'Female':
					female = 1

				if age == '13-17':
					age_13_17 = 1
				if age == '25-34':
					age_25_34 = 1
				if age == '35-44':
					age_35_44 = 1
				if age == '45-54':
					age_45_54 = 1
				if age == '55-64':
					age_55_64 = 1
				if age == '65+':
					age_65 = 1

				df_input = pd.DataFrame({"Impressions" : impression, 
                         "Amount spent (MYR)"  : amountSpent,
                         "Reach" : reach, 
                         "Day" : date.day,
                         "Month" : date.month,
                         "gdr_female" : 1,
                         "gdr_male" : 0,
                         "age_13-17": 0,
                         "age_25-34": 1,
                         "age_35-44": 0,
                         "age_45-54": 0,
                         "age_55-64": 0,
                         "age_65+": 0,
                        }, index=[0])

				prediction = model.predict(df_input)
				st.write("Predicted Sales: ", round(prediction[0],2))
		except:
			st.warning('Oops error!')

# prediction = model.predict(df_test)