import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import datetime

model_sales, model_imp, model_reach = xgb.XGBRegressor(), xgb.XGBRegressor(), xgb.XGBRegressor()
model_sales.load_model("model_sales.json")
model_imp.load_model("model_impressions.json")
model_reach.load_model("model_reach.json")

st.set_page_config(page_title ="Kapten Batik Sales Prediction",
                    initial_sidebar_state="collapsed",
                    page_icon="🔮")

st.title('📈 Kapten Batik Prediction')
st.write('This app enables you to predict Kapten Batik\'s sale, impressions and reach numbers based on their previous ad campaigns.')

with st.beta_container():
	st.subheader('Dataset 🏋️')
	st.write('The dataset used is from Kapten Batik\'s Facebook data about their ad campaigns. Here the app will try to predict the sales, impressions and reach based on the given parameters.')
	st.write('Please note that because of the low number of usable data after cleaning there is dummy data inside the dataset.')

	with st.beta_expander('Parameters Used'):
		st.markdown('The dataset is cleaned and the given parameters are chosen to determine the sales.')
		st.write('Age : Age of the customer.')
		st.write('Gender : Gender of the customer')
		st.write('Day : Date of the sale. You will input a start date and end date of the sale.')
		st.write('Holiday : The holiday period that affect the sales.')
		st.write('Discount : The percentage of discount applied during the ad campaign for the products.')
		st.write('Amount Spent (MYR) : The amount spent in MYR for the ad.')

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
		start_date = st.date_input(
			"Select Day to start sale",value = datetime.date(2021, 1, 13), min_value=datetime.date(2021, 1, 13)
			)
		end_date = st.date_input(
			"Select Day to end sale",value = start_date, min_value=start_date
			)
		date_list = pd.date_range(start_date,end_date)

	with st.beta_expander('Holiday'):
		holiday = st.selectbox(
			'Select Holiday',
			('None', 'Chinese New Year', 'Hari Raya', 'Year End')
			)

	with st.beta_expander('Discount'):
		discount = st.selectbox(
			'Select Discount',
			('No Discount', '20%', '50%', '70%')
			)

	with st.beta_expander('Amount Spent (MYR)'):
		amountSpent = st.number_input(value=0.0, label="Input amount spent", min_value=0.0, max_value=1000.0)

with st.beta_container():
	st.subheader("Prediction 🔮")
	st.write('Model will try to predict sales, impressions and reach from the input parameters given.')

	if st.button('Predict!', key='predict'):
			with st.spinner('Predicting...'):

				male, female, pr_0_10, pr_10_30, pr_30_60, pr_60_90, pr_90_1000, dr_0_10, dr_10_20, dr_20_31, q1, q2, q3, q4, age_13_17, age_25_34, age_35_44, age_45_54, age_55_64, age_65, dis_0, dis_20, dis_50, dis_70, campNone, campCNY, campRaya, campYearEnd = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

				if gender == 'Male':
					male = 1
				if gender == 'Female':
					female = 1

				if amountSpent > 0 and amountSpent <= 10:
					pr_0_10 = 1
				elif amountSpent > 10 and amountSpent <= 30:
					pr_10_30 = 1
				elif amountSpent > 30 and amountSpent <= 60:
					pr_30_60 = 1
				elif amountSpent > 60 and amountSpent <= 90:
					pr_60_90 = 1
				elif amountSpent > 90 and amountSpent <= 1000:
					pr_90_1000 = 1

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

				
				if holiday == 'None':
					campNone = 1
				if holiday == 'Chinese New Year':
					campCNY = 1
				if holiday == 'Hari Raya':
					campRaya = 1
				if holiday == 'Year End':
					campYearEnd = 1

				if discount == 'No Discount':
					dis_0 = 1
				if discount == '20%':
					dis_20 = 1
				if discount == '50%':
					dis_50 = 1
				if discount == '70%':
					dis_70 = 1

				row = []
				for date in date_list:

					if date.day > 0 and date.day <= 10:
						dr_0_10 = 1
					elif date.day > 10 and date.day <= 20:
						dr_10_20 = 1
					elif date.day > 20 and date.day <= 31:
						dr_20_31 = 1

					if date.month > 0 and date.month <= 3:
						q1 = 1
					elif date.month > 3 and date.month <= 6:
						q2 = 1
					elif date.month > 6 and date.month <= 9:
						q3 = 1
					elif date.month > 9 and date.month <= 12:
						q4 = 1

					row.append([pr_0_10, pr_10_30, pr_30_60, pr_60_90, pr_90_1000, dr_0_10, dr_10_20, dr_20_31, q1, q2, q3, q4, campCNY, campNone, campRaya, campYearEnd, female, male, age_13_17, age_25_34, age_35_44, age_45_54, age_55_64, age_65, dis_0, dis_20, dis_50, dis_70]) 

				df_input = pd.DataFrame(row, columns=["pr_0-10", "pr_10-30", "pr_30-60", "pr_60-90", "pr_90-1000", "dr_0-10", "dr_10-20", "dr_20-31", "mr_q1", "mr_q2", "mr_q3", "mr_q4", "camp_CNY Campaign", "camp_None", "camp_Raya Campaign", "camp_Year End Campaign", "gdr_female", "gdr_male", "age_13-17", "age_25-34", "age_35-44", "age_45-54", "age_55-64", "age_65+", "dis_0", "dis_20", "dis_50", "dis_70"])
				sales_pred = model_sales.predict(df_input)
				imp_pred = model_imp.predict(df_input)
				reach_pred = model_reach.predict(df_input)

				data = {'Date': date_list.date, 'Sales': sales_pred, 'Impressions': imp_pred, 'Reach': reach_pred}
				df_pred = pd.DataFrame(data)

				fig = px.line(df_pred, x="Date", y=df_pred.columns,
				              hover_data={"Date": "|%B %d, %Y"},
				              title='Prediction Chart')
				fig.update_xaxes(
				    dtick="M1",
				    tickformat="%b\n%Y")

				st.plotly_chart(fig, use_container_width=True)
