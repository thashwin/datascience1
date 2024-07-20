import streamlit as st
import pandas as pd
import numpy as np
import random
from model_metrics import models, imputer, label_encoder, scaler, data_main, X_train, y_train, Scores

# Extract min and max values
min_longitude, max_longitude = data_main['longitude'].min(), data_main['longitude'].max()
min_latitude, max_latitude = data_main['latitude'].min(), data_main['latitude'].max()
min_housing_median_age, max_housing_median_age = data_main['housing_median_age'].min(), data_main['housing_median_age'].max()
min_total_rooms, max_total_rooms = data_main['total_rooms'].min(), data_main['total_rooms'].max()
min_total_bedrooms, max_total_bedrooms = data_main['total_bedrooms'].min(), data_main['total_bedrooms'].max()
min_population, max_population = data_main['population'].min(), data_main['population'].max()
min_households, max_households = data_main['households'].min(), data_main['households'].max()
min_median_income, max_median_income = data_main['median_income'].min(), data_main['median_income'].max()

# Generate value lists for selectbox and sort them
longitude_values = sorted(list(data_main['longitude'].unique()))
latitude_values = sorted(list(data_main['latitude'].unique()))
housing_median_age_values = list(range(int(min_housing_median_age), int(max_housing_median_age) + 1))
total_rooms_values = list(range(int(min_total_rooms), int(max_total_rooms) + 1))
total_bedrooms_values = list(range(int(min_total_bedrooms), int(max_total_bedrooms) + 1))
population_values = list(range(int(min_population), int(max_population) + 1))
households_values = list(range(int(min_households), int(max_households) + 1))
median_income_values = sorted(list(data_main['median_income'].unique()))
ocean_proximity_values = sorted(list(data_main['ocean_proximity'].unique()))

# Initialize session state with random values
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.longitude = random.choice(longitude_values)
    st.session_state.latitude = random.choice(latitude_values)
    st.session_state.housing_median_age = random.choice(housing_median_age_values)
    st.session_state.total_rooms = random.choice(total_rooms_values)
    st.session_state.total_bedrooms = random.choice(total_bedrooms_values)
    st.session_state.population = random.choice(population_values)
    st.session_state.households = random.choice(households_values)
    st.session_state.median_income = random.choice(median_income_values)
    st.session_state.ocean_proximity = random.choice(ocean_proximity_values)

# Title of the app
st.title("🏘️California House Price Prediction")

# Sidebar for model selection
model_name = st.sidebar.selectbox('Choose a model', ('Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting'))

# Load the selected model
model = models[model_name]

# Train and evaluate models
model.fit(X_train, y_train)

# Create a form for the input data
with st.form(key='input_form'):
    st.write("### Enter Input Details")
    st.session_state.longitude = st.selectbox('Longitude', options=longitude_values, index=longitude_values.index(st.session_state.longitude))
    st.session_state.latitude = st.selectbox('Latitude', options=latitude_values, index=latitude_values.index(st.session_state.latitude))
    st.session_state.housing_median_age = st.selectbox('Housing Median Age', options=housing_median_age_values, index=housing_median_age_values.index(st.session_state.housing_median_age))
    st.session_state.total_rooms = st.selectbox('Total Rooms', options=total_rooms_values, index=total_rooms_values.index(st.session_state.total_rooms))
    st.session_state.total_bedrooms = st.selectbox('Total Bedrooms', options=total_bedrooms_values, index=total_bedrooms_values.index(st.session_state.total_bedrooms))
    st.session_state.population = st.selectbox('Population', options=population_values, index=population_values.index(st.session_state.population))
    st.session_state.households = st.selectbox('Households', options=households_values, index=households_values.index(st.session_state.households))
    st.session_state.median_income = st.selectbox('Median Income', options=median_income_values, index=median_income_values.index(st.session_state.median_income))
    st.session_state.ocean_proximity = st.selectbox('Ocean Proximity', options=ocean_proximity_values, index=ocean_proximity_values.index(st.session_state.ocean_proximity))

    # Submit button for the form
    submit_button = st.form_submit_button(label='Predict')

# Perform the prediction if the form is submitted
if submit_button:
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'longitude': [st.session_state.longitude],
        'latitude': [st.session_state.latitude],
        'housing_median_age': [st.session_state.housing_median_age],
        'total_rooms': [st.session_state.total_rooms],
        'total_bedrooms': [st.session_state.total_bedrooms],
        'population': [st.session_state.population],
        'households': [st.session_state.households],
        'median_income': [st.session_state.median_income],
        'ocean_proximity': [st.session_state.ocean_proximity],
        'median_house_value': np.nan
    })

    combined_df = pd.concat([data_main, input_data], ignore_index=True)
    combined_df[['total_bedrooms', 'median_house_value']] = imputer.fit_transform(combined_df[['total_bedrooms', 'median_house_value']])
    combined_df['ocean_proximity'] = label_encoder.fit_transform(combined_df['ocean_proximity'])

    scaledfeatures = scaler.fit_transform(combined_df[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']])
    scaledfeatures_df = pd.DataFrame(scaledfeatures, columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity'])

    # Predict using the selected model
    prediction = model.predict(scaledfeatures_df.tail(1))
    
    # Display the prediction
    st.markdown(
        f"""
        <div style="background-color: #000000; padding: 5px; border-radius: 7px; border: 8px solid #fff; text-align: center;">
            <h3 style="color: #4CAF50;">Predicted price</h3>
            <p style="font-size: 24px; color: #white;">${prediction[0]:,.2f}</p>
        </div>
        """, unsafe_allow_html=True
    )

if st.sidebar.button("Model Scores"):
    scores = Scores()
    for score in scores:
        st.sidebar.write("-" * 10)
        for key, value in score.items():
            st.sidebar.write(f"{key}")
            for k, v in value.items():
                st.sidebar.write(f"{k}:{v}")