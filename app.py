import streamlit as st
import pandas as pd

st.title("Rain Prediction for Tomorrow – India")

model = joblib.load("rain_predictor.pkl")
#inputs
col1, col2 = st.columns(2)
with col1:
    min_temp   = st.number_input("Min Temp (°C)", 0.0, 50.0, 20.0)
    humidity9  = st.number_input("Humidity 9 am (%)", 0, 100, 70)
    pressure9  = st.number_input("Pressure 9 am (hPa)", 900, 1100, 1010)
with col2:
    max_temp   = st.number_input("Max Temp (°C)", 0.0, 55.0, 30.0)
    humidity3  = st.number_input("Humidity 3 pm (%)", 0, 100, 60)
    pressure3  = st.number_input("Pressure 3 pm (hPa)", 900, 1100, 1005)

rain_today = st.radio("Rain Today?", ('No', 'Yes')) == 'Yes'

sample = {
    'MinTemp':min_temp, 'MaxTemp':max_temp,
    'Humidity9am':humidity9, 'Humidity3pm':humidity3,
    'Pressure9am':pressure9, 'Pressure3pm':pressure3,
    'WindSpeed9am':0, 'WindSpeed3pm':0,       # set others to 0/default
    'RainToday':int(rain_today),
    'Month':1, 'Day':1, 'DayOfWeek':0,
    'DeltaHum':humidity3 - humidity9,
    'DeltaPress':pressure3 - pressure9,
    'DeltaWind':0, 'AvgTemp':(min_temp+max_temp)/2
}
for col in model.get_booster().feature_names:
    if col not in sample:
        sample[col] = 0


X_new = pd.DataFrame([sample])
expected_features = ['MinTemp', 'MaxTemp', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
                             'WindSpeed9am', 'WindSpeed3pm', 'RainToday', 'Year', 'DayOfWeek', 'IsWeekend',
                             'DeltaHum', 'DeltaPress', 'DeltaWind', 'AvgTemp',
                             'Location_Chennai', 'Location_Kolkata', 'Location_Mumbai', 'Location_New Delhi']

X_new = X_new[expected_features]

if st.button("Predict"):
    prob = model.predict_proba(X_new)[0,1]
    st.write(f"**Probability of Rain Tomorrow:** {prob:.2%}")
    st.success("Carry an umbrella! ☔") if prob>=0.5 else st.info("Likely no rain.")
