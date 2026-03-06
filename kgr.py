import streamlit as st
import pandas as pd
import joblib

#load our model and encoder
model = joblib.load("movie_model.pkl")
encoder = joblib.load("movie_encoder.pkl")

goal = st.selectbox("Select your goal:", ['job', 'Freelancing', 'Business'])
hobby = st.selectbox("Select your hobby", ['Programming', 'Design', 'Editing'])

st.title("Movie Recommendation System")


if st.button("Submit"):
    data = pd.DataFrame({
        "goal": [goal],
        "hobby": [hobby]
        })
    data['goal'] = data['goal'].str.lower
    data['hobby'] = data['hobby'].str.lower
        
    encoded_data = encoder.transform(data)
        
    prediction = model.predict(encoded_data)
    predicted = prediction[0]
    st.success(f"Recommended movie: {predicted}")
        
