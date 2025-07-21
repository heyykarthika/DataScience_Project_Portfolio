# Save this as app.py
import streamlit as st
import pandas as pd
import pickle

# Load model
with open("log_reg_titanic.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚢 Titanic Survival Predictor")

# Inputs from user
pclass = st.selectbox("Passenger Class:", [1, 2, 3])
sex    = st.selectbox("Sex:", ["male", "female"])
age    = st.slider("Age:", 0, 80, 30)
sibsp  = st.number_input("Siblings/Spouses Aboard:", 0, 8, 0)
parch  = st.number_input("Parents/Children Aboard:", 0, 6, 0)
fare   = st.number_input("Fare Paid (£):", 0.0, 600.0, 32.2)
embark = st.selectbox("Embarked From:", ["S", "C", "Q"])

# Prepare input
input_df = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embark]
})

if st.button("Predict Survival"):
    prob = model.predict_proba(input_df)[0, 1]
    st.success(f"Estimated Survival Probability: **{prob:.2%}**")
