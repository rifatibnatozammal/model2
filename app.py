import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Streamlit app
def main():
    st.title("Predictive Analysis with Random Forest")

    # Load the saved Random Forest model
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Define inputs for features
    CM = st.number_input("CM", value=0.0, step=0.1)
    TS = st.number_input("TS", value=0.0, step=0.1)
    EN = st.number_input("EN", value=0.0, step=0.1)
    VC = st.number_input("VC", value=0.0, step=0.1)
    IC = st.number_input("IC", value=0.0, step=0.1)
    OR = st.number_input("OR", value=0.0, step=0.1)
    PC = st.number_input("PC", value=0.0, step=0.1)
    NI = st.number_input("NI", value=0.0, step=0.1)
    EF = st.number_input("EF", value=0.0, step=0.1)

    if st.button("Predict"):
        # Prepare the input data for prediction
        input_data = np.array([[CM, TS, EN, VC, IC, OR, PC, NI, EF]])
        prediction = model.predict(input_data)
        st.write("Prediction (Encoded):", prediction[0])

if __name__ == "__main__":
    main()
