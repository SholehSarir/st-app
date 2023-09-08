import pandas as pd
import streamlit as st
from absenteeism_module import absenteeism_model
import warnings

# Disable warnings for cleaner output
warnings.filterwarnings("ignore")

def input_output():
    """
    Streamlit app for absenteeism prediction.

    Collects user input via file upload, and makes predictions based on the uploaded data.
    """
    st.title("Absenteeism Prediction")

    data = st.file_uploader("Please Upload Your File Here", type=["csv", "txt"])
    
    if data is not None:
        df = pd.read_csv(data)
        st.write(df)
        
        model = absenteeism_model('model', 'scaler')
        model.load_and_clean_data('Absenteeism_new_data.csv')

        result = ""
        
        if st.button("Click here to Predict"):
            result = model.predicted_outputs()
            st.balloons() 

        st.success('The output is as follows: ')
        st.write(result)

if __name__ == '__main__':
    input_output()
