import streamlit as st

import MyFirstWebApp as mfwa

import MyFirstWebApp_ml as mfml

def main():
    # EDA
    mfwa.main()

    st.header("LogisticRegression Predictor")

    # Predictor
    mfml.main()



if(__name__ == '__main__'):
    main()
