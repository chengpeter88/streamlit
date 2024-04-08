import streamlit as st
import pandas as pd 
import numpy as np  
import seaborn as sns  
import streamlit.components.v1 as components    
import matplotlib.pyplot as plt

# Load EDA
from eda import run_eda_app
from ml import run_ml_app   



html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Early Stage DM Risk Data App </h1>
		<h4 style="color:white;text-align:center;">Diabetes </h4>
		</div>
		"""
desc_temp = """
			### Early Stage Diabetes Risk Predictor App
			This dataset contains the sign and symptoms data of newly diabetic or would be diabetic patient.
			#### Datasource
				- https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.
			#### App Content
				- EDA Section: Exploratory Data Analysis of Data
				- ML Section: ML Predictor App

			"""


def main(): 
    st.title('Main App')
    st.markdown(html_temp, unsafe_allow_html=True)  
    st.markdown(desc_temp, unsafe_allow_html=True)
    menu = ['Home','EDA','ML', 'About']    
    choice = st.sidebar.selectbox('Menu', menu) 
    if choice == 'Home':
        st.subheader('Home') 
    elif choice == 'EDA':   
        st.subheader('EDA')
        run_eda_app()   
    
    elif choice == 'ML':    
        st.subheader('ML')  
        run_ml_app()     
    elif choice == 'About': 
        st.subheader('About')


if __name__ == '__main__':
    main()  

