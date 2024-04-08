import streamlit as st  
import pandas as pd 
# load EDA  data visuyalization libraries   
import matplotlib.pyplot as plt
import seaborn as sns   
plt.style.use('fivethirtyeight')
import warnings  
warnings.filterwarnings('ignore')   
import plotly.express as px 
st.cache_data() 
def load_data(data):
    df = pd.read_csv(data)
    return df      


def run_eda_app():
    st.subheader('Exploratory Data Analysis')
    #df = pd.read_csv('data/diabetes_data_upload.csv')
    df = load_data('data/diabetes_data_upload.csv') 
    df_encode = load_data('data/diabetes_data_upload_clean.csv')
    freq = load_data('data/freqdist_of_age_data.csv')  
    # st.dataframe(df)
    submenu = st.sidebar.selectbox('Submenu', ['Descriptive Statistics', 'Plots'])  
    if submenu == 'Descriptive Statistics':
        st.dataframe(df)
        with st.expander('Data Types'):
            st.write(df.dtypes)

        with st.expander('Descriptive Statistics'):
            st.dataframe(df.describe())
        
        with st.expander('Class Distribution'):
            st.write(df['class'].value_counts())
        
        with st.expander('Gender Distribution'):    
            st.dataframe(df['Gender'].value_counts())

    elif submenu == 'Plots':
        st.subheader('Plots')
        col1, col2 = st.columns([2,1])
        with col1:
            #gender distribution  & info
            with st.expander('Dis plot of gender'):
                # fig = plt.figure(figsize=(6,6))
                # sns.countplot(df['Gender'])  
                # st.pyplot(fig)
                gen_df = df['Gender'].value_counts().to_frame()
                # st.dataframe(gen_df)
                gen_df = gen_df.reset_index()
                gen_df.columns = ['Gender Type', 'Counts']
                #st.dataframe(gen_df)
                p1 = px.pie(gen_df, names='Gender Type', values='Counts')   
                st.plotly_chart(p1,use_container_width=True) # 把大小調整到container的大小
            #for class distribution 
            with st.expander('Dis Plot of Class'):
                fig = plt.figure(figsize=(6,6))
                sns.countplot(df['class'])
                st.pyplot(fig)  

        
        
        with col2:  
            with st.expander('Gender Distribution'):    
                st.dataframe(gen_df)

            with st.expander('Class Distribution'): 
                st.write(df['class'].value_counts())


        #   freq distribution of age  
        with st.expander('Frequency Distribution of Age'):
            fig = plt.figure(figsize=(6,6))
            p2 =px.bar(freq, x='Age', y='count', color='Age', height=600)  
            st.plotly_chart(p2, use_container_width=True)   
            sns.histplot(df['Age'], kde=True)
            st.pyplot(fig)  
        
        # outliner decection    
        with st.expander('Outliner Detection'):
            fig = plt.figure(figsize=(6,6))
            sns.boxplot(df['Age'])
            st.pyplot(fig)  

            p3 = px.box(df, x='Age', points='all',color='class')
            st.plotly_chart(p3, use_container_width=True)

        # correlation matrix    
        with st.expander('Correlation Matrix'):
            fig = plt.figure(figsize=(10,8))
            sns.heatmap(df_encode.corr(), annot=True)
            st.pyplot(fig)  


            p4 = px.imshow(df_encode.corr())            
            st.plotly_chart(p4, use_container_width=True)   
         
                                    

            

        # if st.checkbox('Correlation Matrix'):
        #     st.write(sns.heatmap(df.corr(), annot=True))
        #     st.pyplot()
        # if st.checkbox('Pairplot'):
        #     st.write(sns.pairplot  (df, hue='class'))
        
        