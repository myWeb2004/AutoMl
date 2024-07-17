import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


#machine stuff
from pycaret.classification import setup,pull,compare_models,save_model

if os.path.exists("source.csv"):
    df=pd.read_csv("source.csv")

with st.sidebar:
    st.title("Auto Data Science")
    st.image("https://i.pinimg.com/236x/1e/f6/f1/1ef6f1f8febeedc48fc4a0840e9a54a2.jpg")
    choice=st.radio("Navigation",["Upload","Profiling","Training","Download"])
    



if choice=="Upload":
    file=st.file_uploader("Upload Your Dataset")
    if file:
        df=pd.read_csv(file)
        st.dataframe(df)
        df.to_csv("source.csv",index=None)
if choice=="Profiling":
    profile_report=ProfileReport(df,title="report")
    gen_button=st.button("Generate Report")
    if gen_button:
        st_profile_report(profile_report)
if choice=="Training":
    target=st.selectbox("Select Your Target",df.columns)
    gen_button=st.button("Train")
    if gen_button:
        setup(df,target=target)
        setup_df=pull()
        st.dataframe(setup_df)
        bestmodels=compare_models()
        compare_df=pull()
        st.dataframe(compare_df)
        bestmodels
        save_model(bestmodels,"best_model")


if choice=="Download":
    with open("best_model.pkl","rb") as f:
        st.download_button("download model",f,"trained_model.pkl")