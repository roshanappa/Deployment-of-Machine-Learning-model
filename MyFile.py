import pandas as pd
import streamlit as st
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write('''

# Simple Irish Flower Prediction App
This app provide the Irish Flower type!
'''
)

st.sidebar.header('User Input Parameter')

def user_input_features():
    sepal_length= st.sidebar.slider('sepal_length',4.3,7.9,5.4)
    sepal_width = st.sidebar.slider('sepal_width',2.0,4.3,3.4)
    petal_length = st.sidebar.slider('petal_length',1.0,6.9,1.3)
    petal_width = st.sidebar.slider('petal_width',0.1,2.5,0.2)

    data = {'sepal_length':sepal_length,'sepal_width':sepal_width,'petal_length':petal_length,'petal_width':petal_width}
    feature = pd.DataFrame(data,index=[0])
    return feature

df = user_input_features()
st.subheader('User Input Parameter')
st.write(df)

irish = datasets.load_iris()
x = irish.data
y = irish.target
clf = RandomForestClassifier()
clf.fit(x,y)

prediction = clf.predict(df)
prediction_prob = clf.predict_proba(df)
st.subheader("Class Labels and their Corresponding index Number.")
st.write(irish.target_names)

st.subheader('Prediction')
st.write(irish.target_names[prediction])

st.subheader('Prediction Probablity')
st.write(prediction_prob)



