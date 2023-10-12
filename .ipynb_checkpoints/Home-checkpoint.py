import streamlit as st
import seaborn as sns
import pandas as pd
import pickle
import sklearn

# st.set_page_config(layout='wide')

st.title("Our page")

st.write("This page is beautiful.")

def predict(new_peng):
    scaled_new_peng = my_scaler.transform(new_peng)
    prediction = my_kmeans.predict(scaled_new_peng)
    return prediction

peng_df = sns.load_dataset('penguins')

with st.expander("List the penguins"):

    st.dataframe(peng_df)

    
option = st.selectbox('Select species', ('Adelie','Chinstrap','Gentoo') )    
    
col1, col2, col3 = st.columns(3)    

if option == 'Adelie':
    with col1:
        st.image('Images/Adelie.png')
        
if option == 'Chinstrap':        
    with col2:
        st.image('Images/Chinstrap.png')
        
if option == 'Gentoo':        
    with col3:
        st.image('Images/Gentoo.png')
        
my_scaler = pickle.load(open('myscaler.pickle','rb'))
my_kmeans = pickle.load(open('mykmeans.pickle','rb'))

bill_length = st.slider('What is the bill lenght?',30,65,40)
bill_depth = st.slider('What is the bill depth?',10,25,15)
flipper_length = st.slider('What is the flipper length?',150,250,200)
body_mass = st.slider('What is the body_mass?',2500,6500,4000)

if st.button("Predict"):
    new_peng = pd.DataFrame({'bill_length_mm':[bill_length],'bill_depth_mm':[bill_depth],'flipper_length_mm':[flipper_length],'body_mass_g':[body_mass]})
    st.dataframe(new_peng)
    prediction = predict(new_peng)
    st.write(prediction)
    
