import pickle
import streamlit as st
import pandas as pd
import numpy
import requests


import processor
from processor import chatbot_response

st.header('Rekomendasi NTT ')
ntt = pickle.load(open('ntt_list.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))
cosine_sim_df = pd.DataFrame(similarity, index=ntt['nama'], columns=ntt['nama'])

def recommend(ntt_title):
    recommendations = ntt_rec_CBF(ntt_title)
    recommended_ntt_names = []
    recommended_ntt_posters = []
    for ind in recommendations.index:
        ##ntt_id = (recommendations['id'][ind])
        recommended_ntt_posters.append((recommendations['image_url'][ind]))
        recommended_ntt_names.append(recommendations['nama'][ind])

    return recommended_ntt_names, recommended_ntt_posters

def ntt_rec_CBF(title_ntt, similarity_data=cosine_sim_df, items=ntt[['nama', 'objek', 'image_url']], k=10):
    index = similarity_data.loc[:,title_ntt].to_numpy().argpartition(
        range(-1, -k, -1))
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(title_ntt, errors='ignore')
    return pd.DataFrame(closest).merge(items).head(k)

ntt_list = ntt['nama'].values
selected_ntt = st.selectbox(
    "Nama Lokasi",
    ntt_list
)

if st.button('Rekomendasikan'):
    recommended_ntt,recommended_ntt_posters = recommend(selected_ntt)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_ntt[0])
        st.image(recommended_ntt_posters[0])
    with col2:
        st.text(recommended_ntt[1])
        st.image(recommended_ntt_posters[1])
    with col3:
        st.text(recommended_ntt[2])
        st.image(recommended_ntt_posters[2])
    with col4:
        st.text(recommended_ntt[3])
        st.image(recommended_ntt_posters[3])
    with col5:
        st.text(recommended_ntt[4])
        st.image(recommended_ntt_posters[4])



st.title("Wisatabot")

the_question = st.text_input('')
st.write(processor.chatbot_response(the_question))

