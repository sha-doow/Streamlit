import streamlit as st
from joblib import load

# Load the model and vectorizer
clf = load('hate_speech_model.pkl')
cv = load('count_vectorizer.pkl')

def hate_speech_detection(tweet):
    data = cv.transform([tweet]).toarray()
    prediction = clf.predict(data)
    return prediction[0]

st.title("Hate Speech Detection")
prompt = st.chat_input(placeholder="Enter Your Tweet", key=None, max_chars=None, disabled=False, on_submit=None, args=None, kwargs=None)

if prompt:
    prediction = hate_speech_detection(prompt)
    st.chat_input(f"Prediction: {prediction}")
    
    
