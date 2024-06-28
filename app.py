import streamlit as st
from joblib import load

# Load the model and vectorizer
clf = load('hate_speech_model.pkl')
cv = load('count_vectorizer.pkl')

def hate_speech_detection(tweet):
    data = cv.transform([tweet]).toarray()
    prediction = clf.predict(data)
    return prediction[0]

st.title("Hate Speech Detection Chatbot")

# Initialize the session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def add_to_chat_history(user_input, prediction):
    st.session_state.chat_history.append({"user": user_input, "prediction": prediction})

# User input
user_input = st.text_input("You:", key="user_input")

if st.button("Send") and user_input:
    prediction = hate_speech_detection(user_input)
    add_to_chat_history(user_input, prediction)

# Display chat history
for chat in st.session_state.chat_history:
    st.write(f"**You:** {chat['user']}")
    st.write(f"**Prediction:** {chat['prediction']}")

# Display prediction options
if user_input:
    prediction = hate_speech_detection(user_input)
    st.write(f"Prediction: {prediction}")
