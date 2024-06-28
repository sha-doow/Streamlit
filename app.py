import streamlit as st
from joblib import load

# Load the model and vectorizer
clf = load('hate_speech_model.pkl')
cv = load('count_vectorizer.pkl')

def hate_speech_detection(tweet):
    data = cv.transform([tweet]).toarray()
    prediction = clf.predict(data)
    return prediction[0]

st.title("Shadow's Hate Speech Detection Chatbot")

# Initialize the session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def add_to_chat_history(user_input, prediction):
    st.session_state.chat_history.append({"user": user_input, "prediction": prediction})

# User input
user_input = st.text_input("Enter Tweet", key="user_input")

if st.button("") and user_input:
    prediction = hate_speech_detection(user_input)
    add_to_chat_history(user_input, prediction)
    
# Display chat history in chat format
st.markdown(
    """
    <style>
    .user-message {
        background-color: #000000;
        padding: 10px;
        border-radius: 10px;
        text-align: right;
        margin-bottom: 5px;
    }
    .bot-response {
        background-color: #023020;
        padding: 10px;
        border-radius: 10px;
        text-align: left;
        margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True
)

for chat in st.session_state.chat_history:
    st.markdown(f"<div class='user-message'>User: {chat['user']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-response'>Responds: {chat['prediction']}</div>", unsafe_allow_html=True)
