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
prompt = st.chat_input("Enter Your Tweet")
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

 # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

response = f"Echo: {prompt}"
# Display assistant response in chat message container
with st.chat_message("assistant"):
    st.markdown(response)
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response})

if prompt:
    prediction = hate_speech_detection(prompt)
    st.write(f"Prediction: {prediction}")
    
    
