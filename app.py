import streamlit as st
import os
from groq import Groq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']

def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

def create_conversation(model, memory_length):
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
    )
    memory = ConversationBufferWindowMemory(k=memory_length)
    return ConversationChain(
        llm=groq_chat,
        memory=memory
    )

def main():
    st.title("Groq Chat App")

    initialize_session_state()

    # Sidebar for model selection and memory length
    st.sidebar.title('Chat Settings')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama-3.2-3b-preview', 'llama-3.3-70b-versatile','llama-3.1-405b-reasoning', 'mixtral-8x7b-32768', 'llama3-70b-8192', 'llama3-8b-8192', 'llama-3.1-70b-versatile', 'gemma-7b-it','whisper-large-v3']
    )
    memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)

    # Create or update conversation object if settings change
    if st.session_state.conversation is None or st.sidebar.button('Update Settings'):
        st.session_state.conversation = create_conversation(model, memory_length)

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What is your question?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in st.session_state.conversation.stream(prompt):
                full_response += response['response']
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
