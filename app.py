import streamlit as st
import chainLLM
from io import BytesIO
from tempfile import NamedTemporaryFile

#DECLARE YOUR MODEL FROM OPTIONS HERE:
# - Llama3-8b
# - Mistral0.2-7B
# - Gemma1.1-7B
# - ChatGPT-4o
#USE THIS EXACT NAMES
model_choice = "ChatGPT-4o" #CHANGE ME

st.set_page_config(
    layout="wide",
    page_title="Know Your PDF",
    page_icon=":book:",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("Know Your PDF")
    st.write("A Chat with PDF Assistant, upload you PDF and ask questions.")
    uploaded_file = st.file_uploader("Choose a PDF file", accept_multiple_files=False)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if uploaded_file is not None:
    if "db" not in st.session_state:
        with st.spinner("Processing PDF file..."):
            with NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                data = chainLLM.pdf_data(temp_file.name)
            with st.spinner("Processing PDF file..."):
                st.session_state.db = chainLLM.create_chunks(data)

    if prompt := st.chat_input("How can I help you today?",max_chars=500, key="user_input"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        output = chainLLM.get_output(prompt, st.session_state.db, model_choice)
        response = f"{output}"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.chat_input("How can I help you today?",max_chars=500, key="user_input")
    st.success("Please upload a PDF file to continue.")