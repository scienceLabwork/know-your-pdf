__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
import os
import streamlit as st
import dotenv
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPEN_AI_KEY"]

def llm_model(model_type):
    if(model_type=="ChatGPT-4o"):
        llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        max_tokens=None,
        timeout=None,
        max_retries=2
        )
    else:
        models = {
            "Llama3-8b":"meta-llama/Meta-Llama-3-8B-Instruct",
            "Gemma1.1-7B":"google/gemma-1.1-7b-it",
            "Mistral0.2-7B":"mistralai/Mistral-7B-Instruct-v0.2"
        }
        llm = HuggingFaceEndpoint(
            repo_id=models[model_type],
            temperature=0.5,
        )   
    return llm

def pdf_data(pdf_path):
    doc = PyPDFLoader(pdf_path)
    data = doc.load()
    return data

def create_chunks(data):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)
    embedding_function = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = Chroma.from_documents(docs, embedding_function)
    return db

def get_relevant_context(query, db,top_k=3):
    results = db.similarity_search_with_score(query, k=top_k)
    # print(results)
    return '<NextContext>'.join([doc.page_content for doc, score in results])

def prompt_template():
    template = """
    CONTEXT: {context}
    __________________

    QUESTION: {question}

    Note: You are context chatbot. Answer the users QUESTION using the CONTEXT above.
    Keep your answer ground in the facts of the CONTEXT.
    If the CONTEXT doesn't contain the facts to answer the QUESTION return something in small talk.
    The context sometimes refers to pdf. If use ask for summary of the pdf, return the summary of the pdf. 
    If user tries for small talk, try to reply to it.
    You have to output answer in properly formated markdown format. There should be no error in output format. If you give correct answer you will get 100 points.
    The answer should not contain any irrevelant information. Don't include ```markdown``` key word to identify the code is starting directly start wrting markdown.

    Answer: 
    ."""
    prompt = PromptTemplate.from_template(template)
    return prompt

def get_output(question, db, model="ChatGPT-4o"):
    prompt = prompt_template()
    context = get_relevant_context(question, db)
    llm = llm_model(model)
    llm_chain = prompt | llm
    if(model=="ChatGPT-4o"):
        output = llm_chain.invoke(input={'context':context, 'question':question})
        output = output.content
        return output
    else:
        output = llm_chain.invoke(input={'context':context, 'question':question})
        return output
