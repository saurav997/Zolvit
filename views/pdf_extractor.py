import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the extracted text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save the vector store for the text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to configure the question-answering chain
def get_conversational_chain():
    prompt_template = """
    You are an expert in understanding invoices. Provide detailed answers based on the input context
    that includes customer details, money paid itemwise, invoice number, items purchased, region, date etc. 
    If the answer is not available in the context, say 'Answer not found in the provided document.'

    Context: {context}
    Question: {question}

    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

# Function to handle user input, search in vector store, and return the response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    return response["output_text"]

# Streamlit app layout

st.title("PDF Invoice Information Extractor")

# Input prompt for user questions
user_question = st.text_input("Enter a question about the PDF invoice")

# Upload the PDF files (accept multiple files)
pdf_docs = st.file_uploader("Upload your PDF Invoices", accept_multiple_files=True, type=["pdf"])

# Button to create vector store
if st.button("Submit and Process PDF Invoice"):
    if pdf_docs:
        with st.spinner("Processing PDF invoices..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Vector store created successfully!")
    else:
        st.error("Please upload one or more PDF files.")

# Button to get response from the vector store based on user input
if st.button("Submit Query"):
    if user_question:
        with st.spinner("Searching for relevant information..."):
            response = user_input(user_question)
            st.success("Information extracted successfully!")
            
            # Display the response in a neat text area
            st.text_area("Extracted Information", response, height=200)
    else:
        st.error("Please enter a question to extract information.")

