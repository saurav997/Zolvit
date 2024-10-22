import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import  FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain 
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

from io import BytesIO  # Import BytesIO to handle bytes as file-like objects

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(BytesIO(pdf.read()))  # Convert bytes to a file-like object
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(text):
    text_splitter = Re(chunk_size  =10000,chunk_overlap = 1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # Create directory if it doesn't exist
    if not os.path.exists("faiss_index"):
        os.makedirs("faiss_index")
    # Save FAISS index
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question within the iven context in a detailed professional and succinct manner. If a particular information is not available
    within the provided context do not answer. Do not put any padding to your response, only give the answer to the query within the context
    given below:
    Context: \n {context}\n
    Query: \n{question}\n 
    Answer:
    """
    model = ChatGoogleGenerativeAI(model = "gemini-pro",temparature = .3)
    prompt = PromptTemplate(template = prompt_template,input_variables = ["context","question"])
    chain = load_qa_chain(model,chain_type = "stuff",prompt = prompt)
    return chain

# Modify the FAISS loading part to allow potentially dangerous deserialization.
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)  # Allow deserialization
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "questions": user_question},
        return_only_outputs=True
    )
    print(response)
    st.write("Reply:", response["output_text"])


def main():
    st.set_page_config(("Invoice information extractor"))
    st.header("Extract information from invoices using Gemini")
    user_query = st.text_input("Enter Your Query.")

    if user_query:
        user_input(user_query)
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF Invoices")
        if st.button("Submit"):
            with st.spinner("Vectorizing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__=="__main__":
    main()