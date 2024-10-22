import streamlit as st
import os


st.title("Invoice Information Extractor")
st.subheader("Choose the format of the invoice from the Sidebar")

# Page Header
st.write("""
Welcome to the **Invoice Information Extractor** application. This app allows you to extract structured information from invoices, either in **JPG** or **PDF** format, using Google's **Gemini AI** model. Below are detailed instructions on how to use this tool, along with the complete documentation, including the `requirements.txt` file and the source code for both the **JPG Invoice Extractor** and **PDF Invoice Extractor**.
""")

# Instructions Section
st.header("📋 Instructions for Use")
# Section 1: Image Invoice Information Extractor
st.subheader("Image Invoice Information Extractor")
st.image("Images/JPG_Guide.jpg", caption="Steps 1 to 3 for Image Invoice Information Extractor")

st.markdown("""
- **Step 1** : Browse and upload the image of the invoice (Supported formats: JPG, JPEG, PNG).
- **Step 2** : Write your query in the provided input box (e.g., "What is the total amount on the invoice?").
- **Step 3** : Use the 'Submit Query' button to retrieve the relevant information from the invoice image.
""")

# Section 2: PDF Invoice Information Extractor
st.subheader("PDF Invoice Information Extractor")
st.image("Images/PDF_Guide.jpg", caption="Steps 1 to 4 for PDF Invoice Information Extractor")

st.markdown("""
- **Step 1**: Browse and upload the PDF of the invoice (Supported format: PDF).
- **Step 2**: Press the button labeled 'Submit and Process PDF Invoice' to extract the text and create the vector store for processing.
- **Step 3**: Include your query in the provided text box (e.g., "What is the due date of the invoice?").
- **Step 4**: Submit your query using the 'Submit Query' button to get a response based on the extracted data from the PDF invoice.
""")

st.write("For further assistance or troubleshooting, please contact Saurav R: +91 9745073293.")


# Documentation Section
st.header("📝 Documentation")
st.subheader("Requirements (`requirements.txt`)")
st.markdown("""
This project uses several libraries that you can install via **conda** by creating an environment as follows:

```bash
conda create ./venv
```

and activating it using the below command:

```bash
conda activate ./venv
```

and installing requirements with the below command:

```bash
pip install -r requirements.txt
```

#### List of Dependencies:
- `streamlit`: Frontend framework for interactive web apps.
- `google-generativeai`: Google Gemini model API for content generation.
- `python-dotenv`: To load environment variables from a `.env` file.
- `langchain`: Framework for building language model-based applications.
- `PyPDF2`: A library for reading PDF files and extracting text.
- `chromadb`: Vector database for storage of embeddings.
- `faiss-cpu`: Facebook AI Similarity Search, for efficient similarity search and clustering.
- `langchain_google_genai`: Integration for Google Gemini and Langchain.
- `langchain-community`: Community-supported Langchain utilities.
- `pandas`: Data manipulation and analysis library.
- `requests`: Library for making HTTP requests.

it is also required to have a .env folder with the Google Gemini API key:

```python
# The below key is fake
GOOGLE_API_KEY = "zaSyDbawA_GoQO3UahxLXaRarNCv"
```
""")

# Code Box for JPG Invoice Extractor
st.subheader("JPG Invoice Extractor Code (`jpg_invoice_app.py`)")
st.code("""
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import streamlit as st
import os
import pathlib
import textwrap
from PIL import Image


import google.generativeai as genai


os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Function to load OpenAI model and get respones

def get_gemini_response(input,image,prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input,image[0],prompt])
    return response.text
    

def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")


##initialize our streamlit app

# st.set_page_config(page_title="Gemini Image Demo")

st.header("Image Invoice Extractor")
input=st.text_input("Input Prompt: ",key="input")
uploaded_file = st.file_uploader("Choose an image of an invoice...", type=["jpg", "jpeg", "png"])
image=""   
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Invoice.", use_column_width=True)


submit=st.button("Tell me about the invoice")

input_prompt = "You are an expert in understanding and processing invoices. You will receive input images as invoices & you will have to answer questions based on the input image.Stick with the context of the image provided including customer details, money paid itemwise, invoice number,items purchased, region, date etc."

## If ask button is clicked

if submit:
    image_data = input_image_setup(uploaded_file)
    response=get_gemini_response(input_prompt,image_data,input)
    st.subheader("The Response is")
    st.write(response)
""", language="python")
st.markdown("""
1. **Environment Setup**: Environment variables are loaded and the Google Generative AI API is configured to handle queries and generate responses.
2. **Image Upload & Display**: Users can upload an invoice image in JPG, JPEG, or PNG formats. The image is displayed on the page for confirmation.
3. **Processing & Response Generation**: Upon submitting a query, the uploaded image and input prompt are processed by Google's Gemini model to generate an invoice-specific response.
4. **Output Display**: The generated response is displayed directly on the page, providing clear and accurate information extracted from the image.
""")

# Code Box for PDF Invoice Extractor
st.subheader("PDF Invoice Extractor Code (`pdf_invoice_app.py`)")
st.code("""
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
    prompt_template = "
    You are an expert in understanding invoices. Provide detailed answers based on the input context
    that includes customer details, money paid itemwise, invoice number, items purchased, region, date etc. 
    If the answer is not available in the context, say 'Answer not found in the provided document.'

    Context: {context}
    Question: {question}

    Answer:
    "
    
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



""", language="python")
st.markdown("""
1. **Environment Setup**: The code loads environment variables for API keys and configures the Google Generative AI API for embedding and querying purposes.
2. **PDF Upload & Processing**: Users can upload multiple PDFs. The text is extracted from each PDF using the PyPDF2 library, then split into smaller chunks for easier processing.
3. **Vector Store Creation**: Once the text chunks are extracted, they are embedded using Google's AI embeddings and stored locally as a FAISS vector store for future querying.
4. **Query Submission**: Users can input a query about the uploaded invoices. The question-answering chain uses the FAISS index to search for relevant documents and returns an accurate, context-based response.
5. **Output Display**: The system displays the extracted information neatly in a text area, making it user-friendly and intuitive.
""")

# Conclusion
st.header("💡 Conclusion")
st.write("""
This documentation provides a comprehensive overview of how to use the **Invoice Information Extractor**. 
The application is built using advanced language models from Google Gemini and is designed to help extract 
valuable data from either **JPG** or **PDF** invoices. Ensure you follow the instructions carefully for optimal results.
""")

st.markdown("""
### Scalability and Performance Optimization for Future Development:

1. **Batch Processing for PDFs and Images**: Implement batch processing to handle multiple PDFs or image invoices simultaneously. This will increase throughput and enable efficient handling of large-scale uploads.
   
2. **Asynchronous Processing**: Introduce asynchronous processing for file uploads and queries. By leveraging Python's `asyncio` or similar frameworks, we can reduce wait times and handle multiple requests concurrently, improving the user experience.

3. **Cloud-Based Vector Store**: Move from local FAISS storage to a cloud-based vector database (e.g., Pinecone, Weaviate) for large-scale deployments. This ensures better scalability and performance, especially when dealing with large datasets across multiple users.

4. **Model Optimization**: Use model quantization and fine-tuning techniques to optimize the performance of Google Generative AI models. This will speed up response times, reduce memory usage, and improve the overall performance of the app.

5. **Caching Mechanism**: Implement a caching mechanism to store frequent queries and results, minimizing redundant computations. This can be done using in-memory databases like Redis, which will drastically improve response times for repeated queries.

6. **Load Balancing & Horizontal Scaling**: For large-scale usage, deploy the app on cloud platforms with load balancing and auto-scaling capabilities (e.g., AWS, GCP). This ensures high availability and distributes traffic effectively across multiple instances.

7. **Serverless Architecture**: Consider using a serverless architecture (e.g., AWS Lambda, GCP Cloud Functions) for handling invoice extraction tasks. This reduces operational overhead and scales automatically based on the number of requests.

8. **Monitoring and Logging**: Integrate a robust monitoring and logging system (e.g., Prometheus, Grafana) to track performance metrics and identify bottlenecks. Continuous monitoring will help in proactively scaling the system and improving fault tolerance.

9. **Data Compression and Optimized File Handling**: Optimize the handling of large files by implementing data compression techniques. This reduces file size and accelerates both uploads and processing times, contributing to a smoother user experience.
""")



### Key Features of the Main Page:
st.markdown("""
1. **Instructions**: Clear and concise instructions on how to use the app.
2. **Requirements**: Documentation of the `requirements.txt` and libraries needed to run the app.
3. **Code Explanation**: Displays the full source code for both `jpg_invoice_app.py` and `pdf_invoice_app.py` in well-formatted code blocks with comments.
4. **Professional Layout**: Well-structured layout with headers, subsections, and code blocks to provide a clean and professional look.
5. **Conclusion**: Wraps up the page with a short conclusion for clarity and ease of use.

This page can serve as the documentation and landing page for anyone using your invoice extractor app. The sidebar will be available for selecting between the two formats (JPG and PDF).
""")