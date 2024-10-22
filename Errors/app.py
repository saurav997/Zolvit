from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from PIL import Image
import google.generativeai as genai
import os

#get the apikey from .env folder
genai.configure(api_key = os.getenv("GOOGLE_AP_KEY"))

#Gemini Pro vision model instance declaration
model = genai.GenerativeModel("gemini-pro-vision")

#function to extract model response
def get_response(input,image,prompt):
    response = model.generate_content([input,image[0],prompt])
    return response.text

#streamlit frontend for file input and uploading
st.header("Medical Invoice Informaton Extractor")

#specific input prompt about the uploaded file that is required to be extracted
input = st.text_input("Input Prompt:",key = "input")

uploaded_file = st.file_uploader("Input the file (image or pdf)")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption = "Invoice",use_column_width = True)


submit = st.button("Extract Information about the image")

#main system prompt that would accompany the inputat every instance
input_prompt = """You are an expert in extracting information from the image 
                  of a medical invoice and presenting its information in a professional
                  and accurate manner in accordance with the accompanying query. 
                  Note that the invoce may be in multiple languages so account for this."""

#input image processing
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        image = [
            {
                "mime_type":uploaded_file.type,
                "data":uploaded_file.getvalue()
            }
        ]
        return image
    else:
        raise FileNotFoundError("No file is submitted!")

# response window
if submit:
    image_data = input_image_setup(uploaded_file)
    response = get_response(input_prompt, image_data, input)
    st.subheader("Extracted information:")
    st.write(response)