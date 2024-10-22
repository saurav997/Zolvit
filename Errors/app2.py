from dotenv import load_dotenv
import streamlit as st
from PIL import Image
import google.generativeai as genai
import os
from pdf2image import convert_from_bytes  # to handle PDFs
import io


load_dotenv()

# Get the API key from .env folder
genai.configure(api_key=os.getenv("GOOGLE_AP_KEY"))

# Gemini Pro vision model instance declaration
model = genai.GenerativeModel("gemini-pro-vision")

# Function to extract model response
def get_response(input, image, prompt):
    response = model.generate_content([input, image[0], prompt])
    return response.text

# Streamlit frontend for file input and uploading
st.header("Medical Invoice Information Extractor")

# Input prompt for specific information to extract
input = st.text_input("Input Prompt:", key="input")

uploaded_file = st.file_uploader("Upload the invoice (image or PDF)")

# Function to process the uploaded file
def input_image_setup(uploaded_file):
    if uploaded_file.type == "application/pdf":
        # Convert PDF to image (assuming single-page PDFs)
        image = convert_from_bytes(uploaded_file.read())
        image = images[0]  # Take the first page
        return image,[{"mime_type": "image/jpeg", "data": image.tobytes()}]  
    elif uploaded_file.type.startswith("image/"):
        image = Image.open(uploaded_file)
        return image, [{"mime_type": uploaded_file.type, "data": uploaded_file.getvalue()}]
    else:
        raise ValueError("Unsupported file type!")


# Optimized prompt
input_prompt = """as you are an invoice image extracting expert Extract the relevant details 
                  from this medical invoice image. Ensure accuracy and professionalism in the output. 
                  The invoice may be multilingual."""

# Handle form submission
if uploaded_file is not None:
    try:
        # Display the uploaded invoice as an image (whether PDF or image)
        image, image_data = input_image_setup(uploaded_file)
        st.image(image, caption="Invoice", use_column_width=True)
        # Submit button to extract information
        submit = st.button("Extract Information")
        if submit:
            response = get_response(input_prompt, image_data, input)
            st.subheader("Extracted Information:")
            st.write(response)

    except Exception as e:
        st.error(f"Error: {e}")

