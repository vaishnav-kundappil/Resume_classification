
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import fitz  # PyMuPDF for PDF processing
import docx
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load the trained model and vectorizer
model = joblib.load('resume_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Category mapping
category_mapping = {
    'Peoplesoft Resume': 0,
    'React Developer': 1,
    'SQL Developer': 2,
    'workday': 3
}

# Reverse mapping for prediction output
reverse_category_mapping = {v: k for k, v in category_mapping.items()}

# Page Configuration
st.set_page_config(page_title="Resume Classifier", layout="wide", page_icon="📄")

# Custom CSS for Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background-color: #007BFF;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.title("📄 Resume Classification App")
st.markdown("""
### Upload a resume and get classification results instantly!
Supports **PDF, DOCX, and DOC** formats.
""")

# Function to extract text from uploaded file
def extract_text(file):
    """Extracts text from PDF, DOCX, or DOC files."""
    if file.type == "application/pdf":
        pdf_reader = fitz.open(stream=file.read(), filetype="pdf")
        text = "\n".join([page.get_text("text") for page in pdf_reader])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":  # DOCX
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif file.type == "application/msword":  # DOC
        text = extract_text_from_doc(file)
    else:
        text = ""  # Unsupported format
    return text

def extract_text_from_doc(file):
    """Extracts text from a DOC file (Windows-only, requires Microsoft Word)."""
    try:
        import comtypes.client
        word = comtypes.client.CreateObject("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(file.name)
        text = doc.Content.Text
        doc.Close(False)
        word.Quit()
        return text
    except Exception as e:
        return f"Error processing DOC file: {e}"

# Preprocessing Function
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>|http\S+|\d+', '', text)  # Remove HTML, URLs, and numbers
    
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in tokens if len(w) > 2 and w not in stop_words]

    return " ".join(filtered_words)

# File Upload Section
uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "doc"])

if uploaded_file:
    # Extract text
    resume_text = extract_text(uploaded_file)
    
    if resume_text:
        # Preprocess text
        cleaned_text = preprocess(resume_text)
        
        # Transform using the vectorizer
        transformed_text = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(transformed_text)
        
        # Map prediction to category name
        category_name = reverse_category_mapping.get(prediction[0], "Unknown Category")
        
        # Display output
        st.markdown("### 🏆 Classification Result")
        st.success(f"The model predicts: **{category_name}**")
    else:
        st.error("Error: Unable to extract text from the uploaded file.")
