# Chat-with-PDF-using-Gemini
This Repro explain the pipeline of Building a Chatbot using Gemini with the help of API keys used to chat with single or multiple pdf files

Note: ADD the .env file using the GOOGLE_API_KEY="generated key" in the same folder with your main code.
The API key can be generated from "https://aistudio.google.com/app/apikey"

Requirements:
To run the provided code, you'll need the following modules:

1. **streamlit**: A Python library for creating web applications.
   - Installation: `pip install streamlit`

2. **PyPDF2**: A library for reading PDF files in Python.
   - Installation: `pip install PyPDF2`

3. **langchain**: A Python library for natural language processing tasks.
   - Installation: This library seems to be custom or not publicly available. You may need to check its source or documentation for installation instructions.

4. **googletrans**: A Python library for Google Translate API.
   - Installation: `pip install googletrans==4.0.0-rc1`

5. **Pillow (PIL)**: A Python Imaging Library for opening, manipulating, and saving many different image file formats.
   - Installation: `pip install Pillow`

6. **fitz (PyMuPDF)**: A Python binding for MuPDF - a lightweight PDF and XPS viewer.
   - Installation: `pip install pymupdf`

7. **pytesseract**: A Python wrapper for Google's Tesseract-OCR Engine.
   - Installation: `pip install pytesseract`
