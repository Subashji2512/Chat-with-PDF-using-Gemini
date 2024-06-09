import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from PIL import Image
import io
import fitz  # PyMuPDF
import pytesseract
from googletrans import Translator

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize translator
translator = Translator()

def pdf_to_image(pdf_file):
    """Convert a single-page PDF to an image using PyMuPDF."""
    document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    page = document.load_page(0)  # Load the first (and only) page
    pix = page.get_pixmap()
    image = pix.tobytes("ppm")
    return image

def ocr_image(image_data):
    """Extract text from an image using Tesseract OCR."""
    image = Image.open(io.BytesIO(image_data))
    return pytesseract.image_to_string(image)

def translate_text(text, dest_language='en'):
    """Translate text to the specified language."""
    translated = translator.translate(text, dest=dest_language)
    return translated.text

def clean_text(text):
    """Clean the extracted text."""
    # Remove non-ASCII characters and extra whitespace
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = ' '.join(text.split())
    return text

def preprocess_text(text):
    """Preprocess text for LLM processing: cleaning, translation, and tokenization."""
    text = clean_text(text)
    text = translate_text(text)
    return text

def pdf_to_text(pdf_file):
    """Convert a single-page PDF to text with preprocessing suitable for LLMs."""
    image_data = pdf_to_image(pdf_file)
    text = ocr_image(image_data)
    preprocessed_text = preprocess_text(text)
    return preprocessed_text

def save_text_to_file(text, file_path="processed_text.txt"):
    """Save the processed text to a file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

def get_text_chunks(text):
    """Split text into chunks suitable for processing with LLMs."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create and save a FAISS vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Set up the conversational chain with the appropriate prompt and model."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """Handle user input and get the response from the LLM based on the question."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini")

    # Input for user's question
    user_question = st.text_input("Ask a Question from the PDF Files")

    # If a question is asked, process it
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        # File uploader for PDF documents
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = ""
                # Process each uploaded PDF file
                for pdf_file in pdf_docs:
                    raw_text += pdf_to_text(pdf_file) + "\n"
                # Save the processed text to a file
                save_text_to_file(raw_text, "processed_text.txt")
                # Split text into chunks and create a vector store
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
