import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import requests
import time


# Set up OpenAI API credentials

#model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

def read_pdf(file_path):
    # Open the PDF file
    pdf_document = fitz.open(file_path)

    information = []

    # Iterate through pages
    for page_num in range(pdf_document.page_count):
        # Get the page
        page = pdf_document[page_num]

        # Extract text from the page
        text = page.get_text()

        information.append(text)

    pdf_document.close()
    return information

# Replace 'your_pdf_file.pdf' with the actual path to your PDF file
pdf_file_path = 'City of Stuart v. 3M Lawsuit.pdf'
start_time = time.time()
information = read_pdf(pdf_file_path)

from langchain import HuggingFaceHub

summarizer = HuggingFaceHub(
    huggingfacehub_api_token = "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",
    repo_id="facebook/bart-large-cnn",
    model_kwargs={"temperature":0, "max_length":180}
)
def summarize(llm, text) -> str:
    return llm(f"Summarize this: {text}!")

x = summarize(summarizer, information)

print(x)
