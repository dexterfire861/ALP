import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import requests
import time


# Set up OpenAI API credentials

model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

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

key = "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"

client = OpenAI(api_key=key)


query = "Why did the City of Stuart file a lawsuit against 3M?"

query_emb = model.encode(query)
information_emb = model.encode(information)

scores = util.dot_score(query_emb, information_emb)[0].cpu().tolist()

doc_score_pairs = list(zip(information, scores))

doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

for doc, score in doc_score_pairs:
    print(score, doc)

end_time = time.time()

print(f"Execution time of processing function: {end_time - start_time} seconds")









