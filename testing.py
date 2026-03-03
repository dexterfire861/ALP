"""
testing.py — Single-document RAG pipeline for rapid prototyping.

Processes a single PDF (``va-code.pdf``) through the full RAG pipeline to
quickly test changes to embedding, retrieval, or prompting strategies before
running the heavier multi-document pipeline in ``main.py``.

Note:
    Contains a syntax error on line 203 (missing comma) from early development.
    Use ``main.py`` for production runs.
"""

import pandas as pd
import openai
import os
from dotenv import load_dotenv
import requests
import pypdf

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import MultiQueryRetriever
from langchain import hub

from litellm import embedding, LiteLLM, completion
import numpy as np
import faiss

load_dotenv()
api_key = os.getenv("UF_API_KEY")
base_url = "https://api.ai.it.ufl.edu/"
client = openai.OpenAI(
    api_key=api_key,
    base_url=base_url
)

openai_api_key = os.getenv("OPENAI_API_KEY")


# If it's a .xlsx file
questions_dir = 'Questions.xlsx'
spreadsheet = pd.read_excel(questions_dir, sheet_name=None)  # sheet_name=None reads all sheets


# Access a specific sheet by its name and get all of the values
prompt_engineering = spreadsheet['Prompt Engineering'].columns[0]

queries = []

worksheet = spreadsheet['Questions']
print(worksheet)

queries = [{'question': 'Which state, area, or territory is the reference law regarding to? The answer should be only the name of the state, area, or territory, without any explanation.',
  'type': 'Categorical',
  'tags': '',
  'id': 1},
 {'question': 'List energy sources that receives multifold credit (e.g. doubled, tripled) towards RPS/CES commitment if any. The answer should only include the name of the energy sources and their corresponding multiple in numerical value(e.g. double = 2). Energy sources should be separated by bracket and newline. In each energy source, the energy name and its corresponding multiple should be separated by a colon. If no energy sources receive multifold credits, give "None" as the response. Do not provide any explanation.',
  'type': 'Categorical',
  'tags': '',
  'id': 3},
 {'question': 'What are the categories of sales that are excluded and/or exempted from the commitment? If so, list the conditions or categories that define these exclusions from the list below: BTM (behind the meter), IOU (investor-owned utlity), Community Choice Aggregator, Co-op, Municipal, State, and Federal. Do not provide any explanation. Answer "None" as a response if there is no categories of generation that are excluded from the law.',
  'type': 'Categorical',
  'tags': '',
  'id': 4},
 {'question': 'What is the date at which the commitment passed into law. It is ususally the date that the document is filed or approved. Answer the question with only the year (e.g. YYYY), without any explanation.',
  'type': 'Dates',
  'tags': '',
  'id': 5},
 {'question': "What's the RPS/CES commitment in the policy and what's the year by which the commitment must be met? If the policy specifies multiple commitments at different target dates, answer with the commitment associated with the farthest date. The answer should only include the percentage of commitment (e.g. 10%) and the target year, separated by a colon. Do not provide any explanation.",
  'type': 'Numerical',
  'tags': '',
  'id': 6},
 {'question': 'What is the RPS/CES commitment for the year the policy was introduced? The answer should be a percentage (e.g. 10%). Do not answer with any explanation.',
  'type': 'Numerical',
  'tags': '',
  'id': 7},
 {'question': "What's the percentage of the RPS/CES commitment that is voluntary? If the policy has multiple commitments at different dates, show the voluntary percentage of the commitment associated with the largest date. The answer should be a percentage with two decimal places (e.g. 10.00%). If the text does not mention voluntary percentage, the answer should be 0.00%. Do not answer with any explanation.",
  'type': 'Numerical',
  'tags': '',
  'id': 8},
 {'question': 'Given the following technologies:\nBiofuel, Liquid\nBiogas\nBiogas, Anaerobic Digestion\nBiogas, Landfill\nBiomass\nBiomass, Municipal Solid Waste\nCarbon Capture, Partial\nCarbon Free\nCarbon Neutral\nCoal Gasification\nCoal Mine Methane\nCoal Mine Waste\nCoal, Advanced\nCogeneration, Industrial\nCombined Heat and Power\nCombined Heat and Power, Residential\nDemand Reduction\nDistributed Generation\nEnergy Efficiency\nFuel Cells\nFuel Cells, Residential\nGeothermal, Geoelectric\nHydroelectric\nHydroelectric, Small\nHydroelectric, Small Falling Water\nNatural Gas\nNear Zero Carbon\nNew Technology\nNuclear\nNuclear, Non-Fossil Fuel Carbon Free\nOcean, Tidal, Wave\nRecycled Energy\nSolar\nSolar, Small\nTire-Derived Fuel\nWind\nWind, Small\n\nIdentify energy sources compliant with RPS criteria. Then, for each energy source, select one corresponding technology from the provided list. List the selected technologies, each on a separate line, mirroring the exact name of the technology. Ensure the number of technologies listed matches the number of identified energy sources. Listed technologies can be repeated if multiple energy sources correspond to a same technology. Do not provide explanations or additional words in the response.',
  'type': 'Categorical',
  'tags': '',
  'id': 10},
 {'question': 'Are there any voluntary components of the commitment associated with the farthest date?  Answer the question with “Yes” or “No” without providing any explanation.',
  'type': 'Binary',
  'tags': '',
  'id': 11}
]

models = [
    "mixtral-8x7b-instruct",
    "gemma-1.1-7b-it",
    "gpt-4o",
    "gpt-4-turbo",
    "mistral-7b-instruct",
    "llama3-70b-instruct",
    "llama3-8b-instruct",
    "gpt-4o-mini",
    "gpt-3.5-turbo"
]

lite_llm_client = LiteLLM(api_key=api_key)

# Define the file to query
file_to_query = "va-code.pdf"

# Load the PDF document
loader = PyPDFLoader(file_to_query)
documents = loader.load()

if not documents:
    print("ERROR: No documents loaded.")
else:
    print("Documents loaded successfully.")

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

embedding_model = "text-embedding-ada-002"

if not splits:
    print("ERROR: No splits generated.")
else:
    print(f"Generated {len(splits)} splits.")

# Create embeddings for the document chunks using LiteLLM
embeddings = []
try:
    print("Generating embeddings with LiteLLM...")
    for split in splits:
        try:
            embedding_result = embedding(model=embedding_model, input=split.page_content)
            embedding_vector = np.array(embedding_result.data[0]['embedding'])  #
            embedding_vector_np = np.array(embedding_vector, dtype='float32')
            
            # Store the embedding and corresponding metadata
            embeddings.append({"embedding": embedding_vector_np, "metadata": split.metadata, "text": split.page_content})
        except Exception as e:
            print(f"Error generating embedding for chunk: {e}")

    print("Embeddings generated successfully.")
except Exception as e:
    print("An error occurred while generating embeddings:", e)

# Convert embeddings into a FAISS index for efficient similarity search
d = len(embeddings[0]['embedding'])  # Dimension of the embedding vectors
index = faiss.IndexFlatL2(d)  # Create a FAISS index
embedding_vectors = np.vstack([emb["embedding"] for emb in embeddings]).astype('float32')  # Convert embeddings to float32

# Add vectors to the FAISS index
index.add(embedding_vectors)

# Retrieval function using FAISS
def retrieve_documents(query, embeddings, index, top_k=3):
    # Generate query embedding using LiteLLM with a supported model
    try:
        # Call the embedding API and extract the embedding vector
        embedding_result = embedding(api_key=openai_api_key, model=embedding_model, input=query)  # Adjusted to use the correct parameters
        query_embedding = embedding_result.data[0]['embedding']  # Extract the embedding vector
        
        # Ensure the embedding is a flat list of floats
        if not isinstance(query_embedding, list) or not all(isinstance(x, float) for x in query_embedding):
            raise ValueError("Query embedding is not a flat list of floats.")
        
        # Convert query embedding to a numpy array in float32 format
        query_embedding_np = np.array(query_embedding, dtype='float32')
    except Exception as e:
        print(f"Error generating embedding for query '{query}': {e}")
        return []

    # Perform search in the FAISS index
    distances, indices = index.search(np.array([query_embedding_np]), top_k)  # Perform search
    
    # Retrieve and return relevant documents based on search results
    return [embeddings[i] for i in indices[0]]  # Retrieve relevant documents

# Create results directory if not exists
if not os.path.exists("results"):
    os.makedirs("results")



for model in models:
    print(f"\nProcessing with model: {model}")

    results = []

    # Iterate through queries and get responses
    for query in queries:
        # Perform retrieval based on the query
        relevant_docs = retrieve_documents(query['question'], embeddings, index)
        context = "\n\n".join(doc['metadata'].get('source', 'Unknown') + ": " + doc['text'][:200] + "..." for doc in relevant_docs)

        print(f"Query: {query['question']}\nRetrieved Context: {context}")

        # Generate response using LiteLLM chat completion
        try:

            model_prefix = "openai/"


            
            
            response = completion(
                api_key=api_key,
                base_url=base_url,
                model=model_prefix + model,  # Use the current model for completions
                custom_llm_provider="openai"
                messages=[
                    {"role": "system", "content": "We are researchers specializing in renewable energy law, with a keen interest in analyzing and understanding the various commitments states have made toward renewable energy. Our work involves examining state bills and legislation to gather detailed insights into how different states are approaching renewable energy. I am currently working on a project where I need to compile and analyze data regarding the specific commitments and targets set by each state in the realm of renewable energy. To facilitate this, I'm aiming to fill out an Excel spreadsheet with diverse values representing these commitments. Could you assist me in understanding these commitments better and help gather objective values from the bills to attain the relevant information. Provide only the numerical values for all of the commitment levels as needed. Do not consider any text that has been striked-through in your responses."},
                    {"role": "user", "content": f"Question: {query['question']}\nContext: {context}\nAnswer:"}
                ],
                extra_headers= {"no_cache": False}
            )
            answer = response['choices'][0]['message']['content']
            print("Response: ", answer)

        except Exception as e:
            print(f"Error generating response for query '{query['question']}' with model {model}: {e}")
            answer = "Error: " + str(e)
        
        results.append({
            "Question ID": query['id'],
            "Question": query['question'],
            "Answer": answer,
            "Sources": context
        })

        # Print the response
        print(f"Question ID: {query['id']}")
        print(f"Question: {query['question']}")
        print(f"Answer: {response['choices'][0]['message']['content']}")
        print(f"Sources: {context}")
        print("-" * 50)
    
    df = pd.DataFrame(results)

    

    # Save the results to a CSV file
    pdf_name = os.path.splitext(os.path.basename(file_to_query))[0]  # Extract PDF name without extension
    output_file = f"results/{pdf_name}_{model}.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

        