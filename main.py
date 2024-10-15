import pandas as pd
import openai
import os
from dotenv import load_dotenv
import requests
import pypdf
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatAnthropic
from langchain_huggingface import HuggingFaceEndpoint
from litellm import embedding, LiteLLM, completion
from langchain.prompts import ChatPromptTemplate
import numpy as np
import faiss
import ast
from langchain_community.chat_models import ChatAnthropic

from huggingface_hub import InferenceClient
import anthropic





load_dotenv()

# Load API keys and initialize clients
api_key = os.getenv("UF_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
claude_api_key = os.getenv("CLAUDE_API_KEY")
llama_api_key = os.getenv("LLAMA_API_KEY")
hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")

base_url = "https://api.ai.it.ufl.edu/v1/"




lite_llm_client = LiteLLM(api_key=api_key, base_url=base_url)

# Load questions from Excel
questions_dir = 'Questions.xlsx'
spreadsheet = pd.read_excel(questions_dir, sheet_name=None)
prompt_engineering = spreadsheet['Prompt Engineering'].columns[0]
queries = []
worksheet = spreadsheet['Questions']

for _, row in worksheet.iterrows():
    query = {
        'question': row['Questions'],
        'type': row['Type'],
        'tags': row['Tags'] if 'Tags' in row else '',
        'id': row['ID']
    }
    queries.append(query)

# Load ground truth data
gt_df = pd.read_excel('Database June 20 2022.xlsx', sheet_name='master')

reencoded_credit_multiples = pd.read_excel('reencoded_col_BD_from_database.xlsx')
technology_mappings = pd.read_excel('technology_mapping.xlsx')

def retrieve_ground_truth_values(df, id, reference_law):
    row = df.loc[df['reference law'] == reference_law]
    if id == 3:  # credit multiplier
        str_multiple_list = reencoded_credit_multiples.loc[reencoded_credit_multiples['rps_law'] == reference_law]["credit_multiplier"].iloc[0]
        return ast.literal_eval(str_multiple_list)
    elif id == 4:  # categories of sales excluded or exempt
        cats_of_sales = row.iloc[:, 16].values[0]  # col Q
        return cats_of_sales.split(", ")
    elif id == 5:  # date commitment passed into law
        return str(row.iloc[:, 5].values[0])[:4]  # column F
    elif id == 6:  # RPS and ces commitment and year that it must be met
        rps_commit_perc = row.iloc[:, 8:10].values[0]
        return f"{rps_commit_perc[0]}:{rps_commit_perc[1]}"
    elif id == 7:  # RPS initial commit as a %
        return row.iloc[:, 10].values[0]
    elif id == 8:  # % of RPS and CES that is voluntary
        return row.iloc[:, 17].values[0]
    elif id == 10:  # energy sources
        energy_sources_all = row.iloc[:, 135:172]
        return [column for column in energy_sources_all.columns if pd.notna(energy_sources_all[column].values[0])]
    elif id == 11:  # voluntary components of the commitment associated with the farthest date
        voluntary_perc = float(row.iloc[:, 17].values[0])
        return "Yes" if voluntary_perc > 0 else "No"
    else:
        return "QUESTION ID NOT RECOGNIZED"

models = [
    "mixtral-8x7b-instruct",
    "gemma-1.1-7b-it",
    "gpt-4o",
    "gpt-4-turbo",
    "mistral-7b-instruct",
    "claude-3-5-sonnet-20240620",
    "Meta-Llama-3.1-8B-Instruct"
    # "llama3-70b-instruct", These llama models do not work currently or their input format is different
    # "llama3-8b-instruct",
    "gpt-4o-mini",
    "gpt-3.5-turbo"
]

# Process PDF files from the directory
pdf_directory = os.path.join(os.getcwd(), 'Laws')
print(f"Looking for files in: {pdf_directory}")

pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    if not documents:
        print(f"ERROR: No documents loaded from {file_path}.")
        return None, None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    if not splits:
        print("ERROR: No splits generated.")
        return None, None
    
    return documents, splits

def generate_embeddings(splits):
    embeddings = []
    embedding_model = "text-embedding-ada-002"
    try:
        print("Generating embeddings with LiteLLM...")
        for split in splits:
            try:
                input_text = str(split.page_content)
                embedding_result = embedding(model=embedding_model, input=input_text)
                embedding_vector = np.array(embedding_result.data[0]['embedding'])
                embedding_vector_np = np.array(embedding_vector, dtype='float32')
                embeddings.append({"embedding": embedding_vector_np, "metadata": split.metadata, "text": split.page_content})
            except Exception as e:
                print(f"Error generating embedding for chunk: {e}")
        print("Embeddings generated successfully.")
    except Exception as e:
        print("An error occurred while generating embeddings:", e)
    
    return embeddings
    
def create_faiss_index(embeddings):
    d = len(embeddings[0]['embedding'])  # Dimension of the embedding vectors
    index = faiss.IndexFlatL2(d)  # Create a FAISS index
    embedding_vectors = np.vstack([emb["embedding"] for emb in embeddings]).astype('float32')
    index.add(embedding_vectors)
    return index

def retrieve_documents(query, embeddings, index, top_k=3):
    try:
        query_text = str(query)
        embedding_result = embedding(model="text-embedding-ada-002", input=query_text)
        query_embedding = embedding_result.data[0]['embedding']

        if not isinstance(query_embedding, list) or not all(isinstance(i, float) for i in query_embedding):
            raise ValueError("Invalid embedding format")

        query_embedding_np = np.array(query_embedding, dtype='float32')
        distances, indices = index.search(np.array([query_embedding_np]), top_k)
        return [embeddings[i] for i in indices[0]]
    except Exception as e:
        print(f"Error retrieving documents for query '{query}': {e}")
        return []

def process_queries(queries, embeddings, index, model, prompt_engineering, api_key, base_url, file_name):
    results = []
    for query in queries:
        reference_law = file_name[:-4]
        gt_df = pd.read_excel('Database June 20 2022.xlsx', sheet_name='master')
        ground_truth = retrieve_ground_truth_values(gt_df, query['id'], reference_law)
        relevant_docs = retrieve_documents(query['question'], embeddings, index)  # Pass the FAISS index, not the model
        context = "\n\n".join(doc['text'] for doc in relevant_docs)

        if "claude" in model:
            try:
                response = anthropic.Anthropic(api_key=claude_api_key).messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=1024,
                    system= str(prompt_engineering),
                    messages=[
                        {"role": "user", "content": f"Question: {query['question']}\nContext: {context}\nAnswer:"}
                    ]
                )
                answer = response.content[0].text
            except Exception as e:
                print(f"Error generating response for query '{query['question']}' with model {model}: {e}")
                answer = f"Error: {str(e)}"
        
        elif "Llama" in model:
            try:
                client = InferenceClient(api_key=hugging_face_api_key)

                response = client.chat_completion(
                    model="meta-llama/" + model,
                    messages=[
                        {"role": "system", "content": str(prompt_engineering)},
                        {"role": "user", "content": f"Question: {query['question']}\nContext: {context}\nAnswer:"}
                    ],
                )

                answer = response.choices[0].message.content

                

            except Exception as e:
                print(f"Error generating response for query '{query['question']}' with model {model}: {e}")
                answer = f"Error: {str(e)}"

        else:
            try:
                model_prefix = "openai/"
                response = completion(
                    api_key=api_key,
                    base_url=base_url,
                    model=model_prefix + model,
                    messages=[
                        {"role": "system", "content": str(prompt_engineering)},
                        {"role": "user", "content": f"Question: {query['question']}\nContext: {context}\nAnswer:"}
                    ],
                )
                answer = response['choices'][0]['message']['content']
            except Exception as e:
                print(f"Error generating response for query '{query['question']}' with model {model}: {e}")
                answer = f"Error: {str(e)}"
        
        results.append({
            "Type": query['type'],
            "Tags": query['tags'],
            "Question ID": query['id'],
            "Question": query['question'],
            "Ground Truth": ground_truth,
            "Response": answer,
            "Source": context,
        })
        
        print(f"Question ID: {query['id']}")
        print(f"Question: {query['question']}")
        print(f"Answer: {answer}")
        print(f"Sources: {context}")
        print(f"Ground Truth: {ground_truth}")
        print("-" * 50)
    
    return results

for file_name in pdf_files:
    file_path = os.path.join(pdf_directory, file_name)
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        continue
    
    print(f"\nProcessing file: {file_name}")
    
    # Load and process the PDF
    documents, splits = process_pdf(file_path)
    if not documents or not splits:
        print(f"Error processing {file_name}. Skipping to next file.")
        continue

    # Generate embeddings and create vector store
    embeddings = generate_embeddings(splits)
    if not embeddings:
        print(f"Error generating embeddings for {file_name}. Skipping to next file.")
        continue

    # Create FAISS index
    index = create_faiss_index(embeddings)

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Process each model
    for model in models:
        print(f"\nProcessing with model: {model}")

        results = process_queries(
            queries, 
            embeddings,
            index,
            model, 
            prompt_engineering, 
            api_key, 
            base_url, 
            file_name
        )

        # Save the results to an Excel file in the results directory
        df = pd.DataFrame(results)
        output_file = os.path.join("results", f"{os.path.splitext(file_name)[0]}_{model}.xlsx")
        df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
