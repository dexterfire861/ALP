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
from langchain_community.chat_models import ChatLiteLLM
import numpy as np
import faiss

from langchain_community.chat_models import ChatAnthropic

from huggingface_hub import InferenceClient
import anthropic


load_dotenv()
uf_api_key = os.getenv("UF_API_KEY")
base_url = "https://api.ai.it.ufl.edu/"
client = openai.OpenAI(
    api_key=uf_api_key,
    base_url=base_url
)

openai_api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("UF_API_KEY")
claude_api_key = os.getenv("CLAUDE_API_KEY")
llama_api_key = os.getenv("LLAMA_API_KEY")
hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")



questions_dir = 'Questions.xlsx'
questions_df = pd.read_excel(questions_dir, sheet_name="Questions")  # sheet_name=None reads all sheets

prompt_engineering = pd.read_excel(questions_dir, sheet_name="Prompt Engineering").columns[0]
queries = []

count = 0
for index, row in questions_df.iterrows():
    query_dict = {}
    query_dict['question'] = row[0]
    query_dict['type'] = row[1]
    query_dict['tags'] = row[2]
    query_dict['id'] = int(row[3])
    queries.append(query_dict)

# files_to_query = ['VA HB 1451.pdf', 'ND HB 1506.pdf']
files_to_query = [ # all files
    "AZ AAC R14-2-1618.pdf",
    "AZ AAC R14-2-1801 et seq..pdf",
    "CA SB 100.pdf",
    "CA SB 107.pdf",
    "CA SB 1078.pdf",
    "CA SB 350.pdf",
    "CA SB X1 2.pdf",
    "CO CRS 40-2-124.pdf",
    "CO SB 236.pdf",
    "CT HB 5005.pdf",
    "CT HB 733.pdf",
    "CT HB 7432.pdf",
    "CT SB-9.pdf",
    "D.C. Statutes 15-340.pdf",
    "D.C. Statutes 17-250.pdf",
    "D.C. Statutes 21-154.pdf",
    "D.C. Statutes 22-257.pdf",
    "DE S.B. 199.pdf",
    "DE S.B. 33 [not yet included].pdf",
    "DE SB 74.pdf",
    "Guam Public Statutes 29-62.pdf",
    "HI HB 1464.pdf",
    "HI HB 173.pdf",
    "HI HB 623.pdf",
    "HI SB 2474.pdf",
    "IL 20 ILCS 3855.pdf",
    "IN SB 251.pdf",
    "Iowa Code 476.41 et seq..pdf",
    "Kansas Statutes  66-1256, et seq. 2009.pdf",
    "Kansas Statutes  66-1256, et seq. 2015.pdf",
    "MA 310 CMR 7.75.pdf",
    "MA General Statutes C. 25A S. 11F.pdf",
    "MD SB 209.pdf",
    "MD SB 516.pdf",
    "MD SB 791.pdf",
    "MD SB 869.pdf",
    "ME 1997 PL C. 316.pdf",
    "ME PL C. 403.pdf",
    "ME PL C. 447.pdf",
    "MI SB 213.pdf",
    "MI SB 438.pdf",
    "MN SF 146.pdf",
    "MO R.S. 393.1020 et seq.pdf",
    "MP PL 15-23.pdf",
    "MP PL 18-62.pdf",
    "MT MCA 69-3-2001 et seq..pdf",
    "NC SB 3.pdf",
    "ND HB 1506.pdf",
    "NH Title XXXIV Section 362F.pdf",
    "NJ PL 1999 Ch. 23.pdf",
    "NJ SB 1925.pdf",
    "NJ SB 3723.pdf",
    "NM SB 43.pdf",
    "NM SB 489.pdf",
    "NM Stat. § 62-15-34 et seq..pdf",
    "NV AB 226.pdf",
    "NV SB 358 2009.pdf",
    "NV SB 358 2019.pdf",
    "NV SB 372.pdf",
    "NY Public Service Comission Order Adopting a Clean Energy Standard.pdf",
    "NY Public Service Comission Order Approving Renewable Portfolio Standard.pdf",
    "NY Public Service Comission Order Establishing New RPS Goal and Resolving Main Tier Issues.pdf",
    "NY SB 06599.pdf",
    "OH HB 6.pdf",
    "OH SB 221.pdf",
    "OK HB 3028.pdf",
    "OR SB 1547.pdf",
    "OR SB 838.pdf",
    "PA SB 1030.pdf",
    "PR Act 82-2010.pdf",
    "PR SB 1121.pdf",
    "RI HB 7413.pdf",
    "RI S 2082.pdf",
    "RPS Document Key.pdf",
    "SC SB 1189.pdf",
    "SD HB 1123.pdf",
    "Tex. Utilities Code Ann. §39.904..pdf",
    "USVI Act 7075.pdf",
    "UT SB 0202.pdf",
    "VA Code § 56-585.2.pdf",
    "VA HB 1451.pdf",
    "VT HB 40.pdf",
    "WA Initiative 936.pdf",
    "WA Initiative 937.pdf",
    "WI State Legislature CH 196 .pdf",
    "WV HB 103.pdf"
]


import pandas as pd
import math
import ast

reencoded_credit_multiples = pd.read_excel('reencoded_col_BD_from_database.xlsx')
technology_mappings = pd.read_excel('technology_mapping.xlsx')

def retrieve_ground_truth_values(df, id, reference_law):
  row = df.loc[df['reference law'] == reference_law]

  if id == 3: # credit multiplier
    str_multiple_list = reencoded_credit_multiples.loc[reencoded_credit_multiples['rps_law'] == reference_law]["credit_multiplier"].iloc[0]
    return ast.literal_eval(str_multiple_list)

  elif id == 4: # categories of sales excluded or exempt
    # headings = df.loc[df['reference law'] == 'area names'].iloc[:, 13:19].columns.values # columns N-R
    cats_of_sales = row.iloc[:, 16].values[0] # col Q
    cats_of_sales = cats_of_sales.split(", ")
    # excluded_or_exempt = dict(zip(headings, cats_of_sales))
    return cats_of_sales

  elif id == 5: # date commitment passed into law
    date_commit = str(row.iloc[:, 5].values[0])[:4] # column F
    return date_commit

  elif id == 6: # RPS and ces commitment and year that it must be met
    rps_commit_perc = row.iloc[:, 8:10].values[0]
    for i in range(len(rps_commit_perc)):
      if math.isnan(rps_commit_perc[i]):
        rps_commit_perc[i] = "None"
    rps_val = f"{rps_commit_perc[0]}:{rps_commit_perc[1]}"
    if rps_val == "None:None": # if CES commit is non-existant
      rps_commit_perc = row.iloc[:, 6:8].values[0]
      for i in range(len(rps_commit_perc)):
        if math.isnan(rps_commit_perc[i]):
          rps_commit_perc[i] = "None"
      rps_val = f"{rps_commit_perc[0]}:{rps_commit_perc[1]}"
    return rps_val

  elif id == 7: # RPS initial commit as a %
    rps_initial = row.iloc[:, 10].values[0]
    return rps_initial

  elif id == 8: # % of RPS and CES that is voluntary, one value for both
    rps_voluntary = row.iloc[:, 17].values[0]
    return rps_voluntary

  elif id == 10: # energy sources
    energy_sources_all = row.iloc[:, 135:172]
    energy_sources = []
    for column in energy_sources_all.columns:
      if pd.notna(energy_sources_all[column].values[0]):
        energy_sources.append(column)
    ret = []
    for energy_source in energy_sources:
      ret.append(technology_mappings.loc[technology_mappings['Technology'] == energy_source]["Category"].iloc[0])
    return ret

  elif id == 11: # voluntary components of the commitment associated with the farthest date
    voluntary_perc = float(row.iloc[:, 17].values[0])
    if voluntary_perc <= 0:
      return "No"
    else:
      return 'Yes'

  else:
    return "QUESTION ID NOT RECOGNIZED"



import os

os.environ["LANGCHAIN_API_KEY"]  = os.getenv("LANGCHAIN_KEY")


# iterate through each folder in case_files
import pandas as pd
import openpyxl
import csv
from csv import DictWriter
import langchain_core
# import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI 

gt_df = pd.read_excel('Database June 20 2022.xlsx', sheet_name='master')
reldir = 'Laws'

i = 0

not_working = ["WV HB 103.pdf","WA Initiative 936.pdf","D.C. Statutes 22-257.pdf"]

print("accessing...")


for subdir, dirs, files in os.walk(reldir):
    for file in files:
        if file in files_to_query:
            if file in not_working:
               print(f"{file} not currently working")
               continue
            print(f"\nchecking if {file} is accessible in database...")
            reference_law = file[:-4]
            if gt_df.loc[gt_df['reference law'] == reference_law].empty:
                print(f"{file} not accessible from database! Skipping")
                continue # go to next iteration
            
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

            for _, _, output_files in os.walk("output"):
              for outfile in output_files:
                for model in models:
                  if outfile == f"{reference_law}_({model}).xlsx":
                      models.remove(model)

            if models == []:
              print(f"already queried for all models on {reference_law}!")
              continue
            
            # load documents
            print(f"loading pages from {file}...")
            path = reldir + "/" + file
            loader = PyPDFLoader(path)
            documents = loader.load()

            if not documents:
                print("ERROR: No documents loaded.")
                continue
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
            documents = loader.load()

            # check if loaded file is empty
            if documents == []:
                print("ERROR: loading documents from folder resulted in empty list")
                continue
                
            # Create embeddings for the document chunks using LiteLLM
            embeddings = []
            try:
                # print(f"{bcolors.WARNING}Warning: No active frommets remain. Continue?{bcolors.ENDC}")
                print(f"Generating embeddings with LiteLLM...")
                for split in splits:
                    try:
                        embedding_result = embedding(model=embedding_model, input=split.page_content)
                        embedding_vector = np.array(embedding_result.data[0]['embedding'])
                        embedding_vector_np = np.array(embedding_vector, dtype='float32')
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
            index.add(embedding_vectors) # Add vectors to the FAISS index

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
                df = pd.DataFrame(columns=["Type","Tags","Question ID","Question","Ground Truth","Response","Source"])

                # iterate through queries
                print(f"querying with {model}...")
                for query in queries:
                    # retrieval augmentation
                    relevant_docs = retrieve_documents(query['question'], embeddings, index)
                    context = "\n########## SOURCE ##########".join(doc['metadata'].get('source', 'Unknown') + ": " + doc['text'][:200] + "..." for doc in relevant_docs)

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
                      model_prefix = "openai/"            
                      prompt = f"Question: {query['question']}\nAddtional Information: {query['tags']}\nContext: {context}\nAnswer:"
                      response = completion(
                          api_key=uf_api_key,
                          base_url=base_url,
                          model=model_prefix + model,  # Use the current model for completions
                          messages=[
                              {"role": "system", "content": prompt_engineering},
                              {"role": "user", "content": prompt}
                          ],
                          # extra_headers= {"no_cache": False}
                      )
                    
                    answer = response['choices'][0]['message']['content']
                    # print("Response: ", answer)
                    ground_truth = retrieve_ground_truth_values(gt_df, query['id'], reference_law) # hard-coded ground-truth values
                    row = [query['type'], query['tags'], query['id'], query['question'], ground_truth, answer, context]
                    df.loc[len(df)] = row
                    filename = "output/" + reference_law + "_(" + model + ").xlsx"
                    with pd.ExcelWriter(filename) as writer:
                        df.to_excel(writer)

                    # except Exception as e:
                    #     print(f"Error generating response for query '{query['question']}' with model {model}: {e}")
                    #     answer = "Error: " + str(e)

        else:
            print(f"skipping {file}")



