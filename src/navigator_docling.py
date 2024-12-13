# import docling
import langchain_core
from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docling.document_converter import DocumentConverter
# from tempfile import TemporaryDirectory
# from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEndpoint
from typing import Iterable
from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
from langchain import hub
import pandas as pd
import math
import ast
import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "../data"
EMBEDDINGS_DIR = f"{DATA_DIR}/embeddings"
REENCODED_CREDIT_MULTIPLES = pd.read_excel(f'{DATA_DIR}/reencoded_col_BD_from_database.xlsx')
TECHNOLOGY_MAPPINGS = pd.read_excel(f'{DATA_DIR}/technology_mapping.xlsx')

# CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
# LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
# HF_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

UF_API_KEY = os.getenv("UF_API_KEY")
BASE_URL = "https://api.ai.it.ufl.edu/"

QUESTIONS_DIR = f'{DATA_DIR}/Questions.xlsx'
QUESTIONS_DF = pd.read_excel(QUESTIONS_DIR, sheet_name="Questions")
PROMPT_ENGINEERING = pd.read_excel(QUESTIONS_DIR, sheet_name="Prompt Engineering").columns[0]
FILES_TO_QUERY_TXT = "files_to_query.txt"

GT_DF = pd.read_excel(f'{DATA_DIR}/Database June 20 2022.xlsx', sheet_name='master')

MODELS = [
    # "gemma-1.1-7b-it", # doesnt work
    "llama-3.0-70b-instruct",
    # "llama-3.0-8b-instruct", # doesnt work
    "mixtral-8x7b-instruct",
    "mistral-7b-instruct",
    "gpt-3.5-turbo",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini"
    # "claude-3.5-sonnet" # need to request access for
]

OUTPUT_DIR = "../output/docling_12_12_24"

def retrieve_ground_truth_values(df, id, reference_law):
    row = df.loc[df['reference law'] == reference_law]

    if id == 3: # credit multiplier
        str_multiple_list = REENCODED_CREDIT_MULTIPLES.loc[REENCODED_CREDIT_MULTIPLES['rps_law'] == reference_law]["credit_multiplier"].iloc[0]
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
            ret.append(TECHNOLOGY_MAPPINGS.loc[TECHNOLOGY_MAPPINGS['Technology'] == energy_source]["Category"].iloc[0])
        return ret

    elif id == 11: # voluntary components of the commitment associated with the farthest date
        voluntary_perc = float(row.iloc[:, 17].values[0])
        if voluntary_perc <= 0:
            return "No"
        else:
            return 'Yes'

    else:
        return "QUESTION ID NOT RECOGNIZED"
    
queries = []

count = 0
for index, row in QUESTIONS_DF.iterrows():
    query_dict = {}
    query_dict['question'] = row[0]
    query_dict['type'] = row[1]
    query_dict['tags'] = row[2]
    query_dict['id'] = int(row[3])
    queries.append(query_dict)

files_to_query = []
with open(f"{DATA_DIR}/{FILES_TO_QUERY_TXT}", "r") as file:
    files_to_query = [line.strip() for line in file]
    
    
# Load embedding model
HF_EMBED_MODEL_ID = "BAAI/bge-large-en-v1.5"
EMBEDDINGS = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL_ID)

not_working = ["WV HB 103.pdf","WA Initiative 936.pdf","D.C. Statutes 22-257.pdf","DE S.B. 33 [not yet included].pdf"]

reldir = f'{DATA_DIR}/Laws'
i = 0
print("initilizing...")
for subdir, dirs, files in os.walk(reldir):
    for file in files:
        if file in files_to_query: # hardcoded
            MODELS = [
                # "gemma-1.1-7b-it", # doesnt work
                "llama-3.0-70b-instruct",
                # "llama-3.0-8b-instruct", # doesnt work
                "mixtral-8x7b-instruct",
                "mistral-7b-instruct",
                "gpt-3.5-turbo",
                "gpt-4-turbo",
                "gpt-4o",
                "gpt-4o-mini"
                # "claude-3.5-sonnet" # need to request access for
            ]
            
            print(f"\n{file}")
            
            if file in not_working:
                print(f"SKIPPING {file}, not currently working")
                continue
            
            reference_law = file[:-4]
            if GT_DF.loc[GT_DF['reference law'] == reference_law].empty:
                print(f"SKIPPING {file}, not accessible from database")
                continue # go to next iteration
            
            if not os.path.exists(f"{EMBEDDINGS_DIR}/{reference_law}"):
                print(f"SKIPPING {file}, no embeddings directory")
                continue
            
            for model in MODELS:
                if os.path.exists(f"{OUTPUT_DIR}/{reference_law}_({model}).xlsx"):
                    MODELS.remove(model)
            if MODELS == []:
                print(f"SKIPPING {file}, already queried for all models on {reference_law}!")
                continue
            
            # load embeddings
            vectorstore = FAISS.load_local(f"{EMBEDDINGS_DIR}/{reference_law}", EMBEDDINGS, allow_dangerous_deserialization=True)
            
            # iterate through models
            for model in MODELS:
                # load llm
                llm = ChatOpenAI(
                    openai_api_key=UF_API_KEY,
                    openai_api_base=BASE_URL,
                    model = model,
                    temperature=0.1 # low temp for deterministic output
                )

                # build retriever
                from langchain.retrievers.multi_query import MultiQueryRetriever
                retriever = MultiQueryRetriever.from_llm(
                    retriever=vectorstore.as_retriever(), llm=llm
                )

                # build prompt
                prompt = hub.pull("rlm/rag-prompt")
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                from langchain_core.runnables import RunnableParallel
                rag_chain_from_docs = (
                    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                # intialize rag_chain
                rag_chain_with_source = RunnableParallel(
                    {"context": retriever, "question": RunnablePassthrough()} # change retriever as necessary
                ).assign(answer=rag_chain_from_docs)
            
                df = pd.DataFrame(columns=["Type","Tags","Question ID","Question","Ground Truth","Response","Source"])
                # iterate through questions
                for query in queries:
                    prompt = f"{query['question']}\nAddtional Information: {query['tags']}"
                    answer = rag_chain_with_source.invoke(prompt)
                    # print(answer)
                    ground_truth = retrieve_ground_truth_values(GT_DF, query['id'], reference_law) # hard-coded ground-truth values
                    row = [query['type'], query['tags'], query['id'], query['question'], ground_truth, answer['answer'], answer['context']]
                    df.loc[len(df)] = row
                    
                filename = OUTPUT_DIR + "/" + reference_law + "_(" + model + ").xlsx"
                with pd.ExcelWriter(filename) as writer:
                    df.to_excel(writer)
                    
            
