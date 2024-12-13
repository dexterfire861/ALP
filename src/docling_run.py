import pandas as pd
import math
import ast

data_dir = "/blue/rcstudents/smaley/alp/ALP_updated/data"

reencoded_credit_multiples = pd.read_excel(f'{data_dir}/reencoded_col_BD_from_database.xlsx')
technology_mappings = pd.read_excel(f'{data_dir}/technology_mapping.xlsx')

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
from dotenv import load_dotenv

if not load_dotenv():
    print("failed to load env keys")

# uf_api_key = os.getenv("UF_API_KEY")
# base_url = "https://api.ai.it.ufl.edu/"
# client = openai.OpenAI(
#     api_key=uf_api_key,
#     base_url=base_url
# )

# openai_api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("UF_API_KEY")
claude_api_key = os.getenv("CLAUDE_API_KEY")
llama_api_key = os.getenv("LLAMA_API_KEY")
hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")

questions_dir = f'{data_dir}/Questions.xlsx'
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


import docling
import langchain_core
from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docling.document_converter import DocumentConverter
from tempfile import TemporaryDirectory
from langchain_milvus import Milvus
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
from langchain import hub

class DoclingPDFLoader(BaseLoader):

    def __init__(self, file_path: str | list[str]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter()

    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            dl_doc = self._converter.convert(source).document
            text = dl_doc.export_to_markdown()
            yield LCDocument(page_content=text)
            


os.environ['HF_HOME'] = '/blue/rcstudents/smaley/alp/ALP_updated/models' # for huggingface model install
HF_API_KEY = os.environ.get("PERSONAL_HUGGING_FACE_API_KEY")



gt_df = pd.read_excel(f'{data_dir}/Database June 20 2022.xlsx', sheet_name='master')
reldir = f'{data_dir}/Laws'

i = 0

not_working = ["WV HB 103.pdf","WA Initiative 936.pdf","D.C. Statutes 22-257.pdf","DE S.B. 33 [not yet included].pdf"]

print("initilizing...")

for subdir, dirs, files in os.walk(reldir):
    for file in files:
        if file in files_to_query: # hardcoded
            print(f"\n{file}")
            if file in not_working:
                print(f"SKIPPING {file}, not currently working")
                continue
            print(f"checking if {file} is accessible in database...")
            reference_law = file[:-4]
            if gt_df.loc[gt_df['reference law'] == reference_law].empty:
                print(f"{file} not accessible from database! Skipping")
                continue # go to next iteration
            
            models = [
                "mixtral-8x7b-instruct",
            #     "gemma-1.1-7b-it",
            #     "gpt-4o",
            #     "gpt-4-turbo",
            #     "mistral-7b-instruct",
            #     "llama3-70b-instruct",
            #     "llama3-8b-instruct",
            #     "gpt-4o-mini",
            #     "gpt-3.5-turbo"
            ]
            
            for _, _, output_files in os.walk("output"):
                for outfile in output_files:
                    for model in models:
                        if outfile == f"{reference_law}_({model}).xlsx":
                            models.remove(model)

            if models == []:
                print(f"already queried for all models on {reference_law}!")
                continue
            
            # load and splitdocuments
            print(f"loading pages from {file}...")
            path = reldir + "/" + file
            loader = DoclingPDFLoader(file_path=path)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            docs = loader.load()
            splits = text_splitter.split_documents(docs)
            
            if not splits:
                print("ERROR: No splits generated.")
            else:
                print(f"Generated {len(splits)} splits.")
            
            # load embedding model
            HF_EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
            embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL_ID)
            
            # build vector db
            MILVUS_URI = os.environ.get(
                "MILVUS_URI", f"{(tmp_dir := TemporaryDirectory()).name}/milvus_demo.db"
            )
            vectorstore = Milvus.from_documents(
                splits,
                embeddings,
                connection_args={"uri": MILVUS_URI},
                drop_old=True,
            )
            
            # load llm
            HF_LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3" # hardcoded mistral
            llm = HuggingFaceEndpoint(
                repo_id=HF_LLM_MODEL_ID,
                huggingfacehub_api_token=HF_API_KEY,
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
            
            try:
                # iterate through models
                for model in models:
                    df = pd.DataFrame(columns=["Type","Tags","Question ID","Question","Ground Truth","Response","Source"])
                    # iterate through questions
                    for query in queries:
                        prompt = f"{query['question']}\nAddtional Information: {query['tags']}"
                        answer = rag_chain_with_source.invoke(prompt)
                        # print(answer)
                        ground_truth = retrieve_ground_truth_values(gt_df, query['id'], reference_law) # hard-coded ground-truth values
                        row = [query['type'], query['tags'], query['id'], query['question'], ground_truth, answer['answer'], answer['context']]
                        df.loc[len(df)] = row
                    filename = "/blue/rcstudents/smaley/alp/ALP_updated/output/docling_11_19_24/" + reference_law + "_(" + model + ").xlsx"
                    with pd.ExcelWriter(filename) as writer:
                        df.to_excel(writer)
            except:
                print(f"failed on {reference_law}")
        
            
            