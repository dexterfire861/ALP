import docling
import langchain_core
from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docling.document_converter import DocumentConverter
from tempfile import TemporaryDirectory
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
import pandas as pd
import math
import ast
import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re

class DoclingPDFLoader(BaseLoader):
    def __init__(self, file_path: str | list[str]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter()
    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            dl_doc = self._converter.convert(source).document
            text = dl_doc.export_to_markdown()
            yield LCDocument(page_content=text)

load_dotenv()
os.environ['HF_HOME'] = '/blue/rcstudents/smaley/alp/ALP_updated/models' # for huggingface model install


HF_API_KEY = os.environ.get("PERSONAL_HUGGING_FACE_API_KEY")
DATA_DIR = "../data"
OUTPUT_DIR = "../data/embeddings"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
NOT_WORKING = "AZ AAC R14-2-1618-checkpoint.pdf"


reldir = f'{DATA_DIR}/Laws'
i = 0

for subdir, dirs, files in os.walk(reldir):
    for file in files:
        print(f"\nLoading pages from {file}...")
        path = reldir + "/" + file
        reference_law = file[:-4]
        
        for _, dirs, _ in os.walk(OUTPUT_DIR):
            for dir_name in dirs:
                if dir_name == reference_law:
                    print("already queried!")
                    continue
        
        loader = DoclingPDFLoader(file_path=path)

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        docs = loader.load()
        splits = text_splitter.split_documents(docs)

        if not splits:
            print("ERROR: No splits generated.")
            continue
        else:
            print(f"Generated {len(splits)} splits.")

        # Load embedding model
        HF_EMBED_MODEL_ID = "BAAI/bge-large-en-v1.5" # ~64% accuracy on MTEB
        embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL_ID)

        vectorstore = FAISS.from_documents(splits, embeddings)
        
       
        FAISS_INDEX_FILE = f"{OUTPUT_DIR}/{reference_law}"
        vectorstore.save_local(FAISS_INDEX_FILE)
        
        print(f"FAISS index saved to {FAISS_INDEX_FILE}")
