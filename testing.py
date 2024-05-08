import os
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

openai_api_key = "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"

# enter questions here
questions = [
    "Who were the plaintiffs and who were the defendents?"
]

# iterate through each folder in case_files
reldir = './case_files'
i = 0
for subdir, dirs, files in os.walk(reldir):
    # dont want parent dir
    if i != 0 and i < 3:
        # get subdir path
        path = os.path.join(subdir)
        # name of subdir        
        curr_dir = path[13:]
        loader = PyPDFDirectoryLoader(path)

        # load documents
        print(f"\nloading documents from {curr_dir}...")
        documents = loader.load()
        print(documents)

    i += 1