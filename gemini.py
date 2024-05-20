import os
import getpass
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain import PromptTemplate
from langchain import hub
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma
     

GOOGLE_API_KEY = "AIzaSyBqo_mqOAw3S_MdDqikDCO2SfiowCX5Gyo"

#add environment variable for google api key

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def load_document(filename):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(filename)
        documents = loader.load()
    elif filename.endswith(".txt"):
        loader = TextLoader(filename)
        documents = loader.load()
    else:
        raise ValueError("Invalid file type")

    text_splitter = CharacterTextSplitter(chunk_size=1000,
                                          chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    return docs

#docus = load_document("va-code.pdf")

loader = PyPDFDirectoryLoader("va-code.pdf")
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
docs = text_splitter.split_documents(documents=docs)

          # print(docs)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print(splits)

docs = splits

#convert the document to langchain format

from langchain_google_genai import GoogleGenerativeAIEmbeddings

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

vectorstore = Chroma.from_documents(
                     documents=docs,                 # Data
                     embedding=gemini_embeddings,    # Embedding model
                     persist_directory="./chroma_db" # Directory to save data
                     )

vectorstore_disk = Chroma(
                        persist_directory="./chroma_db",       # Directory of db
                        embedding_function=gemini_embeddings   # Embedding model
                   )

retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 1})

print(len(retriever.get_relevant_documents("MMLU")))

from langchain_google_genai import ChatGoogleGenerativeAI


llm = ChatGoogleGenerativeAI(model="gemini-pro",
                 temperature=0.7, top_p=0.85)

llm_prompt_template = """We are researchers specializing in renewable energy law, with a keen interest in analyzing and understanding the various commitments states have made toward renewable energy. Our work involves examining state bills and legislation to gather detailed insights into how different states are approaching renewable energy. I am currently working on a project where I need to compile and analyze data regarding the specific commitments and targets set by each state in the realm of renewable energy. To facilitate this, I'm aiming to fill out an Excel spreadsheet with diverse values representing these commitments. Could you assist me in understanding these commitments better and help gather objective values from the bills to attain the relevant information. Provide only the numerical values for all of the commitment levels as needed."""


llm_prompt = PromptTemplate.from_template(llm_prompt_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | llm_prompt
    | llm
    | StrOutputParser()
)


resp2 = rag_chain.invoke("What's the date at which the commitment passed into law. It is ususally the date that the document is filed or approved.")
print("RESPONSE: ", resp2)