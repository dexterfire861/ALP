# implementation from https://python.langchain.com/docs/use_cases/question_answering/sources

import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

openai_api_key = "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"

# enter folder name
path = "City of Stuart v. 3M"
loader = PyPDFDirectoryLoader(path)

documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
docs = text_splitter.split_documents(documents=documents)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# doing source retrieval
from langchain_core.runnables import RunnableParallel

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

# enter query
result = rag_chain_with_source.invoke("Who was affected by PFAS and to what extent?")
print("\n############### RESPONSE ###############")
print(result["answer"])
# fetch the 3 sources
for i in range(3):
    print(f"\n############### SOURCE #{i + 1} ###############")
    print(result["context"][i].page_content)
    print("\n" + result["context"][i].metadata['source'])
