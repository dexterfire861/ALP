# implementation from https://python.langchain.com/docs/use_cases/question_answering/sources

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
    # first 5 subdirs
    if i != 0 and i < 6:
        # get subdir path
        path = os.path.join(subdir)
        # name of subdir        
        curr_dir = path[13:]
        loader = PyPDFDirectoryLoader(path)

        # load documents
        print(f"\nloading documents from {curr_dir}...")
        documents = loader.load()

        # split documents and store chunks
        print("spliting documents...")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        docs = text_splitter.split_documents(documents=documents)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))

        # Retrieve and generate using the relevant snippets of the text
        # simple retriever
        retriever = vectorstore.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0)

        # multiquery retriever, appears to have better results
        from langchain.retrievers.multi_query import MultiQueryRetriever
        retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(), llm=llm
        )
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
            {"context": retriever, "question": RunnablePassthrough()} # change retriever as necessary
        ).assign(answer=rag_chain_from_docs)

        # ask questions
        print("querying documents...")
        # print answer to a text document
        filename = "output/" + curr_dir + ".txt"
        file = open(filename, "w")
        for question in questions:
            result = rag_chain_with_source.invoke(question)

            query = "\n############### QUESTION ###############\n" + question
            answer = "\n############### RESPONSE ###############\n" + result["answer"]
            sources = ""
            for j in range(len(result["context"])):
                sources += f"\n############### SOURCE #{j + 1} ###############\n"
                sources += result["context"][j].metadata['source'] + "\n"
                sources += result["context"][j].page_content + "\n"

            file.write(query)
            file.write(answer)
            file.write(sources)
            # print(answer)
            # print(sources)

        file.close()

        # clear chroma database
        # delete this line to query against ALL documents
        vectorstore.delete_collection()

    i += 1
