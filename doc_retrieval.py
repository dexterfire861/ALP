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
    "What were the costs of mitigating PFAs exposure? (environmental remediation, healthcare costs)?",
    "When were the plaintiffs exposed to PFAs?,"
    "What were the costs of mitigating PFAs exposure? We are looking for a numeric dollar value that might include costs of sampling, environmental remediation, healthcare, or any other mentioned costs plaintiffs have encountered due to PFAs exposure.",
    "What were the levels of PFAs found? This would be a value indicating the PFAs concentration found in water bodies, in blood, etc. (e.g. 80 ppt)",
    "What is the size of the population at risk for PFAs contamination? For instance, if a drinking water system was contaminated with PFAs, we want to know the number of people that water system serves.",
    "Who are the plaintiffs and the defendants?",
    "How were the plaintiffs exposed to PFAs? We want to know the exact pathway through which the plaintiff encountered PFAs. For instance, this might be through pathways such as product manufacturing, product use, or a contaminated drinking water supply.",
    "What were the health impacts of PFAs contamination? We are looking for information regarding any bodily/genetic effects PFAS had on the plaintiffs. This may include cancers, diseases, or other human health risks.",
    "What were the environmental impacts of PFAs contamination? This might include contamination of groundwater, soils, surface water, etc.",
    "Any scientific evidence presented regarding the persistence, bioaccumulation, toxicological or any other damaging effects on PFAs?",
    "What are the geographical boundaries of PFAs contamination?",
    "What were the compensations the plaintiffs asked for?",
    "Did 3M and the other defendants conceal the dangers of PFAS from the government and public?",
    "Does this case involve aqueous film-forming foam (AFFF)?",
    "Was the case settled?"
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
