
from langchain import PromptTemplate
from langchain import hub
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnableParallel

path = "va-code.pdf"
loader = PyPDFLoader(path)

documents = loader.load()

google_gemini_api_key = "AIzaSyBqo_mqOAw3S_MdDqikDCO2SfiowCX5Gyo"

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
docs = text_splitter.split_documents(documents=documents)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key = google_gemini_api_key)

vectorstore = Chroma.from_documents(documents=splits,
                                            embedding=embeddings)

prompt = hub.pull("rlm/rag-prompt")
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_gemini_api_key,
                 temperature=0.1, top_p=0.85)



retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(), llm=llm)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | llm
            | StrOutputParser()
        )

rag_chain_with_source = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()} # change retriever as necessary
        ).assign(answer=rag_chain_from_docs)

all_questions = [
    "Which state, area, or territory is the reference law regarding to? The answer should be only the name of the state, area, or territory, without any explanation.",
    "What is the legal commitment name? It is the varbatim name of the commitment in law. Answer should be just the name (e.g. Renewable Portfolio Standard), without any explanation.",
    """List energy sources that meet the criteria of RPS. Categorize the energy sources you found into the following options: "biogas", "biomass", "liquid biofuel", "res fuel cells", "fuel cells", "hydroelectric", "small falling water", "small hydro", "solar", "solar small", "ocean tidal wave thermal", "wind", "wind small", "geothermal geoelectric", "nuclear", "landfill gas", "demand reduction", "energy efficiency", "chp", "res chp", "msw", "anaerobic digestion", "coal mine methane", "tire derived", "advanced coal", "coal mine waste", "coal gasification", "natural gas", "dist generation", "recycled energy", "carbon free", "non ff nuclear carbon free", "partial carbon capture", "industrial cogeneration", "near zero carbon", and "new technology". The answer should only include the name of the energy sources (the name from the text, do not change) and the option. Energy sources should be separated by bracket and newline. The energy name and its option should be separated by a colon. Do not provide any explanation.""",
    "List energy sources that receives multifold credit (e.g. doubled, tripled) towards RPS/CES commitment if any. The answer should only include the name of the energy sources and their corresponding multiple in numerical value(e.g. double = 2). Energy sources should be separated by bracket and newline. In each energy source, the energy name and its corresponding multiple should be separated by a colon. If no energy sources receive multifold credits, give 'None' as the response. Do not provide any explanation.",
    "What is the date at which the commitment passed into law. It is usually the date that the document is filed or approved. Answer the question in the date format YYYY/MM/DD, without any explanation.",
    "What's the RPS/CES commitment in the policy and what's the year by which the commitment must be met? If the policy has multiple commitments at different dates, show the commitment associated with the largest date. The answer should only include the percentage (e.g. 10%) and year, separated by a colon. Do not provide with any explanation.",
    "What is the RPS commitment for the year the policy was introduced? The answer should be a percentage (e.g. 10%). Do not answer with any explanation.",
    "What's the percentage of the RPS/CES commitment that is voluntary? If the policy has multiple commitments at different dates, show the voluntary percentage of the commitment associated with the largest date. The answer should be a percentage with two decimal places (e.g. 10.00%). If the text does not mention voluntary percentage, the answer should be 0.00%. Do not answer with any explanation."
]


llm_prompt_template = "We are researchers specializing in renewable energy law, with a keen interest in analyzing and understanding the various commitments states have made toward renewable energy. Our work involves examining state bills and legislation to gather detailed insights into how different states are approaching renewable energy. I am currently working on a project where I need to compile and analyze data regarding the specific commitments and targets set by each state in the realm of renewable energy. To facilitate this, I'm aiming to fill out an Excel spreadsheet with diverse values representing these commitments. Could you assist me in understanding these commitments better and help gather objective values from the bills to attain the relevant information. Provide only the numerical values for all of the commitment levels as needed. Do not consider any text that has been striked-through in your responses."


llm_prompt = PromptTemplate.from_template(llm_prompt_template)

print("LLM Prompt: ", llm_prompt)

for question in all_questions:
    result = rag_chain_with_source.invoke({"context": question, "retriever": retriever})

    print("Question: ", question)
    print("Answer: ", result["answer"])

