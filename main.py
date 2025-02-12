import pandas as pd
import os
from dotenv import load_dotenv
import ast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()

# Load API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
claude_api_key = os.getenv("CLAUDE_API_KEY")
hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Load questions from Excel
questions_df = pd.read_excel('Questions.xlsx', sheet_name='Questions')
queries = questions_df.to_dict('records')

# Load ground truth data
gt_df = pd.read_excel('Database June 20 2022.xlsx', sheet_name='master')
reencoded_credit_multiples = pd.read_excel('reencoded_col_BD_from_database.xlsx')

def retrieve_ground_truth_values(df, id, reference_law):
    row = df.loc[df['reference law'] == reference_law]
    if id == 3:  # credit multiplier
        str_multiple_list = reencoded_credit_multiples.loc[reencoded_credit_multiples['rps_law'] == reference_law]["credit_multiplier"].iloc[0]
        return ast.literal_eval(str_multiple_list)
    elif id == 4:  # categories of sales excluded or exempt
        cats_of_sales = row.iloc[:, 16].values[0]  # col Q
        return cats_of_sales.split(", ")
    elif id == 5:  # date commitment passed into law
        return str(row.iloc[:, 5].values[0])[:4]  # column F
    elif id == 6:  # RPS and ces commitment and year that it must be met
        rps_commit_perc = row.iloc[:, 8:10].values[0]
        return f"{rps_commit_perc[0]}:{rps_commit_perc[1]}"
    elif id == 7:  # RPS initial commit as a %
        return row.iloc[:, 10].values[0]
    elif id == 8:  # % of RPS and CES that is voluntary
        return row.iloc[:, 17].values[0]
    elif id == 10:  # energy sources
        energy_sources_all = row.iloc[:, 135:172]
        return [column for column in energy_sources_all.columns if pd.notna(energy_sources_all[column].values[0])]
    elif id == 11:  # voluntary components of the commitment associated with the farthest date
        voluntary_perc = float(row.iloc[:, 17].values[0])
        return "Yes" if voluntary_perc > 0 else "No"
    else:
        return "QUESTION ID NOT RECOGNIZED"

# Define models
models = {
    "openai": ChatOpenAI(api_key=openai_api_key, model_name='gpt-4o' ),
    "anthropic": ChatAnthropic(model_name='claude-3-5-sonnet-20241022', api_key=claude_api_key),
    "huggingface": HuggingFaceEndpoint(
                    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                    task="question-answering",
                    max_new_tokens=100,
                    do_sample=False,
                    api_key=hugging_face_api_key
                )
}

# Process PDF files
pdf_directory = os.path.join(os.getcwd(), 'Laws')
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    return documents, splits

def create_vector_store(splits):
    return FAISS.from_documents(splits, embeddings)

def setup_rag_pipeline(vector_store, llm):
    retriever = vector_store.as_retriever()
    
    system_prompt = (
        "We are researchers specializing in renewable energy law, with a keen interest in analyzing and understanding the various commitments states have made toward renewable energy. Our work involves examining state bills and legislation to gather detailed insights into how different states are approaching renewable energy. I am currently working on a project where I need to compile and analyze data regarding the specific commitments and targets set by each state in the realm of renewable energy. To facilitate this, I'm aiming to fill out an Excel spreadsheet with diverse values representing these commitments. Could you assist me in understanding these commitments better and help gather objective values from the bills to attain the relevant information. Provide only the numerical values for all of the commitment levels as needed. Do not consider any text that has been striked-through in your responses."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

def process_queries(queries, file_name, rag_chain):
    results = []
    for query in queries:
        reference_law = file_name[:-4]
        ground_truth = retrieve_ground_truth_values(gt_df, query['ID'], reference_law)
        
        response = rag_chain.invoke({"input": query['Questions']})
        
        results.append({
            "Type": query['Type'],
            "Tags": query['Tags'],
            "Question ID": query['ID'],
            "Question": query['Questions'],
            "Ground Truth": ground_truth,
            "Response": response['answer'],
            "Source": response.get('source_documents', [])
        })
    
    return results

# Main execution
for file_name in pdf_files:
    file_path = os.path.join(pdf_directory, file_name)
    print(f"\nProcessing file: {file_name}")
    
    documents, splits = process_pdf(file_path)
    vector_store = create_vector_store(splits)

    os.makedirs("results", exist_ok=True)

    for model_name, llm in models.items():
        print(f"\nProcessing with model: {model_name}")

        rag_chain = setup_rag_pipeline(vector_store, llm)
        results = process_queries(queries, file_name, rag_chain)

        print("Results:")
        for result in results:
            print(result)

        df = pd.DataFrame(results)
        output_file = os.path.join("results", f"{os.path.splitext(file_name)[0]}_{model_name}.xlsx")
        df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")