from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
import getpass
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


loader = PyPDFLoader("database.pdf")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector = embeddings.embed_query("hello, world!")

vectorstore = Chroma.from_documents(documents=docs, embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})
retrieved_docs = retriever.invoke("What is stress?")
print(retrieved_docs[5].page_content)

import warnings
warnings.filterwarnings("ignore", message="Your application has authenticated using end user credentials")

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    # Check if the environment variable is already set
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GOOGLE_API_KEY is not set in environment variables.")
    else:
        # Access the API key from the environment variable
        google_api_key = os.environ["GOOGLE_API_KEY"]
        print(f"Google API key loaded successfully: {google_api_key[:5]}****")
except Exception as e:
    print(f"An error occurred: {e}")

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.7, max_tokens=500)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
   '''You are a mental health virtual assistant  Alexa  designed to support students with their emotional and mental health concerns.
You can understand and respond with empathy while offering advice and resources based on the queries.
You should assist in helping students manage stress, anxiety, academic pressures, and any other emotional or mental health concerns.
Use your knowledege and sense to understand emotions, maintain a comforting tone, and offer solutions based on the context but be empathic and kind in nature all time.
When needed, retrieve helpful resources or suggestions from the knowledge base for students to explore.'''
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]

)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

try:
  response = rag_chain.invoke({"input": "how to manage stress of studies "})
  print(response["answer"])
except Exception as e:
  print(f"An error occurred: {e}")


from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# System prompt for mental health assistant
system_prompt = (
    '''You are a mental health virtual assistant, Alexa, designed to support students with their emotional and mental health concerns.
    You can understand and respond with empathy while offering advice and resources based on the queries.
    You should assist in helping students manage stress, anxiety, academic pressures, and any other emotional or mental health concerns.
    Use your knowledge and sense to understand emotions, maintain a comforting tone, and offer solutions based on the context but be empathetic and kind at all times.
    When needed, retrieve helpful resources or suggestions from the knowledge base for students to explore.
    '''
    "{context}"
)

# Creating the prompt with system message and user input
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Memory for chat history retention and context
memory = ConversationBufferMemory(
    memory_key="chat_history",  # Store conversation history
    return_messages=True        # Return previous messages as part of the context
)

# Create the question-answer chain (does not use retriever directly)
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Fetch relevant documents using the retriever separately
def fetch_documents(input_text):
    # Retrieve relevant documents from the knowledge base
    docs = retriever.get_relevant_documents(input_text)
    return docs

def process_input(input_text):
    try:
        # Step 1: Fetch documents using the retriever
        docs = fetch_documents(input_text)

        # Step 2: Prepare the input with chat history (context) and user input
        chat_history = memory.load_memory_variables({})["chat_history"]

        # Step 3: Use question-answering chain to process the retrieved documents
        response = question_answer_chain.invoke({
            "input_documents": docs,
            "input": input_text,
            "context": chat_history
        })

        # Debugging: Print the full response to see its structure
        print("Full response:", response)

        # Step 4: Handle the response based on its actual structure
        if isinstance(response, dict) and "output" in response:
            return response["output"]
        else:
            # If it's not a dict, return the raw response or handle it accordingly
            return str(response)

    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

# Example usage
if __name__ == "__main__":
    input_query = "how to manage stress of studies"
    answer = process_input(input_query)
    print("Chatbot response:", answer)
    
input_query = "having trouble concentrating, or worried about upcoming exams?"
answer = process_input(input_query)
print("Chatbot response:", answer)

