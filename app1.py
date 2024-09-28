from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import logging


# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory





logging.basicConfig(level=logging.DEBUG)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Setup the psychology chatbot components
loader = PyPDFLoader("database.pdf")
data = loader.load()

# Split data into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Embed documents and set up the Chroma vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

# Create the retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

# System prompt for the chatbot
system_prompt = (
    '''You are a mental health virtual assistant, designed to support students with their emotional and mental health concerns.
    Respond with empathy and understanding while offering practical advice for managing stress, anxiety, and academic pressures.'''
    "{context}"
)

# Set up Google Generative AI for the response
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, max_tokens=500)

# Set up a memory buffer for conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompt template for chat history and human input
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Question-answer chain (combining retriever results and LLM)
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Process the input query
def process_input(input_text):
    try:
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(input_text)
        chat_history = memory.load_memory_variables({})["chat_history"]

        # Use the chain to generate a response
        response = question_answer_chain.invoke({
            "input_documents": docs,
            "input": input_text,
            "context": chat_history
        })

        # Extract the answer from the response
        return response.get("output", str(response))

    except Exception as e:
        print(f"An error occurred: {e}")
        return "I'm sorry, I couldn't process your request right now."

# Flask route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Flask route for the chatbot
@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Ensure the request has JSON data
        if not request.is_json:
            return jsonify({'response': "Invalid request: JSON required"}), 400

        # Parse JSON request
        data = request.get_json()

        # Check if 'message' key is present in the request
        user_message = data.get('message')
        if not user_message:
            return jsonify({'response': "Error: 'message' field is missing"}), 400

        # For now, return a simple response (replace this logic with your AI response later)
        bot_response = f"Echo: {user_message}"

        # Return the bot's response as JSON
        return jsonify({'response': bot_response}), 200

    except Exception as e:
        # Log the exception for debugging purposes and return a generic error message
        print(f"An error occurred: {e}")
        return jsonify({'response': "I'm sorry, I couldn't process your request right now."}), 500

if __name__ == '__main__':
    app.run(debug=True)

