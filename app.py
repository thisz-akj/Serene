


from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


from langchain_community.vectorstores import Chroma
from langchain_community.llms import ChatGoogleGenerativeAI

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Setup the psychology chatbot components
loader = PyPDFLoader("database.pdf")
data = loader.load()

from langchain.llms import ChatGoogleGenerativeAI

# Set up Google Generative AI for the response (using Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, max_tokens=500)

# Now pass the LLM instance to the chain

# Split data into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Embed documents and set up the Chroma vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

# Create the retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

# System prompt for the chatbot
# System prompt including context for documents
system_prompt = (
    '''You are a mental health virtual assistant, designed to support students with their emotional and mental health concerns.
    Respond with empathy and understanding while offering practical advice for managing stress, anxiety, and academic pressures.
    
    Here is some relevant information for you to consider:
    {context}'''
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Update the prompt to accept 'context' and 'input'
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Now pass the LLM instance (e.g., Google Generative AI) to the chain
question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)


# Process input function
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
        app.logger.error(f"An error occurred: {e}")
        return "I'm sorry, I couldn't process your request right now."


# Flask route for the chatbot
@app.route('/chat', methods=['POST'])
def chat():
    try:
        if not request.is_json:
            return jsonify({'response': "Invalid request: JSON required"}), 400

        data = request.get_json()
        user_message = data.get('message')

        if not user_message:
            return jsonify({'response': "Error: 'message' field is missing"}), 400

        # Process the input message and generate a bot response
        bot_response = process_input(user_message)

        # Return the bot's response as JSON
        return jsonify({'response': bot_response}), 200

    except Exception as e:
        # Log the exception for debugging purposes
        app.logger.error(f"Error during chat processing: {e}")
        return jsonify({'response': "I'm sorry, I couldn't process your request right now."}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

    app.run(debug=True)
