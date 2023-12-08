import os
import streamlit as st
from typing import Literal
from dataclasses import dataclass

import qdrant_client
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import Qdrant

from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain,LLMChain, RetrievalQA
from langchain.chains.conversation.memory import ConversationSummaryMemory

os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
QDRANT_HOST = st.secrets["QDRANT_HOST"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

# Streamlit Header
st.set_page_config(page_title="RAG-Chat - An LLM-powered chat bot")
st.title("RAGChat")
st.write("This is a chatbot for your custom knowledge database")

# Defining message class
@dataclass
class Message :
    """Class for keepiong track of chat Message."""
    origin : Literal["Customer","elsa"]
    Message : "str"

#Load QDRANT client
def load_db():
    client = qdrant_client.QdrantClient(
        url=QDRANT_HOST,
        api_key=QDRANT_API_KEY,
    )
    embeddings = CohereEmbeddings(model="embed-english-v2.0")
    vector_store = Qdrant(
        client = client,
        collection_name = "hotelDataCollection",
        embeddings = embeddings
    )
    print("connection established !")
    return vector_store

def initialize_session_state() :
    vector_store = load_db()

    if "history" not in st.session_state:
        st.session_state.history = []
    
    if "chain" not in st.session_state : 
        prompt_template = """
        You are a Hotel Receptionist at "Four Points by Sheraton" hotel.

        You will be given a context of the conversation made so far followed by a customer's question, 
        give the answer to the question using the context. 
        The answer should be short, straight and to the point. If you don't know the answer, reply that the answer is not available.
        Never Hallucinate

        Context: {context}

        Question: {question}
        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain_type_kwargs = { "prompt" : PROMPT }
        llm = Cohere(model = "command", temperature=0.5)

    # build your rag chain
    st.session_state.chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type="stuff",
        memory = ConversationSummaryMemory(llm = llm, memory_key='chat_history', input_key='question', output_key= 'answer', return_messages=True),
        retriever = vector_store.as_retriever(),
        return_source_documents=False,
        combine_docs_chain_kwargs=chain_type_kwargs,
    )