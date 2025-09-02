from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

def initialize_chat_engine(data_dir="data/", model_name="gemini-2.5-pro"):
    # Load and split documents
    all_docs = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(data_dir, file))
            all_docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(all_docs)

    # Create embedding and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)

    # Create Gemini chat model
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    # Memory for conversation
    memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)

    #Set up the chain with prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=True
    )

    return qa_chain