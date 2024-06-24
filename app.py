import streamlit as st
import PyPDF2
import os
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.document import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
key=st.secrets['GOOGLE_API_KEY']
load_dotenv()
GOOGLE_API_KEY='AIzaSyC-2p9sD2V3WdBHeK0B-c1SMeKim_G6PoQ'
# Make sure the environment variable is loaded correctly
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

genai.configure(api_key=api_key)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, vector_store):
    retriever = vector_store.as_retriever()
    docs = retriever.get_relevant_documents(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    # Record chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Insert the latest chat at the beginning
    st.session_state.chat_history.insert(0, {
        "user": user_question,
        "bot": response["output_text"]
    })

    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Assignment")
    st.header("AI NXT Assignment")

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question and st.session_state.vector_store:
        user_input(user_question, st.session_state.vector_store)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vector_store = get_vector_store(text_chunks)
                st.session_state.raw_text = raw_text  # Store raw text in session state
                st.success("Done")

        # Display the uploaded PDF content
        st.subheader("Uploaded PDF Content")
        if 'raw_text' in st.session_state and st.session_state.raw_text:
            st.text_area("PDF Content", value=st.session_state.raw_text, height=400)

    # Display chat history
    st.subheader("Chat History")
    if 'chat_history' in st.session_state:
        for chat in st.session_state.chat_history:
            st.write(f"**User:** {chat['user']}")
            st.write(f"**Bot:** {chat['bot']}")
            st.write("---")

if __name__ == "__main__":
    main()
