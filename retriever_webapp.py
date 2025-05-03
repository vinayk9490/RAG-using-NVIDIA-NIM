import streamlit as st
st.title("Q/A RAG Atomic Habits")

import os
os.environ["NVIDIA_API_KEY"] = "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"

from langchain_nvidia_ai_endpoints import ChatNVIDIA,NVIDIAEmbeddings,NVIDIARerank
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


def vector_embeddings():
    st.session_state.embeddings = NVIDIAEmbeddings(model="NV-Embed-QA")
    pdf_filepath = r"G:\LangChain\Atomic habits ( PDFDrive ).pdf"
    loader = PyPDFLoader(pdf_filepath)
    docs = loader.load()

    # Filter out empty documents
    docs = [doc for doc in docs if doc.page_content.strip()]
    st.session_state.docs = docs

    # Split text
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    split_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)

    # Filter out empty chunks (again)
    split_docs = [doc for doc in split_docs if doc.page_content.strip()]

    # Create Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=st.session_state.embeddings,
        persist_directory="./vecembeddings_chroma_db"
    )
    st.session_state.vectorstore = vectorstore


prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

user_prompt=st.text_input("Dear user, please enter the query to be answered")

llm = ChatNVIDIA(model="meta/llama3-8b-instruct",
                 temperature=0.6)

if(st.button("Create Vector Embeddings:")):
    vector_embeddings()
    st.write("The Vector Embeddings are Successfully Created")



system_prompt = (
"You are an assistant for question-answering tasks. "
"Use the following pieces of retrieved context to answer "
"the question. If you don't know the answer, say that you "
"don't know. Use three sentences maximum and keep the "
"answer concise."
"\n\n"
"{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

if(user_prompt):
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":1})
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": "What are the four laws of Atomic habits can you explain in one line about each law in a short way?"})
    #print(response["answer"])
    st.write(response["answer"])