import os
os.environ["NVIDIA_API_KEY"] = "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"

from langchain_nvidia_ai_endpoints import ChatNVIDIA,NVIDIAEmbeddings,NVIDIARerank
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


llm = ChatNVIDIA(model="meta/llama3-8b-instruct",
                 temperature=0.6)

embeddings = NVIDIAEmbeddings(model="NV-Embed-QA")

pdf_filepath = r"G:\LangChain\Atomic habits ( PDFDrive ).pdf"
loader = PyPDFLoader(pdf_filepath)
data = loader.load()

#now use recursive character text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
docs = text_splitter.split_documents(data)

#Here the chromaDB takes the documents as the input and generates the embeddings from them using "LLAMA EMBEDDING MODEL".
#these embeddings are stored in chromaDB
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,  # Use `embedding` instead of `embedding_function`
    persist_directory="./vecembeddings_chroma_db"
)

print("Chroma DB created successfully!")

retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":1})
retrieved_docs = retriever.invoke("Who is sachin tendulkar?")

#creating a retriever chain 
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

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

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "What are the four laws of Atomic habits can you explain in one line about each law in a short way?"})
print(response["answer"])


