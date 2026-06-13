from dotenv import load_dotenv
import langchain_community.document_loaders as loaders
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import streamlit as st


@st.cache_resource
def build_chain():
    load_dotenv()
    loaddoc = loaders.PyPDFLoader("Docs/article1.pdf").load()
    textsplit = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splitdocs = textsplit.split_documents(loaddoc)
    # Create an embeddings object.
    embeddings = OpenAIEmbeddings() 
    # Use Chroma.from_documents()
    if os.path.exists("chroma_db"):
        vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    else:
        vector_db = Chroma.from_documents(splitdocs, embeddings, persist_directory="chroma_db")
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )
    def format_docs(docs):
        formatted = "\n\n".join([d.page_content for d in docs])
        return formatted  
    chain = (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
         | prompt
         | ChatOpenAI(model="gpt-4o-mini")
         | StrOutputParser()
    )
    return chain
 
chain = build_chain()
st.title("Manufacturing RAG Agent")

# 3. Chat history lives in st.session_state (this SURVIVES reruns,
#    unlike normal variables). Pattern:
#    - if "messages" not in st.session_state: initialize it to []
#    - loop over st.session_state.messages and render each one with
#      st.chat_message(role) + st.markdown(content)
#    This redraws the whole conversation on every rerun — which is
#    why history has to live in session_state, not a plain list.
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. The input + response cycle:
#    - prompt = st.chat_input("Ask about the documents...")
#    - if prompt:
#         * append the user message to session_state.messages, render it
#         * call chain.invoke(prompt) to get the answer
#         * append + render the assistant message
prompt = st.chat_input("Ask about the documents...")
if prompt:
    # append the user message to session_state.messages, render it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    # call chain.invoke(prompt) to get the answer
    response = chain.invoke(prompt)
    # append + render the assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
