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


load_dotenv()
multiple_docs = loaders.PyPDFDirectoryLoader("Docs/").load()
textsplit = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splitdocs = textsplit.split_documents(multiple_docs)
# Create an embeddings object.
embeddings = OpenAIEmbeddings() 
# Use Chroma.from_documents()
if os.path.exists("chroma_db"):
    vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
else:
    vector_db = Chroma.from_documents(splitdocs, embeddings, persist_directory="chroma_db")
# 1. Define a question as a plain string variable.
q1 = "how do you identify waste in the manufacturing process and what are different types of waste?"
# Out-of-scope question to test the "don't know" part of the prompt:
# q1 = "what are the best practices for employee onboarding in the tech industry?"
# 2. RETRIEVE: Pull the .page_content out of each returned chunk and join them into one
#    big "context" string. 
results = vector_db.similarity_search(q1)
context = "\n\n".join([r.page_content for r in results])
# 3. BUILD THE PROMPT: write a prompt that instructs the model to answer the question using
message = f"""Answer the question using ONLY the context below. If the answer is not in the context, say you don't know.
Context:
{context}
Question: {q1}
"""
# 4. TEST THE PROMPT: feed it to the model and see what happens.
llm = ChatOpenAI(model="gpt-4o-mini")

retriever = vector_db.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    formatted = "\n\n".join([d.page_content for d in docs])
    return formatted

prompt = ChatPromptTemplate.from_template(
    "Answer the question using ONLY the context below. "
    "If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)
chain = (
    {"context": retriever | format_docs,
     "question": RunnablePassthrough()}
     | prompt
     | llm
     | StrOutputParser()
)

response = chain.invoke(q1)

print(response)