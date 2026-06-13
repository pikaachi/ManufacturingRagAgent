# (keep your loading + chunking code that produces the list of chunks)
from dotenv import load_dotenv
import langchain_community.document_loaders as loaders
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()
loaddoc = loaders.PyPDFLoader("Docs/article1.pdf").load()

textsplit = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

splitdocs = textsplit.split_documents(loaddoc)
# 1. Import OpenAIEmbeddings from langchain_openai.
#    Create an embeddings object. (It reads your API key from the
#    environment, same as the chat model did.)

# Create an embeddings object.
embeddings = OpenAIEmbeddings() 
# Use Chroma.from_documents()
db = Chroma.from_documents(splitdocs, embeddings, persist_directory="chroma_db")

# Now TEST RETRIEVAL
results = db.similarity_search("how do you identify waste in the manufacturing process?")
print(results[0].page_content)
print(results[0].metadata)
