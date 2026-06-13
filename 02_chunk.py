# loading code
import langchain_community.document_loaders as loaders
# 2. Point it at your file. 
loaddoc = loaders.PyPDFLoader("Docs/article1.pdf").load()

# 1. Import RecursiveCharacterTextSplitter from langchain.text_splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 2. Create a splitter with a chunk_size and chunk_overlap.
textsplit = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# 3. Feed your loaded documents to the splitter's
#    .split_documents() method. 
textsplit.split_documents(loaddoc)

# 4. INVESTIGATE
# print(len(textsplit.split_documents(loaddoc)))
# print(textsplit.split_documents(loaddoc)[0].page_content)
# print(textsplit.split_documents(loaddoc)[0].metadata)