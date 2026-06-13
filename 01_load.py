# 1. Import the loader from langchain_community.document_loaders
import langchain_community.document_loaders as loaders
# 2. Point it at your file. 
loaders.PyPDFLoader("Docs/article1.pdf").load()

# 3. INVESTIGATE 
# print(len(loaders.PyPDFLoader("Docs/article1.pdf").load()))
# print(loaders.PyPDFLoader("Docs/article1.pdf").load()[0].page_content)
# print(loaders.PyPDFLoader("Docs/article1.pdf").load()[0].metadata)