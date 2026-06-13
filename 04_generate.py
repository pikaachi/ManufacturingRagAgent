from dotenv import load_dotenv
import langchain_community.document_loaders as loaders
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

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
# print(results[0].page_content)

# 1. Define a question as a plain string variable.
# q1 = "how do you identify waste in the manufacturing process?"
# Out-of-scope question to test the "don't know" part of the prompt:
q1 = "what are the best practices for employee onboarding in the tech industry?"
# 2. RETRIEVE: use your store to get the relevant chunks for that
#    question (the similarity_search you already know). Pull the
#    .page_content out of each returned chunk and join them into one
#    big "context" string. Look at it — this is literally what you're
#    about to feed the model.
results = db.similarity_search(q1)
context = "\n\n".join([r.page_content for r in results])
# 3. BUILD THE PROMPT: construct a message that contains, clearly
#    separated:
#      - an instruction setting the rules
#      - the context (your joined chunks)
#      - the question
#    The instruction is the heart of it. Something along the lines of:
#    "Answer the question using ONLY the context below. If the answer
#    is not in the context, say you don't know." Write it in your own
#    words — phrasing it yourself is part of the lesson.
message = f"""Answer the question using ONLY the context below. If the answer is not in the context, say you don't know.
Context:
{context}
Question: {q1}
"""
# 4. TEST THE PROMPT: feed it to the model and see what happens.
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke(message)
print(response.content)