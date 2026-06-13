import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()
key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("Say 'connection works' and nothing else.")
print(response.content)