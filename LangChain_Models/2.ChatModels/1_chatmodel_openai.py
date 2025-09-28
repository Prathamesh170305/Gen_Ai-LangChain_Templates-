from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI(model='gpt-4')

result=model.invoke("What is capital of India?",tempreature=0)

print(result)
#answer will not be the simple string with metadata

print(result.content)
#string main content ke liye 