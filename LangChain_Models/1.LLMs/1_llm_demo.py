from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm=OpenAI(model='gpt-3.5-turbo-instruct')

#invoke sabse jyada imp h , yahi answer leke aate hai 
result=llm.invoke("What is the Capital of India?")

print(result) 
#answer ek plain text string rahega