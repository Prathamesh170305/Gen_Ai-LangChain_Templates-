from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=OpenAIEmbeddings(model='text-embeddings-3-large',dimensions=32)

documents=[
    "Delhi is the capital of India",
    "Kolkata is the captital of West Bengal",
    "Paris is the capital of France"
]

result=embedding.embed_documents(documents)

#result iss documents ka vector representation hai basically embeddings h 
print(str(result))