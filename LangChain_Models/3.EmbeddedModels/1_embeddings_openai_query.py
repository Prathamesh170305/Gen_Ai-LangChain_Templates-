from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=OpenAIEmbeddings(model='text-embeddings-3-large',dimensions=32)

result=embedding.embed_query("Delhi is the Capital of India")

#result iss sentence ka vector representation hai basically embeddings h 
print(str(result))