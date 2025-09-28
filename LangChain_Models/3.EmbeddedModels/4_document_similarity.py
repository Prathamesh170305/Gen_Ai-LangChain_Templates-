#from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()


#embedding=OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)
embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


documents=[
    "Virat Kohli is an Indian Cricketer known for his aggressive batting and leadership",
    "Rohit Sharma is the Indian World t20 cup winning captain",
    "Sachin is the god of cricket"
]

query="tell me about Rohit sharma"

doc_embeddings=embedding.embed_documents(documents)
query_embedding=embedding.embed_query(query)

scores=(cosine_similarity([query_embedding],doc_embeddings)[0])

index,score=(sorted(list(enumerate(scores)),key=lambda x:x[1])[-1])

print(documents[index])
print("Similarity score is:",score)

