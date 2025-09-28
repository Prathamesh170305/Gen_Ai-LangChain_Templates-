from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text="Delhi is the capital of India"

vector=embedding.embed_query(text)

print(str(vector))

#we can also pass the documents as :
# documents=[
#     "Delhi is the capital of India",
#     "Kolkata is the captital of West Bengal",
#     "Paris is the capital of France"
# ]