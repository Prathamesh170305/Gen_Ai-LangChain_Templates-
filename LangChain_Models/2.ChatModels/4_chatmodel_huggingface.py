from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

from langchain_huggingface import ChatHuggingFace
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")
print("Response:", result.content)
