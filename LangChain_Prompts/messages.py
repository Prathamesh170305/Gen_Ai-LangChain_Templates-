from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is LangChain?"),
]

res=model.invoke(messages)
messages.append(AIMessage(content=res.content))

print(messages)
