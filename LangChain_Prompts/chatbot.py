from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)

chat_history=[
    SystemMessage(content='you are a helpful chat assistant'),
    #HumanMessage(content='Hello, who are you?')
]

while True:
    user_input=input('You:')
    chat_history.append(HumanMessage(content=user_input))
    if user_input=='exit':
        break
    res=model.invoke(chat_history)
    chat_history.append(AIMessage(content=res.content)x)
    print('AI:',res.content)
    
print(chat_history)