from langchain_core.prompts import chatPromptTemplate ,MessagesPlaceholder
from langchain_core.messages import HumanMessage

#chat template 
chat_template=chatPromptTemplate(
    [
        ('system','You are a {domain} assistant.'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human','{query}')
    ]
)


#load chat history
chat_history=[] 
with open('chat_history.txt','r') as file:
    chat_history.extend(file.readlines())
    

print(chat_history)

#create the prompt
prompt=chat_template.invoke({'domain':'Customer Care','chat_history':chat_history,'query':'Where is my refund?'})