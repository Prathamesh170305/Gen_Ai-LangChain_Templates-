from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize plain text generation LLM
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model=ChatHuggingFace(llm=llm)


#Task->
    # topic->llm->report->llm->summary
    
prompt1=PromptTemplate(
    template="Give me a detailed report on the following topic : {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Give me a concise summary of the following report : {report}",
    input_variables=["report"]
)

parser=StrOutputParser()

chain=prompt1 | model | parser | prompt2 | model | parser 

res=chain.invoke({"topic":"God of Cricket"})

print(res)