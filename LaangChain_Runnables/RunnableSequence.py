from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

load_dotenv()

# Initialize plain text generation LLM
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Write a detailed explanation of the following joke : {joke}",
    input_variables=["joke"]
)

parser=StrOutputParser()

chain=RunnableSequence([prompt, model, parser,prompt2, model, parser])

res=chain.invoke({"topic":"programming"})

print(res)