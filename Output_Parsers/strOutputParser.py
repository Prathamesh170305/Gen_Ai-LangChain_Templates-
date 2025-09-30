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

# 1st prompt
template1 = PromptTemplate(
    template="Write a detailed report of the following topic: {topic}",
    input_variables=["topic"]
)
prompt1 = template1.invoke({"topic": "Artificial Intelligence"})
res = model.invoke(prompt1)

# 2nd prompt
template2 = PromptTemplate(
    template="Write a concise summary of the following text: {text}",
    input_variables=["text"]
)
prompt2 = template2.invoke({'text':res.content})
res2 = model.invoke(prompt2)

print("Detailed Report:\n", res)
print("\nSummary:\n", res2)

parser = StrOutputParser()
chain=template1 | model | parser | template2 | model | parser

result=chain.invoke({"topic":"Artificial Intelligence"})

print("\nSummary using chain:\n", result)
