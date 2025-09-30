from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch ,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# Initialize plain text generation LLM
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['Positive', 'Negative', 'Neutral'] = Field(description="The sentiment of the text")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

# FIXED prompt: tell LLM to output only valid JSON object
prompt1 = PromptTemplate(
    template=(
        "Classify the sentiment of the following text: {text}\n"
        "Give your answer as Positive or Negative.\n"
        "Respond ONLY with a valid JSON object.\n"
        "{format_instruction}"
    ),
    input_variables=["text"],
    partial_variables={"format_instruction": parser2.get_format_instructions()}
)  

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="Write an appropriate response for the following positive review : {text}",
    input_variables=["text"] 
)
prompt3 = PromptTemplate(
    template="Write an appropriate response for the following negative review : {text}",
    input_variables=["text"] 
)

# Branch checks must match capitalization in schema
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'Positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'Negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "Neutral review, no response needed")
)

chain = classifier_chain | branch_chain

# Run chain
res = chain.invoke({"text": "The product quality is really good and I am satisfied with my purchase."})

print("Final Result:", res)
