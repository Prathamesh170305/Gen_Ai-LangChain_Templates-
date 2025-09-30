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

prompt=PromptTemplate(
    template="What is a good name for a company that makes {product}?",
    input_variables=["product"]
)

parser=StrOutputParser()

chain=prompt | model | parser #prompt to model to parser
final_res=chain.invoke({"product":"Condoms"})

print(final_res)

#chain.get_graph().print_ascii()