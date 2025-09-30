from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

# Define schema
class Person(BaseModel):
    name: str = Field(description="The name of the person")
    city: str = Field(description="The city where the person lives")
    age: int = Field(description="The age of the person")
    
parser = PydanticOutputParser(pydantic_object=Person)

# Correct template (note: format_instructions)
template = PromptTemplate(
    template="Give me the name, city and age of the fictional person from {place}\n{format_instructions}",
    input_variables=['place'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# # Build prompt
# prompt = template.format(place="India")

# # Call model
# res = model.invoke(prompt)

# # Parse response into Person
# final_res = parser.parse(res.content)

chain=template | model | parser
final_res=chain.invoke({"place":"Germany"})

print(final_res)
