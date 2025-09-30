from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser , JsonOutputParser

load_dotenv()

# Initialize plain text generation LLM
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model=ChatHuggingFace(llm=llm)

parser=JsonOutputParser()

template=PromptTemplate(
    template="Give me the name , city and age of the fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)

# prompt=template.format()

# res=model.invoke(prompt)

# final_res=parser.parse(res.content)


chain=template | model | parser 
final_res=chain.invoke({})
print(final_res)