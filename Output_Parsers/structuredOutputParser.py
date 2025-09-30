from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import StructuredOutputParser , ResponseSchema

load_dotenv()

# Initialize plain text generation LLM
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model=ChatHuggingFace(llm=llm)

schema=[
    ResponseSchema(name='fact_1',description='fact number 1 on the topic'),
    ResponseSchema(name='fact_2',description='fact number 2 on the topic'),
    ResponseSchema(name='fact_3',description='fact number 3 on the topic')
]

parser=StructuredOutputParser.from_response_schemas(schema)

template=PromptTemplate(
    template="Give me 3 interesting facts on the following topic : {topic} \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)

# Prompt=template.invoke({"topic":"Artificial Intelligence"})

# res=model.invoke(Prompt)

# final_res=parser.parse(res.content)

chain=template | model | parser 
final_res=chain.invoke({"topic":"Artificial Intelligence"})

print(final_res)