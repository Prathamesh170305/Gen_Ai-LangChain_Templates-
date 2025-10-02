from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough , RunnableLambda,RunnableBranch

load_dotenv()

# Initialize plain text generation LLM
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

Prompt=PromptTemplate(
    template="Write a detailed report about {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Write a concise summary of the following text : {text}",
    input_variables=["text"]
)

parser=StrOutputParser()

report_gen_chain=Prompt | model | parser

branch_chain=RunnableBranch(
    (lambda x : len(x.split())>500, RunnableSequence(prompt2 , model , parser)),
    RunnablePassthrough()
)

final_chain=report_gen_chain | branch_chain

print(final_chain.invoke({"topic":"Cricket"}))