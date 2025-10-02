from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel

load_dotenv()

# Initialize plain text generation LLM
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="Generate the tweet about {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Generate the linkedIn post on: {topic}",
    input_variables=["topic"]
)

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'tweet': RunnableSequence(prompt1 | model | parser),
    'hashtags': RunnableSequence(prompt2 | model | parser)
})

res=parallel_chain.invoke({"topic":"AI"})

print(res)
