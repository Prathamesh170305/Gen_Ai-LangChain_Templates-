from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough

load_dotenv()

# Initialize plain text generation LLM
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Write a detailed explanation of the following joke : {joke}",
    input_variables=["joke"]
)

parser = StrOutputParser()

# Joke generator chain
joke_gen_chain = prompt | model | parser

# Parallel: pass joke directly + explanation
parallel_gen = RunnableParallel({
    'joke': RunnablePassthrough(),   # just forwards the joke string
    'explanation': prompt2 | model | parser
})

# Final chain: generate joke → feed into parallel
final_chain = joke_gen_chain | parallel_gen

res = final_chain.invoke({"topic":"programming"})

print(res)
