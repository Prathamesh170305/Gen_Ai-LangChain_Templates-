from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough , RunnableLambda

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

parser = StrOutputParser()

# Joke generator (fix: pipe instead of list)
joke_gen_chain = prompt | model | parser

# Parallel branch: forward joke + compute word count
parallel_chain = RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'word_count': RunnableLambda(lambda x: len(x.split(" ")))
    }
)

# Final chain: generate joke -> send to parallel
final_chain = joke_gen_chain | parallel_chain

print(final_chain.invoke({"topic":"programming"}))
