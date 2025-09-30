from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

# Initialize plain text generation LLM
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model1=ChatHuggingFace(llm=llm)
model2=ChatHuggingFace(llm=llm)


#task->
# text(detailed text on the topic)->geneate notes + quiz ->user

prompt1=PromptTemplate(
    template="Generate short and simple notes on the following text: {text}",
    input_variables=["text"]
)

prompt2=PromptTemplate(
    template="Generate a quiz (5 questions) with answers on the following notes: {notes}",
    input_variables=["notes"]
)

prompt3=PromptTemplate(
    template="merge the provided notes and quiz into single document \n notes->{notes} , quiz->{quiz}",
    input_variables=["notes","quiz"]
)

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser}
)

merge_chain= prompt3 | model1 | parser

chain=parallel_chain | merge_chain

res=chain.invoke({"text":"India, officially the Republic of India,[j][20] is a country in South Asia. It is the seventh-largest country by area; the most populous country since 2023;[21] and, since its independence in 1947, the world's most populous democracy.[22][23][24] Bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast, it shares land borders with Pakistan to the west;[k] China, Nepal, and Bhutan to the north; and Bangladesh and Myanmar to the east. In the Indian Ocean, India is near Sri Lanka and the Maldives; its Andaman and Nicobar Islands share a maritime border with Myanmar, Thailand, and Indonesia."})