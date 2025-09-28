from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Literal

load_dotenv()

model=ChatOpenAI()

#schema
class Review(TypedDict):
    key_themes:Annotated[list[str],"A list of key themes mentioned in the review"]
    summary: Annotated[str,'A brief summary of the review']
    sentiment: Annotated[str,"The sentiment of the review, either 'positive', 'negative', or 'neutral'"]
    pros: Annotated[list[str],"A list of positive aspects mentioned in the review"]
    cons: Annotated[list[str],"A list of negative aspects mentioned in the review"]

structured_model=model.with_structured_output(Review)

res=structured_model.invoke("// Give a brief summary and sentiment of the following review: 'The food was delicious but the service was terrible.'")

print(res)