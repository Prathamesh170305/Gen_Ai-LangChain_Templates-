from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Literal
from pydantic import BaseModel, Field

load_dotenv()

model=ChatOpenAI()

#schema
json_schema={
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "description": "A list of key themes mentioned in the review",
      "items": {
        "type": "string"
      }
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "description": "The sentiment of the review, either 'positive', 'negative', or 'neutral'",
      "enum": ["positive", "negative", "neutral"]
    },
    "pros": {
      "type": "array",
      "description": "A list of positive aspects mentioned in the review",
      "items": {
        "type": "string"
      }
    },
    "cons": {
      "type": "array",
      "description": "A list of negative aspects mentioned in the review",
      "items": {
        "type": "string"
      }
    }
  },
  "required": ["key_themes", "summary", "sentiment", "pros", "cons"]
}

    

structured_model=model.with_structured_output(json_schema)

res=structured_model.invoke("// Give a brief summary and sentiment of the following review: 'The food was delicious but the service was terrible.'")

print(res.summary)