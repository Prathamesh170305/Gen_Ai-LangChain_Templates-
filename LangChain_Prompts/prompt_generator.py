from langchain_core.prompts import PromptTemplate

# template
template = PromptTemplate(
    template="Summarize the research paper titled '{paper_input}' in a '{style_input}' style and '{length_input}' length.",
    input_variables=['paper_input','style_input','length_input']
)

template.save('template.json')