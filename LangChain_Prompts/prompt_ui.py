from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
# from langchain_openai import OpenAIEmbeddings
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

st.header("Research Tool")

paper_input = st.selectbox("Search Research Paper", [
    'A Survey on Image Data Augmentation for Deep Learning',
    'Attention Is All You Need',
    'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
    'Deep Residual Learning for Image Recognition',
    'Generative Adversarial Nets'
])

style_input = st.selectbox("Select Style", ['Beginner','Intermediate','Expert'])
length_input = st.selectbox("Select Length", ['Short(1-2 paragraph)','Medium(3-4 paragraph)','Long(5-6 paragraph)'])

template=load_prompt('template.json')

# fill the placeholders
prompt = template.format(
    paper_input=paper_input,
    style_input=style_input,
    length_input=length_input
)

if st.button('Summary'):
    res = model.invoke(prompt)
    st.write(res.content)
