import streamlit as st
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch
import os

st.title("My IS883 Week7 Assignment")

prompt = st.text_input("Share with us your experience of the latest trip")

### Load your API Key
os.environ["OPENAI_API_KEY"] = st.secrets["OpenAIkey"]

llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.environ["OPENAI_API_KEY"]
)

### Create a template to detect airline experience
airline_experience_template = """You are an expert at interpreting airline customer experiences.

From the following text, determine whether the customer had a positive or negative experience.
If it is negative, identify whether the cause was the airline's fault (e.g., lost luggage) or beyond the airline's control (e.g., a weather-related delay).

You may only respond with one of the following:
1. Positive experience
2. Negative experience caused by the airline's fault
3. Negative experience beyond the airline's control

Do not provide any response other than these three options.

Text:
{request}

"""


### Create the decision-making chain

flight_experience_chain = (
    PromptTemplate.from_template(airline_experience_template)
    | llm
    | StrOutputParser()
)

# Negative Experiences Caused by the Airline 
negative_byairline_chain = PromptTemplate.from_template(
    """You are an airline counter staff and will talk to customers in person. 
    Your task is to offer empathetic reassurance to the customer regarding their issue and inform them that customer service will contact them soon to resolve the issue or provide compensation shortly.
    Respond in first-person mode.
    
    Your response should follow these guidelines:
    1. Address the customer directly
    2. Inform customers that customer service will contact them soon to resolve the issue or provide compensation shortly.


Text:
{text}

"""
) | llm

# Negative Experiences Beyond the Airline's Control
negative_beyondcontrol_chain = PromptTemplate.from_template(
    """You are an airline counter staff and will talk to customers in person. 
    Your task is to offer empathetic reassurance to the customer regarding their issue and inform them that the airline is not liable in such situations.
    Respond in first-person mode.

Your response should follow these guidelines:
    1. Address the customer directly
    2. Inform customers that the airline is not liable in such situations.


Text:
{text}

"""
) | llm

#Positive Experiences
positive_chain = PromptTemplate.from_template(
    """You are an airline counter staff and will talk to customers in person. 
    Your task is to express gratitude to customers for their valuable feedback and for choosing to fly with the airline.
    Respond in first-person mode.

Your response should follow these guidelines:
    1. Address the customer directly
    2. thank them for their feedback and for choosing to fly with the airline


Text:
{text}

"""
) | llm

### Routing/Branching chain
branch = RunnableBranch(
    (lambda x: "negative experience caused by the airline's fault" in x["experience_type"].lower(), negative_byairline_chain),
    (lambda x: "negative experience beyond the airline's control" in x["experience_type"].lower(), negative_beyondcontrol_chain),
    positive_chain,
)

### Put all the chains together
full_chain = {"experience_type": flight_experience_chain, "text": lambda x: x["request"]} | branch


### Display
reply = full_chain.invoke({"request": prompt})
st.write( prompt )
st.write( reply.content )
