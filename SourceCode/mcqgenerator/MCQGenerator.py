import os
import json
import pandas as pd
import traceback
import getpass
import os

from SourceCode.mcqgenerator.utils import read_file, get_table_data
from SourceCode.mcqgenerator.logger import logging

if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

from langchain_groq import ChatGroq

model = ChatGroq(model="llama3-8b-8192",temperature=0.5)

# importing necessary packages from langchain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

# importing necessary packages for callbacks from langchain_core
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

RESPONSE_JSON = {
    "1": {
        "Q1": "Multiple Choice Question",
        "Options": {
            "a": "choose this",
            "b": "choose this",
            "c": "choose this",
            "d": "choose this",
        },
        "Correct": "Correct Answer"
    },
    "2": {
        "Q2": "Multiple Choice Question",
        "Options": {
            "a": "choose this",
            "b": "choose this",
            "c": "choose this",
            "d": "choose this",
        },
        "Correct": "Correct Answer"
    },
    "3": {
        "Q3": "Multiple Choice Question",
        "Options": {
            "a": "choose this",
            "b": "choose this",
            "c": "choose this",
            "d": "choose this",
        },
        "Correct": "Correct Answer"
    },
}

template = """
Text:{text}
You are an expert AI MCQ maker. From the above given text, it is you job to \
create a quiz of {number} multiple choice questions for {subject} students in a {difficult} level.
Make Sure the questions do not get repeated and check all the questions to be vomforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""
quiz_Generation_Template = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=template,
    )
quiz_chain=LLMChain(llm=model, prompt=quiz_Generation_Template, output_key="quiz", verbose=True)

TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""
quiz_evaluation_prompt=PromptTemplate(
    input_variables=["subject", "quiz"], 
    template=TEMPLATE2
    )
review_chain=LLMChain(llm=model, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)

generate_evaluate_chain=SequentialChain(chains=[quiz_chain, review_chain], input_variables=["text", "number", "subject", "tone", "response_json", "difficult"],
                                        output_variables=["quiz", "review"], verbose=True,)
