import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.query_engine import PandasQueryEngine
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import ReActAgent
from llama_index.llms import OpenAI
from pdf import canada_engine
from note_engine import note_engine
from prompts import new_prompt, instruction_str, context

load_dotenv()

# Load population data
population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

# Initialize PandasQueryEngine for population data
population_query_engine = PandasQueryEngine(
    df=population_df, verbose=True, instruction_str=instruction_str
)

# Initialize tools
tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="This tool provides information about world population and demographics.",
        ),
    ),
    QueryEngineTool(
        query_engine=canada_engine,
        metadata=ToolMetadata(
            name="canada_data",
            description="This tool provides detailed information about Canada.",
        ),
    ),
]

# Initialize AI model
llm = OpenAI(model="gpt-3.5-turbo-0613")

# Initialize reactive agent
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

# Streamlit interface
st.title("Interactive Query Tool")

query_prompt = st.text_input("Enter your query:")
if query_prompt:
    result = agent.query(query_prompt)
    st.write(result.response)
