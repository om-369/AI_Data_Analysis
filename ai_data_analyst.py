import json
import tempfile
import csv
import streamlit as st 
import pandas as pd


from agno.models.openai import OpenAIChat
from phi.agent.duckdb import DuckDbAgent
from agno.tools.pandas import PandasTools
import re

# function to preprocess and save the uploaded file
def preprocess_and_save(file):
    try:
        # read the uploaded file into a dataframe
        if file.name.endswith(".csv"):
            df = pd.read_csv(file, encoding="utf-8", na_values=["NA", "N/A", "missing"])
            
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file, na_values=["NA", "N/A", "missing"])
        else:
            st.error("Unsupported file format. Please Upload a CSV or Excel file")
            return None, None, None
        
        # #Ensure string columns are properly qouted 
        for col in df.select_dtypes(include=['object']):
            df['col'] = df[col].astype(str).replace({r'"':'""'}, regex=True)
            
            
        # Parse dates and numeric columns
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df['col'] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
                
        # create a temporary file to store the processed data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            
            df.to_csv(temp_path, index=False,quoting=csv.QUOTE_ALL)
            # return the temporary file name and the dataframe
        return temp_path, df.columns.tolist(),df   # Return the Data Frame as well
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

# streamlit app 
st.title("📊 Data Analyst Agent")

# Sidebar for API Keys
with st.sidebar:
    st.header("API Keys")
    openai_key = st.text_input("Enter your OpenAI API Key:",type="password")
    if openai_key:
        st.session_state.openai_key = openai_key
        st.success("API key Saved!")
    else:
        st.warning("Please enter your OpenAI API Key.")
        
# file upload widget 
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None and "openai_key" in st.session_state:
    # preprocess and save the uploaded file
    temp_path, columns, df = preprocess_and_save(uploaded_file)
    
    if temp_path and columns and df is not None:
        # Display the uploaded data as table
        st.write("Uploaded Data:")
        st.dataframe(df)
        
        
        # Display the column of the uploaded data
        st.write("Uploaded columns:", columns)
        
        # Configure the semantic model with the temporary file path
        semantic_model = {
            "tables":[
                {
                    "name": "uploaded_data",
                    "description":"Contains the uploaded dataset.",
                    "path": temp_path,
                
            }
                ]
        }
        
        # Initialize the DuckDbAgent for sql generation 
        duckdb_agent = DuckDbAgent(
            model=OpenAIChat(model="gpt-4", api_key=st.session_state.openai_key),
            semantic_model=json.dumps(semantic_model),
            tools=[PandasTools()],
            markdown_mode=True,
            add_history_to_messages = False,
            followups = False,
            read_tool_call_history = False,
            system_propmt = "You are a expert data analyst. Generate SQL Queries to solve the user's query. Return only the SQL query, enclosed in ```sql ``` and give the final answer. ",
            
        )
        
        
        # Initialize code storage in sesion state
        if "generated_code" not in st.session_state:
            st.session_state.generated_code = None
            
            
        # Main query input widget
        user_query = st.text_area("Ask a query about the data:")
        
        # add info message about terminal output
        st.info("💡 check your terminal for a clearer output of the agents response")
        
        if st.button("Submit Query"):
            if user_query.strip() == "":
                st.warning("please enter a query")
            else:
                try:
                    # show loading spinner while processing
                    with st.spinner("Processing your query..."):
                        
                        # get the response from DuckdbAgent
                        response1 = duckdb_agent.run(user_query)
                        
                        # Extract the content from the RunResponse object
                        if hasattr(response1, 'content'):
                            response_content = response1.content
                        else:
                            response_content = str(response1)
                        response = duckdb_agent.print_response(
                            user_query,
                            stream=True,
                        )
                        
                    # Display the response in streamlit
                    st.markdown(response_content)
                
                except Exception as e:
                    st.error(f"Error generating response from the DuckDbAgent: {e}")
                    st.error("Please try rephrasing your query or check if the data format is correct")
            