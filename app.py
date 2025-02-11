import streamlit as st
import pandas as pd
import sqlite3
import langchain_openai
import io
import json
from sdv.metadata import Metadata
# import cohere
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os, time
import pickle
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI
llm = AzureChatOpenAI(
    azure_deployment="bfsi-genai-demo-gpt-4o",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-05-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
def save_to_pickle(df, filename='replica_df.pkl'):
    """"
    Save DataFrame to a pickle file
    """
    try:
        with open(filename, 'wb') as file:
            pickle.dump(df, file)
        return True
    except Exception as e:
        print(f"Error saving DataFrame to pickle: {e}")
        return False
    
def load_from_pickle(filename='replica_df.pkl'):
    """"
    Load DataFrame from a pickle file
    """
    try:
        with open(filename, 'rb') as file:
            df = pickle.load(file)
        return df
    except Exception as e:
        print(f"Error loading DataFrame from pickle: {e}")
        return None
        
def call_llm_for_sql(user_prompt, table_name="customer", db_name="customer_testbed.db"):
    try:

        # Generate table schema dynamically
        conn = sqlite3.connect(db_name)
        schema_query = f"PRAGMA table_info({table_name})"
        table_schema = pd.read_sql(schema_query, conn)
        conn.close()

        # Build schema description for the LLM
        schema_description = "\n".join(
            [
                f"{row['name']} ({row['type']})"
                for _, row in table_schema.iterrows()
            ]
        )

        # Construct the LLM prompt
        prompt = f"""
        You are a SQL expert. Generate a valid SQL query based on the following conditions and table schema. Follow these instructions strictly:

        1. The table name is `{table_name}`.
        2. The schema of the table is as follows:
        {schema_description}

        3. **User Prompt**:
        {user_prompt}

        4. Ensure that the SQL query is valid and strictly adheres to the schema.
        5. Only return the SQL query. Do not include explanations or extra text.

        **Example**:
        User Prompt: "Select all rows where Age is greater than 30 and Subscription Type is 'Premium'."
        Table Schema:
        - Name (TEXT)
        - Age (INTEGER)
        - Gender (TEXT)
        - Country (TEXT)
        - Subscription Type (TEXT)
        - Active User (BOOLEAN)
        
        Output:
        SELECT * FROM customer WHERE Age > 30 AND Subscription_Type = 'Premium';
        """
        
### replace with azure chatgpt 4.0 code
        test_response = llm.invoke("Hello")
        print(test_response)
        response = llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        # Extract the SQL query from the response
        generated_sql = extract_sql_query(response.content)
        return generated_sql

    except Exception as e:
        print(f"Error generating SQL: {e}")
        return None


def call_llm_for_pandas_query(user_prompt, table_name="customer"):
    try:
        # Generate table schema dynamically
        replica_df = load_from_pickle()
        schema_description = "\n".join(
            [
                f"{col} ({dtype})"
                for col, dtype in replica_df.dtypes.items()
            ]
        )

        # Construct the LLM prompt
        prompt = f"""
        You are a Pandas expert. Generate a valid Pandas query based on the following conditions and table schema. Follow these instructions strictly:

        1. The DataFrame name is `df`.
        2. The schema of the DataFrame is as follows:
        {schema_description}

        3. **User Prompt**:
        {user_prompt}

        4. Ensure that the Pandas query is valid and strictly adheres to the schema.
        5. Only return the Pandas query. Do not include explanations or extra text.

        **Example**:
        User Prompt: "Select all rows where Age is greater than 30 and Subscription Type is 'Premium'."
        DataFrame Schema:
        - Name (TEXT)
        - Age (INTEGER)
        - Gender (TEXT)
        - Country (TEXT)
        - Subscription Type (TEXT)
        - Active User (BOOLEAN)
        
        Output:
        df[(df['Age'] > 30) & (df['Subscription_Type'] == 'Premium')]
        """
        
        response = llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        # Extract the Pandas query from the response
        generated_query = clean_pandas_query(response.content)
        return generated_query

    except Exception as e:
        print(f"Error generating Pandas query: {e}")
        return None

def create_replica_db(original_db="customer.db", replica_db="customer_testbed.db"):
    try:
        if data_storage_option == "SQLite":
            # Connect to the original database
            conn_original = sqlite3.connect(original_db)
            conn_replica = sqlite3.connect(replica_db)

            # Fetch all table names from the original database
            cursor_original = conn_original.cursor()
            cursor_original.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor_original.fetchall()

            # Copy data for each table
            for table_name_tuple in tables:
                table_name = table_name_tuple[0]

                # Read data from the original table
                data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn_original)

                # Insert data into the replica database
                data.to_sql(table_name, conn_replica, if_exists="replace", index=False)

            # Close connections
            conn_original.close()
            conn_replica.close()
        else:
            conn_original = sqlite3.connect(original_db)
            
        return True

    except Exception as e:
        print(f"Error creating replica database: {e}")
        return False


def upload_to_db(file, db_name="customer.db"):
    try:
        # Read CSV file into DataFrame
        df = pd.read_csv(file)

        if data_storage_option == "SQLite":
            # Connect to SQLite database
            conn = sqlite3.connect(db_name)

            # Insert data into the database
            table_name = "customer"
            
            # Convert all string data to lowercase
            df = df.map(lambda x: x.lower() if isinstance(x, str) else x)
            
            df.to_sql(table_name, conn, if_exists="replace", index=False)

            # Fetch table info for display
            query = f"PRAGMA table_info({table_name})"
            column_info = pd.read_sql(query, conn)
            data = pd.read_sql(f"SELECT * FROM {table_name}", conn)

            conn.close()
        else:
            table_name = "customer"
            column_info = pd.DataFrame(
                {
                    "name": df.columns,
                    "type": df.dtypes
                }
            )
            data = df
        
        return table_name, column_info, data

    except Exception as e:
        st.error(f"Error uploading file to database: {e}")
        return None, None, None
    
    
    
def call_llm_to_generate_test_data(test_condition, table_name):
    try:
        if data_storage_option == "SQLite":
            # Generate data structure from the database
            # data_structure = generate_data_structure(table_name)

            # Construct the prompt
            system_prompt =f"""You are an advanced data generator. Based on the given conditions and structure, you need to create realistic test data in a tabular format. Follow these instructions strictly:"""
            user_prompt = f"""        
            1. **Conditions**: Use the following test condition provided by the user:
            {test_condition}
            
            3. **reference dataset**:
            index,acct_num,ccid,zip4_code,lobs,govt_issued_id,phone_number,first_name,last_name,address_line_one,city_name,state_code,postal_code,country
            1234566,9100055032276,CC00000001,Null,RETAIL,3340783723534,2175557570,LOIS,WILLIAMS,936 CARDINAL LANE,CHAMPAIGN,IL,61820,UNITED STATES
            1234567,9100050039685,CC00000002,56799,CARD,4413416413453,9185552699,JULIE,BELL,3714 HENRY FORD AVENUE,AFTON,OK,74331,UNITED STATES
            1234568,5189413110708,CC00000003,Null,WEATH MANAGEMENT,9093134663453,5595550732,JESUS,MARESCOT,178 CARLING CT,SAN JOSE,CA,95111,UNITED STATES
            1234569,54241810713641,CC00000004,Null,CORPORATE,4357890873453,Null,KING,MAKER,13TH ST,PLANO,TX,75074,UNITED STATES
            1234570,517908006006,CC00000005,Null,CARD,50202221712312,7015555173,SUSAN,BLAKELY,3056 COURTRIGHT STREET,DICKINSON,ND,58601,UNITED STATES
            1234571,5179080064702,CC00000006,67564,MORGAGE,5524154083242,45676549,ERICA,HWANG,486 DUCK CREEK ROAD,SAN FRANCISCO,CA,94107,UNITED STATES
            1234572,5424181321715,CC00000007,9466,WEATH MANAGEMENT,3700047723423,78987675,MARCELLA,MCDONNELL,12242 E WHITMORE AVE,HUGHSON,CA,95326,UNITED STATES
            1234573,5285001306647,CC00000008,2715,RETAIL,3423423422342,Null,SAM,SAMY,6500 CAMPUS CIRCLE DR E,IRVING,TX,75063,UNITED STATES
            1234574,5424185127468,CC00000009,2127,CORPORATE,7467698866572,Null,NIH,FYTF HIUHIU,2887 MARK TWAIN DR,FARMERS BRANCH,TX,75234,UNITED STATES
            1234575,9100055619353,CC00000010,Null,CARD,765220380223,4805550065,KRISTIE,COOPER,3924 ELMWOOD AVENUE,PHOENIX,AZ,85003,UNITED STATES
            1234576,5179080561458,CC00000011,Null,WEATH MANAGEMENT,7510525336786,8085558139,SOO,CHRISTENSON,4996 ARRON SMITH DRIVE,HONOLULU,HI,96814,UNITED STATES
            1234577,5189410433475,CC00000012,1910,CARD,9314259531231,Null,APRIL,SANCHEZ,10623 N OSCEOLA DR,WESTMINSTER,CO,80031,UNITED STATES
            1234578,51790005549197,CC00000013,166,MORGAGE,34560000053,Null,SEE,SAMM,3323 BALCONES DR,IRVING,TX,75063,UNITED STATES
            1234579,9100057952612,CC00000014,Null,CORPORATE,3685820105756,2485556850,VANESSA,WILLIAMSON,2967 CORPENING DRIVE,PONTIAC,MI,48342,UNITED STATES


            4. **Output Requirements**:
            - Generate a minimum of 10 rows.
            - always look at the reference dataset and generate data within records value.
            - when you generate records, you must maintain the same value for others fileds except some filds data which is given in user prompt
            - learn the pattern and generate data as instructed.But maintain records data for CCID
            - keep acct_num,govt_issued_id , zip4_code lenght and format same as exmple input
            - Ensure that the test data adheres strictly to the condition provided.        
            - Return the data only in CSV format.
            - maintain the order of csv header as given example
            - index is a unique fileds. never duplicate the value. keep maintain the exaple format data
            

            5. **Example Output**:
            index,acct_num,ccid,zip4_code,lobs,govt_issued_id,phone_number,first_name,last_name,address_line_one,city_name,state_code,postal_code,country
            1234566,9100055032276,CC00000001,Null,RETAIL,3340783723534,2175557570,LOIS,WILLIAMS,936 CARDINAL LANE,CHAMPAIGN,IL,61820,UNITED STATES
            1234567,9100050039685,CC00000002,56799,CARD,4413416413453,9185552699,JULIE,BELL,3714 HENRY FORD AVENUE,AFTON,OK,74331,UNITED STATES
            1234568,5189413110708,CC00000003,Null,WEATH MANAGEMENT,9093134663453,5595550732,JESUS,MARESCOT,178 CARLING CT,SAN JOSE,CA,95111,UNITED STATES
            
            Now, generate realistic test data that strictly adheres to the given conditions. Return only CSV data that can be directly loaded into pandas. No  backticks, no explanations, and no extra text like ```sql, python, code, etc.```.
            """
            # Call LLM to generate test data
            
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            # Get the response text
            generated_text = response.content
            
            # Convert the CSV response into a pandas DataFrame
            test_data = pd.read_csv(io.StringIO(generated_text))
        else:
            # Construct the LLM prompt for pandas DataFrame
            system_prompt =f"""You are an advanced data generator. Based on the given conditions and structure, you need to create realistic test data in a tabular format. Follow these instructions strictly:"""
            user_prompt = f"""
            
                
                Follow these instructions strictly:
                1. **Conditions**: Use the following test condition provided by the user:
            {test_condition}
            
                2. Generate a minimum of 10 rows.
                - Generate a minimum of 10 rows.
                - always look at the reference dataset and generate data within records value.
                - when you generate records, you must maintain the same value for others fileds except some filds data which is given in user prompt
                - learn the pattern and generate data as instructed.But maintain records data for CCID
                - keep acct_num,govt_issued_id , zip4_code lenght and format same as exmple input
                - Ensure that the test data adheres strictly to the condition provided.        
                - Return the data only in CSV format.
                - maintain the order of csv header as given example
                - index is a unique fileds. never duplicate the value. keep maintain the exaple format data
            
                Reference dataset:
                index,acct_num,ccid,zip4_code,lobs,govt_issued_id,phone_number,first_name,last_name,address_line_one,city_name,state_code,postal_code,country
                1234566,9100055032276,CC00000001,Null,RETAIL,3340783723534,2175557570,LOIS,WILLIAMS,936 CARDINAL LANE,CHAMPAIGN,IL,61820,UNITED STATES
                1234567,9100050039685,CC00000002,56799,CARD,4413416413453,9185552699,JULIE,BELL,3714 HENRY FORD AVENUE,AFTON,OK,74331,UNITED STATES
                1234568,5189413110708,CC00000003,Null,WEATH MANAGEMENT,9093134663453,5595550732,JESUS,MARESCOT,178 CARLING CT,SAN JOSE,CA,95111,UNITED STATES
               
                5. **Example Output**:
                    index,acct_num,ccid,zip4_code,lobs,govt_issued_id,phone_number,first_name,last_name,address_line_one,city_name,state_code,postal_code,country
                    1234566,9100055032276,CC00000001,Null,RETAIL,3340783723534,2175557570,LOIS,WILLIAMS,936 CARDINAL LANE,CHAMPAIGN,IL,61820,UNITED STATES
                    1234567,9100050039685,CC00000002,56799,CARD,4413416413453,9185552699,JULIE,BELL,3714 HENRY FORD AVENUE,AFTON,OK,74331,UNITED STATES
                    1234568,5189413110708,CC00000003,Null,WEATH MANAGEMENT,9093134663453,5595550732,JESUS,MARESCOT,178 CARLING CT,SAN JOSE,CA,95111,UNITED STATES
                
                Now, generate realistic test data that strictly adheres to the given conditions. Return only CSV data that can be directly loaded into pandas. No  backticks, no explanations, and no extra text like ```sql, python, code, etc.```.
            """
            
            
            
            
            # print(user_prompt)
            # Call Cohere to generate test data
            ### replace with azure chatgpt 4.0 code
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])

            # # Get the response text
            generated_text = response.content
            # response = co.chat(
            #     model='command-r-plus-08-2024',
            #     messages = [{"role":"system","content":system_prompt},
            #                 {"role":"user","content":user_prompt}],
            
            # )

            # Get the response text
            # generated_text = response.message.content[0].text
            print(generated_text)
            # Convert the CSV response into a pandas DataFrame
        
            test_data = pd.read_csv(io.StringIO(generated_text))

        return test_data

    except Exception as e:
        st.error(f"Error generating test data: {e}")
        return None

def generate_data_structure(table_name, db_name="customer.db"):
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_name)
        query = f"PRAGMA table_info({table_name})"
        column_info = pd.read_sql(query, conn)
        conn.close()

        # Build the data structure
        data_structure = []
        for _, row in column_info.iterrows():
            column_name = row["name"]
            data_type = "String" if row["type"] in ["TEXT", "VARCHAR"] else \
                        "Integer" if row["type"] in ["INTEGER", "INT"] else \
                        "Boolean" if row["type"] in ["BOOLEAN"] else \
                        "Float" if row["type"] in ["REAL", "FLOAT"] else "String"
            example = "Example: " + ("Alice" if data_type == "String" else "25" if data_type == "Integer" else "True" if data_type == "Boolean" else "10.5")
            data_structure.append(f"- {column_name} ({data_type}, {example})")
        return data_structure
    except Exception as e:
        print(f"Error generating data structure: {e}")
        return None

# Function to create SDV metadata
def create_sdv_metadata(column_types):
    metadata = {
        "columns": {}
    }
    for column, column_type in column_types.items():
        metadata["columns"][column] = {
            "sdtype": column_type
        }
    return metadata

# Function to execute SQL on the database
def execute_sql_on_db(sql_query, db_name="customer_testbed.db"):
    try:
        conn = sqlite3.connect(db_name)
        results = pd.read_sql(sql_query, conn)
        conn.close()
        
        print(results)
        return results
    except Exception as e:
        st.error(f"Error executing SQL: {e}")
        return None

def execute_pandas_query(query_text, df=None):
    """
    Execute the pandas query on the replica pandas dataframe and return the results.
    """
    try:
        # Execute the Pandas query
        df = load_from_pickle() if df is None else df
        
        if df is None:
            st.error("No replica DataFrame found to execute the query.")
            return None
        
        local_namespace = {"df": df}    
        
        results = eval(query_text, {}, local_namespace)
        if isinstance(results, pd.DataFrame):
            return results
        else:
            st.error("The query did not return a Pandas DataFrame.")
            return None
        return results
    except Exception as e:
        st.error(f"Error executing Pandas query: {e}")
        return None


def insert_test_data_incrementally(test_data_df, replica_db="customer_testbed.db", table_name="customer"):
    try:
        if data_storage_option == "SQLite":
            # Connect to the replica database
            conn_replica = sqlite3.connect(replica_db)

            cursor = conn_replica.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in cursor.fetchall()]
            
            test_data_df = test_data_df[columns]  # Ensure the columns match the table schema
            
            # Fetch existing data from the replica table
            query = f"SELECT * FROM {table_name}"
            existing_data_df = pd.read_sql(query, conn_replica)

            # Find new rows to insert (rows not already in the replica table)
            new_data_df = pd.concat([test_data_df, existing_data_df]).drop_duplicates(keep=False)

            if new_data_df.empty:
                print("No new rows to insert.")
            else:
                # Insert the new rows into the replica table
                new_data_df.to_sql(table_name, conn_replica, if_exists="append", index=False)
                print(f"Inserted {len(new_data_df)} new rows into '{table_name}'.")

            # Close the connection
            conn_replica.close()
            return True
        else:
            existing_data_df = load_from_pickle()
            if existing_data_df is None:
                st.error("No existing data found in the replica DataFrame.")
                return False
            
            # Ensure the columns match the DataFrame schema
            test_data_df = test_data_df[existing_data_df.columns]
            
            new_data_df = pd.concat([test_data_df, existing_data_df]).drop_duplicates(keep=False)

            if new_data_df.empty:
                print("No new rows to insert.") 
            else:
                # Combine and save back to the pickle file
                updated_data_df = pd.concat([existing_data_df, new_data_df]).drop_duplicates()
                if save_to_pickle(updated_data_df):
                    print(f"Inserted {len(new_data_df)} new rows into the replica DataFrame.")
                    return True
        return False

    except Exception as e:
        print(f"Error inserting test data incrementally: {e}")
        return False

def train_ctgan_model(data: pd.DataFrame,metadata):
    # Initialize CTGAN model
    #model = CTGAN()
    #model = GaussianCopula()   
    # Train the model
    # metadata= metadata.load_from_json('metadata\my_metadata_v1_1737839165.json')
    synthesizer = CTGANSynthesizer(metadata)
    # synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    return synthesizer

def auto_detect_meta_data(data: pd.DataFrame):
    """Detect metadata from a single DataFrame"""
    metadata = Metadata.detect_from_dataframe(
        data=data,
        table_name='data'  # Provide a table name for the metadata
    )
    metadata.save_to_json('data_meta1.json')
    return metadata

def create_meta_file(data: pd.DataFrame):
    meta_data = {
        "columns": [
            {"name": col, "type": "categorical" if data[col].dtype.name == 'category' else "numerical"}
            for col in data.columns
        ],
        "num_rows": len(data)
    }
    with open('data_meta.json', 'w') as meta_file:
        json.dump(meta_data, meta_file, indent=4)
    return meta_data   
        
        
def insert_synthetic_data_into_db(data: pd.DataFrame):
    try:
        # Insert the synthetic data into the replica database
        success = insert_test_data_incrementally(data)  # You can reuse your earlier insert function
        if success:
            st.success("Synthetic data inserted into replica database successfully!")
        else:
            st.error("Failed to insert synthetic data into the replica database.")
    except Exception as e:
        st.error(f"Error inserting synthetic data: {e}")
        
def generate_synthetic_data(ctgan_model, num_rows) -> pd.DataFrame:
    synthetic_data = ctgan_model.sample(num_rows)
    return synthetic_data      
def clean_sql(sql_text):
    """
    Removes Markdown-style formatting from an SQL query.
    Handles cases where the text starts with ```sql or just ```.
    """
    # Remove ```sql or ``` from the beginning
    if sql_text.startswith("```sql"):
        sql_text = sql_text[len("```sql"):].strip()
    elif sql_text.startswith("```"):
        sql_text = sql_text[len("```"):].strip()
    
    # Remove trailing ``` if present
    if sql_text.endswith("```"):
        sql_text = sql_text[:-len("```")].strip()
    
    return sql_text              

def clean_pandas_query(query_text):
    """
    Removes Markdown-style formatting from a Pandas query.
    Handles cases where the text starts with ```python or just ```.

    Args:
        query_text (str): The Pandas query text.

    Returns:
        str: The cleaned Pandas query.
    """
    # Remove ```python or ``` from the beginning
    if query_text.startswith("```python"):
        query_text = query_text[len("```python"):].strip()
    elif query_text.startswith("```"):
        query_text = query_text[len("```"):].strip()
    
    # Remove trailing ``` if present
    if query_text.endswith("```"):
        query_text = query_text[:-len("```")].strip()
    
    return query_text

import re

def extract_sql_query(response_text):
    # Define a regex pattern to match SQL queries
    pattern = re.compile(r"(SELECT\s.*?;)", re.IGNORECASE | re.DOTALL)
    match = pattern.search(response_text)
    if match:
        return match.group(1).strip()
    else:
        return None

# Set page configuration for better layout
st.set_page_config(page_title="CSV to DB Manager", layout="wide")

# Initialize Cohere API
# co = cohere.ClientV2('xz6VhvXnfldENCPxSIWa19qEumptDOjH2tPoXA1F')  # Replace with your actual API key

# Page Title
st.title("Intelligent Test Data Generator")

# Sidebar for actions
with st.sidebar:
    st.header("Workflow steps")
    
    st.markdown("1. Load the data into database")
    st.markdown("2. Generate SQL from user test condition – using LLM")
    st.markdown("3. Fetching data from TestBed db using generated SQL")
    st.markdown("4. If records found, fetch data from test-bed database.")
    st.markdown("5. If no records found, generate test data using LLM.")
    st.markdown("6. Incrementally Insert Test data into Test-Bed database.")
    st.markdown("**For high volume synthetic data**")
    st.markdown("7. Column Mapping – Mark the categorical and numerical columns.")
    st.markdown("8, Train the statiscal model using LLM generated test data")
    st.markdown("9. Generate Synthetic Data – based on the trained model.")
    st.markdown("10, Incrementally Insert Test data into Test-Bed database.")

# Add a toggle button to select between SQLite and Pandas DataFrame
st.sidebar.subheader("Select Data Storage Option")
data_storage_option = st.sidebar.radio(
    "Choose the data storage option:",
    ("SQLite", "Pandas DataFrame")
)

# Step 1: File Upload Section
st.subheader("Load data File")
uploaded_file = st.file_uploader("📤 Upload your data file here:", type=["csv"], help="Upload a data file to populate the database.")

if uploaded_file:
    # Upload and process the file
    st.write("**File Uploaded Successfully!**")
    table_name = "customer"
    table_name, column_info, data = upload_to_db(uploaded_file)
    st.session_state.column_info = column_info

    if table_name:
        st.success(f"Data successfully inserted into table: `{table_name}`.")
        st.markdown("### Table Preview")
        st.dataframe(
            data,
            column_config={
                "acct_num": st.column_config.NumberColumn(format="%d"),
                "zip4_code": st.column_config.NumberColumn(format="%d"),
                "govt_issued_id": st.column_config.NumberColumn(format="%d"),
                "phone_number": st.column_config.NumberColumn(format="%d"),
                "postal_code": st.column_config.NumberColumn(format="%d"),
                "index": st.column_config.NumberColumn(format="%d"),
            },
            hide_index=True,
        )  # Show a preview of the table
        if data_storage_option == "Pandas DataFrame":
            if save_to_pickle(data):
                st.success("Data saved to pickle file successfully!")
            else:
                st.error("Failed to save data to pickle file.")
    else:
        st.error("Failed to upload and process the file.")

# Step 2: Create Replica DB
if data_storage_option == "SQLite":
    st.subheader("Create Test Bed Database")
    if st.button("Create Test Bed Database"):
        success = create_replica_db()
        if success:
            st.success("TestBed database `customer_testbed.db` created successfully!")
        else:
            st.error("Failed to create Test Bed database.")

# Step 3: Column Mapping and Metadata
# if uploaded_file:
#     st.subheader("Step 3: Define Column Types")
#     if not column_info.empty:
#         column_types = {}
#         cols = st.columns(4)  # Display 4 columns per row
#         for index, column_name in enumerate(column_info["name"]):
#             col = cols[index % 4]  # Assign to one of the 4 columns
#             with col:
#                 column_type = st.selectbox(
#                     f"Type for `{column_name}`",
#                     ["categorical", "numerical", "boolean", "ID"],
#                     key=column_name
#                 )
#                 column_types[column_name] = column_type

#         # Save metadata
#         if st.button("Save Column Metadata"):
#             metadata = create_sdv_metadata(column_types)
#             st.json(metadata)
#             st.success("Column metadata saved successfully!")

# Step 4: SQL Generation
st.subheader("Describe your test case scenario")

user_prompt = st.text_area(
    "🔍 Describe your test condition (e.g., 'find all records where ccid is same for different LOBs')",
    help="write in english natural language to generate data ."
)

if st.button("Generate test data"):
    if user_prompt:
        st.info("Generating Query, please wait...")
        table_name ="customer"
        if data_storage_option == "SQLite":
            query_text = call_llm_for_sql(user_prompt, table_name)
            query = clean_sql(query_text)
            print(query)
            st.toast("Generated SQL query: " + query)
        else:
            query_text = call_llm_for_pandas_query(user_prompt, table_name)
            st.info(f"Generated Pandas query: {query_text} ")
            if query_text:
                query = clean_pandas_query(query_text)
                st.info("Generated Pandas query:" + query)
            else:
                query = None # or some default value
                st.error("Failed to generate Pandas query. Please try again.")
        
        if query:
            st.subheader("Generated Query")
            st.code(query, language="sql" if data_storage_option == "SQLite" else "python")
            
            # Execute the Query
            st.write("**Executing Query in test-bed DB...**")            
            if data_storage_option == "SQLite":
                results = execute_sql_on_db(query)
            else:
                results = execute_pandas_query(query)
                
            print("testing......")
            print(results)
            if results is not None and not results.empty:
                st.success("Records found!")
                st.subheader("Search Results")
                st.dataframe(
                    results,
                    column_config={
                        "acct_num": st.column_config.NumberColumn(format="%d"),
                        "zip4_code": st.column_config.NumberColumn(format="%d"),
                        "govt_issued_id": st.column_config.NumberColumn(format="%d"),
                        "phone_number": st.column_config.NumberColumn(format="%d"),
                        "index": st.column_config.NumberColumn(format="%d"),
                    },
                    hide_index=True,
                )  # Show a preview 
                st.session_state.test_data_from_llm = results
            else:
                st.warning("No test data found! ")
                with st.spinner("Generating test data using AI... This might take a moment."):
                    if "test_data_generated" not in st.session_state:
                        st.session_state.test_data_generated = False

                    test_data = call_llm_to_generate_test_data(user_prompt, table_name)

                if test_data is not None:
                    st.session_state.test_data_generated = True
                    st.session_state.test_data_from_llm= test_data
                    st.subheader("Generated Test Data")
                        
                    st.dataframe(
                        test_data,
                        column_config={
                            "acct_num": st.column_config.NumberColumn(format="%d"),
                            "zip4_code": st.column_config.NumberColumn(format="%d"),
                            "govt_issued_id": st.column_config.NumberColumn(format="%d"),
                            "phone_number": st.column_config.NumberColumn(format="%d"),
                            "index": st.column_config.NumberColumn(format="%d"),
                        },
                        hide_index=True,
                    )  # Show a preview 
                    def convert_df_to_csv(df):
                        return df.to_csv(index=False).encode('utf-8')

                    csv_data = convert_df_to_csv(test_data)
                    # print(csv_data)
                    #save csv_data temporarily
                    test_data.to_csv("temp_data.csv",index=False)
                    try:
                          
                        success = insert_test_data_incrementally(test_data)

                        if success:
                            st.success("Test data inserted into Test Bed database successfully!")

                            # Fetch and display the incremental data
                            st.write("**Displaying newly inserted incremental test data...**")
                            if data_storage_option == "SQLite":
                                with st.spinner("Fetching newly inserted records..."):
                                    conn_replica = sqlite3.connect("customer_testbed.db")
                                    # Fetch the latest records added to the table
                                    query = f"SELECT * FROM customer ORDER BY ROWID DESC"
                                    incremental_data = pd.read_sql(query, conn_replica)
                                    conn_replica.close()

                                if not incremental_data.empty:
                                    st.subheader("Newly Inserted Incremental Test Data")
                                    st.dataframe(
                                        incremental_data.head(500),
                                        column_config={
                                            "acct_num": st.column_config.NumberColumn(format="%d"),
                                            "zip4_code": st.column_config.NumberColumn(format="%d"),
                                            "govt_issued_id": st.column_config.NumberColumn(format="%d"),
                                            "phone_number": st.column_config.NumberColumn(format="%d"),
                                            "postal_code": st.column_config.NumberColumn(format="%d"),
                                            "index": st.column_config.NumberColumn(format="%d"),                                                                       },
                                        hide_index=True,
                                    )  # Show a preview
                                else:
                                    st.warning("No new data found!")
                            else:
                                #fetch and display the replica data using pandas
                                with st.spinner("Fetching newly inserted records..."):
                                    replica_df = load_from_pickle()
                                    
                                    if replica_df is not None and not replica_df.empty:
                                        st.subheader("The TestBed Data with newly inserted records")
                                        st.dataframe(
                                            replica_df.head(500),
                                            column_config={
                                                "acct_num": st.column_config.NumberColumn(format="%d"),
                                                "zip4_code": st.column_config.NumberColumn(format="%d"),
                                                "govt_issued_id": st.column_config.NumberColumn(format="%d"),
                                                "phone_number": st.column_config.NumberColumn(format="%d"),
                                                "postal_code": st.column_config.NumberColumn(format="%d"),
                                                "index": st.column_config.NumberColumn(format="%d"),                                                                       },
                                            hide_index=True,
                                        )  # Show a preview
                                    else:
                                        st.warning("No new data found!")
                        else:
                            st.error("Failed to  insert test data into replica database.")
                    except Exception as e:
                        st.error(f"Error processing or inrting test data: {e}")   
                    # Generate more than using the LLM generated  test data by calling the statistical model method
                    
        else:
            st.error("Failed to generate test data. Please try again.")
    else:
        st.warning("Please provide a condition to generate data.")
if st.button("Generate high volume Synthetic data using statistical model", key="more_data") :
    try:
#st.dataframe(synthetic_data)
        print("\n----generating more data----\n")
        test_data = st.session_state.test_data_from_llm
        metafile = Metadata.load_from_json(filepath="data_meta.json")
        
        # metafile.save_to_json("metadata1.json")
        try:
            with st.spinner("Training statistical model with AI generated test data."):
                ctgan_model = train_ctgan_model(test_data,metafile)
                with st.spinner("Generating synthetic data..."):
                    synthetic_data = generate_synthetic_data(ctgan_model, num_rows=100)
                    st.session_state.synthetic_data = synthetic_data
            st.success("Synthetic data generated successfully!")
            #st.dataframe(synthetic_data)
            st.dataframe(
                synthetic_data.head(459),
                column_config={
                    "acct_num": st.column_config.NumberColumn(format="%d"),
                    "zip4_code": st.column_config.NumberColumn(format="%d"),
                    "govt_issued_id": st.column_config.NumberColumn(format="%d"),
                    "phone_number": st.column_config.NumberColumn(format="%d"),
                    "postal_code": st.column_config.NumberColumn(format="%d"),
                    "index": st.column_config.NumberColumn(format="%d"),
                },
                hide_index=True,
            )  # Show a preview 
        except Exception as e:
            st.error(f"Error training statistical model: {e}")
    except Exception as e:
        st.error(f"Error generating synthetic data: {e}")

# Display generated synthetic data
if "synthetic_data" in st.session_state:
    st.subheader("New Synthetic Data:")
    st.dataframe(st.session_state.synthetic_data)
    st.download_button(
        label="Download Generated Data",
        data=st.session_state.synthetic_data.to_csv(index=False),
        file_name=f"synthetic_data_{int(time.time())}.csv",
        mime="text/csv",
        key="download_more_data"
    )

# Footer
st.markdown("---")
st.caption("@Copyright - Tata Consultqancy Services | 2025")

