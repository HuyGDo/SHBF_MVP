import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Optional
import google.generativeai as genai

# Load GOOGLE_API_KEY from .env file
# Ensure your .env file has GOOGLE_API_KEY="your_api_key"
# from dotenv import load_dotenv
# load_dotenv()
# For MVP, we assume GOOGLE_API_KEY is set in the environment directly
# or will be handled by the Gradio app's environment setup.

# Simplified schema for MVP. In a real scenario, this would be more dynamic
# or fetched from a database metadata store.
DEFAULT_DB_SCHEMA = '''
Table: DEBT_CUSTOMER_LD_DETAIL
Columns: 
  - SO_HOP_DONG (VARCHAR2(15))
  - CUSTOMER_ID (NUMBER)
  - KY_HAN_VAY (NUMBER)
  - LAI_SUAT (NUMBER)
  - SO_TIEN_GIAI_NGAN (NUMBER)
  - DU_NO (NUMBER)
  - DU_NO_LAI (NUMBER)
  - NGAY_GIAI_NGAN (DATE)
  - NGAY_KY_HOP_DONG (DATE)
  - MUC_DICH_VAY (VARCHAR2(250))
  - NHOM_NO_THEO_HD_DPD (VARCHAR2(3))
  - STATUS (derived from internal DPD columns)

Table: DEBT_LD_REPAY_SCHEDULE
Columns:
  - SO_HOP_DONG (VARCHAR2(20))
  - KY_THANH_TOAN (NUMBER)
  - TU_NGAY (DATE)
  - DEN_NGAY (DATE)
  - GOC (NUMBER)
  - LAI (NUMBER)
  - EMI_AMOUNT (NUMBER)
  - DU_NO (NUMBER)

Table: DEBT_LD_PAID_HISTORY
Columns:
  - SO_HOP_DONG (VARCHAR2(20))
  - NGAY_THANH_TOAN (DATE)
  - KY_THANH_TOAN (NUMBER)
  - GOC_DA_THU (NUMBER)
  - LAI_DA_THU (NUMBER)
  - TONG_TIEN_THU (NUMBER)
  - DPD_HOP_DONG_SAU_TT (NUMBER)

Table: DEBT_CUSTOMERS
Columns:
  - CUSTOMER_ID (NUMBER)
  - HO_TEN_DAY_DU (VARCHAR2(250))
  - SO_CMND_THE_CAN_CUOC (VARCHAR2(50))
  - SO_DIEN_THOAI (VARCHAR2(50))
  - EMAIL (VARCHAR2(100))
  - DU_NO (NUMBER)
  - DU_NO_LAI (NUMBER)
  - BUCKET_CODE (VARCHAR2(20))

Relationships:
- DEBT_CUSTOMER_LD_DETAIL.SO_HOP_DONG can be joined with DEBT_LD_REPAY_SCHEDULE.SO_HOP_DONG
- DEBT_CUSTOMER_LD_DETAIL.SO_HOP_DONG can be joined with DEBT_LD_PAID_HISTORY.SO_HOP_DONG
- DEBT_CUSTOMER_LD_DETAIL.CUSTOMER_ID can be joined with DEBT_CUSTOMERS.CUSTOMER_ID
'''

def get_text_to_sql_llm_mvp(google_api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
    """
    Initializes and returns the Text-to-SQL LLM.

    Args:
        google_api_key (Optional[str]): Google API Key. If None, it's expected to be in the environment.
        model_name (str): The Gemini model name to use (e.g., "gemini-2.0-flash").

    Returns:
        A LangChain LCEL chain configured for SQL generation.
    """
    if google_api_key is None:
        google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    # Configure the Google Generative AI library
    genai.configure(api_key=google_api_key)

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_api_key,
            temperature=0,
            convert_system_message_to_human=True
        )
    except Exception as e:
        raise ValueError(f"Failed to initialize Gemini model: {str(e)}")

    prompt_template = """
    You are an expert SQL generation assistant.
    Given a user's question and the database schema, generate a syntactically correct SQL query to answer the question.
    Only output the SQL query. Do not include any other text or explanations.
    If you cannot generate a query for any reason, output "Invalid Question".

    Database Schema:
    {schema}

    User Question: {question}

    SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(template=prompt_template)

    parser = StrOutputParser()

    chain = prompt | llm | parser
    return chain

if __name__ == '__main__':
    # This is an example of how to use the get_text_to_sql_llm_mvp function.
    # Ensure GOOGLE_API_KEY is set in your environment variables.

    # from dotenv import load_dotenv
    # load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY not set. Please set it in your environment or a .env file for this example to run.")
    else:
        print("GOOGLE_API_KEY found. Initializing Text-to-SQL LLM...")
        t2sql_llm_chain = get_text_to_sql_llm_mvp()
        print("Text-to-SQL LLM Chain initialized.")

        sample_question_1 = "What are the loan types and their amounts for customer with id CUST001?"
        print(f"\nTesting with: '{sample_question_1}'")
        try:
            sql_query_1 = t2sql_llm_chain.invoke({"question": sample_question_1, "schema": DEFAULT_DB_SCHEMA})
            print(f"Generated SQL 1: {sql_query_1}")

            sample_question_2 = "Show all active loans with interest rate less than 5%."
            print(f"\nTesting with: '{sample_question_2}'")
            sql_query_2 = t2sql_llm_chain.invoke({"question": sample_question_2, "schema": DEFAULT_DB_SCHEMA})
            print(f"Generated SQL 2: {sql_query_2}")

            sample_question_3 = "Who is the Prime Minister of Canada?"
            print(f"\nTesting with (invalid): '{sample_question_3}'")
            sql_query_3 = t2sql_llm_chain.invoke({"question": sample_question_3, "schema": DEFAULT_DB_SCHEMA})
            print(f"Generated SQL 3: {sql_query_3}")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure your GOOGLE_API_KEY is valid and the Gemini API is enabled for your project.")
            print("You might also need to install necessary libraries: pip install langchain-google-genai python-dotenv langchain")
