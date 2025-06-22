from langchain.tools import BaseTool
from typing import Type, Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from core_ai_mvp.llm.text_to_sql_llm_mvp import get_text_to_sql_llm_mvp, DEFAULT_DB_SCHEMA
from data_access_mvp.postgres_mvp import execute_validated_sql_query, SQLValidationError
import psycopg2 # For catching specific db errors
import logging # Added logging

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# Basic config will be inherited if this module is imported by another that sets it up.
# For standalone script execution, basicConfig might be needed in if __name__ == '__main__'.

class T2SqlToolInput(BaseModel):
    sql_query_prompt: str = Field(description="The natural language question to be converted into a SQL query.")
    # db_schema is implicitly used by the text_to_sql_llm_mvp using DEFAULT_DB_SCHEMA

class T2SqlToolMvp(BaseTool):
    name: str = "database_query_executor"
    description: str = ("Useful for answering questions about user-specific data by generating and executing a SQL query. "
                        "Input should be a natural language question about data in the database.")
    args_schema: Type[BaseModel] = T2SqlToolInput

    def _run(self, sql_query_prompt: str, **kwargs: Any) -> Union[List[Dict[str, Any]], Dict[str, str]]:
        """Executes the Text-to-SQL tool."""
        if not sql_query_prompt:
            return {"error": "No SQL query prompt provided."}

        logger.info(f"Received natural language query: '{sql_query_prompt}'")

        try:
            # 1. Get the Text-to-SQL LLM chain
            t2sql_chain = get_text_to_sql_llm_mvp()
            logger.info(f"Generating SQL query for: '{sql_query_prompt}' using schema:\n{DEFAULT_DB_SCHEMA}")
            
            # 2. Generate SQL query with specific guidance for active loans query
            if "active loans" in sql_query_prompt.lower():
                # Following intent_map.yml for query_list_active_loans
                sql_template = '''
                SELECT 
                    SO_HOP_DONG, 
                    DU_NO, 
                    DU_NO_LAI, 
                    MUC_DICH_VAY, 
                    STATUS
                FROM DEBT_CUSTOMER_LD_DETAIL
                WHERE CUSTOMER_ID = {}
                '''
                # Extract customer ID from the prompt
                import re
                customer_id = re.search(r'\b\d+\b', sql_query_prompt)
                if customer_id:
                    generated_sql = sql_template.format(customer_id.group())
                else:
                    return {"error": "Could not extract customer ID from the query"}
            else:
                # For other queries, use the LLM
                generated_sql = t2sql_chain.invoke({
                    "question": sql_query_prompt,
                    "schema": DEFAULT_DB_SCHEMA 
                })

            logger.info(f"Generated SQL: '{generated_sql}'")

            if not generated_sql or generated_sql.strip().upper() == "INVALID QUESTION":
                return {"info": "Could not generate a valid SQL query for the question.", "original_prompt": sql_query_prompt}

            # Clean the generated SQL
            cleaned_sql = generated_sql.strip()
            if cleaned_sql.startswith("```sql"):
                cleaned_sql = cleaned_sql[len("```sql"):]
            if cleaned_sql.startswith("```"):
                cleaned_sql = cleaned_sql[len("```"):]
            if cleaned_sql.endswith("```"):
                cleaned_sql = cleaned_sql[:-len("```")]
            cleaned_sql = cleaned_sql.strip()

            # 3. Execute SQL Query
            logger.info(f"Executing validated SQL query: '{cleaned_sql}'")
            sql_results = execute_validated_sql_query(cleaned_sql)

            if not sql_results:
                return {"info": "Query executed successfully, but returned no results.", "sql_query": cleaned_sql}
            
            logger.info(f"SQL execution returned {len(sql_results)} row(s).")
            return sql_results

        except SQLValidationError as ve:
            logger.error(f"SQL Validation Error: {ve} for query '{cleaned_sql if 'cleaned_sql' in locals() else 'N/A'}'")
            return {"error": f"Generated SQL query failed validation: {str(ve)}", "cleaned_sql": cleaned_sql if 'cleaned_sql' in locals() else 'N/A'}
        except psycopg2.Error as db_err:
            logger.error(f"Database execution error: {db_err} for query '{cleaned_sql if 'cleaned_sql' in locals() else 'N/A'}'")
            return {"error": f"Database error while executing SQL: {str(db_err)}", "cleaned_sql": cleaned_sql if 'cleaned_sql' in locals() else 'N/A'}
        except Exception as e:
            logger.error(f"Error during Text-to-SQL processing: {e}")
            error_sql_context = cleaned_sql if 'cleaned_sql' in locals() else sql_query_prompt
            return {"error": f"Failed to process Text-to-SQL request due to: {str(e)}", "context": error_sql_context}

    async def _arun(self, sql_query_prompt: str, **kwargs: Any) -> Union[List[Dict[str, Any]], Dict[str, str]]:
        raise NotImplementedError("T2SqlToolMvp does not support async execution yet.")

if __name__ == '__main__':
    # Setup basic logging for standalone script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Testing T2SqlToolMvp...")
    logger.info("Ensure PostgreSQL server is running and GOOGLE_API_KEY is set in the environment.")
    logger.info(f"Make sure your Gemini model for SQL generation is available.")
    logger.info(f"The tool will use the default schema:\n{DEFAULT_DB_SCHEMA}")
    # You will also need the 'loans' and 'customers' tables in your PG database for meaningful results.

    t2sql_tool = T2SqlToolMvp()

    # Test Case 1: Valid question that should generate a SELECT query
    # Assumes GOOGLE_API_KEY is set in environment
    query1 = "What are the names and email addresses of all customers?"
    logger.info(f"\n--- Test Case 1: Query = '{query1}' ---")
    try:
        results1 = t2sql_tool.run({"sql_query_prompt": query1})
        logger.info("Results 1:")
        if isinstance(results1, list):
            for row in results1[:3]: # Print first 3 results
                logger.info(f"  {row}")
            if len(results1) > 3:
                logger.info(f"  ... and {len(results1) - 3} more rows.")
        else:
            logger.info(f"  {results1}") # Error or info message
    except Exception as e:
        logger.error(f"Runtime error in Test Case 1: {e}")

    # Test Case 2: Question that might be hard for a simple T2SQL model or lead to no results
    query2 = "Find loans with an interest rate of exactly 3.14159 percent that started on a Tuesday."
    logger.info(f"\n--- Test Case 2: Query = '{query2}' ---")
    try:
        results2 = t2sql_tool.run(sql_query_prompt=query2)
        logger.info("Results 2:")
        logger.info(f"  {results2}")
    except Exception as e:
        logger.error(f"Runtime error in Test Case 2: {e}")

    # Test Case 3: Question that should result in "Invalid Question" or similar from the LLM
    query3 = "What is the meaning of life?"
    logger.info(f"\n--- Test Case 3: Query = '{query3}' ---")
    try:
        results3 = t2sql_tool.run(tool_input={"sql_query_prompt": query3})
        logger.info("Results 3:")
        logger.info(f"  {results3}")
    except Exception as e:
        logger.error(f"Runtime error in Test Case 3: {e}")

    # Test Case 4: An empty prompt
    query4 = ""
    logger.info(f"\n--- Test Case 4: Empty Query ---")
    try:
        results4 = t2sql_tool.run({"sql_query_prompt": query4})
        logger.info("Results 4:")
        logger.info(f"  {results4}")
    except Exception as e:
        logger.error(f"Runtime error in Test Case 4: {e}")

    logger.info("\nNote: For these tests to pass and return data, you need:")
    logger.info("1. A running PostgreSQL instance with the DSN configured in .env.")
    logger.info("2. Tables 'loans' and 'customers' with some data, matching DEFAULT_DB_SCHEMA.")
    logger.info("3. GOOGLE_API_KEY environment variable set and valid for Gemini API access.")
    logger.info("4. `text_to_sql_llm_mvp.py` and `postgres_mvp.py` to be functional.")
