import yaml
import logging
from pathlib import Path
from langchain.tools import BaseTool
from typing import Type, Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
import psycopg2

from core_ai_mvp.llm.text_to_sql_llm_mvp import get_text_to_sql_llm_mvp, DEFAULT_DB_SCHEMA
from data_access_mvp.postgres_mvp import execute_validated_sql_query, SQLValidationError

# --- Logger Setup ---
logger = logging.getLogger(__name__)

# --- Helper Function to Load Intent Map ---
def load_intent_map() -> Dict[str, Any]:
    """Loads the intent map from the YAML file."""
    # Construct a path to the file relative to this script's location
    # This makes it robust to where the script is run from.
    yaml_path = Path(__file__).parent.parent.parent.parent / "metadata" / "intent_map.yml"
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Intent map file not found at: {yaml_path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        return {}

def build_dynamic_schema(intents: List[str], intent_map: Dict[str, Any]) -> str:
    """Builds a dynamic schema string based on the detected intents."""
    if not intents or not intent_map:
        return DEFAULT_DB_SCHEMA

    # Use sets to avoid duplicates
    tables_in_scope = set()
    table_details = {} # Store columns per table
    joins_in_scope = set()

    for intent in intents:
        intent_data = intent_map.get(intent)
        if not intent_data:
            continue

        # Collect tables and columns
        for table_info in intent_data.get('tables', []):
            table_name = table_info['table']
            tables_in_scope.add(table_name)
            if table_name not in table_details:
                table_details[table_name] = set()
            table_details[table_name].update(table_info['columns'])

        # Collect join information
        for join_info in intent_data.get('joins', []):
            joins_in_scope.add(f"{join_info['table_1']} can be joined with {join_info['table_2']} on {join_info['join_key']}")

    if not tables_in_scope:
        return DEFAULT_DB_SCHEMA

    # Construct the dynamic schema string for the LLM
    schema_parts = ["This query should be answerable using the following tables:"]
    for table_name in sorted(list(tables_in_scope)):
        columns_str = ", ".join(sorted(list(table_details[table_name])))
        schema_parts.append(f"{table_name}({columns_str})")

    if joins_in_scope:
        schema_parts.append("\nThese tables can be joined using:")
        schema_parts.extend(sorted(list(joins_in_scope)))

    dynamic_schema = "\n".join(schema_parts)
    logger.info(f"Built dynamic schema for intents {intents}:\n{dynamic_schema}")
    return dynamic_schema


class T2SqlToolInput(BaseModel):
    sql_query_prompt: str = Field(description="The natural language question to be converted into a SQL query.")
    intents: Optional[List[str]] = Field(None, description="A list of intents detected from the user query to guide schema selection.")


class T2SqlToolMvp(BaseTool):
    name: str = "database_query_executor"
    description: str = ("Useful for answering questions about user-specific data by generating and executing a SQL query. "
                        "Input should be a natural language question and can include a list of intents.")
    args_schema: Type[BaseModel] = T2SqlToolInput
    
    # Load the intent map once when the tool is initialized for efficiency
    intent_map: Dict[str, Any] = Field(default_factory=load_intent_map)


    def _run(self, sql_query_prompt: str, intents: Optional[List[str]] = None, **kwargs: Any) -> Union[List[Dict[str, Any]], Dict[str, str]]:
        """Executes the Text-to-SQL tool using a dynamic schema based on intents."""
        if not sql_query_prompt:
            return {"error": "No SQL query prompt provided."}

        logger.info(f"Received natural language query: '{sql_query_prompt}' with intents: {intents}")

        try:
            # 1. Determine which schema to use
            if intents and self.intent_map:
                schema_for_llm = build_dynamic_schema(intents, self.intent_map)
            else:
                logger.warning("No intents provided or intent map not loaded. Falling back to default schema.")
                schema_for_llm = DEFAULT_DB_SCHEMA
            
            # 2. Get the Text-to-SQL LLM chain
            t2sql_chain = get_text_to_sql_llm_mvp()
            logger.info(f"Generating SQL query for: '{sql_query_prompt}'")
            
            # 3. Generate SQL query using the determined schema
            generated_sql = t2sql_chain.invoke({
                "question": sql_query_prompt,
                "schema": schema_for_llm
            })

            logger.info(f"Generated SQL: '{generated_sql}'")

            if not generated_sql or "I don't know" in generated_sql or generated_sql.strip().upper() == "INVALID QUESTION":
                return {"info": "Could not generate a valid SQL query for the question.", "original_prompt": sql_query_prompt}

            # 4. Clean the generated SQL
            cleaned_sql = generated_sql.strip()
            if cleaned_sql.startswith("```sql"):
                cleaned_sql = cleaned_sql[len("```sql"):]
            if cleaned_sql.startswith("```"):
                cleaned_sql = cleaned_sql[len("```"):]
            if cleaned_sql.endswith("```"):
                cleaned_sql = cleaned_sql[:-len("```")]
            cleaned_sql = cleaned_sql.strip()

            # 5. Execute SQL Query
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
            logger.error(f"Error during Text-to-SQL processing: {e}", exc_info=True)
            error_sql_context = cleaned_sql if 'cleaned_sql' in locals() else sql_query_prompt
            return {"error": f"Failed to process Text-to-SQL request due to: {str(e)}", "context": error_sql_context}

    async def _arun(self, **kwargs: Any) -> Union[List[Dict[str, Any]], Dict[str, str]]:
        raise NotImplementedError("T2SqlToolMvp does not support async execution yet.")
