import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from config_mvp.settings_mvp import POSTGRES_DSN
from typing import List, Dict, Any

# Basic validation: allow only SELECT and forbid destructive keywords
ALLOWED_SQL_OPERATIONS = ["SELECT"]
FORBIDDEN_SQL_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE", "GRANT", "REVOKE"]

class SQLValidationError(ValueError):
    """Custom exception for SQL validation errors."""
    pass

def execute_validated_sql_query(query: str, params: tuple = None) -> List[Dict[str, Any]]:
    """
    Connects to PostgreSQL, executes a validated SELECT SQL query, and returns the results.

    Args:
        query (str): The SQL query string to execute. Must be a SELECT query.
        params (tuple, optional): Parameters to pass to the SQL query for safe substitution. Defaults to None.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a row 
                               with column names as keys. Returns an empty list if no results.

    Raises:
        SQLValidationError: If the query is not a SELECT statement or contains forbidden keywords.
        psycopg2.Error: For database connection or query execution errors.
    """
    normalized_query = query.strip().upper()
    
 
    if not any(normalized_query.startswith(op) for op in ALLOWED_SQL_OPERATIONS):
        raise SQLValidationError(
            f"Invalid SQL operation. Only {', '.join(ALLOWED_SQL_OPERATIONS)} statements are allowed. "
            f"Query provided: {query[:100]}..."
        )

  
    query_parts = normalized_query.replace('(', ' ').replace(')', ' ').replace(';', ' ').split()

    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if keyword in query_parts:
            raise SQLValidationError(
                f"Forbidden SQL keyword '{keyword}' found in query. "
                f"Query provided: {query[:100]}..."
            )

    conn = None
    results = []
    try:
        print(f"[PostgresMvp] Attempting to connect with DSN: {POSTGRES_DSN}")
        conn = psycopg2.connect(POSTGRES_DSN)
        # Using RealDictCursor to get results as dictionaries
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if params:
                cur.execute(query, params)
            else:
                cur.execute(query)
            
            # Fetch results if it's a SELECT query (which it should be if validation passed)
            if cur.description: # Check if the cursor has a description (i.e., it returned data)
                results = cur.fetchall()
        conn.commit()
    except psycopg2.Error as e:
        if conn:
            conn.rollback() 
        print(f"PostgreSQL Error: {e}")
        raise 
    finally:
        if conn:
            conn.close()
    return results
