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
    # Basic Validation
    normalized_query = query.strip().upper()
    
    # Ensure the query starts with a valid operation (e.g., "SELECT")
    # The previous check `startswith(tuple(op + " " for op in ALLOWED_SQL_OPERATIONS))` was too strict 
    # as it required a space after SELECT, failing for queries with newlines after SELECT.
    if not any(normalized_query.startswith(op) for op in ALLOWED_SQL_OPERATIONS):
        raise SQLValidationError(
            f"Invalid SQL operation. Only {', '.join(ALLOWED_SQL_OPERATIONS)} statements are allowed. "
            f"Query provided: {query[:100]}..."
        )

    # Check for forbidden keywords more carefully
    # This is a basic check; a real SQL parser would be more robust.
    # We split by common delimiters to avoid false positives on keywords within identifiers or strings.
    # For example, a column named 'delete_flag' should not trigger 'DELETE'.
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
        conn.commit() # Not strictly necessary for SELECT, but good practice if DDL/DML were allowed.
    except psycopg2.Error as e:
        if conn:
            conn.rollback() # Rollback in case of error
        print(f"PostgreSQL Error: {e}")
        raise  # Re-raise the exception after logging
    finally:
        if conn:
            conn.close()
    return results

if __name__ == '__main__':
    print(f"Attempting to connect to PostgreSQL using DSN: {POSTGRES_DSN}")
    print("Please ensure your PostgreSQL server is running and the DSN is correct.")
    print("For this example to run meaningful SELECT queries, you need tables like 'loans' and 'customers'.")

    # Example Test Queries
    # 1. Valid SELECT query (assuming 'loans' table exists)
    valid_query_loans = "SELECT loan_id, loan_amount, status FROM loans LIMIT 2;"
    # 2. Valid SELECT query with parameters (assuming 'customers' table exists)
    valid_query_customers = "SELECT customer_id, customer_name FROM customers WHERE customer_id = %s;"
    customer_id_param = ("CUST001",) # Example customer ID
    # 3. Invalid query - attempts UPDATE
    invalid_query_update = "UPDATE loans SET status = 'defaulted' WHERE loan_id = 'L001';"
    # 4. Invalid query - contains DROP
    invalid_query_drop = "SELECT * FROM users; DROP TABLE loans;"
    # 5. Invalid query - not starting with SELECT
    invalid_query_not_select = "SHOW TABLES;"

    test_queries = [
        {"name": "Valid Loans Query", "query": valid_query_loans, "params": None, "should_pass": True},
        {"name": "Valid Customers Query with Params", "query": valid_query_customers, "params": customer_id_param, "should_pass": True},
        {"name": "Invalid UPDATE Query", "query": invalid_query_update, "params": None, "should_pass": False},
        {"name": "Invalid DROP Query", "query": invalid_query_drop, "params": None, "should_pass": False},
        {"name": "Invalid Non-SELECT Query", "query": invalid_query_not_select, "params": None, "should_pass": False},
        {"name": "Valid query with keyword in name (should pass for MVP simple check)", "query": "SELECT column_named_delete FROM test_table;", "params": None, "should_pass": True},
        {"name": "Invalid query with space before keyword", "query": "SELECT * FROM test; DELETE FROM users;", "params": None, "should_pass": False},
    ]

    for test in test_queries:
        print(f"\n--- Testing: {test['name']} ---")
        print(f"Query: {test['query']}")
        try:
            results = execute_validated_sql_query(test["query"], test["params"])
            if test["should_pass"]:
                print("Validation PASSED (as expected)")
                print(f"Results: {results}")
            else:
                print("Validation PASSED (UNEXPECTEDLY) - Check validation logic!")
        except SQLValidationError as e:
            if not test["should_pass"]:
                print(f"Validation FAILED (as expected): {e}")
            else:
                print(f"Validation FAILED (UNEXPECTEDLY): {e}")
        except psycopg2.Error as e:
            print(f"Database Execution Error: {e}")
            print("This might be due to the table/columns not existing, or connection issues.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    # Note: The example with "column_named_delete" might pass with very simple keyword checking
    # but would ideally be caught by a more robust SQL parser. The current validation
    # is a basic MVP safeguard.
