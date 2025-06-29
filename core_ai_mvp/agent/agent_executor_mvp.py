import os
import json
from typing import Dict, Any, List, Optional
import google.generativeai as genai
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core_ai_mvp.llm.main_llm_mvp import get_main_llm_mvp, Plan
from core_ai_mvp.agent.tools.rag_tool_mvp import RagToolMvp
from core_ai_mvp.agent.tools.t2sql_tool_mvp import T2SqlToolMvp
# Assuming GOOGLE_API_KEY is in environment. Settings can be used for other configs.
# from config_mvp.settings_mvp import GOOGLE_API_KEY 

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# Enhanced log format for better readability
LOG_FORMAT = '%(asctime)s - %(levelname)-8s - %(name)s.%(funcName)s:%(lineno)d - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# Configuration for the LLMs used in this executor
GEMINI_MODEL_FOR_SYNTHESIS = "gemini-2.0-flash" # Or your preferred Gemini model for generation
GEMINI_MODEL_FOR_PLANNING = "gemini-2.0-flash" # Consistent with main_llm_mvp.py default

DEFAULT_LANGUAGE = "vi" # Default language if not specified in plan

def format_rag_results_for_llm(rag_output: List[Dict[str, Any]]) -> str:
    if not rag_output:
        return "No policy information was retrieved or applicable."
    
    if isinstance(rag_output, list) and rag_output and "error" in rag_output[0]:
        logger.error(f"RAG tool returned an error: {rag_output[0]['error']}")
        return f"Could not retrieve policy documents: {rag_output[0]['error']}"
    if isinstance(rag_output, list) and rag_output and "info" in rag_output[0]:
        logger.info(f"RAG tool returned info: {rag_output[0]['info']}")
        return f"Policy document information: {rag_output[0]['info']}"

    snippets = []
    for i, item in enumerate(rag_output):
        content = item.get('content', 'N/A')
        source = item.get('source', 'N/A')
        score = item.get('score', 'N/A')
        snippets.append(f"Snippet {i+1}:\n  Source: {source}\n  Score: {score:.4f}\n  Content: {content}")
    return "\n".join(snippets) if snippets else "No specific policy documents found matching your query."

def format_sql_results_for_llm(t2sql_output: Any) -> str:
    if not t2sql_output:
        # This case might be covered if t2sql_output is an empty list from successful query
        return "No database information was retrieved or applicable."

    if isinstance(t2sql_output, dict):
        if "error" in t2sql_output:
            err_msg = f"Error querying database: {t2sql_output['error']}"
            if "generated_sql" in t2sql_output:
                err_msg += f" (Attempted SQL: {t2sql_output['generated_sql']})"
            return err_msg
        elif "info" in t2sql_output:
            info_msg = f"Database query status: {t2sql_output['info']}"
            if "generated_sql" in t2sql_output:
                info_msg += f" (Attempted SQL: {t2sql_output['generated_sql']})"
            return info_msg
    
    if isinstance(t2sql_output, list):
        if not t2sql_output: # Empty list from a successful query
            return "Query executed successfully but found no matching records in the database."
        try:
            # Pretty print JSON for better LLM readability
            return json.dumps(t2sql_output, indent=2, ensure_ascii=False)
        except TypeError:
            return f"Database query returned data that could not be formatted as JSON: {str(t2sql_output)}"
    
    return f"Unexpected format or no information retrieved from the database: {str(t2sql_output)}"

def agent_orchestrator_mvp(user_query: str, history_string: str) -> str:
    """
    Orchestrates the AI's response generation process.
    1. Generates a plan using the Main LLM.
    2. Validates the plan; if not actionable, asks for clarification using an LLM.
    3. Executes tools (RAG, SQL) based on the plan.
    4. Synthesizes a final response using an LLM, based on tool outputs and history.
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        return "Error: GOOGLE_API_KEY not found. Please set it in your environment variables."

    # Configure the Google Generative AI library globally
    try:
        genai.configure(api_key=google_api_key)
    except Exception as e:
        logger.error(f"Error configuring Gemini API: {e}")
        return "I'm having trouble connecting to the AI service. Please check the API configuration."

    # Initialize planning LLM chain
    try:
        main_llm_planning_chain = get_main_llm_mvp(google_api_key=google_api_key, model_name=GEMINI_MODEL_FOR_PLANNING)
    except Exception as e:
        logger.error(f"Error initializing planning LLM: {e}")
        return "I'm having trouble starting up my planning module. Please check the API configuration and try again later."

    # Initialize tools
    rag_tool = RagToolMvp()
    t2sql_tool = T2SqlToolMvp()

    # Initialize base LLM for synthesis and clarification with better error handling
    try:
        base_llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_FOR_SYNTHESIS,
            google_api_key=google_api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
    except Exception as e:
        logger.error(f"Error initializing synthesis LLM: {e}")
        return "I'm having trouble initializing my response generation module. Please try again later."

    # Initialize clarification chain
    clarification_prompt = ChatPromptTemplate.from_template(
        """
        I need to ask for clarification about this query: {initial_query}
        
        IMPORTANT:
        1. Generate a polite response in Vietnamese ONLY
        2. Do not add any translations
        3. Do not add any explanations or meta-commentary
        4. Keep the response concise and direct
        5. Ask specifically what additional information you need from the user
        
        Your response in Vietnamese:
        """
    )
    clarification_chain = clarification_prompt | base_llm | StrOutputParser()

    # 1. Planning Phase
    try:
        # Clean the query text by removing extra whitespace and newlines
        cleaned_query = " ".join(user_query.split())
        logger.info(f"Generating plan for query: '{cleaned_query}' with history: '{history_string}'")
        
        # Pre-process query to help with routing
        has_customer_id = any(pattern in cleaned_query.lower() for pattern in ["mã id", "id của tôi", "id là"])
        has_loan_query = any(pattern in cleaned_query.lower() for pattern in ["khoản vay", "hiệu lực"])
        
        # Force SQL route for customer ID + loan queries before even calling the LLM
        if has_customer_id and has_loan_query:
            logger.info("Query contains customer ID and loan information - forcing SQL route")
            plan = Plan(
                route="SQL",
                intents=["get_active_loans", "view_loan_status"],
                entities={"customer_id": cleaned_query.split(":")[-1].strip() if ":" in cleaned_query else ""},
                sql_prompt=f"Find all active loans for customer with ID {cleaned_query.split(':')[-1].strip() if ':' in cleaned_query else ''}",
                policy_query=None,
                language="vi"
            )
            logger.info(f"Forced SQL plan: \n{json.dumps(plan.dict(), indent=4, ensure_ascii=False)}") # Pretty print JSON
            return handle_sql_query(plan, cleaned_query, history_string, base_llm, t2sql_tool)
            
        # Only proceed with LLM planning if not a clear SQL case
        plan_dict = main_llm_planning_chain.invoke({
            "query": cleaned_query,
            "history": history_string
        })
        logger.info(f"Raw plan dict: \n{json.dumps(plan_dict, indent=4, ensure_ascii=False)}") # Pretty print JSON
        
        # Convert the dictionary to a Plan object
        try:
            plan = Plan(**plan_dict)
            logger.info(f"Plan generated and validated: \n{json.dumps(plan.dict(), indent=4, ensure_ascii=False)}") # Pretty print JSON
        except Exception as e:
            logger.error(f"Error converting plan dict to Plan object: {e}")
            # Fallback for parsing errors. If customer_id was part of the original intent, try basic SQL.
            if has_customer_id: # 'has_customer_id' was defined earlier based on cleaned_query
                logger.info("Defaulting to SQL route due to plan parsing error and presence of customer ID indicators.")
                # Simplified plan for direct SQL attempt
                customer_id_entity = cleaned_query.split(":")[-1].strip() if ":" in cleaned_query else "unknown" # Basic extraction
                # Attempt to create a generic SQL prompt from the original query
                fallback_sql_prompt = f"Retrieve relevant information for customer based on query: '{cleaned_query}'"
                if customer_id_entity != "unknown":
                    fallback_sql_prompt = f"Retrieve relevant information for customer ID {customer_id_entity} based on query: '{cleaned_query}'"

                return handle_sql_query(
                    Plan(
                        route="SQL",
                        intents=["fallback_get_customer_data"], # Generic fallback intent
                        entities={"customer_id": customer_id_entity} if customer_id_entity != "unknown" else {},
                        sql_prompt=fallback_sql_prompt,
                        policy_query=None,
                        language=DEFAULT_LANGUAGE
                    ),
                    cleaned_query,
                    history_string,
                    base_llm, # 'base_llm' was initialized earlier
                    t2sql_tool # 't2sql_tool' was initialized earlier
                )
            # If no clear customer ID, or SQL fallback is not appropriate, then ask for clarification.
            return clarification_chain.invoke({"initial_query": cleaned_query}) # 'clarification_chain' was initialized earlier

        # 2. Execution Phase based on Plan
        final_response = ""
        if plan.route == "SQL":
            if not plan.sql_prompt:
                logger.warning("Plan specified SQL route but no sql_prompt was provided.")
                final_response = "Xin lỗi, tôi không thể xử lý yêu cầu SQL của bạn vì thiếu thông tin chi tiết."
            else:
                final_response = handle_sql_query(plan, cleaned_query, history_string, base_llm, t2sql_tool)
        
        elif plan.route == "RAG":
            if not plan.policy_query:
                logger.warning("Plan specified RAG route but no policy_query was provided.")
                final_response = "Xin lỗi, tôi không thể xử lý yêu cầu tra cứu chính sách của bạn vì thiếu thông tin chi tiết."
            else:
                final_response = handle_rag_query(plan, cleaned_query, history_string, base_llm, rag_tool)
        
        elif plan.route == "BOTH":
            if not plan.sql_prompt or not plan.policy_query:
                logger.warning("Plan specified BOTH route but sql_prompt or policy_query was missing.")
                final_response = "Xin lỗi, tôi không thể xử lý yêu cầu kết hợp của bạn vì thiếu thông tin chi tiết."
            else:
                final_response = handle_both_query(plan, cleaned_query, history_string, base_llm, t2sql_tool, rag_tool)
            
        else: # Includes CLARIFY or unknown routes or if plan.route is not one of SQL, RAG, BOTH
            logger.warning(f"Plan indicates clarification or unknown/unsupported route: {plan.route}")
            # If the plan itself suggests clarification or is unclear
            # We can use a more specific clarification prompt if plan has details, or generic
            # For MVP, if route is not SQL, RAG, or BOTH, we treat as needing clarification.
            final_response = clarification_chain.invoke({"initial_query": cleaned_query})

        return final_response

    except Exception as e:
        logger.error(f"Error during planning or execution: {e}")
        # General fallback clarification if anything else goes wrong in the main orchestration logic
        return clarification_chain.invoke({"initial_query": user_query}) # Use original user_query for clarification context

def handle_sql_query(plan: Plan, query: str, history: str, llm: Any, sql_tool: Any) -> str:
    """Helper function to handle SQL queries"""
    logger.info(f"Handling SQL query with plan: \n{json.dumps(plan.dict(), indent=4, ensure_ascii=False)}") # Pretty print JSON
    language = plan.language or DEFAULT_LANGUAGE # Added language parameter
    try:
        t2sql_output = sql_tool.run({"sql_query_prompt": plan.sql_prompt})
        # sql_results_str is already formatted by format_sql_results_for_llm, which includes JSON dump if it's a list
        # If t2sql_output (raw) needs to be pretty-printed:
        # logger.debug(f"Raw SQL output: \n{json.dumps(t2sql_output, indent=4, ensure_ascii=False)}")
        sql_results_str = format_sql_results_for_llm(t2sql_output)
        logger.info(f"SQL results: \n{sql_results_str}") # Log the formatted string
        
        synthesis_prompt = ChatPromptTemplate.from_template('''
        Based on the user's query about their loan information and the database results, provide a clear response.
        
        IMPORTANT:
        1. Respond ONLY in {language}.
        2. Be direct and concise.
        3. If no data is found, politely inform the user.
        4. Do not add translations or explanations.
        
        User Query: {query}
        Database Results: {results}
        
        Your response in {language}: 
        ''') # Added {language} to prompt
        synthesis_chain = synthesis_prompt | llm | StrOutputParser()

        return synthesis_chain.invoke({
            "query": query,
            "results": sql_results_str,
            "language": language # Added language to invocation
        })
    except Exception as e:
        logger.error(f"Error in SQL handling: {e}")
        # Fallback message in the target language
        if language == "vi":
            return "Xin lỗi, tôi không thể truy xuất thông tin khoản vay của bạn lúc này. Vui lòng thử lại sau."
        else:
            return "Sorry, I couldn't retrieve your loan information at this time. Please try again later."

def handle_rag_query(plan: Plan, query: str, history: str, llm: Any, rag_tool: Any) -> str:
    """Helper function to handle RAG queries"""
    logger.info(f"Handling RAG query with plan: \n{json.dumps(plan.dict(), indent=4, ensure_ascii=False)}") # Pretty print JSON
    language = plan.language or DEFAULT_LANGUAGE
    try:
        rag_output = rag_tool.run({"policy_query_prompt": plan.policy_query})
        # rag_results_str is already formatted by format_rag_results_for_llm, can be logged directly
        # If rag_output itself (the list of dicts) needs to be pretty-printed before formatting:
        # logger.debug(f"Raw RAG output: \n{json.dumps(rag_output, indent=4, ensure_ascii=False)}")
        rag_results_str = format_rag_results_for_llm(rag_output)
        logger.info(f"RAG results: \n{rag_results_str}") # Log the formatted string

        synthesis_prompt_template = """
        Based on the user's query about bank policies and the retrieved policy information, provide a clear response.
        
        IMPORTANT:
        1. Respond ONLY in {language}.
        2. Be direct and concise.
        3. If no relevant policy information is found, politely inform the user.
        4. Do not add translations or explanations beyond what's requested.
        
        User Query: {query}
        Policy Information: {results}
        
        Your response in {language}:
        """
        synthesis_prompt = ChatPromptTemplate.from_template(synthesis_prompt_template)
        synthesis_chain = synthesis_prompt | llm | StrOutputParser()

        return synthesis_chain.invoke({
            "query": query,
            "results": rag_results_str,
            "language": language
        })
    except Exception as e:
        logger.error(f"Error in RAG handling: {e}")
        # Fallback message in the target language
        if language == "vi":
            return "Xin lỗi, tôi không thể truy xuất thông tin chính sách lúc này. Vui lòng thử lại sau."
        else:
            return "Sorry, I couldn't retrieve policy information at this time. Please try again later."

def handle_both_query(plan: Plan, query: str, history: str, llm: Any, sql_tool: Any, rag_tool: Any) -> str:
    """Helper function to handle combined RAG and SQL queries"""
    logger.info(f"Handling BOTH query with plan: \n{json.dumps(plan.dict(), indent=4, ensure_ascii=False)}") # Pretty print JSON
    language = plan.language or DEFAULT_LANGUAGE
    try:
        # Execute RAG part
        rag_output = rag_tool.run({"policy_query_prompt": plan.policy_query})
        rag_results_str = format_rag_results_for_llm(rag_output)
        logger.info(f"RAG results (for BOTH): \n{rag_results_str}")

        # Execute SQL part
        t2sql_output = sql_tool.run({"sql_query_prompt": plan.sql_prompt})
        # sql_results_str is already formatted by format_sql_results_for_llm, which includes JSON dump if it's a list
        # If t2sql_output (raw) needs to be pretty-printed:
        # logger.debug(f"Raw SQL output: \n{json.dumps(t2sql_output, indent=4, ensure_ascii=False)}") 
        sql_results_str = format_sql_results_for_llm(t2sql_output)
        logger.info(f"SQL results (for BOTH): \n{sql_results_str}")

        synthesis_prompt_template = """
        Based on the user's query, the retrieved policy information, and the database results, provide a comprehensive and clear response.
        
        IMPORTANT:
        1. Respond ONLY in {language}.
        2. Synthesize information from BOTH policy and database results.
        3. Be direct and concise.
        4. If no data is found for either part, politely inform the user about what was found and what wasn't.
        5. Do not add translations or explanations beyond what's requested.
        
        User Query: {query}
        Policy Information: {rag_results}
        Database Results: {sql_results}
        
        Your response in {language}:
        """
        synthesis_prompt = ChatPromptTemplate.from_template(synthesis_prompt_template)
        synthesis_chain = synthesis_prompt | llm | StrOutputParser()

        return synthesis_chain.invoke({
            "query": query,
            "rag_results": rag_results_str,
            "sql_results": sql_results_str,
            "language": language
        })
    except Exception as e:
        logger.error(f"Error in BOTH handling: {e}")
        if language == "vi":
            return "Xin lỗi, tôi không thể xử lý yêu cầu kết hợp của bạn lúc này. Vui lòng thử lại sau."
        else:
            return "Sorry, I couldn't process your combined request at this time. Please try again later."

if __name__ == '__main__':
    logger.info("Testing Agent Orchestrator MVP...")
    logger.info("Ensure GOOGLE_API_KEY is set. For full tests, ensure backend services (Qdrant, PostgreSQL, Embedding Model) are running.")

    # Mock history and query
    sample_history = "Human: Hi!\nAI: Hello! How can I assist you today with SHBFinance services?"
    
    # Test Case 1: RAG Query (requires Qdrant & embedding model)
    # To mock, you might need to adjust RagToolMvp or qdrant_mvp to return mock data if services aren't up.
    # For a real test, ensure data is ingested into Qdrant.
    # query1 = "What are the general conditions for loan approval?"
    # logger.info(f"\n--- Test Case 1: RAG Query: '{query1}' ---")
    # response1 = agent_orchestrator_mvp(query1, sample_history)
    # logger.info(f"Response 1: {response1}")

    # Test Case 2: SQL Query (requires PostgreSQL & Text2SQL LLM)
    # To mock, T2SqlToolMvp or postgres_mvp could return mock data.
    # For a real test, ensure PG has 'loans', 'customers' tables with data.
    # query2 = "Show me my loan with ID L002 and the customer name associated with it."
    # logger.info(f"\n--- Test Case 2: SQL Query: '{query2}' ---")
    # response2 = agent_orchestrator_mvp(query2, sample_history)
    # logger.info(f"Response 2: {response2}")

    # Test Case 3: BOTH Query
    # query3 = "What is the policy on early loan repayment and can you show my active loans?"
    # logger.info(f"\n--- Test Case 3: BOTH Query: '{query3}' ---")
    # response3 = agent_orchestrator_mvp(query3, sample_history)
    # logger.info(f"Response 3: {response3}")

    # Test Case 4: Query that might lead to clarification (if LLM plan is poor or misses prompts)
    # This depends heavily on the planning LLM's output for such a vague query.
    # query4 = "Tell me about loans."
    # logger.info(f"\n--- Test Case 4: Vague Query (potential clarification): '{query4}' ---")
    # response4 = agent_orchestrator_mvp(query4, sample_history)
    # logger.info(f"Response 4: {response4}")

    # Test Case 5: Planning LLM fails (e.g. if API key is wrong temporarily for planning LLM only)
    # This is harder to simulate deterministically without changing the code, 
    # but the general error path is there.

    # Test Case 6: Simple RAG query that should work if services are up
    query_simple_rag = "What are interest rates?"
    logger.info(f"\n--- Test Case (Simple RAG): Query = '{query_simple_rag}' ---")
    # response_simple_rag = agent_orchestrator_mvp(query_simple_rag, sample_history)
    # logger.info(f"Response (Simple RAG): {response_simple_rag}")
    logger.info("Uncomment test cases and ensure all services (PostgreSQL, Qdrant, Embedding Model, Google Gemini API) are running and configured to test thoroughly.")
    logger.info("A simple test for API key presence:")
    if not os.getenv("GOOGLE_API_KEY"):
        logger.critical("CRITICAL: GOOGLE_API_KEY is not set in environment. The orchestrator will not function.")
    else:
        logger.info("GOOGLE_API_KEY is found.")
        # Example of a very simple query to trigger the planning phase (may require services for full execution)
        # logger.info("\n--- Basic Sanity Test Run (may require services for full execution) ---")
        # sanity_response = agent_orchestrator_mvp("hello", "")
        # logger.info(f"Sanity test response: {sanity_response}")
        logger.info("Please run the main Gradio app (app_mvp.py) for interactive testing once all components are ready.")
