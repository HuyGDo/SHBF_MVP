import os
import json
from typing import Dict, Any, List, Optional
# import google.generativeai as genai
import logging
import datetime
from decimal import Decimal

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core_ai_mvp.llm.main_llm_mvp import get_main_llm_mvp, Plan
from core_ai_mvp.agent.tools.rag_tool_mvp import RagToolMvp
from core_ai_mvp.agent.tools.t2sql_tool_mvp import T2SqlToolMvp
# Assuming GOOGLE_API_KEY is in environment. Settings can be used for other configs.
from config_mvp.settings_mvp import MAIN_LLM_API_URL

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# Enhanced log format for better readability
LOG_FORMAT = '%(asctime)s - %(levelname)-8s - %(name)s.%(funcName)s:%(lineno)d - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# Configuration for the LLMs used in this executor
# GEMINI_MODEL_FOR_SYNTHESIS = "gemini-1.5-flash" # Or your preferred Gemini model for generation
# GEMINI_MODEL_FOR_PLANNING = "gemini-1.5-flash" # Consistent with main_llm_mvp.py default
LOCAL_MODEL_FOR_SYNTHESIS = "local-model" # Name of your model in LM Studio
LOCAL_MODEL_FOR_PLANNING = "local-model" # Name of your model in LM Studio


DEFAULT_LANGUAGE = "vi" # Default language if not specified in plan

def json_serial_default(o: Any) -> Any:
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()
    if isinstance(o, Decimal):
        # Using str is safer for precision than float
        return str(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

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
            # Convert RealDictRow objects to standard dicts
            results_as_dicts = [dict(row) for row in t2sql_output]
            # Pretty print JSON for better LLM readability, with custom serializer
            return json.dumps(results_as_dicts, indent=2, ensure_ascii=False, default=json_serial_default)
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
    # google_api_key = os.getenv("GOOGLE_API_KEY")
    # if not google_api_key:
    #     return "Error: GOOGLE_API_KEY not found. Please set it in your environment variables."

    # # Configure the Google Generative AI library globally
    # try:
    #     genai.configure(api_key=google_api_key)
    # except Exception as e:
    #     logger.error(f"Error configuring Gemini API: {e}")
    #     return "I'm having trouble connecting to the AI service. Please check the API configuration."

    # Initialize planning LLM chain
    try:
        main_llm_planning_chain = get_main_llm_mvp(model_name=LOCAL_MODEL_FOR_PLANNING)
    except Exception as e:
        logger.error(f"Error initializing planning LLM: {e}")
        return "I'm having trouble starting up my planning module. Please check the API configuration and try again later."

    # Initialize tools
    rag_tool = RagToolMvp()
    t2sql_tool = T2SqlToolMvp()

    # Initialize base LLM for synthesis and clarification with better error handling
    try:
        # base_llm = ChatGoogleGenerativeAI(
        #     model=GEMINI_MODEL_FOR_SYNTHESIS,
        #     google_api_key=google_api_key,
        #     temperature=0.7,
        #     convert_system_message_to_human=True
        # )
        base_llm = ChatOpenAI(
            model=LOCAL_MODEL_FOR_SYNTHESIS,
            temperature=0.7,
            base_url=MAIN_LLM_API_URL,
            api_key="lm-studio" # Not used by LM Studio but required by the library
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
        logger.info(f"--- Step 1: Generating Plan ---")
        logger.info(f"Query: '{cleaned_query}' | History: '{history_string}'")
        
        # The main planning LLM is robust enough to handle routing.
        # The previous hardcoded SQL route has been removed to avoid misclassifying RAG queries.
        # All queries will now go through the LLM planner.
            
        plan_dict = main_llm_planning_chain.invoke({
            "query": cleaned_query,
            "history": history_string
        })
        logger.info(f"--- Step 2: Plan Received --- \n{json.dumps(plan_dict, indent=2, ensure_ascii=False)}")
        logger.info(f"--- Step 2.1: Query --- \n{json.dumps(cleaned_query, indent=2, ensure_ascii=False)}")
        logger.info(f"--- Step 2.2: History --- \n{json.dumps(cleaned_query, indent=2, ensure_ascii=False)}")
        
        # Convert the dictionary to a Plan object
        try:
            plan = Plan(**plan_dict)
            logger.info(f"Plan validated successfully: Route='{plan.route}'")
        except Exception as e:
            logger.error(f"Fatal: Could not parse the plan from the LLM. Error: {e}", exc_info=True)
            # Fallback for critical parsing errors.
            has_customer_id = any(pattern in cleaned_query.lower() for pattern in ["mã id", "id của tôi", "id là", "hđ"])
            if has_customer_id:
                logger.warning("Plan parsing failed, but customer identifiers found. Attempting a direct SQL fallback.")
                customer_id_entity = cleaned_query.split(":")[-1].strip() if ":" in cleaned_query else "unknown"
                fallback_sql_prompt = f"Retrieve relevant information for customer based on query: '{cleaned_query}'"
                if customer_id_entity != "unknown":
                    fallback_sql_prompt = f"Retrieve relevant information for customer ID {customer_id_entity} based on query: '{cleaned_query}'"

                # Directly call handle_sql_query with a constructed plan
                return handle_sql_query(
                    plan=Plan(
                        route="SQL",
                        intents=["fallback_get_customer_data"],
                        entities={"customer_id": customer_id_entity} if customer_id_entity != "unknown" else {},
                        sql_prompt=fallback_sql_prompt,
                        policy_query=None,
                        language=DEFAULT_LANGUAGE
                    ),
                    query=cleaned_query,
                    history=history_string,
                    llm=base_llm,
                    sql_tool=t2sql_tool
                )
            
            # If no customer ID, clarification is the only safe option.
            logger.warning("Plan parsing failed and no clear fallback. Asking for clarification.")
            return clarification_chain.invoke({"initial_query": cleaned_query})

        # 2. Execution Phase based on Plan
        final_response = ""
        logger.info(f"--- Step 3: Executing Plan (Route: {plan.route}) ---")
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
        logger.error(f"Critical error in orchestrator: {e}", exc_info=True)
        # General fallback clarification if anything else goes wrong in the main orchestration logic
        return clarification_chain.invoke({"initial_query": user_query}) # Use original user_query for clarification context

def handle_sql_query(plan: Plan, query: str, history: str, llm: Any, sql_tool: Any) -> str:
    """Helper function to handle SQL queries"""
    logger.info(f"Handling SQL query with plan: \\n{json.dumps(plan.dict(), indent=2, ensure_ascii=False)}")
    language = plan.language or DEFAULT_LANGUAGE # Added language parameter
    try:
        t2sql_output = sql_tool.run({
            "sql_query_prompt": plan.sql_prompt,
            "intents": plan.intents # Pass intents to the tool
        })
        # sql_results_str is already formatted by format_sql_results_for_llm, which includes JSON dump if it's a list
        # If t2sql_output (raw) needs to be pretty-printed:
        # logger.debug(f"Raw SQL output: \\n{json.dumps(t2sql_output, indent=4, ensure_ascii=False)}")
        sql_results_str = format_sql_results_for_llm(t2sql_output)
        logger.info(f"SQL results: \\n{sql_results_str}") # Log the formatted string
        
        synthesis_prompt = ChatPromptTemplate.from_template('''
        You are an assistant for SHBFinance. Based on the user's query and the following database results, provide a clear and helpful response in Vietnamese.

        **Crucial Instructions:**
        1.  **DO NOT** output the raw JSON. Summarize the key information from the database results in a natural, conversational way.
        2.  Respond **ONLY** in {language}.
        3.  Address the user's query directly.
        4.  If the database results are empty, politely inform the user that no information was found.
        5.  Do not add any meta-commentary or explanations about the data source.

        **User Query:** "{query}"

        **Database Results (in JSON format):**
        {results}

        **Your helpful, conversational response in {language}:**
        ''')
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
    logger.info(f"Handling RAG query with plan: \\n{json.dumps(plan.dict(), indent=2, ensure_ascii=False)}")
    language = plan.language or DEFAULT_LANGUAGE
    try:
        rag_output = rag_tool.run({
            "policy_query_prompt": plan.policy_query,
            "top_k": 5, # Increase top_k to get more context
            "score_threshold": 0.3 # Add a threshold to filter irrelevant results
        })
        # rag_results_str is already formatted by format_rag_results_for_llm, can be logged directly
        # If rag_output itself (the list of dicts) needs to be pretty-printed before formatting:
        # logger.debug(f"Raw RAG output: \\n{json.dumps(rag_output, indent=4, ensure_ascii=False)}")
        rag_results_str = format_rag_results_for_llm(rag_output)
        logger.info(f"RAG results: \\n{rag_results_str}") # Log the formatted string

        synthesis_prompt_template = """
        You are an assistant for SHBFinance. Based on the user's query and the retrieved policy documents, provide a clear and helpful response in Vietnamese.

        **Crucial Instructions:**
        1.  Synthesize the information from the policy snippets into a coherent answer. Do not just list the snippets.
        2.  Respond **ONLY** in {language}.
        3.  Address the user's query directly.
        4.  If the retrieved information is not relevant, state that you couldn't find specific policy details on that topic.
        5.  Do not add any meta-commentary or explanations about the data source.

        **User Query:** "{query}"

        **Retrieved Policy Information:**
        {results}

        **Your helpful, conversational response in {language}:**
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
    logger.info(f"Handling BOTH query with plan: \\n{json.dumps(plan.dict(), indent=2, ensure_ascii=False)}")
    language = plan.language or DEFAULT_LANGUAGE
    try:
        # Execute RAG part
        rag_output = rag_tool.run({
            "policy_query_prompt": plan.policy_query,
            "top_k": 5, # Increase top_k to get more context
            "score_threshold": 0.3 # Add a threshold to filter irrelevant results
        })
        rag_results_str = format_rag_results_for_llm(rag_output)
        logger.info(f"RAG results (for BOTH): \\n{rag_results_str}")

        # Execute SQL part
        t2sql_output = sql_tool.run({
            "sql_query_prompt": plan.sql_prompt,
            "intents": plan.intents # Pass intents to the tool
        })
        # sql_results_str is already formatted by format_sql_results_for_llm, which includes JSON dump if it's a list
        # If t2sql_output (raw) needs to be pretty-printed:
        # logger.debug(f"Raw SQL output: \\n{json.dumps(t2sql_output, indent=4, ensure_ascii=False)}") 
        sql_results_str = format_sql_results_for_llm(t2sql_output)
        logger.info(f"SQL results (for BOTH): \\n{sql_results_str}")

        synthesis_prompt_template = """
        You are an assistant for SHBFinance. Based on the user's query, the retrieved policy information, and the user's data from the database, provide a comprehensive and helpful response in Vietnamese.

        **Crucial Instructions:**
        1.  **DO NOT** output the raw JSON from the database.
        2.  Synthesize the information from **both** the policy documents and the database results into a single, coherent answer.
        3.  Respond **ONLY** in {language}.
        4.  Address all parts of the user's query.
        5.  If information is missing from either the policy or the database, mention it gracefully.
        6.  Do not add any meta-commentary or explanations about the data sources.

        **User Query:** "{query}"

        **Retrieved Policy Information:**
        {rag_results}

        **User's Database Results (in JSON format):**
        {sql_results}

        **Your comprehensive, conversational response in {language}:**
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

