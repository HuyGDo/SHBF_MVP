import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Optional, Any
import google.generativeai as genai


# Load GOOGLE_API_KEY from .env file
# Ensure your .env file has GOOGLE_API_KEY="your_api_key"
# from dotenv import load_dotenv
# load_dotenv()
# For MVP, we assume GOOGLE_API_KEY is set in the environment directly
# or will be handled by the Gradio app's environment setup.

# Define the Pydantic model based on plan_schema_2.json
class Plan(BaseModel):
    route: str = Field(description="Primary execution path: pure SQL, pure RAG, or combined.", enum=["SQL", "RAG", "BOTH"])
    intents: List[str] = Field(description="List of normalised intent identifiers.", min_items=1)
    entities: Dict[str, Any] = Field(description="Key-value map of extracted entities. Values may be string, number, boolean, array, object or null.")
    sql_prompt: Optional[str] = Field(description="Concise English question given to the Text-to-SQL model.", default=None)
    policy_query: Optional[str] = Field(description="Free-text query used for vector search if `route` is RAG or BOTH.", default=None)
    language: str = Field(description="Language in which the final answer should be generated.", enum=["vi", "en"], default="vi")

def get_main_llm_mvp(google_api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
    """
    Initializes and returns the main LLM for plan generation.

    Args:
        google_api_key (Optional[str]): Google API Key. If None, it's expected to be in the environment.
        model_name (str): The Gemini model name to use (e.g., "gemini-2.0-flash").

    Returns:
        A LangChain LCEL chain configured for plan generation.
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

    parser = JsonOutputParser(pydantic_object=Plan)

    prompt_template = """
    You are an AI assistant for a Vietnamese retail bank.  
Your job is to read the user’s request and produce a **plan** – a single JSON object that tells downstream agents what to do.

─────────────────────────  CORE TASK  ─────────────────────────
Given {history} and the current user query {query}:

1. Decide the execution **route**  
   • "SQL"  – query needs personal or contract-specific data  
   • "RAG"  – query asks for general policies / definitions / procedures  
   • "BOTH" – needs both personal data **and** policy context

2. Produce (when applicable)  
   • a list of **intents** (see intent catalogue below)  
   • an **entities** object containing only keys from the Entity Catalogue  
   • a short **sql_prompt** (Vietnamese or English) if route is SQL or BOTH  
   • a **policy_query** in **Vietnamese** if route is RAG or BOTH

────────────────────────  ROUTING RULES  ───────────────────────
1. SQL route if the user:
   • Mentions their customer / contract / loan IDs  
   • Requests balances, rates, instalments, fees, etc. for *their* loan  
   • Uses phrases like “khoản vay của tôi”, “mã id …”, “HĐ-…”, “LMS…”, etc.

2. RAG route **only** if:
   • The user asks about general policy, definitions, requirements or procedures  
   • No personal identifiers or customer-specific data are requested

3. BOTH route if both 1 & 2 are true in the same question.

───────────────────────  OUTPUT FORMAT  ───────────────────────
Return **only** a JSON object that can be parsed by `json.loads`, exactly:

{
  "route": "SQL" | "RAG" | "BOTH",
  "intents": [ ... ],                 // **Empty array [] when route == "RAG"**
  "entities": {                       // Only keys from Entity Catalogue
      "contract_id": "HD123456",
      "customer_id": 12345,
      ...
  },
  "sql_prompt":  "…",                 // null if not needed
  "policy_query": "…",                // null if not needed
  "language": "vi"
}

──────────────────────── INTENT CATALOGUE ─────────────────────
query_list_active_loans, query_loan_details_by_identifier, …  (full list unchanged)

──────────────────────── ENTITY CATALOGUE ─────────────────────
Use these keys **only** & follow patterns / enums exactly.

[
  { "name":"contract_id",        "type":"string",  "pattern":"^[A-Z0-9]{6,}$",              "example":"HD123456" },
  { "name":"customer_id",        "type":"number",                                                  "example":34567 },
  { "name":"los_cif_no",         "type":"string",                                                  "example":"LOS000987" },
  { "name":"lms_cif_no",         "type":"string",                                                  "example":"LMS001122" },
  { "name":"loan_id",            "type":"string",                                                  "example":"LD_NO=00001" },
  { "name":"date",               "type":"date",    "format":"YYYY-MM-DD",                         "example":"2025-06-15" },
  { "name":"date_from",          "type":"date",    "format":"YYYY-MM-DD",                         "example":"2025-01-01" },
  { "name":"date_to",            "type":"date",    "format":"YYYY-MM-DD",                         "example":"2025-06-30" },
  { "name":"amount",             "type":"number",  "unit":"VND",                                  "example":10000000 },
  { "name":"currency",           "type":"string",  "enum":["VND","USD"],                          "example":"VND" },
  { "name":"percentage_rate",    "type":"number",  "range":[0,100],                               "example":5.5 },
  { "name":"loan_status",        "type":"string",  "enum":["active","paid_off","overdue"],        "example":"active" },
  { "name":"loan_product",       "type":"string",                                                  "example":"Unsecured Consumer" },
  { "name":"interest_rate_type", "type":"string",  "enum":["fixed","variable"],                   "example":"fixed" },
  { "name":"loan_officer_id",    "type":"string",                                                  "example":"EMP0012" },
  { "name":"installment_no",     "type":"integer",                                                 "example":5 },
  { "name":"payment_id",         "type":"string",                                                  "example":"PMT7890" },
  { "name":"days_overdue",       "type":"integer",                                                 "example":7 },
  { "name":"period_months",      "type":"integer",                                                 "example":24 },
  { "name":"period_years",       "type":"integer",                                                 "example":2 },
  { "name":"fee_type",           "type":"string",  "enum":["origination","late","early_repay"],   "example":"late" },
  { "name":"policy_article",     "type":"string",                                                  "example":"Article 12" },
  { "name":"policy_clause",      "type":"string",                                                  "example":"Clause 3" },
  { "name":"officer_phone",      "type":"string",  "pattern":"^0\\d{8,10}$",                      "example":"0912345678" },
  { "name":"officer_email",      "type":"string",  "format":"email",                              "example":"support@bank.vn" },
  { "name":"address",            "type":"string",                                                  "example":"123 Lê Lợi, Q1" },
  { "name":"loan_type",          "type":"string",  "enum":["secured","unsecured"],                "example":"unsecured" },
  { "name":"overdue_flag",       "type":"boolean",                                                 "example":true },
  { "name":"bucket_code",        "type":"string",                                                  "example":"B2" },
  { "name":"stage_code",         "type":"string",                                                  "example":"STG01" },
  { "name":"group_code",         "type":"string",                                                  "example":"GRP_SOFTCALL" },
  { "name":"user_code",          "type":"string",                                                  "example":"COLL0123" }
]

──────────────────────────  EXAMPLES  ──────────────────────────
SQL:
User → “Tất cả khoản vay còn hiệu lực của tôi. ID: 1001”
Assistant → {
  "route":"SQL",
  "intents":["query_list_active_loans"],
  "entities":{"customer_id":1001},
  "sql_prompt":"Find all active loans for customer 1001",
  "policy_query":null,
  "language":"vi"
}

RAG: (no intents needed)
User → “Quy định về phí trả nợ trước hạn?”
Assistant → {
  "route":"RAG",
  "intents":[],
  "entities":{},
  "sql_prompt":null,
  "policy_query":"Quy định về phí trả nợ trước hạn?",
  "language":"vi"
}

BOTH:
User → “HĐ HD001. Phí trả nợ trước hạn bao nhiêu?”
Assistant → {
  "route":"BOTH",
  "intents":["query_loan_details_by_identifier","query_other_applicable_fees"],
  "entities":{"contract_id":"HD001"},
  "sql_prompt":"Get contract HD001 to calculate pre-payment fee.",
  "policy_query":"Chính sách hiện hành về phí trả nợ trước hạn là gì?",
  "language":"vi"
}"""

    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | llm | parser
    return chain

if __name__ == '__main__':
    # This is an example of how to use the get_main_llm_mvp function.
    # Ensure GOOGLE_API_KEY is set in your environment variables.
    # You might need to install python-dotenv and create a .env file:
    # GOOGLE_API_KEY="your_actual_google_api_key"

    # from dotenv import load_dotenv
    # load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY not set. Please set it in your environment or a .env file for this example to run.")
    else:
        print("GOOGLE_API_KEY found. Initializing Main LLM...")
        main_llm_chain = get_main_llm_mvp()
        print("Main LLM Chain initialized.")

        sample_query = "What is the interest rate for a home loan and what are my current active loans?"
        sample_history = "Human: Hi\nAI: Hello! How can I help you today?"

        print(f"\nTesting with sample query: '{sample_query}'")
        try:
            plan_output = main_llm_chain.invoke({"query": sample_query, "history": sample_history})
            print("\nPlan Output:")
            import json
            print(json.dumps(plan_output, indent=2))

            # Example for a RAG query
            sample_rag_query = "What are the general conditions for loan approval?"
            print(f"\nTesting with RAG query: '{sample_rag_query}'")
            plan_rag_output = main_llm_chain.invoke({"query": sample_rag_query, "history": ""})
            print("\nPlan RAG Output:")
            print(json.dumps(plan_rag_output, indent=2))

            # Example for a SQL query
            sample_sql_query = "Show me my loan with ID 12345."
            print(f"\nTesting with SQL query: '{sample_sql_query}'")
            plan_sql_output = main_llm_chain.invoke({"query": sample_sql_query, "history": ""})
            print("\nPlan SQL Output:")
            print(json.dumps(plan_sql_output, indent=2))

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure your GOOGLE_API_KEY is valid and the Gemini API is enabled for your project.")
            print("You might also need to install necessary libraries: pip install langchain-google-genai python-dotenv langchain")
