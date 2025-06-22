import gradio as gr
from core_ai_mvp.agent.agent_executor_mvp import agent_orchestrator_mvp
from core_ai_mvp.memory.memory_manager_mvp import get_memory_manager_mvp, DEFAULT_MEMORY_KEY, DEFAULT_AI_PREFIX, DEFAULT_HUMAN_PREFIX

# Initialize conversational memory globally
# k=3 means it will remember the last 3 pairs of human/AI interactions.
chat_memory = get_memory_manager_mvp(
    k=3, 
    memory_key=DEFAULT_MEMORY_KEY, 
    ai_prefix=DEFAULT_AI_PREFIX, 
    human_prefix=DEFAULT_HUMAN_PREFIX
)

print("Chat memory initialized for Gradio app.")

def chat_with_agent(message: str, history: list):
    """
    This function is called by the Gradio interface for each chat interaction.

    Args:
        message (str): The new user message.
        history (list): The chat history maintained by Gradio (list of [user_msg, ai_msg] pairs).
                        While Gradio provides this, we use our own `chat_memory` for LLM context.

    Returns:
        str: The AI agent's response.
    """
    print(f"\n[Gradio App] Received message: '{message}'")
    print(f"[Gradio App] Gradio history (for UI): {history}")

    # Load the current history from our ConversationBufferWindowMemory
    # This history_for_llm is what the agent_orchestrator and underlying LLMs will see.
    # It's already formatted and windowed by chat_memory.
    loaded_memory = chat_memory.load_memory_variables({})
    history_for_llm = loaded_memory.get(chat_memory.memory_key, "")
    print(f"[Gradio App] History for LLM (from chat_memory, window k={chat_memory.k}):\n'''{history_for_llm}'''")

    # Call the agent orchestrator
    # The agent_orchestrator_mvp expects the history *before* the current user message is added.
    ai_response = agent_orchestrator_mvp(user_query=message, history_string=history_for_llm)
    print(f"[Gradio App] Agent orchestrator returned: '{ai_response}'")

    # Save the current interaction to our ConversationBufferWindowMemory
    chat_memory.save_context({"input": message}, {"output": ai_response})
    # For debugging, let's see memory after save
    # new_loaded_memory = chat_memory.load_memory_variables({})
    # print(f"[Gradio App] chat_memory after save_context: '{new_loaded_memory.get(chat_memory.memory_key)}'")

    return ai_response

# Setup Gradio Chat Interface
demo = gr.ChatInterface(
    fn=chat_with_agent,
    title="SHBFinance MVP Chatbot",
    description=("Ask questions about loan policies or your (mock) loan data. "
                 "Powered by local LLMs via LM Studio and Gemini API."),
    examples=[
        ["What are the general conditions for loan approval?"],
        ["Show me my loan with ID L001 and the customer name associated with it."],
        ["What is the policy on early loan repayment and can you show my active loans with type 'Business Loan'?"],
        ["What are the interest rates?"],
        ["Xin chào, tôi muốn hỏi về các khoản vay."]
    ],
    retry_btn=None,
    undo_btn="Delete Previous Turn",
    clear_btn="Clear Chat",
)

if __name__ == "__main__":
    print("Launching Gradio MVP app...")
    print("Please ensure all backend services are running: ")
    print("  - LM Studio (Embedding Model, potentially Main/Text2SQL if not using Gemini API for all)")
    print("  - Google Gemini API accessible (GOOGLE_API_KEY set)")
    print("  - PostgreSQL Database")
    print("  - Qdrant Vector Database")
    demo.launch() # Share=False by default. Set share=True to create a public link.
