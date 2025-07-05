from langchain.memory import ConversationBufferWindowMemory

DEFAULT_MEMORY_KEY = "history" # Default key for storing history in LLM prompts
DEFAULT_AI_PREFIX = "AI"
DEFAULT_HUMAN_PREFIX = "Human"

def get_memory_manager_mvp(
    k: int = 3, 
    memory_key: str = DEFAULT_MEMORY_KEY,
    ai_prefix: str = DEFAULT_AI_PREFIX,
    human_prefix: str = DEFAULT_HUMAN_PREFIX
    ) -> ConversationBufferWindowMemory:
    """
    Initializes and returns a ConversationBufferWindowMemory instance.

    Args:
        k (int): The number of past conversation turns to remember. Default is 3.
        memory_key (str): The key under which the conversation history is stored and 
                          expected in the prompt. Default is "history".
        ai_prefix (str): The prefix for AI messages in the history. Default is "AI".
        human_prefix (str): The prefix for human messages in the history. Default is "Human".

    Returns:
        ConversationBufferWindowMemory: An instance of the memory manager.
    """
    return ConversationBufferWindowMemory(
        k=k, 
        memory_key=memory_key,
        ai_prefix=ai_prefix,
        human_prefix=human_prefix,
        return_messages=False # For MVP, simpler string history is fine
    )

