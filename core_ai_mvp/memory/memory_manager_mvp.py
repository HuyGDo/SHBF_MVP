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

if __name__ == '__main__':
    print(f"Initializing memory manager with k=2, memory_key='{DEFAULT_MEMORY_KEY}'")
    memory = get_memory_manager_mvp(k=2)

    # Simulate a conversation
    print("\nSimulating conversation...")

    # Turn 1
    user_input_1 = "Hello, I need help with my loan."
    ai_response_1 = "Hi there! I can help with that. What is your loan ID?"
    memory.save_context({"input": user_input_1}, {"output": ai_response_1})
    print(f"Human: {user_input_1}")
    print(f"AI: {ai_response_1}")
    current_history = memory.load_memory_variables({})
    print(f"Memory ({DEFAULT_MEMORY_KEY}):\n{current_history.get(DEFAULT_MEMORY_KEY)}")

    # Turn 2
    user_input_2 = "My loan ID is 12345."
    ai_response_2 = "Thank you. Looking up loan ID 12345..."
    memory.save_context({"input": user_input_2}, {"output": ai_response_2})
    print(f"\nHuman: {user_input_2}")
    print(f"AI: {ai_response_2}")
    current_history = memory.load_memory_variables({})
    print(f"Memory ({DEFAULT_MEMORY_KEY}):\n{current_history.get(DEFAULT_MEMORY_KEY)}")

    # Turn 3
    user_input_3 = "What is its status?"
    ai_response_3 = "Your loan 12345 is currently active."
    memory.save_context({"input": user_input_3}, {"output": ai_response_3})
    print(f"\nHuman: {user_input_3}")
    print(f"AI: {ai_response_3}")
    current_history = memory.load_memory_variables({})
    print(f"Memory ({DEFAULT_MEMORY_KEY}):\n{current_history.get(DEFAULT_MEMORY_KEY)}")
    print("\nNote: The first turn should now be excluded from history due to k=2.")

    # Turn 4 (to show the windowing effect more clearly)
    user_input_4 = "And what is the interest rate?"
    ai_response_4 = "The interest rate is 3.5%."
    memory.save_context({"input": user_input_4}, {"output": ai_response_4})
    print(f"\nHuman: {user_input_4}")
    print(f"AI: {ai_response_4}")
    current_history = memory.load_memory_variables({})
    print(f"Memory ({DEFAULT_MEMORY_KEY}):\n{current_history.get(DEFAULT_MEMORY_KEY)}")
    print("\nNote: Now the second turn (loan ID 12345) should be excluded.")

    print("\nTesting memory with different prefixes and k=1:")
    memory_custom = get_memory_manager_mvp(k=1, ai_prefix="Bot", human_prefix="User", memory_key="chat_log")
    memory_custom.save_context({"input": "Hi"}, {"output": "Hello"})
    memory_custom.save_context({"input": "How are you?"}, {"output": "I am good."})
    current_history_custom = memory_custom.load_memory_variables({})
    print(f"Memory (chat_log):\n{current_history_custom.get('chat_log')}")
