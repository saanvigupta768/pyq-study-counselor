from dotenv import load_dotenv
from groq import Groq
from mem0 import MemoryClient
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
memory = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))

USER_ID = "saanvi"  # identifies whose memory this is

print("Agent ready. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        break

    # --- Search memory for anything relevant to this message ---
    search_result = memory.search(user_input, filters={"user_id": USER_ID})
    relevant_memories = search_result.get("results", [])
    memory_context = "\n".join([m["memory"] for m in relevant_memories])

    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Relevant known facts about the user:\n{memory_context}"},
        {"role": "user", "content": user_input}
    ]

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages
    )

    reply = response.choices[0].message.content
    print(f"Agent: {reply}\n")

    # --- Let Mem0 decide what's worth remembering from this exchange ---
    memory.add(
        [{"role": "user", "content": user_input}, {"role": "assistant", "content": reply}],
        user_id=USER_ID
    )