from dotenv import load_dotenv
from groq import Groq
import os
import json
from collections import defaultdict

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYLLABUS_TOPICS = [
    "Introduction to Data Structures (Arrays, Complexity Analysis)",
    "Linked Lists",
    "Stacks and Queues",
    "Trees",
    "Graphs",
    "Sorting and Searching",
    "Hashing"
]


def load_pyqs():
    # Use your real papers if pyqs.txt exists, otherwise the demo file
    path = "pyqs.txt" if os.path.exists("pyqs.txt") else "sample_pyqs.txt"
    if not os.path.exists(path):
        print("No PYQ file found. Create pyqs.txt or sample_pyqs.txt next to this script.")
        raise SystemExit
    print(f"Reading questions from {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_chunks(text, max_chars=3000):
    # Long input is sent in pieces so the model doesn't get overwhelmed or cut off
    chunks, current = [], ""
    for line in text.splitlines():
        if len(current) + len(line) > max_chars and current:
            chunks.append(current)
            current = ""
        current += line + "\n"
    if current.strip():
        chunks.append(current)
    return chunks


def segregate_chunk(chunk, topics):
    prompt = f"""You are an expert exam paper analyst. Below is a list of syllabus topics and a block of raw exam questions.

Syllabus topics: {topics}

Raw questions:
{chunk}

For each question you find, identify:
1. The question text
2. Which syllabus topic it belongs to (must be exactly one of the topics listed above)
3. The marks it's worth (if mentioned, otherwise put 0)

Respond ONLY with a JSON array, no other text, in this exact format:
[{{"question": "...", "topic": "...", "marks": 5}}]"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0  # no randomness, so repeated runs give the same answer
    )

    reply = response.choices[0].message.content
    # Grab just the JSON array, even if the model added extra text around it
    start, end = reply.find("["), reply.rfind("]")
    try:
        return json.loads(reply[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        print("Warning: couldn't read the model's reply for one chunk, skipping it.")
        return []


def segregate_questions(raw_text, topics):
    results = []
    for chunk in split_into_chunks(raw_text):
        for q in segregate_chunk(chunk, topics):
            # Reject any topic the model invented that isn't in the syllabus
            if q.get("topic") not in topics:
                q["topic"] = "Unclassified"
            try:
                q["marks"] = int(q.get("marks", 0))
            except (TypeError, ValueError):
                q["marks"] = 0
            results.append(q)
    return results


def score_topics(questions):
    stats = defaultdict(lambda: {"count": 0, "total_marks": 0})
    for q in questions:
        stats[q["topic"]]["count"] += 1
        stats[q["topic"]]["total_marks"] += q["marks"]
    # Most marks first; if tied, the topic asked more often comes first
    return sorted(stats.items(),
                  key=lambda x: (x[1]["total_marks"], x[1]["count"]),
                  reverse=True)


# --- Run it ---
raw_pyqs = load_pyqs()
segregated = segregate_questions(raw_pyqs, SYLLABUS_TOPICS)

print(f"\nFound {len(segregated)} questions:\n")
for q in segregated:
    print(f"[{q['topic']}] ({q['marks']} marks): {q['question']}")

print("\n--- Topic Importance Ranking ---\n")
ranked = score_topics(segregated)
for topic, s in ranked:
    print(f"{topic}: appeared {s['count']} times, {s['total_marks']} total marks")

asked = {topic for topic, _ in ranked}
never_asked = [t for t in SYLLABUS_TOPICS if t not in asked]
if never_asked:
    print("\nSyllabus topics that never appeared in these papers:")
    for t in never_asked:
        print(f"- {t}")