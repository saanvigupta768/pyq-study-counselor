from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
import os
import re
import json
import base64
import hashlib
from collections import defaultdict

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

TEXT_MODEL = "openai/gpt-oss-20b"
# Model used to read photos of papers. Groq renames models often. If photo reading
# stops working, check console.groq.com/docs/vision and put VISION_MODEL=... in your .env file.
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")

PYQ_FOLDER = "pyqs"              # put your papers (.pdf, .txt, .png, .jpg) in this folder
DEMO_FILE = "sample_pyqs.txt"    # used only if the folder is empty or missing
SYLLABUS_FILE = "syllabus.txt"   # optional: one syllabus topic per line
CACHE_FILE = "analysis_cache.json"

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
MAX_IMAGE_BYTES = 4 * 1024 * 1024

DEFAULT_TOPICS = [
    "Introduction to Data Structures (Arrays, Complexity Analysis)",
    "Linked Lists",
    "Stacks and Queues",
    "Trees",
    "Graphs",
    "Sorting and Searching",
    "Hashing"
]


def load_syllabus():
    """Uses syllabus.txt if you made one (one topic per line), otherwise the Data Structures topics."""
    if os.path.exists(SYLLABUS_FILE):
        with open(SYLLABUS_FILE, "r", encoding="utf-8") as f:
            topics = [line.strip() for line in f if line.strip()]
        if topics:
            print(f"Using the {len(topics)} syllabus topics from {SYLLABUS_FILE}")
            return topics
    return DEFAULT_TOPICS


SYLLABUS_TOPICS = load_syllabus()


# ---------- Step 1: read the papers ----------

def read_pdf(path):
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_image(path):
    """Sends a photo of a paper to a vision model and gets the text back."""
    ext = os.path.splitext(path)[1].lower()
    if os.path.getsize(path) > MAX_IMAGE_BYTES:
        raise ValueError("the image is bigger than 4 MB, please shrink it")
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "This is a photo of an exam paper. Transcribe all the text exactly, "
                                         "keeping question numbers and marks. Output only the text."},
                {"type": "image_url", "image_url": {"url": f"data:{IMAGE_TYPES[ext]};base64,{encoded}"}},
            ],
        }],
        temperature=0.1,
        max_completion_tokens=4096,
    )
    text = response.choices[0].message.content or ""
    # Some models print their thinking first; remove it
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def load_papers():
    """Returns a list of (paper_name, text) for every paper in the pyqs folder."""
    papers = []
    if os.path.isdir(PYQ_FOLDER):
        for name in sorted(os.listdir(PYQ_FOLDER)):
            path = os.path.join(PYQ_FOLDER, name)
            ext = os.path.splitext(name)[1].lower()
            try:
                if ext == ".pdf":
                    text = read_pdf(path)
                elif ext == ".txt":
                    text = read_txt(path)
                elif ext in IMAGE_TYPES:
                    print(f"Reading photo {name} ...")
                    text = read_image(path)
                else:
                    continue
            except Exception as e:
                print(f"Warning: couldn't read {name} ({e}), skipping it.")
                if ext in IMAGE_TYPES:
                    print("  Tip: if the model name is the problem, check console.groq.com/docs/vision "
                          "and set VISION_MODEL in your .env file.")
                continue

            if not text.strip():
                print(f"Warning: no text found in {name}. If it's a scanned PDF, save the pages as "
                      f".png or .jpg photos instead. Skipping it.")
                continue
            papers.append((name, text))

    if not papers:
        if not os.path.exists(DEMO_FILE):
            print(f"No papers found. Put .pdf, .txt or photo files in a folder called '{PYQ_FOLDER}'.")
            raise SystemExit
        print(f"No papers in '{PYQ_FOLDER}/', using the demo file {DEMO_FILE}")
        papers.append((DEMO_FILE, read_txt(DEMO_FILE)))
    else:
        print(f"Loaded {len(papers)} paper(s): " + ", ".join(name for name, _ in papers))
    return papers


# ---------- Step 2: split long text safely ----------

QUESTION_START = re.compile(r"^\s*(Q\.?\s*\d+|\d+\s*[\.\)])", re.IGNORECASE)


def split_into_chunks(text, max_chars=3000):
    """Cuts long text into pieces, but only between questions, so no question is split in half."""
    chunks, current = [], ""
    for line in text.splitlines():
        too_long = len(current) + len(line) > max_chars and current
        hard_limit = len(current) > max_chars * 2   # safety: papers with no numbering
        if (too_long and QUESTION_START.match(line)) or hard_limit:
            chunks.append(current)
            current = ""
        current += line + "\n"
    if current.strip():
        chunks.append(current)
    return chunks


# ---------- Step 3: sort questions into topics with the AI ----------

def segregate_chunk(chunk, topics):
    """Returns a list of questions, or None if this chunk failed."""
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

    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0  # no randomness, so repeated runs give the same answer
        )
        reply = response.choices[0].message.content
        start, end = reply.find("["), reply.rfind("]")
        return json.loads(reply[start:end + 1], strict=False)   # strict=False allows line breaks inside text
    except Exception as e:
        print(f"Warning: one chunk failed ({e}), skipping it.")
        return None


def segregate_papers(papers, topics):
    """Returns (questions, number_of_failed_chunks)."""
    results, failed = [], 0
    for paper_name, text in papers:
        for chunk in split_into_chunks(text):
            found = segregate_chunk(chunk, topics)
            if found is None:
                failed += 1
                continue
            for q in found:
                if q.get("topic") not in topics:
                    q["topic"] = "Unclassified"
                try:
                    q["marks"] = int(q.get("marks", 0))
                except (TypeError, ValueError):
                    q["marks"] = 0
                q["paper"] = paper_name
                results.append(q)
    return results, failed


def fingerprint(papers, topics):
    h = hashlib.md5()
    for name, text in papers:
        h.update(name.encode("utf-8"))
        h.update(text.encode("utf-8"))
    h.update("|".join(topics).encode("utf-8"))
    return h.hexdigest()


def analyze(papers, topics):
    """Sorts the questions with the AI, or reuses the saved result if the papers haven't changed."""
    key = fingerprint(papers, topics)
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if cache.get("key") == key:
                print("Using the saved analysis (your papers haven't changed).")
                return cache["questions"]
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    questions, failed = segregate_papers(papers, topics)
    if failed:
        print(f"Note: {failed} chunk(s) failed, so this result is incomplete and won't be saved. Run again to retry.")
    elif questions:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"key": key, "questions": questions}, f, indent=2)
    return questions


# ---------- Step 4: rank the topics ----------

def score_topics(questions):
    stats = defaultdict(lambda: {"count": 0, "total_marks": 0, "papers": set()})
    for q in questions:
        s = stats[q["topic"]]
        s["count"] += 1
        s["total_marks"] += q["marks"]
        s["papers"].add(q["paper"])
    # Most marks first; if tied, the topic asked more often comes first
    return sorted(stats.items(),
                  key=lambda x: (x[1]["total_marks"], x[1]["count"]),
                  reverse=True)


def print_ranking(ranked, topics):
    print("\n--- Topic Importance Ranking ---\n")
    for topic, s in ranked:
        print(f"{topic}: {s['count']} questions across {len(s['papers'])} paper(s), "
              f"{s['total_marks']} total marks")

    asked = {topic for topic, _ in ranked}
    never_asked = [t for t in topics if t not in asked]
    if never_asked:
        print("\nSyllabus topics that never appeared in these papers:")
        for t in never_asked:
            print(f"- {t}")


def main():
    papers = load_papers()
    segregated = analyze(papers, SYLLABUS_TOPICS)

    print(f"\nFound {len(segregated)} questions:\n")
    for q in segregated:
        print(f"[{q['topic']}] ({q['marks']} marks, {q['paper']}): {q['question']}")

    print_ranking(score_topics(segregated), SYLLABUS_TOPICS)


if __name__ == "__main__":
    main()