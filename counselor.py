import json
import os
from collections import Counter

from pyq_analyzer import (
    client, TEXT_MODEL, SYLLABUS_TOPICS,
    load_papers, analyze, score_topics, print_ranking,
)

PROFILE_FILE = "student_profile.json"
SAMPLE_PAPER_FILE = "sample_paper.txt"
ASSIGNMENT_FILE = "assignment.txt"
STUDY_PLAN_FILE = "study_plan.txt"


# ---------- helpers ----------

def ask_ai(prompt, temperature=0.7):
    """Sends a prompt to the AI and returns the reply text, or None if something went wrong."""
    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Couldn't get a reply from the AI ({e}). Wait a minute and try again.")
        return None


def extract_json(reply):
    start, end = reply.find("{"), reply.rfind("}")
    try:
        return json.loads(reply[start:end + 1], strict=False)   # strict=False allows line breaks inside text
    except (json.JSONDecodeError, ValueError):
        return None


def load_profile():
    """The profile remembers the student's score for each topic: {topic: {"score": 0-10, "feedback": "..."}}"""
    if os.path.exists(PROFILE_FILE):
        try:
            with open(PROFILE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_profile(profile):
    with open(PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n(Saved to {path})")


# ---------- Phase 4: assess the student ----------

def read_answer():
    """Reads a multi-line answer. Returns (status, text) where status is 'answer', 'skip' or 'stop'."""
    lines = []
    while True:
        line = input()
        word = line.strip().lower()
        if not lines and word in ("skip", "stop"):
            return word, ""
        if word == "":
            if lines:
                return "answer", "\n".join(lines)
            continue
        lines.append(line)


def grade_answer(question, answer):
    prompt = f"""You are a fair university examiner. Grade the student's answer to the exam question below.

Question: {question}

Student's answer:
{answer}

Give a score from 0 to 10 (10 = complete and correct, 0 = wrong or empty) and 1-2 sentences of feedback saying what was good and what was missing.
Respond ONLY with JSON in this exact format: {{"score": 7, "feedback": "..."}}"""

    reply = ask_ai(prompt, temperature=0)
    if reply is None:
        return None
    data = extract_json(reply)
    if not data:
        return None
    try:
        score = max(0, min(10, int(data.get("score", 0))))
    except (TypeError, ValueError):
        return None
    return {"score": score, "feedback": str(data.get("feedback", "")).strip()}


def pick_question(segregated, topic):
    """The highest-marks real question from that topic."""
    options = [q for q in segregated if q["topic"] == topic]
    return max(options, key=lambda q: q["marks"]) if options else None


def run_assessment(segregated, ranked, profile):
    topics = [t for t, _ in ranked if t != "Unclassified"]

    if profile:
        again = input("You already have saved results. Re-test every topic? (y/n): ").strip().lower()
        if again != "y":
            topics = [t for t in topics if t not in profile]
            if not topics:
                print("Every topic is already tested. Choose 'y' next time to retake them.")
                return

    print(f"\nI'll ask you one real exam question from each of {len(topics)} topic(s), most important first.")
    print("Answer from memory like you would in the exam.")

    for i, topic in enumerate(topics, 1):
        q = pick_question(segregated, topic)
        print(f"\n--- Topic {i}/{len(topics)}: {topic} ---")
        print(f"Question ({q['marks']} marks): {q['question']}")
        print("Type your answer, then press Enter on an empty line when you're done.")
        print("(Type 'skip' if you don't know it, or 'stop' to finish for now.)")

        status, answer = read_answer()
        if status == "stop":
            break
        if status == "skip":
            profile[topic] = {"score": 0, "feedback": "The student skipped this question (doesn't know it yet)."}
            print("Skipped. I'll count this as a topic to work on.")
        else:
            result = grade_answer(q["question"], answer)
            if result is None:
                print("I couldn't grade that one, moving on.")
                continue
            profile[topic] = result
            print(f"Score: {result['score']}/10. {result['feedback']}")
        save_profile(profile)

    print("\nAssessment saved. Your results are used for the assignment and the study plan.")


def priority_list(ranked, profile):
    """Ranks topics by (how important they are) x (how weak the student is)."""
    rows_in = [(t, s) for t, s in ranked if t != "Unclassified"]
    use_marks = sum(s["total_marks"] for _, s in rows_in) > 0
    value = (lambda s: s["total_marks"]) if use_marks else (lambda s: s["count"])
    total = sum(value(s) for _, s in rows_in) or 1

    rows = []
    for topic, s in rows_in:
        weight = value(s) / total
        entry = profile.get(topic)
        mastery = entry["score"] / 10 if entry else None
        gap = 1 - mastery if mastery is not None else 0.5   # untested = assume half-known
        rows.append({
            "topic": topic,
            "weight": weight,
            "mastery": mastery,
            "priority": weight * gap,
            "feedback": entry["feedback"] if entry else "",
        })
    rows.sort(key=lambda r: r["priority"], reverse=True)
    return rows


# ---------- Phase 5: generate sample papers and assignments ----------

def build_plan(segregated, ranked, profile, mode):
    """Decides how many questions per topic and how many marks each, copying the real pattern."""
    num_papers = max(1, len({q["paper"] for q in segregated}))
    mastery_of = {r["topic"]: r["mastery"] for r in priority_list(ranked, profile)}

    plan = []
    for topic, s in ranked:
        if topic == "Unclassified":
            continue
        count = max(1, round(s["count"] / num_papers))   # questions per paper, on average

        if mode == "assignment":                         # more practice on weak topics
            m = mastery_of[topic]
            count += 1 if m is None else (2 if m < 0.4 else 1 if m < 0.7 else 0)

        marks_seen = [q["marks"] for q in segregated if q["topic"] == topic and q["marks"] > 0]
        marks_each = Counter(marks_seen).most_common(1)[0][0] if marks_seen else 5
        examples = [q["question"] for q in segregated if q["topic"] == topic][:3]
        plan.append({"topic": topic, "count": count, "marks": marks_each, "examples": examples})
    return plan


def generate_practice(segregated, ranked, profile, mode):
    plan = build_plan(segregated, ranked, profile, mode)
    total = sum(p["count"] * p["marks"] for p in plan)

    spec_lines = []
    for p in plan:
        examples = "\n".join(f"    * {e}" for e in p["examples"])
        spec_lines.append(f"- {p['topic']}: write {p['count']} new question(s), each worth {p['marks']} marks. "
                          f"Real examples of the style:\n{examples}")
    spec = "\n".join(spec_lines)

    if mode == "paper":
        title, path = "Sample Exam Paper", SAMPLE_PAPER_FILE
        layout = ("Number the questions Q1, Q2, ... continuously, with the marks in brackets at the end of each "
                  "question, like (5 marks). Do not add topic headings; mix the topics the way a real exam paper would.")
    else:
        title, path = "Practice Assignment", ASSIGNMENT_FILE
        layout = ("Group the questions under a heading for each topic, number them Q1, Q2, ... continuously, "
                  "with the marks in brackets at the end of each question.")
        if not profile:
            print("Tip: take the assessment first (option 2) and the assignment will focus on your weak topics.")

    prompt = f"""You are an experienced university examiner. Write a {title} that follows the pattern of real previous-year papers.

Requirements per topic (number of questions, marks each, and real examples of the wording and difficulty):
{spec}

Rules:
- Write NEW questions in the same style, wording and difficulty as the examples. Do not copy the examples.
- Use exactly the number of questions and marks listed for each topic (total {total} marks).
- {layout}
- Output only the {title.lower()}, starting with a title line. Nothing else."""

    print(f"\nWriting a {title.lower()} ({total} marks) ...")
    text = ask_ai(prompt, temperature=0.7)
    if text is None:
        return
    print("\n" + text)
    save_text(path, text)


# ---------- Phase 6: the counselor ----------

def ask_number(question, default, low, high):
    raw = input(f"{question} (press Enter for {default}): ").strip()
    if raw == "":
        return default
    try:
        return max(low, min(high, int(raw)))
    except ValueError:
        print(f"That wasn't a number, using {default}.")
        return default


def make_study_plan(segregated, ranked, profile):
    days = ask_number("How many days do you have before the exam?", 7, 1, 60)
    hours = ask_number("How many hours can you study each day?", 2, 1, 12)

    rows = priority_list(ranked, profile)
    topic_lines = []
    for i, r in enumerate(rows, 1):
        mastery = "not tested yet" if r["mastery"] is None else f"scored {round(r['mastery'] * 100)}% on the assessment"
        real = sorted((q for q in segregated if q["topic"] == r["topic"]), key=lambda q: q["marks"], reverse=True)[:2]
        examples = "; ".join(q["question"] for q in real)
        feedback = f" Examiner note: {r['feedback']}" if r["feedback"] else ""
        topic_lines.append(f"{i}. {r['topic']}: {round(r['weight'] * 100)}% of past exam marks, {mastery}.{feedback} "
                           f"Real exam questions: {examples}")
    topics_text = "\n".join(topic_lines)

    if profile:
        assessed = "The student has taken the assessment, so use their scores."
    else:
        assessed = ("The student has NOT taken the assessment, so base the plan on exam importance only, "
                    "say so clearly, and recommend taking the assessment for a better plan.")

    prompt = f"""You are a warm, practical personal tutor and study counselor talking directly to a student ("you").

The student has {days} day(s) before the exam and can study {hours} hour(s) per day.
{assessed}

Topics in priority order (highest priority first; priority = how much exam weight x how weak the student is):
{topics_text}

Write the student's study plan with these parts:
1. Where you stand: 3-4 honest, encouraging sentences.
2. Day-by-day plan for all {days} day(s): which topics, roughly how many of the {hours} hours on each, exactly what to study (use the real exam questions above to name the sub-topics), and how to study it (for example: write the algorithm by hand, solve this type of question, explain it aloud).
3. What to give less time to, and why.
4. One tip for the last day before the exam.

Rules: only use the topics and information above, do not invent syllabus details. Spend more time on higher-priority topics. Keep it clear and friendly."""

    print("\nThinking about your plan ...")
    text = ask_ai(prompt, temperature=0.5)
    if text is None:
        return
    print("\n" + text)
    save_text(STUDY_PLAN_FILE, text)


# ---------- menu ----------

def main():
    papers = load_papers()
    segregated = analyze(papers, SYLLABUS_TOPICS)
    if not segregated:
        print("No questions could be read from the papers, so there is nothing to work with yet. "
              "Check your files and try again.")
        return

    ranked = score_topics(segregated)
    profile = load_profile()
    print(f"\nReady: {len(segregated)} questions from {len(papers)} paper(s).")

    while True:
        print("\nWhat would you like to do?")
        print("  1. See which topics matter most")
        print("  2. Take the assessment (find out what you know)")
        print("  3. Generate a sample exam paper")
        print("  4. Generate a practice assignment")
        print("  5. Get my personal study plan")
        print("  0. Quit")
        choice = input("Choose a number: ").strip()

        if choice == "1":
            print_ranking(ranked, SYLLABUS_TOPICS)
        elif choice == "2":
            run_assessment(segregated, ranked, profile)
        elif choice == "3":
            generate_practice(segregated, ranked, profile, "paper")
        elif choice == "4":
            generate_practice(segregated, ranked, profile, "assignment")
        elif choice == "5":
            make_study_plan(segregated, ranked, profile)
        elif choice == "0":
            print("Good luck with your exams!")
            break
        else:
            print("Please type one of the numbers from the menu.")


if __name__ == "__main__":
    main()