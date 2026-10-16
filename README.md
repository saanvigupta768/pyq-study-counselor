# PYQ Study Counselor

An AI tool that analyzes previous-year exam questions (PYQs) and tells a student which topics matter most. The long-term goal is a personal tutor that also generates practice papers and a study plan.

Built from scratch in Python using the Groq API (free tier) while learning how LLM-based agents work.

## What works today

- Reads exam questions from a text file
- Sorts each question into a syllabus topic and extracts its marks, using an LLM
- Ranks topics by total marks, using frequency to break ties
- Lists syllabus topics that never appeared in the papers

Example output (from `sample_pyqs.txt`):

```
Sorting and Searching: appeared 3 times, 16 total marks
Linked Lists: appeared 2 times, 10 total marks
Stacks and Queues: appeared 2 times, 8 total marks
...
```

## Status: work in progress

Planned next:
- [ ] PDF and image upload (OCR) instead of a text file
- [ ] Assess what the student already knows
- [ ] Generate sample papers that match the real exam pattern
- [ ] Personalized study plan ("what to study first, and why")

Known limitation: topic labels come from an LLM, so messy input can be mislabeled.

## Run it

```
pip install groq python-dotenv
```

Create a `.env` file containing `GROQ_API_KEY=your_key_here` (free key from console.groq.com), then run:

```
python pyq_analyzer.py
```

It reads `pyqs.txt` if you create one, otherwise it uses the demo file `sample_pyqs.txt`.

## Files

- `pyq_analyzer.py`: topic segregation and importance ranking
- `agent.py`: a chat agent with persistent memory (Mem0), my first step before this project
- `test_connection.py`: checks the Groq connection