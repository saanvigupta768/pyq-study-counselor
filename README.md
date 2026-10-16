# PYQ Study Counselor

An AI study counselor for exam preparation. Give it previous-year papers (PYQs) and it works out which topics matter most, finds out what you already know, writes practice papers in the same pattern as your real exam, and builds a study plan around your weak spots.

Built from scratch in Python using the Groq API (free tier) while learning how LLM-based agents work.

## What it does

- **Reads papers in any form:** PDFs with selectable text, text files, and photos of papers (png / jpg / webp), any number of them
- **Sorts every question into a syllabus topic** and extracts its marks, using an LLM
- **Ranks topics by importance:** total marks, with frequency as the tie-break, plus how many papers each topic appeared in
- **Assesses the student:** asks one real exam question per topic, grades the answer out of 10 with feedback, and remembers the results
- **Generates a sample exam paper** that copies the real pattern: same number of questions per topic and same marks
- **Generates a practice assignment** with extra questions on the topics you are weakest in
- **Builds a personal study plan** for the number of days and hours you have, ordered by exam importance x how weak you are, naming exactly what to study and how
- Works for any subject: put your topics in `syllabus.txt`

## Run it

```
pip install groq python-dotenv pypdf mem0ai
```

Create a `.env` file in the project folder:

```
GROQ_API_KEY=your_key_here
MEM0_API_KEY=your_key_here
```

- `GROQ_API_KEY` is free from console.groq.com and is needed for everything.
- `MEM0_API_KEY` is free from app.mem0.ai and is only needed for `agent.py`.
- Optional: `VISION_MODEL=...` if photo reading stops working (Groq renames models often, see console.groq.com/docs/vision).

Create a folder called `pyqs` next to the scripts and put your papers in it. Optionally create `syllabus.txt` with one topic per line (without it, the Data Structures topics are used). Then start the counselor:

```
python counselor.py
```

```
What would you like to do?
  1. See which topics matter most
  2. Take the assessment (find out what you know)
  3. Generate a sample exam paper
  4. Generate a practice assignment
  5. Get my personal study plan
  0. Quit
```

If the `pyqs` folder is empty or missing, it runs on the demo file `sample_pyqs.txt`. To see only the topic ranking, run `python pyq_analyzer.py`.

Your results are saved locally (`student_profile.json`, `sample_paper.txt`, `assignment.txt`, `study_plan.txt`) and are not uploaded to GitHub.

## Status

Done:
- [x] Topic segregation and importance ranking
- [x] PDF, text and photo upload (multiple papers)
- [x] Student assessment
- [x] Sample paper and assignment generation
- [x] Personalized study plan

Planned next:
- [ ] A simple web interface
- [ ] Answer keys for the generated papers

Known limitations:
- Topic labels and answer grading come from an LLM, so they are good estimates, not exact. Messy input can be mislabeled.
- Photo reading depends on a Groq vision model; images over 4 MB are skipped.
- A scanned PDF with no selectable text is skipped; save its pages as images instead.
- Answers are typed in the terminal, so the assessment suits short written answers.
- Free-tier rate limits apply; if a request fails, wait a minute and retry.

## Files

- `counselor.py`: the main program (menu, assessment, practice papers, study plan)
- `pyq_analyzer.py`: reads papers, sorts questions into topics, ranks topics
- `agent.py`: a chat agent with persistent memory (Mem0), my first step before this project
- `test_connection.py`: checks the Groq connection
- `sample_pyqs.txt`: demo questions