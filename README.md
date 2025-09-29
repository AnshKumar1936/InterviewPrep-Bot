## IntPrep ChatBot

A Streamlit application that generates tailored interview questions and model answers for a given role and seniority. It uses Groq’s hosted LLMs to create concise, realistic Q&A, optionally including brief rubrics and follow-up questions.

### What it does
- Takes role, seniority, domain/stack, preferred styles, and number of questions
- Generates numbered interview questions with strong model answers
- Streams output for fast feedback
- Automatically tries multiple Groq models if a selected one is deprecated or unavailable

### Models used
- `gemma2-9b-it`
- `gemma-7b-it`


### Prerequisites
- Python 3.9+
- A Groq API key

### Setup and run (Windows)
1) Create a virtual environment and activate it:
```bat
python -m venv venv
venv\Scripts\activate
```

2) Install requirements:
```bat
pip install -r requirements.txt
```

3) Provide your Groq API key (local dev):
- Create a file named `.env` in the project root with:
```env
GROQ_API_KEY=YOUR_REAL_KEY
```

4) Run the app:
```bat
streamlit run app.py
```

