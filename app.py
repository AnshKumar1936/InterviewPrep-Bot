import streamlit as st
from typing import List

from groq_client import stream_chat_completion


APP_TITLE = "IntPrep ChatBot"
TAGLINE = "You fully interview preparation tool"
MODEL_CANDIDATES = [
	"llama-3.2-90b-text-preview",
	"llama-3.2-11b-text-preview",
	"llama3-70b-8192",
	"llama3-8b-8192",
	"mixtral-8x7b-32768",
	"gemma2-9b-it",
	"gemma-7b-it",
]
DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_TOKENS = 1200


def build_system_prompt() -> str:
	return (
		"You are an expert technical interviewer and career coach. "
		"Generate tailored interview questions and model answers. "
		"Vary difficulty and question styles as requested. Be concise, structured, and actionable."
	)


def build_user_prompt(
	role: str,
	seniority: str,
	domain: str,
	num_questions: int,
	difficulty: str,
	styles: List[str],
	include_rubrics: bool,
) -> str:
	style_line = ", ".join(styles) if styles else "mixed"
	rubric_line = (
		"Include brief evaluation rubrics for each question." if include_rubrics else ""
	)
	return f"""
Role: {role}
Seniority: {seniority}
Domain/Stack Focus: {domain}
Number of Questions: {num_questions}
Difficulty: {difficulty}
Question Styles: {style_line}

Instructions:
- Produce a numbered list of {num_questions} interview questions.
- For each question, include a high-quality model answer.
- Keep each Q&A concise but substantive; avoid fluff.
- If coding, include short code snippets and complexity where relevant.
- Prefer realistic, practical scenarios.
- {rubric_line}
- End with 3 short follow-up questions the interviewer might ask.
""".strip()


def main() -> None:
	st.set_page_config(page_title=APP_TITLE, page_icon="💼", layout="wide")
	st.title(APP_TITLE)
	st.caption(TAGLINE)

	col1, col2 = st.columns(2)
	with col1:
		role = st.text_input("Role", placeholder="e.g. Machine Learning Engineer")
		seniority = st.selectbox(
			"Seniority",
			["intern", "fresher", "Entry level", "Mid", "senior"],
			index=2,
		)
		domain = st.text_input("Domain/Stack", placeholder="e.g. NLP, MLOps, Backend, React, AWS")
	with col2:
		num_questions = st.slider("Number of questions", 3, 15, 8)
		difficulty = st.select_slider(
			"Overall difficulty",
			options=["Easy", "Medium", "Hard", "Mixed"],
			value="Mixed",
		)
		styles = st.multiselect(
			"Question styles",
			["Behavioral", "System Design", "Algorithms", "Coding", "Theory", "Metrics/Analytics"],
			default=["Behavioral", "Coding", "System Design"],
		)
		include_rubrics = st.checkbox("Include brief evaluation rubrics", value=True)

	generate = st.button("Generate Q&A", type="primary")

	if generate:
		if not role:
			st.warning("Please enter a role to continue.")
			st.stop()

		with st.status("Finding a supported model and generating...", expanded=False) as status:
			system_prompt = build_system_prompt()
			user_prompt = build_user_prompt(
				role, seniority, domain, int(num_questions), difficulty, styles, include_rubrics
			)

			placeholder = st.empty()
			buffer = []
			last_error = None
			used_model = None

			for m in MODEL_CANDIDATES:
				try:
					for chunk in stream_chat_completion(
						model=m,
						system_prompt=system_prompt,
						user_prompt=user_prompt,
						temperature=DEFAULT_TEMPERATURE,
						max_tokens=int(DEFAULT_MAX_TOKENS),
					):
						if used_model is None:
							used_model = m
							status.update(label=f"Model: {used_model}")
						buffer.append(chunk)
						placeholder.markdown("".join(buffer))
					break
				except Exception as e:
					last_error = e
					continue

			if used_model is None:
				msg = str(last_error) if last_error else "No supported model worked."
				st.error(
					"All tried models failed or are unsupported. "
					"Please update the internal model list or check the Groq model docs (https://console.groq.com/docs/models).\n\n"
					f"Last error: {msg}"
				)
				st.stop()

		st.success("Done")


if __name__ == "__main__":
	main()
