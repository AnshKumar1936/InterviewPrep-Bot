import os
from typing import Generator

from dotenv import load_dotenv
from groq import Groq


def get_groq_client() -> Groq:
	"""Create and return a Groq client using env var GROQ_API_KEY or Streamlit secrets.

	Order:
	1) Load from .env / environment (preferred locally)
	2) If not set, read Streamlit secrets only if a secrets.toml exists
	"""
	# 1) Environment / .env
	load_dotenv(override=False)
	api_key = os.getenv("GROQ_API_KEY")

	# 2) Streamlit secrets (only if a secrets file exists)
	if not api_key:
		project_secrets = os.path.join(os.getcwd(), ".streamlit", "secrets.toml")
		home_secrets = os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml")
		if os.path.exists(project_secrets) or os.path.exists(home_secrets):
			try:
				import streamlit as st  # type: ignore
				api_key = st.secrets.get("GROQ_API_KEY")  # pylint: disable=no-member
			except Exception:
				api_key = None

	if not api_key:
		raise RuntimeError(
			"GROQ_API_KEY is not set. Provide it via .env locally or Streamlit secrets in the cloud."
		)
	return Groq(api_key=api_key)


def stream_chat_completion(
	model: str,
	system_prompt: str,
	user_prompt: str,
	temperature: float = 0.4,
	max_tokens: int = 1200,
) -> Generator[str, None, None]:
	"""Yield chunks of generated text from Groq chat completions API."""
	client = get_groq_client()
	stream = client.chat.completions.create(
		model=model,
		messages=[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		],
		temperature=temperature,
		max_tokens=max_tokens,
		stream=True,
	)
	for event in stream:
		for choice in getattr(event, "choices", []) or []:
			delta = getattr(choice, "delta", None)
			if delta and getattr(delta, "content", None):
				yield delta.content
