# ManimTired

An LLM-powered Manim video generator.

## Setup

```bash
$ .venv\Scripts\activate
$ uv pip install -r pyproject.toml
```

## Usage

```bash
$ uv run main.py --llm "gpt4o" # opens dearpygui app
```

## LLM
We'll be using `llama-3.3-70b` hosted on Cerebras or `gpt-4o` hosted on OpenAI. API key is stored in `.env` as `CEREBRAS_API_KEY` or `OPENAI_API_KEY`. Use langchain to interface with the LLM.

## Features

- [ ] Generate Manim code from a prompt
- [ ] Render Manim code to video
- [ ] Display thumbnail
- [ ] Play video in desktop UI with media controls
- [ ] Audio narration
- [ ] Agent architecture: planning agent, scripting agent, voice script writing agent, voice agent