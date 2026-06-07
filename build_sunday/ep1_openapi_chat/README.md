# Build Sunday (Episode 1): Production-Ready Smart Q&A CLI Bot

Welcome to **Build Sunday**! This project is a production-style, CLI-based Smart Q&A Bot built using the OpenAI API, Python 3.11, and `asyncio`. 

It is designed to demonstrate key software engineering and production AI architecture concepts in a clean, modular, and easy-to-explain format suitable for developers of all skill levels.

---

## 🛠️ Tech Stack & Key Libraries

- **Python 3.11**: Modern Python with native support for structured asynchronous concurrency (`asyncio`).
- **OpenAI SDK**: The official OpenAI Client library for chat completions.
- **Rich**: A library for rich text and beautiful formatting in terminal interfaces (provides our live-stream layouts, status panels, and statistics tables).
- **Tiktoken**: OpenAI’s open-source fast BPE tokenizer used to count prompt/completion tokens locally.
- **python-dotenv**: Loads variables from `.env` files into environment variables.
- **Pytest**: A testing framework for unit and integration testing.

---

## 🏗️ Architecture Design & Module Responsibilities

The project follows the **Separation of Concerns (SoC)** principle, separating API integration, memory management, configuration, token tracking, and the CLI presenter.

```
EP_1_SMARTQ&A/
├── src/
│   ├── __init__.py
│   ├── config.py         # Loads and validates environment configurations.
│   ├── utils.py          # Backoff retries and general utility methods.
│   ├── token_counter.py  # Local token counting (tiktoken) and billing estimations.
│   ├── conversation.py   # State management (LLM memory) and token-based history pruning.
│   ├── api_client.py     # Asynchronous OpenAI client with error handling.
│   └── main.py           # CLI presentation layer & event loop.
├── tests/
│   ├── __init__.py
│   └── test_bot.py       # Unit test suite verifying components.
├── .env.example          # Environment variable template.
├── requirements.txt      # Python package dependencies.
└── README.md             # This file.
```

### Module Breakdown

1. **`config.py`**: Centralizes application parameters. Validates that critical keys (like `OPENAI_API_KEY`) are loaded securely, refusing to run if placeholders are found.
2. **`utils.py`**: Provides reusable logic. Specifically contains our custom async retry logic.
3. **`token_counter.py`**: Encapsulates all Byte Pair Encoding (BPE) calculations using `tiktoken`. It computes input/output tokens and translates them into cumulative dollar cost based on model-specific pricing sheets.
4. **`conversation.py`**: Simulates stateful conversation memory over stateless LLMs. Implements a rolling-window truncation algorithm that trims older exchanges to prevent context overflow while keeping the base system prompt intact.
5. **`api_client.py`**: Serves as the AI client. Implements a streaming interface with a generator-safe exponential backoff retry loop.
6. **`main.py`**: The application driver. Runs the interactive CLI loop, captures text token flows live, and formats statistics tables via `rich`.

---

## ⚡ Core Production Engineering Patterns Implemented

### 1. Resiliency (Exponential Backoff with Jitter)
Distributed systems frequently experience transient failures (network drops, rate limits `HTTP 429`, or server overload `HTTP 5xx`). We implement:
- **Exponential Backoff**: Multiplying the wait time between retry attempts (e.g. 2s, 4s, 8s) to allow downstream API servers to recover.
- **Jitter**: Introducing random variance to the backoff delays. This prevents multiple client instances from synchronizing their retry cycles (the *Thundering Herd Problem*).
- **Selective Retries**: Checking exception types to retry on transient errors (timeouts, rate limits, 5xx) but fail immediately on developer errors (401 invalid key, 400 bad request).

### 2. Generator-Safe Stream Retries
Standard decorators fail when wrapping Python async generators (`yield`). The exception is raised during stream iteration, not when the generator object is created. We solve this by implementing a generator-safe retry loop inside the stream consumer loop.

### 3. State Management & Rolling Truncation
Because LLMs are stateless, we pass the entire transcription history back with every call. To prevent running out of context tokens, `conversation.py` tracks history size and slices out the oldest query-response pair if limits are breached, preserving the original system instructions.

---

## 🚀 Setup & Installation Instructions

### Prerequisite: Python 3.11
Ensure you have Python 3.11 installed. You can check your version using:
```bash
python --version
```

### Step 1: Clone the project & Navigate to Directory
```bash
cd EP_1_SMARTQ&A
```

### Step 2: Create and Activate a Virtual Environment
```bash
# Create the virtual environment
python3 -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate

# Activate on Windows (CMD):
# venv\Scripts\activate.bat
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
1. Copy the template configuration file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` in your text editor and insert your OpenAI API Key:
   ```env
   OPENAI_API_KEY=sk-proj-YOUR_API_KEY_HERE
   ```

---

## 🏃 Running the Application

Start the interactive CLI session:
```bash
python3 -m src.main
```

### In-App Controls
- Type any question and hit **Enter** to see streaming output.
- Type `clear` to reset conversation context memory.
- Type `exit` or `quit` to exit, or press `Ctrl+C` / `Ctrl+D` at the input prompt.
- Press `Ctrl+C` while the assistant is speaking to immediately stop/truncate the output stream and start a new prompt.

---

## 🧪 Running Automated Tests

We use `pytest` for testing our logic modules. To execute the test suite:
```bash
PYTHONPATH=. pytest tests/
```
