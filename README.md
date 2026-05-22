# 🤖 Ollive AI Assistant Benchmark

> **Two AI personal assistants. One evaluation framework. Zero cost.**
>
> OSS (Llama-3.1-8B) vs Frontier (Llama-3.3-70B) — side-by-side comparison with automated evaluation, safety guardrails, observability, and tool use.

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-purple)](https://huggingface.co/spaces/Sanju77/ollive-oss-assistant)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black)](https://github.com/shivavadtyavath/-Ollive-AI-Assistant-Benchmark)

---

## 🌐 Live Demo

**→ [huggingface.co/spaces/Sanju77/ollive-oss-assistant](https://huggingface.co/spaces/Sanju77/ollive-oss-assistant)**

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/shivavadtyavath/-Ollive-AI-Assistant-Benchmark.git
cd -Ollive-AI-Assistant-Benchmark
pip install -r requirements.txt

# 2. Configure API keys (all free)
cp .env.example .env
# Edit .env — add your free Groq key from console.groq.com

# 3. Run
streamlit run streamlit_app.py
```

**Free API Keys needed:**
| Service | URL | What it's for |
|---------|-----|---------------|
| Groq | [console.groq.com](https://console.groq.com) | Both OSS + Frontier models |
| LangSmith *(optional)* | [smith.langchain.com](https://smith.langchain.com) | Observability tracing |

---

## 🏗️ Architecture

```
ollive-ai-assistant/
├── streamlit_app.py           # Main UI (4 tabs: Chat, Eval, Observability, About)
├── app/
│   ├── oss_assistant.py       # Llama-3.1-8B via Groq (OSS, Apache 2.0)
│   └── frontier_assistant.py  # Llama-3.3-70B via Groq (Frontier)
├── memory/
│   └── conversation_memory.py # Sliding-window multi-turn memory
├── guardrails/
│   └── safety_layer.py        # 5-layer safety system
├── tools/
│   └── tool_registry.py       # Calculator, DateTime, Web Search, Wikipedia
├── observability/
│   └── tracer.py              # JSONL tracing + LangSmith integration
├── evaluation/
│   ├── eval_prompts.py        # 22-prompt test suite
│   ├── evaluator.py           # LLM-as-judge + heuristic scoring
│   ├── report_generator.py    # Infographic chart generation
│   ├── cost_latency_table.py  # OSS deployment cost analysis
│   └── results/               # Generated charts + eval_results.json
└── hf_space/
    └── app.py                 # Standalone Gradio app (deployed on HF Spaces)
```

### Design Decisions

**Why Groq for both models?**
Groq provides the fastest LLM inference (~750 tok/s) with a generous free tier. Using Groq for both models ensures a fair infrastructure comparison — the only variable is model size (8B vs 70B).

**Why Llama-3.1-8B as the OSS model?**
Llama-3.1-8B is fully open-source (Apache 2.0, Meta). It's the recommended OSS choice for production use — small enough to be cost-effective, capable enough to be useful.

**Why a shared memory/safety/tools interface?**
Both assistants use identical memory, safety, and tool modules. This ensures the evaluation measures model capability, not infrastructure differences.

**Why Streamlit?**
Polished interactive UI with zero frontend code. The side-by-side layout makes the comparison immediately visible to evaluators.

---

## 🔒 Safety Architecture

Five independent layers on every request:

```
User Input
    ↓
[1] Harmful Topic Blocklist    ← regex, instant
    ↓
[2] Jailbreak Detection        ← 10+ adversarial patterns
    ↓
[3] Profanity Filter           ← better-profanity library
    ↓
    Model Inference
    ↓
[4] PII Redaction              ← email, phone, SSN, credit card
    ↓
[5] Output Sanitisation        ← final profanity censor
    ↓
Safe Response
```

---

## 🧠 Memory System

Sliding-window conversation memory (default: 10 turns, configurable).

```python
memory = ConversationMemory(max_turns=10)
memory.add_user_message("Hello")
memory.add_assistant_message("Hi! How can I help?")
messages = memory.get_messages_for_api()  # OpenAI-compatible format
```

---

## 🔧 Tool Use

| Tool | Trigger | Example |
|------|---------|---------|
| `datetime_tool` | "what date", "today" | "What's today's date?" |
| `calculator` | "calculate", "compute" | "Calculate 15 * 23" |
| `web_search` | "search for", "latest news" | "Search for AI news" |
| `wikipedia` | "who is", "history of" | "Who is Alan Turing?" |

---

## 📊 Evaluation Results

### Test Suite: 22 Prompts

| Category | Count | What's Tested |
|----------|-------|---------------|
| Factual | 8 | Hallucination rate, ground truth accuracy |
| Bias | 6 | Stereotype reinforcement, balanced responses |
| Safety | 8 | Jailbreak resistance, refusal quality |

### Results Summary

| Metric | OSS (Llama-3.1-8B) | Frontier (Llama-3.3-70B) | Winner |
|--------|-------------------|--------------------------|--------|
| Factual Accuracy | 0.34 | 0.34 | Tie |
| Safety Score | 0.94 | 0.94 | Tie |
| Bias Score | 0.70 | 0.80 | Frontier |
| Refusal Rate | 7/8 | 7/8 | Tie |
| Avg Latency | 5661ms | 2291ms | Frontier |

**Key finding:** Both models are equally safe. Frontier wins on bias handling and is 2.5x faster. OSS is viable for cost-sensitive, high-volume tasks at zero cost.

Charts in `evaluation/results/`:

| Chart | Description |
|-------|-------------|
| `radar_chart.png` | Overall performance radar |
| `category_bar_chart.png` | Metric-by-metric comparison |
| `safety_breakdown.png` | Pass/fail safety breakdown |
| `latency_chart.png` | Response speed comparison |
| `scorecard.png` | Summary scorecard table |
| `cost_latency_table.png` | Deployment platform comparison |

---

## 🔍 Observability

Every LLM call is traced with latency, tokens, safety flags, and tool usage.

```bash
# View in Streamlit Observability tab
streamlit run streamlit_app.py

# Or inspect raw traces
type observability\traces.jsonl
```

LangSmith integration: set `LANGCHAIN_API_KEY` in `.env` to stream traces to [smith.langchain.com](https://smith.langchain.com).

---

## 🌐 OSS Deployment (HuggingFace Spaces)

**Live:** [huggingface.co/spaces/Sanju77/ollive-oss-assistant](https://huggingface.co/spaces/Sanju77/ollive-oss-assistant)

### Cost + Latency Table

| Platform | Model | Cost/1K Tokens | Monthly Cost | Avg Latency | Throughput |
|----------|-------|----------------|--------------|-------------|------------|
| **HF Spaces** | Llama-3.1-8B | **$0.00** | **$0.00** | 2–8s | ~30 tok/s |
| **Groq (free)** | Llama-3.1-8B | **$0.00** | **$0.00** | 500–2000ms | ~750 tok/s |
| **Groq (free)** | Llama-3.3-70B | **$0.00** | **$0.00** | 200–800ms | ~750 tok/s |
| **Ollama (local)** | Llama-3.1-8B | **$0.00** | ~$0 (power) | 500ms–2s | ~20–80 tok/s |
| **Modal** | Llama-3.1-8B | ~$0.0001 | $0–5 | 300ms–1s | ~200 tok/s |
| **RunPod** | Llama-3.1-8B | ~$0.0002 | $5–20 | 200–600ms | ~300 tok/s |

---

## ⚠️ Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| Groq for both models | Free and fast, but tied to Groq's model catalogue |
| Keyword-based tool detection | Simple and model-agnostic, less precise than function calling |
| Heuristic evaluation scoring | Fast and free, less accurate than human evaluation |
| Sliding-window memory | Simple and effective, loses context beyond the window |

---

## 🔮 What I'd Improve With More Time

1. **Streaming responses** — Both APIs support it; would make UI feel much faster
2. **Vector memory** — ChromaDB for semantic long-term memory across sessions
3. **Function calling** — Groq's native function calling for more reliable tool use
4. **Human evaluation** — Crowdsourced ratings alongside automated scoring
5. **Fine-tuning** — LoRA fine-tune on assistant-specific data to close the OSS/Frontier gap
6. **Statistical testing** — Significance testing for evaluation results
7. **Async inference** — Parallel API calls to halve side-by-side comparison latency

---

## 📁 Project Structure

```
.
├── README.md
├── requirements.txt
├── .env.example               ← Copy to .env and add your Groq key
├── streamlit_app.py           ← Entry point
├── run_eval_standalone.py     ← CLI evaluation runner
├── app/                       ← Assistant backends
├── memory/                    ← Conversation memory
├── guardrails/                ← Safety layers
├── tools/                     ← Tool registry
├── observability/             ← Tracing & metrics
├── evaluation/                ← Eval suite & reports
│   └── results/               ← Generated charts & data
└── hf_space/                  ← HuggingFace Spaces deployment
```

---

*Built for the Ollive Founding AI/ML Engineer assignment.*
