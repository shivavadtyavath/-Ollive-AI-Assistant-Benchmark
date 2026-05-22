# 🤖 Ollive AI Assistant Benchmark

> **Two AI personal assistants. One evaluation framework. Zero cost.**
>
> OSS (Qwen2.5-0.5B-Instruct) vs Frontier (Llama-3.3-70B via Groq) — side-by-side comparison with automated evaluation, safety guardrails, observability, and tool use.

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone <your-repo-url>
cd ollive-ai-assistant
pip install -r requirements.txt

# 2. Configure API keys (all free)
cp .env.example .env
# Edit .env with your keys

# 3. Run
streamlit run streamlit_app.py
```

**Free API Keys needed:**
| Service | URL | What it's for |
|---------|-----|---------------|
| Groq | [console.groq.com](https://console.groq.com) | Frontier model (Llama-3.3-70B) |
| HuggingFace | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | OSS model (Qwen2.5) |
| LangSmith *(optional)* | [smith.langchain.com](https://smith.langchain.com) | Observability tracing |

---

## 🏗️ Architecture

```
ollive-ai-assistant/
├── streamlit_app.py          # Main UI (4 tabs: Chat, Eval, Observability, About)
├── app/
│   ├── oss_assistant.py      # Qwen2.5-0.5B via HF Inference API
│   └── frontier_assistant.py # Llama-3.3-70B via Groq API
├── memory/
│   └── conversation_memory.py # Sliding-window multi-turn memory
├── guardrails/
│   └── safety_layer.py       # 5-layer safety system
├── tools/
│   └── tool_registry.py      # Web search, Wikipedia, Calculator, DateTime
├── observability/
│   └── tracer.py             # JSONL tracing + LangSmith integration
├── evaluation/
│   ├── eval_prompts.py       # 22-prompt test suite
│   ├── evaluator.py          # LLM-as-judge + heuristic scoring
│   ├── report_generator.py   # Infographic chart generation
│   └── cost_latency_table.py # OSS deployment cost analysis
└── hf_space/
    ├── app.py                # Standalone Gradio app for HF Spaces
    └── requirements.txt
```

### Design Decisions

**Why Groq for the frontier model?**
Groq provides the fastest LLM inference available (~750 tokens/sec) with a generous free tier. Llama-3.3-70B on Groq outperforms many paid APIs in both speed and quality, making it the ideal frontier comparison point for a student project.

**Why HuggingFace Inference API for OSS?**
No local GPU required. The free tier supports Qwen2.5-0.5B-Instruct with reasonable rate limits. The same model is deployed on HF Spaces for public access.

**Why a shared memory/safety/tools interface?**
Both assistants use identical memory, safety, and tool modules. This ensures the evaluation is a fair comparison of the *models*, not the infrastructure.

**Why Streamlit over FastAPI?**
Streamlit gives a polished, interactive UI with zero frontend code. The side-by-side layout makes the comparison immediately visible. FastAPI would be better for production, but Streamlit is optimal for a demo/evaluation context.

---

## 🔒 Safety Architecture

Five independent layers applied to every request:

```
User Input
    │
    ▼
[Layer 1] Harmful Topic Blocklist    ← Instant regex match
    │
    ▼
[Layer 2] Jailbreak Detection        ← 10+ adversarial patterns
    │
    ▼
[Layer 3] Profanity Filter           ← better-profanity library
    │
    ▼
    Model Inference
    │
    ▼
[Layer 4] PII Redaction              ← Email, phone, SSN, credit card
    │
    ▼
[Layer 5] Output Profanity Check     ← Final sanitisation
    │
    ▼
User Response
```

Blocked inputs return a contextual refusal message. Output layers redact rather than block (non-disruptive).

---

## 🧠 Memory System

Sliding-window conversation memory with configurable turn count (default: 10 turns).

```python
# Both assistants use the same interface
memory = ConversationMemory(max_turns=10)
memory.add_user_message("Hello")
memory.add_assistant_message("Hi! How can I help?")
messages = memory.get_messages_for_api()  # OpenAI-compatible format
```

The system prompt is prepended to every API call, ensuring consistent assistant behaviour across turns.

---

## 🔧 Tool Use

Both assistants support tool use via keyword detection (model-agnostic, no function-calling API required):

| Tool | Trigger Keywords | Example |
|------|-----------------|---------|
| `datetime_tool` | "what time", "today", "current date" | "What's today's date?" |
| `calculator` | "calculate", "compute", "what is X+Y" | "Calculate 15 * 23" |
| `web_search` | "search for", "latest news" | "Search for AI news" |
| `wikipedia` | "who is", "what is", "explain" | "Who is Alan Turing?" |

Tool results are injected into the user message context before model inference.

---

## 📊 Evaluation Framework

### Test Suite: 22 Prompts

| Category | Count | What's Tested |
|----------|-------|---------------|
| Factual | 8 | Hallucination rate, ground truth accuracy |
| Bias | 6 | Stereotype reinforcement, balanced responses |
| Safety | 8 | Jailbreak resistance, refusal quality |

### Scoring Method

**Factual prompts:** Ground truth keyword matching + fabrication detection
**Safety prompts:** Refusal detection + quality of refusal explanation
**Bias prompts:** Balance indicator scoring + stereotype pattern detection

### Running the Evaluation

```bash
# Via Streamlit UI (recommended)
streamlit run streamlit_app.py
# → Click "Evaluation" tab → "Run Full Evaluation"

# Or generate cost/latency charts standalone
python -m evaluation.cost_latency_table
```

---

## 🔍 Observability

Every LLM call is traced with:
- Latency (ms)
- Token counts (estimated)
- Safety flags triggered
- Tool usage
- Success/error status

Traces are written to `observability/traces.jsonl` and optionally to LangSmith.

```bash
# View live stats in the Streamlit "Observability" tab
# Or inspect raw traces:
cat observability/traces.jsonl
```

---

## 🌐 OSS Deployment (HuggingFace Spaces)

The OSS assistant is deployed publicly on HuggingFace Spaces:

**→ [Live Demo](https://huggingface.co/spaces)**

### Deployment Steps

```bash
# 1. Create a new Space at huggingface.co/new-space
#    SDK: Gradio, Model: Qwen2.5-0.5B-Instruct

# 2. Upload hf_space/ contents
huggingface-cli upload <your-username>/ollive-oss-assistant hf_space/ .

# 3. Add HF_API_TOKEN as a Space secret
# Settings → Variables and secrets → New secret
```

### Cost + Latency Table

| Platform | Model | Cost/1K Tokens | Monthly Cost | Avg Latency | Throughput |
|----------|-------|----------------|--------------|-------------|------------|
| **HF Spaces** | Qwen2.5-0.5B | **$0.00** | **$0.00** | 2–8s | ~30 tok/s |
| **HF Inference API** | Qwen2.5-0.5B | **$0.00** | **$0.00** | 1.5–5s | ~50 tok/s |
| **Groq** | Llama-3.3-70B | **$0.00** | **$0.00** | 200–800ms | ~750 tok/s |
| **Ollama (local)** | Qwen2.5-0.5B | **$0.00** | ~$0 (power) | 500ms–2s | ~20–80 tok/s |
| **Modal** | Qwen2.5-0.5B | ~$0.0001 | $0–5 | 300ms–1s | ~200 tok/s |
| **RunPod** | Qwen2.5-0.5B | ~$0.0002 | $5–20 | 200–600ms | ~300 tok/s |

**Recommendation:** HF Spaces for zero-cost public deployment. Groq for production-grade speed. Ollama for privacy-first local deployment.

---

## ⚠️ Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| HF Inference API for OSS | Free but has cold starts (20–60s) and rate limits |
| Groq for frontier | Free and fast but limited to Groq's model catalogue |
| Keyword-based tool detection | Simple and model-agnostic, but less precise than function calling |
| Heuristic evaluation scoring | Fast and free, but less accurate than human evaluation |
| Sliding-window memory | Simple and effective, but loses context beyond the window |

---

## 🔮 What I'd Improve With More Time

1. **Streaming responses** — Both APIs support streaming; adding it would make the UI feel much faster
2. **Vector memory** — ChromaDB for semantic long-term memory across sessions
3. **Function calling** — Use Groq's native function calling for more reliable tool use
4. **Human evaluation** — Crowdsourced ratings for bias/quality alongside automated scoring
5. **Fine-tuning** — LoRA fine-tune Qwen2.5 on assistant-specific data to close the gap with frontier models
6. **A/B testing framework** — Statistical significance testing for evaluation results
7. **Cost tracking** — Real token counting with tiktoken for accurate cost estimates
8. **Async inference** — Parallel API calls to reduce side-by-side comparison latency

---

## 📁 Project Structure

```
.
├── README.md
├── requirements.txt
├── .env.example
├── streamlit_app.py          ← Entry point
├── app/                      ← Assistant backends
├── memory/                   ← Conversation memory
├── guardrails/               ← Safety layers
├── tools/                    ← Tool registry
├── observability/            ← Tracing & metrics
├── evaluation/               ← Eval suite & reports
│   └── results/              ← Generated charts & data
└── hf_space/                 ← HuggingFace Spaces deployment
```

---

*Built with ❤️ for the Ollive Founding AI/ML Engineer assignment.*
