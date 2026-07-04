# Sridevi Turaga

[![Email](https://img.shields.io/badge/Email-sridevit.connect@gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:sridevit.connect@gmail.com)
[![NYU](https://img.shields.io/badge/M.S.-NYU%20Tandon%20%7C%20AI%20%26%20Urban%20Science-57068C?style=flat)](https://engineering.nyu.edu/academics/programs/applied-urban-science-and-informatics-ms)
[![arXiv](https://img.shields.io/badge/arXiv-Published-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.16008)


## ABOUT ME
AI Engineer with around 2 years shipping LLM systems, multi-agent architectures, and production ML pipelines, following a strong 6 years foundation in enterprise delivery. M.S. from NYU Tandon. Published researcher on arXiv. Founding engineer across three US startups spanning healthcare, real estate, and computer vision. 

[Published Research On Multi-Agent Systems](https://arxiv.org/abs/2603.16008)

---

## Shipped Products

### [HouSmart](https://github.com/stme2015/Projects/tree/main/housmart) — Property Investment Intelligence Platform
**FastAPI · Python · PostgreSQL/PostGIS · Asyncio · LangGraph · Gemini · Redis**

Built an end-to-end property evaluation system that autonomously orchestrates 10+ external APIs (RentCast, FEMA, Census) in parallel. Designed human-in-the-loop review states to catch and correct LLM extraction failures. Reduced pipeline runtime by 90% using async execution and Redis-based caching. Cut per-property processing cost to $0.024.

Live: [housmart.ai](https://www.housmart.ai)

### [RemiMinder AI](https://github.com/stme2015/Projects/tree/main/remiminder) — HIPAA-Compliant Patient Care Platform
**Flutter · FastAPI · GCP Cloud Run · Cloud Tasks · Docker · Whisper · Gemini OCR · Supabase RLS · Firebase JWT**

Deployed a multimodal clinical transcription pipeline handling speech-to-text, OCR, and LLM summarization on Cloud Run under a Business Associate Agreement. Benchmarked Whisper variants, GPT-4o, and Gemini across latency and cost to cut inference spend by 60%. Enforced tenant data isolation via Supabase Row-Level Security and Firebase JWT/JWKS authentication.

Live: [remiminderai.com](https://remiminderai.com)

### [StylePilot](https://github.com/stme2015/Projects/tree/main/stylepilot-ai) — Computer Vision Styling Platform
**CLIP ViT-B/32 · GPT-4o · pgvector · HNSW · FastAPI · AWS (EC2, S3, SQS, RDS) · React Native**

Deployed 5 production vision and recommendation models across 4 containerized microservices with SQS worker queues for async reliability. Integrated pgvector HNSW indices with rule-based candidate pruning to keep search latency under 10ms while reducing downstream LLM calls by 90%. Shipped the complete React Native client to 50+ iOS beta users via TestFlight on a $42/month AWS infrastructure baseline.

Live: [stylepilot.ai](https://stylepilot.ai)

---

## Multi-Agent Systems

### [CoDesign AI Urban Planner](https://github.com/stme2015/Projects/tree/main/codesign-ai-urban-planner) — Multi-Agent Urban Design Platform
**React 18 · TypeScript · Node.js · Gemini API · Google Maps API · Firestore · Google Cloud Storage**

Published on arXiv (2026). Architected a multi-user platform where community members co-design street-level urban interventions with specialized AI agents (urban planner, accessibility specialist, facilitator), each operating on distinct tool-calling surfaces backed by spatial mapping APIs. Agents ground proposals in real geographic context, detect conflicts across participants, and synthesize structured outputs from unstructured community dialogue. Session transcripts are frozen to GCS and indexed to Firestore for continuity across rounds.

### [OpenClaw Literature Review Agent](https://github.com/stme2015/Projects/tree/main/openclaw-literature-review-agent) — Multi-Agent Research Pipeline
**Python · OpenClaw · arXiv API · Semantic Scholar · WhatsApp**

Built a 6-agent sequential research assistant on the OpenClaw framework. Role-separated specialist agents coordinate through a dual-memory architecture: a volatile per-run pipeline state file for short-term inter-agent communication and a persistent cross-session ledger for long-term context accumulation. An adversarial critic agent enforces verbatim provenance anchors on all citations and issues loop-back corrections when sources cannot be verified, eliminating hallucination at the output stage.

### [ConResSim](https://github.com/stme2015/Projects/tree/main/conressim-conflict-mediation-agent) — Multi-Agent Conflict Mediation System
**Python · Microsoft AutoGen · Gemini 2.5 Flash · Llama 3.3 · LLM-as-Judge**

Presented at ICUA 2026. A multi-agent simulation system that models real-world group dialogues to detect emerging conflicts and apply theory-grounded mediation. Stakeholder agents operate with distinct roles and goals inside a shared group chat, while a Mediator agent orchestrates conflict detection and structured resolution across the session.


---

## Machine Learning Research

### [FEMA Disaster Assistance Predictor](https://github.com/stme2015/Projects/tree/main/fema-disaster-assistance-predictor) — NYU Tandon (1st Place)
**Python · XGBoost · Wide and Deep Neural Networks · ArcGIS · National Risk Index**

Engineered a geospatial climate predictor integrating National Risk Index climate variables with historical FEMA claim data across 100K+ records. Benchmarked XGBoost against Wide and Deep Neural Networks on accuracy and latency tradeoff, achieving a 6.23% MAPE score. Presented geospatial findings via ArcGIS StoryMap. Won 1st Place at the NYU ML Data Drive Competition.

[View StoryMap](https://storymaps.arcgis.com/stories/249c7616cc87454f8f9058be3afac771)

---

## Production-Grade AI Utilities

A repository organizing standalone production-grade tools across multi-agent orchestration, RAG pipelines, conversational AI, fine-tuning and optimization, predictive analytics, cloud infrastructure, and computer vision.

[View Repository](https://github.com/stme2015/production-ai-utilities)

---

## Core Stack

**Languages and Frameworks:** Python, TypeScript, Next.js, Node.js, FastAPI, Express.js, Flutter, SQL, Docker

**AI and RAG:** Microsoft AutoGen, LangGraph, LangChain, n8n, OpenClaw, CLIP, PyTorch, pgvector, HNSW, Cohere Rerank, HuggingFace

**ML & Modeling:** PyTorch, TensorFlow, XGBoost, Scikit-learn, ONNX, Fine-tuning (DPO, QLoRA, LoRA, SFT, PEFT), Statistics, Modeling, ARIMA, NLP, Computer Vision

**Databases:** PostgreSQL/PostGIS, Supabase, MongoDB, SQLite, Redis

**Cloud and Infrastructure:** AWS (EC2, S3, SQS, RDS), GCP (Cloud Run, Cloud Tasks, Cloud SQL), Supabase, Firebase, Terraform, GitHub Actions
