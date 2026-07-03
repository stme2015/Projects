# OpenClaw Literature Review Agent — Multi-Agent Research Pipeline

Python · OpenClaw · arXiv API · Semantic Scholar · WhatsApp Interface · pypdf

---

## What It Does

This system is an automated 6-agent sequential pipeline that scans academic papers, extracts findings, generates novel research questions, and compiles formatted literature reviews. The pipeline implements a zero-hallucination validation policy by verifying all generated hypotheses directly against source texts before rendering outputs.

## Run State and Memory Architecture

Volatile state variables and raw extracted text structures are isolated in `memory/pipeline-state.md` during execution. Once the pipeline completes a run, the system saves the finalized metrics and session logs to a persistent cross-session ledger (`MEMORY.md`).

## Key Technical Decisions

**Adversarial Audit Loop.** Built a dedicated Critic agent that runs cross-check verification on generated research questions. It audits all citations to ensure they contain exact, verbatim quotes from the target PDFs. If any quote fails validation, it initiates a loop-back instruction to force rewriting, capped at a maximum of 2 iterations.

**Strict Date Filtering.** The Paper Monitor parses arXiv XML responses and enforces a strict 3-month publishing horizon window. It dynamically scales up to 6 months only if fewer than 3 recent papers are returned on the topic.

**Unstructured PDF Extraction.** Integrated `pypdf` parsing libraries to extract raw data blocks from downloaded papers, feeding abstracts and full texts into a shared context window for downstream synthesis.

## Stack

| Layer | Technologies |
|---|---|
| Pipeline Orchestration | OpenClaw Framework, Python |
| Academic Search APIs | arXiv API, Semantic Scholar API |
| Messaging Delivery | WhatsApp (Green API/Twilio Webhook) |
| Local Storage | Markdown-based Volatile & Long-Term memory structures |
| Document Extraction | pypdf |

## Metrics

- Dual-memory pipeline architecture (volatile and persistent stores)
- Zero-hallucination validation via exact-quote citation anchors
- 6-agent sequential workflow pipeline with a maximum 2-iteration critic check
