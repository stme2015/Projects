# CoDesign AI Urban Planner — Multi-Agent Urban Design Platform

**Published Research:** [arXiv:2603.16008](https://arxiv.org/abs/2603.16008)

React 18 · TypeScript · Node.js · Gemini API · Google Maps API · Firestore · Google Cloud Storage

---

## What It Does

CoDesign AI is a multi-user, multi-agent platform designed to facilitate street-level urban interventions. Community members design street layouts in a shared workspace alongside specialized AI agents (Urban Planner, Accessibility Specialist, Facilitator) operating on spatial mapping APIs to resolve layout conflicts and synthesize structured design proposals from conversation logs.

## Collaborative Workspace

![Multi-Agent CoDesign Interface](docs/Multi-Agent-img.png)

## Key Technical Decisions

**Specialized Agent Task-Routing.** Designed three distinct tool-calling surfaces to isolate agent concerns. The Urban Planner queries spatial layout vectors, the Accessibility Specialist audits width/clearance markers against guidelines, and the Facilitator arbitrates disputes. This separation isolates errors and ensures spatial design revisions are grounded in real geographic boundaries.

**Context-Aware Continuity.** Inter-agent state transcripts and participant inputs are serialized, frozen, and backed up to Google Cloud Storage (GCS) at the end of every feedback round. The state schema is indexed in Firestore to retain long-term memory across session restarts.

**Conflict Resolution Engine.** Created a rule-based arbiter that maps spatial constraints against agent critique scores. When the system detects overlapping design coordinate geometries, it triggers a negotiation state that prompts the participants with alternative configurations.

## Stack

| Layer | Technologies |
|---|---|
| Frontend & Maps | React 18, TypeScript, Google Maps API |
| Backend & State | Node.js, Express |
| Multi-Agent Orchestration | Gemini API, Custom Tool Calling |
| Database & Storage | Firestore, Google Cloud Storage |

## Metrics

- Grounded spatial designs using live map APIs
- Cross-agent chat transcripts indexed for multi-round session memory
- Handles coordinate conflict detection to guide group consensus
