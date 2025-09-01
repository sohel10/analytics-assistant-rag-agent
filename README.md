# Clinical Ops Assistant (RAG-lite, FLAN-T5) — Streamlit
streamlit · transformers · flan-t5 · tokenizers · python-docx · healthcare · clinical-trials · RAG

A lightweight, demo-ready assistant for clinical operations that drafts **patient-friendly narratives** from structured JSON and **grounds** them using tiny reference snippets (RAG-lite). Ships with a **Streamlit UI**, **DOCX export**, and **batch CSV** processing.

> **Live app (replace with your link):** https://clinical-ops-flan-t5-agent-<your-handle>.streamlit.app

---

## ✨ Features
- **Patient narrative generation** using a small open LLM (Google **FLAN-T5**) via `transformers`.
- **RAG-lite grounding**: retrieves a few relevant lines from `./sources/*.txt` and cites them in the prompt.
- **Batch mode**: upload a CSV of patient JSON rows and download results.
- **DOCX export**: one-click download of the generated message.

---

## 🧠 How it works (quick)
1. You paste a **patient JSON** (e.g., demographics + conditions).
2. App builds a simple **query** from age/gender/conditions.
3. It retrieves top-K **reference snippets** from `./sources` (token overlap scoring).
4. A **prompt** is constructed with the facts + the snippet titles.
5. **FLAN-T5** generates a short, patient-friendly note (with a required safety closing line).

---

## 🗂 Repo Structure
