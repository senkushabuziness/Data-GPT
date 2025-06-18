
---

# 📘 Data GPT – Phase 1 PRD Documentation

---

## 📌 Project Overview

**Data GPT** is a no-code, Chainlit-based AI platform that enables users to analyze tabular data (CSV files) through natural language queries. The system interprets these queries using open-source LLMs and executes them via SQL on **DuckDB**, returning insights in the form of text, charts, and tables — all without requiring the user to write any code.

The app is modular, and each intern is responsible for a specific component, ensuring scalable development and rapid iteration.

---

## 🎯 Objectives

* Build a natural language interface to interact with CSV data
* Enable SQL-powered analysis using **DuckDB**
* Convert natural language to SQL or Pandas using **open-source LLMs**
* Display results via **textual summaries**, **charts**, and **tables**
* Maintain session memory and user context
* Implement OAuth2-based authentication
* Evaluate multiple open-source LLMs and benchmark them
* Assign individual modules to interns — one developer per module

---

## 🛠️ Technical Stack

| Layer               | Tool/Technology                                |
| ------------------- | ---------------------------------------------- |
| **Frontend/UI**     | Chainlit                                       |
| **ETL Layer**       | Pandas + DuckDB                                |
| **LLM Interface**   | LangChain + OSS LLMs (LLaMA, Mistral, Phi-3)   |
| **Visualization**   | Plotly                                         |
| **Auth**            | OAuth 2.0 (Google/GitHub via Chainlit/FastAPI) |
| **Session Memory**  | LangChain memory + Chainlit context store      |
| **Analytics DB**    | DuckDB                                         |
| **Model Hosting**   | Ollama, LM Studio, HuggingFace Spaces, Colab   |
| **DevOps (future)** | GitHub Actions, Docker, Google Cloud Run       |

---

## 📊 Key Performance Indicators (KPIs)

### 🔹 User Interaction KPIs

* **CSV Load Success Rate** – % of uploaded files successfully parsed and stored
* **Time to First Insight** – Time from file upload to first chart/response
* **Prompt-to-Insight Success** – % of prompts generating meaningful output
* **Chart Accuracy** – Correctness of visualizations based on prompt intent
* **Session Continuity** – % of multi-turn queries maintaining logical flow

### 🔸 ETL + DuckDB KPIs

* **ETL Load Time** – Time from CSV upload to DuckDB table creation
* **Transformation Failures** – % of parsing/type inference errors
* **Query Execution Rate** – % of SQL queries that execute without exception
* **Memory Usage** – DuckDB resource usage per session or user

### 🧠 LLM Evaluation KPIs

* **Accuracy Score** – How close the output SQL/Pandas matches expected logic
* **Latency** – Avg. time per LLM query
* **Token Usage** – Tokens consumed per request (efficiency)
* **Cost to Run** – Estimated runtime cost for hosting/free alternatives

---

## 🏗️ High-Level Design Document (HLDD)

### 📌 System Overview

The system supports an "Upload → Ask → Answer" model:

* CSVs are uploaded and stored as DuckDB tables
* Users ask natural language questions
* Open-source LLMs generate SQL (or Pandas)
* Query is run on DuckDB
* Output is rendered as text, tables, or charts

---

### 🧱 Component Breakdown

| Component           | Description                                                 |
| ------------------- | ----------------------------------------------------------- |
| **Chainlit UI**     | Upload files, enter prompts, show charts and responses      |
| **ETL Layer**       | Clean column names, infer types, and load CSV into DuckDB   |
| **LLM Engine**      | Use OSS LLMs to translate NL to SQL/Pandas via LangChain    |
| **SQL Executor**    | Execute queries on DuckDB and handle errors                 |
| **Chart Generator** | Convert dataframes to interactive charts using Plotly       |
| **Auth Layer**      | Secure access with Google/GitHub login                      |
| **Session Tracker** | Store user context and prompt history                       |
| **Model Evaluator** | Benchmark OSS LLMs to identify best fit for cost + accuracy |

---

### 🖼 Architecture Diagram (Textual)

```
[User]
   │
   ▼
[Chainlit UI]
   │
   ├─── CSV Upload ───────► [ETL Pipeline] ─────► [DuckDB] ◄───┐
   │                        (pandas, cleaning, schema)         │
   │                                                        [SQL Engine]
   ├─── Natural Language Query ────────────────────────────────┘
   │                                   │
   ▼                                   ▼
[LLM via LangChain]             [SQL Output to Table + Plotly Chart]
   │                                   │
   └──────── Response (Text + Visual) ◄┘

```

---

## 🔁 ETL & Storage Strategy

* **Extract**: Users upload CSVs via the Chainlit interface
* **Transform**: Clean headers, infer column types using Pandas
* **Load**: Data is written into DuckDB as a temporary/per-session table
* **Table Naming**: `session_<uuid>_data` to ensure isolation
* **Cleanup**: Periodic or event-based clearing of unused session tables

---

## 🧩 Module Assignments

| Module                      | Description                                                                                         | Assigned Developer |
|-----------------------------|-----------------------------------------------------------------------------------------------------|--------------------|
| Chainlit +  Chart Generator | File upload and natural language input interface + Convert query results into Plotly visualizations | Krish              |
| ETL Handler                 | Transform and load uploaded CSVs into DuckDB                                                        | TBD                |
| LLM Prompt Handler          | Use LLM to translate NL input to SQL or Python                                                      | Pulkit             |
| SQL Executor                | Run generated SQL on DuckDB and return result                                                       | Krish              |
| Auth Setup                  | OAuth2 login with Google/GitHub                                                                     | Khushi             |
| Model Evaluator             | Evaluate open-source LLMs and benchmark them(with POC on titanic dataset)                           | Pulkit             |
| Session Tracker             | Maintain user memory and query context                                                              | Khushi             |

---

## 🚀 Deployment Roadmap (Future Phases)

| Phase   | Objective                                                            |
| ------- |----------------------------------------------------------------------|
| Phase 2 | Add CI/CD using GitHub Actions and Dockerfile   + Subscription Model |

---
