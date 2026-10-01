# job_matcher — Local Job-Search & Job-Matching Pipeline

An intelligent **information-retrieval system** for job hunting — not an LLM
chatbot. It scrapes jobs (JobSpy: LinkedIn + Indeed), stores them in SQLite,
applies **deterministic rules** (experience, seniority, location, career
direction, must-have skills), ranks the survivors with **sentence embeddings**,
and only then asks a **local Qwen model (via Ollama)** to deeply analyse the
top candidates. Final scoring and verdicts are always deterministic.

```
Resume PDF → Qwen profile → search queries → JobSpy → normalise → dedup → SQLite
      → RULE ENGINE (experience / seniority / location / must-have / role relevance)
      → EMBEDDINGS (semantic ranking) → top-K
      → QWEN deep analysis → DETERMINISTIC SCORING
      → matched_jobs.csv + rejected_jobs.csv
```

## Design principles (enforced in code)

| # | Rule | Where enforced |
|---|------|----------------|
| 1 | Never use an LLM for calculations that can be done deterministically | `matching/experience.py`, `matching/scoring.py` |
| 2 | Never let embeddings decide experience eligibility | embeddings only enter as a scored component |
| 3 | Never let Qwen override hard rejection rules | `matching/scoring.py: determine_verdict`, re-check after Qwen in `main.py` |
| 4 | Embeddings find semantic similarity | `matching/semantic_matcher.py` |
| 5 | Qwen adds nuanced reasoning | `llm/prompts.py` (top-K only) |
| 6 | SQLite avoids repeated processing | `database/sqlite_store.py` (content/config hashes) |
| 7 | Structured JSON everywhere | `llm/schemas.py` (Pydantic) |
| 8 | Every job has an explainable reason | `matching/scoring.py: build_explanations`, both CSVs |
| 9 | Relevant experience > total experience for domain jobs | `matching/experience.py: effective_candidate_years` |
| 10 | Career direction ≠ technical capability | `matching/scoring.py: deterministic_career_alignment` |

---

## Quick start

### 1. Install Ollama + a model

```bash
# https://ollama.com/download
ollama pull qwen3:4b          # default; any small Qwen works
ollama serve                  # usually already running
```

### 2. Install Python dependencies

```bash
cd job_matcher
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> CPU-only torch is fine: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### 3. Run

```bash
python main.py \
    --resume data/sample_resume.pdf \
    --sites linkedin indeed \
    --location "Bangalore,Hyderabad,Pune,Remote" \
    --model qwen3:4b \
    --embedding-model BAAI/bge-small-en-v1.5 \
    --max-age-days 14 \
    --top-k 50 \
    --output data/matched_jobs.csv
```

Outputs: `data/matched_jobs.csv` (MATCH + BORDERLINE, best first) and
`data/rejected_jobs.csv` (every rejection **with its reason**).

### 4. Offline demo (no Ollama / no downloads)

```bash
python examples/run_offline_demo.py
# → examples/output/matched_jobs.csv + rejected_jobs.csv
```

Seeds the spec's section-31 scenario (Edge-AI job, 5–8y ML job, generic SWE
job, telecom job, …) and runs the real pipeline with heuristic profile
extraction and deterministic analysis fallbacks.

---

## CLI reference

```text
python main.py [options]

--resume PATH            resume PDF/TXT (cached after first use)
--profile-json PATH      use a pre-extracted profile JSON instead
--config PATH            YAML config (default: config.yaml)
--sites SITE [SITE...]   linkedin indeed glassdoor zip_recruiter google naukri bayt
--location, --locations  comma-separated: "Bangalore,Hyderabad,Remote"
--model NAME             Ollama model (default qwen3:4b) — never hard-coded
--embedding-model NAME   sentence-transformers model (default BAAI/bge-small-en-v1.5)
--max-age-days N         job freshness filter (default 14; unknown dates kept)
--top-k N                jobs sent to Qwen after embedding ranking (default 50)
--max-qwen-jobs N        hard cap on Qwen calls (default 50)
--must-have SKILL...     every recommended job must involve these skills
--soft-experience        near-miss experience becomes BORDERLINE, not REJECTED
--results-wanted N       results per scrape call (default 30)
--output / --output-dir  CSV destinations
--db PATH                SQLite path (default data/jobs.db)
--offline                no LLM at all (heuristic profile + regex extraction)
--scrape-only            fill the database, skip matching
--match-only             match jobs already in the database, skip scraping
--export                 regenerate CSVs from cached results (no LLM, no scrape)
--reprocess              ignore cached Qwen results and re-analyse
--verbose                debug logging
```

Typical workflow:

```bash
python main.py --resume resume.pdf --scrape-only          # 1. gather jobs
python main.py --resume resume.pdf --match-only           # 2. match (reuses cache)
python main.py --export                                   # 3. re-export CSVs anytime
python main.py --resume resume.pdf --reprocess            # 4. force full re-analysis
```

---

## Project structure

```
job_matcher/
├── main.py                  # CLI + pipeline orchestration
├── config.py                # frozen-dataclass config, YAML + CLI override merging
├── config.yaml              # every policy knob, fully commented
├── models.py                # shared enums + records + result (de)serialisation
├── requirements.txt
│
├── scraping/
│   ├── jobspy_scraper.py    # JobSpy wrapper, query generation, per-call isolation
│   └── normalizer.py        # cleaning, URL canonicalisation, dedup, content hashes
├── resume/
│   ├── parser.py            # PyMuPDF/pypdf text extraction
│   └── profile_extractor.py # Qwen profile + heuristic fallback + postprocessing
├── matching/
│   ├── experience.py        # regex extraction, domain classification, the ladder
│   ├── rules.py             # seniority/location/role/freshness/must-have filters
│   ├── skill_taxonomy.py    # categories, aliases, EXACT/RELATED/TRANSFERABLE/MISSING
│   ├── embeddings.py        # provider interface + sentence-transformers backend
│   ├── semantic_matcher.py  # structured text builders, batch ranking, top-K
│   └── scoring.py           # final deterministic score, verdicts, explanations
├── llm/
│   ├── ollama_client.py     # one reusable client, JSON repair, retries, guardrails
│   ├── prompts.py           # the three Qwen prompts (versioned for caching)
│   └── schemas.py           # Pydantic contracts with lenient LLM coercion
├── database/
│   └── sqlite_store.py      # schema, upserts, requirement/match/embedding caches
├── output/
│   └── csv_writer.py        # matched_jobs.csv + rejected_jobs.csv (spec columns)
├── data/
│   ├── sample_resume.pdf    # the section-31 example candidate
│   ├── jobs.db              # created on first run
│   ├── matched_jobs.csv
│   └── rejected_jobs.csv
├── examples/
│   ├── run_offline_demo.py  # full pipeline, no Ollama needed
│   └── demo_config.yaml
└── tests/                   # 180 unit/integration tests (pytest)
```

---

## How decisions are made

### 1. Candidate profile (Qwen, cached)

Qwen returns strict JSON: total vs **relevant** experience, per-domain years
(`machine_learning`, `computer_vision`, `model_deployment`, `quantization`, …),
core/secondary skills, target roles, excluded roles. Seniority is **not** left
to the LLM — it is inferred deterministically from the title (years-based
fallback for unmarked titles). If Qwen is unavailable, a heuristic extractor
(regex + taxonomy scan) keeps the pipeline alive (`--offline`).

### 2. Experience matching (deterministic, domain-aware)

For each job the requirement is extracted by regex
(`3+ years`, `3-5 years`, `2 to 4 years`, `minimum 4 years`, `5 years preferred`,
en-dashes, …) and classified into a domain. Qwen re-extracts for the top-K and
the results are merged (Qwen can only *add* strictness — never relax a
deterministic rejection).

The candidate's **effective years** for a job are:

1. direct years in the required domain (e.g. CV years for a CV requirement)
2. for `software_engineering`: total years (software dev spans a career)
3. adjacent-domain years × configurable credit (default 0.5, CV↔ML etc.)
4. relevant years when the resume gave no domain breakdown
5. for `general`: relevant years (falling back to total)

Then the ladder (evaluated in order):

| situation | verdict |
|---|---|
| no requirement, no seniority to infer from | `UNKNOWN` |
| effective < min, within soft tolerance (`--soft-experience`) | `MARGINAL` |
| effective < min | `UNDERQUALIFIED` (hard reject) |
| effective > reference AND gap ≥ 2y AND ratio ≥ 1.75 | `OVERQUALIFIED` (hard reject) |
| effective > reference AND (gap ≥ 2y OR ratio ≥ 1.75) | `POTENTIALLY_OVERQUALIFIED` (flag) |
| effective > reference, within tolerances | `ACCEPTABLE` |
| inside the required range | `STRONG_MATCH` |

`reference` = job max when given, else job min. Using **both** an absolute gap
and a ratio prevents overly aggressive filtering (5y vs 4y = ACCEPTABLE,
8y vs 4y = OVERQUALIFIED, 10y/4y-ML vs 3–4y = OVERQUALIFIED,
6y-ML vs 3–4y = POTENTIALLY_OVERQUALIFIED — the exact spec table).

Worked examples (candidate: 10y total, 4y ML/CV, 3y deployment, 2y quantization):

| Job | Result | Why |
|---|---|---|
| Senior Edge AI Eng, 3–5y | **MATCH** | 3y deployment ∈ 3–5, ONNX/Quantization exact, TensorRT related |
| ML Engineer, 5–8y ML | REJECTED `UNDERQUALIFIED` | 4y ML < 5y (relevant, not total!) |
| Software Engineer, 2–4y | REJECTED `OVERQUALIFIED`/`LOW_CAREER_ALIGNMENT` | capability ≠ career direction |
| Telecom Network Eng | REJECTED `IRRELEVANT_ROLE` | historical BSNL experience does not steer the career |

### 3. Skill matching (taxonomy, deterministic)

Every job skill is resolved to a canonical name + category
(MODEL_DEPLOYMENT: ONNX, TIDL, QNN, SNPE, TensorRT, OpenVINO, TVM…;
QUANTIZATION: INT8, PTQ, QAT, GPTQ, AWQ, distillation…;
COMPUTER_VISION: BEV, LSS, object detection…; MLOPS; DEEP_LEARNING; …):

* **EXACT** — same canonical skill (Python ↔ Python)
* **RELATED** — same category (job TensorRT vs candidate ONNX Runtime/TIDL/QNN/SNPE)
* **TRANSFERABLE** — adjacent category (job PTQ vs candidate TensorRT)
* **MISSING** — nothing in the same or adjacent family (CUDA with no deployment background)

Required skills weigh 1.0, preferred 0.5. `must_have_skills` (config/CLI) is a
hard filter on the **job** involving those skills.

### 4. Semantic ranking (embeddings)

Structured, truncated text is embedded for the resume (title, target roles,
summary, core skills, domain years) and each job (title, company, skills,
experience, responsibilities, requirements-focused description excerpt).
Cosine similarity ranks the rule-eligible jobs; the top-K
(default 50, `--top-k`) proceed to Qwen. Embeddings never gate experience or
hard requirements — only relevance. Swap models freely via
`--embedding-model`; the `EmbeddingProvider` interface isolates the backend
(an offline hashing provider is used by tests/demo).

### 5. Qwen deep analysis (top-K only)

Qwen receives the profile, the job, the deterministic signals (experience fit,
matched/missing skills) and is explicitly told the experience verdict is final.
It returns 0–100 sub-scores + narrative. Guardrails:

* invalid JSON → repair (fences, `<think>`, trailing commas) → retry →
  schema-guided retry → deterministic fallback (the run never crashes)
* scores are clamped; deviations > 40 from the deterministic value are ignored
* Qwen can only **downgrade** (match_score ≤ 20 caps at BORDERLINE), never rescue

### 6. Final score and verdict (deterministic)

```
final = 0.30·experience + 0.25·skills + 0.20·embedding
      + 0.10·responsibility + 0.10·domain + 0.05·career_alignment
```

* any hard rejection → `REJECTED` regardless of score
* `MARGINAL` / `POTENTIALLY_OVERQUALIFIED` experience → capped at `BORDERLINE`
* score ≥ 75 → `MATCH`; 60–74 → `BORDERLINE`; < 60 → `REJECTED`
* career alignment < 40 → `REJECTED` (`LOW_CAREER_ALIGNMENT`)

All thresholds live in `config.yaml`.

---

## SQLite schema (`data/jobs.db`)

| table | purpose |
|---|---|
| `jobs` | one row per posting: url (PK), company, title, location, description, salary, source, sources_json, posted_date, first_seen, last_seen, content_hash, … |
| `job_requirements` | per-job extraction: min/max experience, experience_domain, required/preferred skills, seniority, work mode, responsibilities, requirements_hash |
| `job_matches` | per-job result: all component scores, final score, experience_fit, verdict, reasoning, analysis_json, result_json, **config_hash** |
| `candidate_profile` | cached Qwen profile extractions (resume hash + model) |
| `skills` | the skill taxonomy (auto-synced each run) |
| `search_runs` | run metadata + stats JSON |
| `embeddings_cache` | job vectors keyed by (embedding model, content hash) |

### Incremental behaviour (section 22)

* same job seen again → `last_seen` updated, sources merged, **no reprocessing**
* requirements cached per `content_hash + model + prompt version`
* matches cached per `config_hash` — change any policy and results recompute
* `--reprocess` forces full re-analysis; `--export` rebuilds CSVs from cache

---

## Output columns

`matched_jobs.csv` (sorted: eligible first, then final score ↓):
Company, Job Title, Platform, Location, Work Mode, Employment Type, Posted Date,
Candidate Total/Relevant Experience, Job Min/Max Experience, Experience Domain,
Experience Fit, Experience Gap, Over/Underqualification Flags,
Required/Preferred/Matched/Related/Missing Skills,
Embedding / Skill / Experience / Domain / Responsibility / Career-Alignment /
Final scores, Verdict, Strengths, Concerns, Reasoning, Job Link.

`rejected_jobs.csv`: Company, Job Title, Platform, Location, Job Experience,
Candidate Experience, **Rejection Reason** (UNDERQUALIFIED, OVERQUALIFIED,
IRRELEVANT_ROLE, MISSING_MUST_HAVE_SKILL, LOCATION_MISMATCH, SENIORITY_MISMATCH,
STALE_JOB, LOW_CAREER_ALIGNMENT, …), Rejection Stage, Detail, Score, Job Link.

---

## Testing

```bash
python -m pytest tests/ -q          # 180 tests, < 1 s, no network/LLM needed
```

Coverage includes the full spec decision table (10y/3y, 10y/8y, 5y/4y,
4 ML vs 5 ML, 4 CV vs 5 ML, unknown experience, "3+ years", "3-5 years",
"5 years preferred", "minimum 4 years", soft mode), over/underqualification
configurability, the skill taxonomy (TensorRT↔ONNX family, Kubeflow↔Docker),
seniority normalisation (Junior→Principal, roman numerals), location
aliasing (Bengaluru→Bangalore, Gurugram→Delhi), deduplication
(URL + company/title/location across sites), Qwen JSON repair
(fences, think-tags, trailing commas, clamping), scoring guardrails
(Qwen cannot rescue a rejection), SQLite round-trips, config merging,
and a full offline end-to-end pipeline run with a fake LLM.

## Performance notes (8 GB RAM friendly)

* one Ollama client, one embedding model load per run; batched encoding
* Qwen sees only top-K jobs with truncated descriptions (~4 KB each)
* everything expensive is cached in SQLite; reruns are near-instant
* qwen3 thinking mode disabled (`/no_think`, `think=false`) for fast JSON
* `num_ctx` 4096 by default — lower it in `config.yaml` for constrained hosts

## Troubleshooting

| symptom | fix |
|---|---|
| `Ollama not reachable` | start `ollama serve`; the pipeline degrades to offline mode automatically |
| LinkedIn returns few / blocked results | guest scraping is rate-limited; set `LINKEDIN_USERNAME`/`LINKEDIN_PASSWORD` env vars, retry later, or rely on Indeed |
| model not found | `ollama pull qwen3:4b` |
| slow first run | the embedding model downloads once (~130 MB for bge-small) |
| empty matched CSV | check `rejected_jobs.csv` reasons; loosen `locations`, thresholds, or run with `--soft-experience` |
