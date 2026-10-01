#!/usr/bin/env python3
"""
Offline end-to-end demo — no Ollama, no downloads, no scraping.

Seeds the five jobs from the spec's section-31 scenario (plus two extra
edge cases) into a demo SQLite database, then runs the REAL pipeline from
main.py in offline mode:

    heuristic profile extraction (from data/sample_resume.pdf)
    -> deterministic rule engine
    -> hashing embeddings (stand-in for bge-small)
    -> no Qwen (deterministic analysis fallbacks)
    -> deterministic scoring
    -> examples/output/matched_jobs.csv + rejected_jobs.csv

Usage:
    cd job_matcher
    python examples/run_offline_demo.py [--verbose]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.sqlite_store import SqliteStore  # noqa: E402
from scraping.normalizer import normalize_job  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "examples" / "output"
DEMO_DB = OUTPUT_DIR / "demo.db"
DEMO_CONFIG = PROJECT_ROOT / "examples" / "demo_config.yaml"

SAMPLE_JOBS = [
    {   # Job A (spec 31): Senior Edge AI Engineer -> expected MATCH
        "site": "linkedin", "title": "Senior Edge AI Engineer", "company": "Qualcomm",
        "job_url": "https://www.linkedin.com/jobs/view/9001/",
        "location": "Bengaluru, Karnataka, India",
        "date_posted": (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d"),
        "description": (
            "We build edge AI platforms for automotive customers.\n\n"
            "Requirements: 3-5 years of experience in edge AI model deployment.\n"
            "Skills: Python, C++, ONNX, TensorRT, Quantization (INT8/PTQ).\n"
            "Responsibilities: model optimization, inference acceleration, "
            "hardware/software co-design, quantization pipeline development.\n"
            "Nice to have: CUDA programming experience."
        ),
    },
    {   # Job B (spec 31): ML Engineer 5-8 years -> UNDERQUALIFIED (4 ML years)
        "site": "indeed", "title": "Machine Learning Engineer", "company": "RecSys Startup",
        "job_url": "https://in.indeed.com/viewjob?jk=demo0002",
        "location": "Hyderabad, Telangana, India",
        "date_posted": (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d"),
        "description": (
            "Build large-scale recommendation systems.\n"
            "Requirements: 5-8 years of machine learning experience.\n"
            "Skills: Python, PyTorch, Kubernetes, MLOps, Spark."
        ),
    },
    {   # Job C (spec 31): generic Software Engineer -> career misalignment
        "site": "indeed", "title": "Software Engineer", "company": "WebCorp",
        "job_url": "https://in.indeed.com/viewjob?jk=demo0003",
        "location": "Pune, Maharashtra, India",
        "date_posted": (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d"),
        "description": (
            "Build REST APIs and web backends.\n"
            "Requirements: 2-4 years of software development experience.\n"
            "Skills: Java, Spring Boot, SQL, Microservices, REST API."
        ),
    },
    {   # Job D (spec 31): telecom role -> IRRELEVANT_ROLE (career direction)
        "site": "indeed", "title": "Telecom Network Engineer", "company": "BSNL",
        "job_url": "https://in.indeed.com/viewjob?jk=demo0004",
        "location": "Chennai, Tamil Nadu, India",
        "date_posted": (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d"),
        "description": (
            "Maintain LTE/5G network operations.\n"
            "Requirements: 5+ years of telecom network experience.\n"
            "Skills: routing, switching, TCP/IP, firewall, network security."
        ),
    },
    {   # Job E: Computer Vision Engineer 3-4y -> expected MATCH
        "site": "linkedin", "title": "Computer Vision Engineer", "company": "DriveAI",
        "job_url": "https://www.linkedin.com/jobs/view/9005/",
        "location": "Remote",
        "date_posted": (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d"),
        "description": (
            "Develop 3D perception (BEV, Lift-Splat-Shoot) for autonomous driving.\n"
            "Requirements: 3-4 years of computer vision experience.\n"
            "Skills: PyTorch, Object Detection, Semantic Segmentation, ONNX.\n"
            "Responsibilities: train and deploy perception models, evaluate on "
            "LiDAR and camera data."
        ),
    },
    {   # Edge case: stale posting (35 days old) -> STALE_JOB
        "site": "indeed", "title": "Deep Learning Engineer", "company": "OldCorp",
        "job_url": "https://in.indeed.com/viewjob?jk=demo0006",
        "location": "Bangalore, Karnataka, India",
        "date_posted": (datetime.now(timezone.utc) - timedelta(days=35)).strftime("%Y-%m-%d"),
        "description": "Train deep learning models. 3+ years of experience in deep learning.",
    },
    {   # Edge case: severely overqualified (2y junior role for a 10y candidate)
        "site": "indeed", "title": "Junior ML Engineer", "company": "TinyCo",
        "job_url": "https://in.indeed.com/viewjob?jk=demo0007",
        "location": "Bangalore, Karnataka, India",
        "date_posted": (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d"),
        "description": (
            "Entry-level ML role. Requirements: 1-2 years of machine learning "
            "experience. Skills: Python, Scikit-learn."
        ),
    },
]


def seed_database() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if DEMO_DB.exists():
        DEMO_DB.unlink()
    records = [normalize_job(row) for row in SAMPLE_JOBS]
    records = [r for r in records if r is not None]
    with SqliteStore(DEMO_DB) as store:
        store.upsert_jobs(records)
    print(f"Seeded {len(records)} demo job(s) into {DEMO_DB}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args, _ = parser.parse_known_args()

    seed_database()

    import main as pipeline_main

    cli_args = [
        "--resume", str(PROJECT_ROOT / "data" / "sample_resume.pdf"),
        "--match-only",
        "--offline",
        "--config", str(DEMO_CONFIG),
        "--db", str(DEMO_DB),
        "--output-dir", str(OUTPUT_DIR),
    ]
    if args.verbose:
        cli_args.append("--verbose")
    print("\nRunning pipeline:", " ".join(cli_args), "\n")
    return pipeline_main.main(cli_args)


if __name__ == "__main__":
    raise SystemExit(main())
