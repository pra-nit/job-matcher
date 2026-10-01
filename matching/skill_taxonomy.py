"""
Skill taxonomy: canonical skill names, categories, aliases and relations.

The taxonomy powers deterministic skill matching (EXACT / RELATED /
TRANSFERABLE / MISSING).  Each canonical skill belongs to exactly one
category; aliases (including abbreviations and common spellings) map to
canonical names.  Category pairs listed in RELATED_CATEGORIES are treated as
"transferable" domains (e.g. QUANTIZATION <-> MODEL_DEPLOYMENT).

This module is intentionally dependency-free so it can be unit tested in
isolation and so the taxonomy can be replaced/extended without touching the
rest of the pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from models import SkillMatch, SkillMatchResult, SkillMatchType

# ---------------------------------------------------------------------------
# Categories -> canonical skills
# ---------------------------------------------------------------------------
SKILL_TAXONOMY: Dict[str, Set[str]] = {
    "PROGRAMMING": {
        "Python", "C++", "C", "Java", "Go", "Rust", "JavaScript", "TypeScript",
        "Bash", "Shell Scripting", "SQL", "MATLAB", "Kotlin", "Scala", "Embedded C",
    },
    "DEEP_LEARNING": {
        "PyTorch", "TensorFlow", "Keras", "JAX", "Transformers", "Hugging Face",
        "CNN", "RNN", "LSTM", "Vision Transformer", "ViT", "Attention Mechanisms",
        "DETR", "YOLO", "ResNet", "U-Net", "BERT", "GPT", "LLM", "Fine-tuning",
        "LoRA", "PEFT", "Diffusion Models", "Neural Networks", "Deep Learning",
        "Transformer Models", "Model Training", "Transfer Learning",
    },
    "COMPUTER_VISION": {
        "Object Detection", "Semantic Segmentation", "Instance Segmentation",
        "Panoptic Segmentation", "BEV", "Bird's Eye View", "LSS",
        "Lift-Splat-Shoot", "3D Perception", "Multi-Object Tracking",
        "Image Processing", "OpenCV", "Point Cloud", "Point Cloud Processing",
        "LiDAR", "SLAM", "Sensor Fusion", "Camera Calibration", "Stereo Vision",
        "Image Classification", "Visual Odometry", "Computer Vision",
        "Optical Flow", "Image Augmentation",
    },
    "CLASSICAL_ML": {
        "Scikit-learn", "XGBoost", "LightGBM", "CatBoost", "Random Forest", "SVM",
        "Decision Trees", "Gradient Boosting", "Clustering", "Regression",
        "Feature Engineering", "Recommendation Systems", "Time Series Forecasting",
        "Machine Learning", "Statistical Modeling", "Model Evaluation",
    },
    "NLP": {
        "Natural Language Processing", "spaCy", "NLTK", "Text Classification",
        "Named Entity Recognition", "NER", "Text Embeddings", "RAG",
        "Retrieval-Augmented Generation", "Vector Database", "Prompt Engineering",
        "Text Summarization", "Question Answering",
    },
    "GENAI": {
        "Generative AI", "vLLM", "LangChain", "LlamaIndex", "Agentic AI",
        "Multi-Agent Systems", "LLM Fine-tuning", "LLM Deployment",
        "LLM Evaluation", "Model Context Protocol",
    },
    "MODEL_DEPLOYMENT": {
        "ONNX", "ONNX Runtime", "TIDL", "TI Jacinto", "QNN", "SNPE", "TensorRT",
        "OpenVINO", "TVM", "TensorFlow Lite", "TFLite", "Core ML", "Edge AI",
        "Embedded AI", "Model Optimization", "Inference Optimization",
        "Inference Acceleration", "CUDA", "cuDNN", "Triton Inference Server",
        "ARM NEON", "DSP", "NPU", "Model Serving", "Model Compilation",
        "Model Deployment", "Model Conversion", "GPU Programming",
    },
    "QUANTIZATION": {
        "INT8", "INT16", "INT4", "FP16", "FP8", "PTQ",
        "Post-Training Quantization", "QAT", "Quantization-Aware Training",
        "GPTQ", "AWQ", "Calibration", "Mixed Precision",
        "Per-channel Quantization", "Knowledge Distillation", "Pruning",
        "Sparsity", "Weight Quantization", "Activation Quantization",
        "Quantization", "SmoothQuant", "Weight Sharing",
    },
    "MLOPS": {
        "Docker", "Jenkins", "CI/CD", "MLflow", "Kubeflow", "Kubernetes",
        "Airflow", "Model Registry", "Model Monitoring", "Feature Store",
        "ML Pipelines", "MLOps", "Experiment Tracking", "Data Versioning",
        "DVC", "Model Versioning", "GitHub Actions",
    },
    "CLOUD": {
        "AWS", "Azure", "GCP", "SageMaker", "Vertex AI", "EC2", "S3", "Lambda",
        "CloudFormation", "Google Colab", "Cloud Deployment", "ECR", "EKS",
    },
    "DATA_ENGINEERING": {
        "Pandas", "NumPy", "Spark", "PySpark", "ETL", "Data Pipelines",
        "Data Warehousing", "Tableau", "Power BI", "Apache Kafka", "Hadoop",
        "Data Modeling", "Big Data",
    },
    "NETWORKING": {
        "TCP/IP", "Routing", "Switching", "BGP", "OSPF", "DNS", "DHCP",
        "Firewall", "Network Security", "LTE", "5G", "Telecom",
        "Telecommunications", "Cisco", "Juniper", "Network Engineering",
        "MPLS", "VLAN", "Optical Networks", "RF Engineering",
    },
    "SOFTWARE_ENGINEERING": {
        "Git", "Linux", "Unit Testing", "Agile", "Scrum", "OOP",
        "Design Patterns", "Microservices", "REST API", "gRPC", "System Design",
        "Data Structures", "Algorithms", "Code Review", "Debugging",
        "Multithreading", "Software Development", "Performance Optimization",
    },
}

# Category pairs considered "transferable" relative to each other.
RELATED_CATEGORIES: Set[frozenset] = {
    frozenset({"DEEP_LEARNING", "COMPUTER_VISION"}),
    frozenset({"DEEP_LEARNING", "CLASSICAL_ML"}),
    frozenset({"DEEP_LEARNING", "NLP"}),
    frozenset({"NLP", "GENAI"}),
    frozenset({"GENAI", "DEEP_LEARNING"}),
    frozenset({"MODEL_DEPLOYMENT", "QUANTIZATION"}),
    frozenset({"MODEL_DEPLOYMENT", "DEEP_LEARNING"}),
    frozenset({"MODEL_DEPLOYMENT", "SOFTWARE_ENGINEERING"}),
    frozenset({"MLOPS", "SOFTWARE_ENGINEERING"}),
    frozenset({"MLOPS", "CLOUD"}),
    frozenset({"MLOPS", "MODEL_DEPLOYMENT"}),
    frozenset({"CLASSICAL_ML", "DATA_ENGINEERING"}),
    frozenset({"DATA_ENGINEERING", "SOFTWARE_ENGINEERING"}),
    frozenset({"CLOUD", "SOFTWARE_ENGINEERING"}),
}

# ---------------------------------------------------------------------------
# Aliases: normalised spelling -> canonical skill
# ---------------------------------------------------------------------------
SKILL_ALIASES: Dict[str, str] = {
    # programming
    "python": "Python", "c++": "C++", "cpp": "C++", "c plus plus": "C++",
    "java": "Java", "golang": "Go", "go lang": "Go", "rust": "Rust",
    "javascript": "JavaScript", "typescript": "TypeScript", "bash": "Bash",
    "shell scripting": "Shell Scripting", "shell": "Shell Scripting",
    "sql": "SQL", "mysql": "SQL", "postgresql": "SQL", "matlab": "MATLAB",
    "embedded c": "Embedded C",
    # deep learning
    "pytorch": "PyTorch", "py torch": "PyTorch", "torch": "PyTorch",
    "tensorflow": "TensorFlow", "tf": "TensorFlow",
    "keras": "Keras", "jax": "JAX",
    "transformer": "Transformers", "transformers": "Transformers",
    "transformer models": "Transformer Models", "hugging face": "Hugging Face",
    "huggingface": "Hugging Face", "hf": "Hugging Face",
    "cnn": "CNN", "convolutional neural network": "CNN",
    "convolutional neural networks": "CNN", "rnn": "RNN",
    "lstm": "LSTM", "vit": "ViT", "vision transformer": "Vision Transformer",
    "attention": "Attention Mechanisms", "attention mechanisms": "Attention Mechanisms",
    "self attention": "Attention Mechanisms", "detr": "DETR", "yolo": "YOLO",
    "resnet": "ResNet", "unet": "U-Net", "u net": "U-Net",
    "bert": "BERT", "gpt": "GPT", "llm": "LLM", "large language model": "LLM",
    "large language models": "LLM", "llms": "LLM",
    "fine tuning": "Fine-tuning", "finetuning": "Fine-tuning",
    "fine-tuning": "Fine-tuning", "lora": "LoRA", "peft": "PEFT",
    "diffusion": "Diffusion Models", "diffusion models": "Diffusion Models",
    "neural network": "Neural Networks", "neural networks": "Neural Networks",
    "deep learning": "Deep Learning", "dl": "Deep Learning",
    "transfer learning": "Transfer Learning", "model training": "Model Training",
    # computer vision
    "computer vision": "Computer Vision", "cv": "Computer Vision",
    "object detection": "Object Detection", "detection": "Object Detection",
    "semantic segmentation": "Semantic Segmentation",
    "instance segmentation": "Instance Segmentation",
    "panoptic segmentation": "Panoptic Segmentation",
    "segmentation": "Semantic Segmentation",
    "bev": "BEV", "birds eye view": "Bird's Eye View",
    "bird's eye view": "Bird's Eye View", "lss": "LSS",
    "lift splat shoot": "LSS", "lift-splat-shoot": "LSS",
    "liftsplatshoot": "LSS",
    "3d perception": "3D Perception", "perception": "3D Perception",
    "multi object tracking": "Multi-Object Tracking", "mot": "Multi-Object Tracking",
    "image processing": "Image Processing", "opencv": "OpenCV",
    "point cloud": "Point Cloud", "point clouds": "Point Cloud",
    "point cloud processing": "Point Cloud Processing",
    "lidar": "LiDAR", "slam": "SLAM", "sensor fusion": "Sensor Fusion",
    "camera calibration": "Camera Calibration", "stereo vision": "Stereo Vision",
    "image classification": "Image Classification",
    "visual odometry": "Visual Odometry", "optical flow": "Optical Flow",
    # classical ML
    "machine learning": "Machine Learning", "ml": "Machine Learning",
    "scikit learn": "Scikit-learn", "sklearn": "Scikit-learn",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
    "random forest": "Random Forest", "svm": "SVM",
    "support vector machine": "SVM", "decision tree": "Decision Trees",
    "gradient boosting": "Gradient Boosting",
    "feature engineering": "Feature Engineering",
    "recommendation systems": "Recommendation Systems",
    "recommendation system": "Recommendation Systems",
    "recommender systems": "Recommendation Systems",
    "time series": "Time Series Forecasting",
    "time series forecasting": "Time Series Forecasting",
    # NLP
    "natural language processing": "Natural Language Processing", "nlp": "Natural Language Processing",
    "spacy": "spaCy", "nltk": "NLTK", "text classification": "Text Classification",
    "named entity recognition": "Named Entity Recognition", "ner": "NER",
    "text embeddings": "Text Embeddings", "embeddings": "Text Embeddings",
    "rag": "RAG", "retrieval augmented generation": "Retrieval-Augmented Generation",
    "retrieval augmented generation rag": "Retrieval-Augmented Generation",
    "vector database": "Vector Database", "vector db": "Vector Database",
    "vectorstore": "Vector Database", "prompt engineering": "Prompt Engineering",
    # genai
    "generative ai": "Generative AI", "genai": "Generative AI",
    "gen ai": "Generative AI", "vllm": "vLLM", "langchain": "LangChain",
    "llamaindex": "LlamaIndex", "agentic ai": "Agentic AI",
    "agent": "Agentic AI", "agents": "Agentic AI",
    "multi agent systems": "Multi-Agent Systems", "llm fine tuning": "LLM Fine-tuning",
    "llm deployment": "LLM Deployment", "llm evaluation": "LLM Evaluation",
    # model deployment
    "onnx": "ONNX", "onnxruntime": "ONNX Runtime", "onnx runtime": "ONNX Runtime",
    "ort": "ONNX Runtime", "tidl": "TIDL", "ti jacinto": "TI Jacinto",
    "jacinto": "TI Jacinto", "qnn": "QNN", "snpe": "SNPE",
    "tensorrt": "TensorRT", "tensor rt": "TensorRT", "trt": "TensorRT",
    "openvino": "OpenVINO", "tvm": "TVM", "tensorflow lite": "TensorFlow Lite",
    "tflite": "TFLite", "tf lite": "TensorFlow Lite", "coreml": "Core ML",
    "core ml": "Core ML", "edge ai": "Edge AI", "embedded ai": "Embedded AI",
    "model optimization": "Model Optimization", "optimization": "Model Optimization",
    "inference optimization": "Inference Optimization",
    "inference acceleration": "Inference Acceleration",
    "inference": "Model Serving", "cuda": "CUDA", "cudnn": "cuDNN",
    "triton": "Triton Inference Server", "triton inference server": "Triton Inference Server",
    "arm neon": "ARM NEON", "neon": "ARM NEON", "dsp": "DSP", "npu": "NPU",
    "model serving": "Model Serving", "serving": "Model Serving",
    "model compilation": "Model Compilation", "model deployment": "Model Deployment",
    "model conversion": "Model Conversion", "gpu programming": "GPU Programming",
    # quantization
    "int8": "INT8", "int 8": "INT8", "int16": "INT16", "int 16": "INT16",
    "int4": "INT4", "fp16": "FP16", "float16": "FP16", "fp8": "FP8",
    "half precision": "FP16", "ptq": "PTQ", "post training quantization": "PTQ",
    "qat": "QAT", "quantization aware training": "QAT",
    "quantization-aware training": "QAT",
    "quantization": "Quantization", "quantisation": "Quantization",
    "quantize": "Quantization", "gptq": "GPTQ", "awq": "AWQ",
    "calibration": "Calibration", "mixed precision": "Mixed Precision",
    "per channel quantization": "Per-channel Quantization",
    "knowledge distillation": "Knowledge Distillation", "distillation": "Knowledge Distillation",
    "pruning": "Pruning", "model pruning": "Pruning", "sparsity": "Sparsity",
    "smoothquant": "SmoothQuant", "weight quantization": "Weight Quantization",
    # mlops
    "docker": "Docker", "containerization": "Docker", "jenkins": "Jenkins",
    "ci cd": "CI/CD", "cicd": "CI/CD", "ci/cd": "CI/CD", "continuous integration": "CI/CD",
    "continuous integration and delivery": "CI/CD", "mlflow": "MLflow",
    "kubeflow": "Kubeflow", "kubernetes": "Kubernetes", "k8s": "Kubernetes",
    "airflow": "Airflow", "apache airflow": "Airflow",
    "model registry": "Model Registry", "model monitoring": "Model Monitoring",
    "feature store": "Feature Store", "ml pipelines": "ML Pipelines",
    "ml pipeline": "ML Pipelines", "pipelines": "ML Pipelines",
    "mlops": "MLOps", "ml operations": "MLOps", "experiment tracking": "Experiment Tracking",
    "dvc": "DVC", "data versioning": "Data Versioning",
    "github actions": "GitHub Actions",
    # cloud
    "aws": "AWS", "amazon web services": "AWS", "azure": "Azure",
    "microsoft azure": "Azure", "gcp": "GCP", "google cloud": "GCP",
    "google cloud platform": "GCP", "sagemaker": "SageMaker",
    "vertex ai": "Vertex AI", "ec2": "EC2", "s3": "S3", "lambda": "Lambda",
    # data
    "pandas": "Pandas", "numpy": "NumPy", "spark": "Spark", "pyspark": "PySpark",
    "apache spark": "Spark", "etl": "ETL", "data pipelines": "Data Pipelines",
    "data pipeline": "Data Pipelines", "data engineering": "Data Pipelines",
    "kafka": "Apache Kafka", "apache kafka": "Apache Kafka",
    "data warehousing": "Data Warehousing", "tableau": "Tableau",
    "power bi": "Power BI", "big data": "Big Data",
    # networking
    "tcp ip": "TCP/IP", "tcp/ip": "TCP/IP", "tcpip": "TCP/IP",
    "routing": "Routing", "switching": "Switching", "bgp": "BGP", "ospf": "OSPF",
    "dns": "DNS", "dhcp": "DHCP", "firewall": "Firewall",
    "network security": "Network Security", "lte": "LTE", "5g": "5G",
    "telecom": "Telecom", "telecom engineering": "Telecommunications",
    "telecommunications": "Telecommunications", "cisco": "Cisco",
    "juniper": "Juniper", "network engineering": "Network Engineering",
    "networking": "Network Engineering", "mpls": "MPLS", "vlan": "VLAN",
    # software engineering
    "git": "Git", "github": "Git", "linux": "Linux", "unix": "Linux",
    "unit testing": "Unit Testing", "unit tests": "Unit Testing",
    "testing": "Unit Testing", "agile": "Agile", "scrum": "Scrum",
    "oop": "OOP", "object oriented programming": "OOP",
    "design patterns": "Design Patterns", "microservices": "Microservices",
    "rest api": "REST API", "rest apis": "REST API", "restful api": "REST API",
    "grpc": "gRPC", "system design": "System Design",
    "data structures": "Data Structures", "algorithms": "Algorithms",
    "dsa": "Data Structures", "code review": "Code Review",
    "multithreading": "Multithreading", "debugging": "Debugging",
    "software development": "Software Development",
    "performance optimization": "Performance Optimization",
}

# ---------------------------------------------------------------------------
# Lookup indexes (built once, treated as immutable)
# ---------------------------------------------------------------------------
_CANONICAL_TO_CATEGORY: Dict[str, str] = {}
for _cat, _skills in SKILL_TAXONOMY.items():
    for _skill in _skills:
        _CANONICAL_TO_CATEGORY[_skill.lower()] = _cat
        SKILL_ALIASES.setdefault(_skill.lower(), _skill)

# Longest aliases first so that "onnx runtime" wins over "onnx".
_ALIASES_BY_LENGTH: List[str] = sorted(SKILL_ALIASES.keys(), key=len, reverse=True)
_SUBSTRING_ALIASES: List[str] = [a for a in _ALIASES_BY_LENGTH if len(a) >= 3]

_WORD_RE = re.compile(r"[a-z0-9#+/.]+")
_NON_WORD_RE = re.compile(r"[^a-z0-9#+/ ]+")


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation that interferes with skill matching."""
    text = (text or "").lower().replace("’", "'")
    text = _NON_WORD_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def resolve_skill(name: str) -> Tuple[str, Optional[str]]:
    """
    Resolve a raw skill string to ``(canonical_name, category)``.

    Unknown skills keep a prettified version of their name and have no
    category (they can still be matched EXACT by name).
    """
    if not name or not name.strip():
        return "", None
    norm = normalize_text(name)
    if not norm:
        return name.strip(), None
    # exact alias hit (covers single-letter skills like "c" safely)
    if norm in SKILL_ALIASES:
        canonical = SKILL_ALIASES[norm]
        return canonical, _CANONICAL_TO_CATEGORY.get(canonical.lower())
    # substring hit for longer aliases, e.g. "cuda kernel programming" -> CUDA
    for alias in _SUBSTRING_ALIASES:
        if f" {alias} " in f" {norm} " or norm.startswith(alias + " ") or norm.endswith(" " + alias):
            canonical = SKILL_ALIASES[alias]
            return canonical, _CANONICAL_TO_CATEGORY.get(canonical.lower())
    return _prettify(name), None


def _prettify(name: str) -> str:
    name = re.sub(r"\s+", " ", name.strip())
    return name[:80]


def category_of(skill: str) -> Optional[str]:
    _, category = resolve_skill(skill)
    return category


def categories_related(category_a: str, category_b: str) -> bool:
    return frozenset({category_a, category_b}) in RELATED_CATEGORIES


def extract_skills_from_text(text: str, max_skills: int = 60) -> List[str]:
    """
    Deterministically detect taxonomy skills mentioned in free text.
    Used to augment / validate LLM extraction and as an offline fallback.
    """
    if not text:
        return []
    norm = f" {normalize_text(text)} "
    found: List[str] = []
    seen: Set[str] = set()
    for alias in _ALIASES_BY_LENGTH:
        if alias in seen:
            continue
        if len(alias) < 3:
            continue  # avoid noisy single/two-letter matches in free text
        if f" {alias} " in norm:
            canonical = SKILL_ALIASES[alias]
            if canonical.lower() not in seen:
                seen.add(canonical.lower())
                found.append(canonical)
            if len(found) >= max_skills:
                break
    return found


# ---------------------------------------------------------------------------
# Skill matching (EXACT / RELATED / TRANSFERABLE / MISSING)
# ---------------------------------------------------------------------------
@dataclass
class _CandidateIndex:
    """Normalised view of the candidate's skills for fast lookup."""

    canonical: Dict[str, str] = field(default_factory=dict)  # canonical -> original
    categories: Dict[str, List[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, skills: Iterable[str]) -> "_CandidateIndex":
        idx = cls()
        for raw in skills:
            if not raw or not str(raw).strip():
                continue
            canonical, category = resolve_skill(str(raw))
            if not canonical:
                continue
            idx.canonical.setdefault(canonical.lower(), str(raw).strip())
            if category:
                idx.categories.setdefault(category, []).append(canonical)
        return idx

    def best_in_category(self, category: str) -> Optional[str]:
        return self.categories.get(category, [None])[0] if self.categories.get(category) else None


def match_skills(
    job_skills: Iterable[str],
    candidate_skills: Iterable[str],
    required: Optional[Iterable[str]] = None,
    preferred: Optional[Iterable[str]] = None,
) -> SkillMatchResult:
    """
    Classify every job skill against the candidate's skills.

    Scoring (deterministic, RULE 1):
        EXACT        -> 1.0
        RELATED      -> 0.75  (same taxonomy category)
        TRANSFERABLE -> 0.5   (adjacent category)
        MISSING      -> 0
    Required skills weigh 1.0, preferred skills 0.5.  A job with no listed
    skills gets a neutral score of 50.
    """
    index = _CandidateIndex.build(candidate_skills)
    required = {resolve_skill(s)[0].lower() for s in (required or []) if str(s).strip()}
    preferred = {resolve_skill(s)[0].lower() for s in (preferred or []) if str(s).strip()}

    matches: List[SkillMatch] = []
    total_weight = 0.0
    weighted_score = 0.0

    seen: Set[str] = set()
    for raw in job_skills:
        if not raw or not str(raw).strip():
            continue
        canonical, category = resolve_skill(str(raw))
        key = canonical.lower()
        if key in seen or not canonical:
            continue
        seen.add(key)

        # 1. EXACT: same canonical skill
        if key in index.canonical:
            matches.append(SkillMatch(canonical, index.canonical[key], SkillMatchType.EXACT, category))
            value = 1.0
        # 2. RELATED: candidate has a skill in the same category
        elif category and index.best_in_category(category):
            matches.append(
                SkillMatch(canonical, index.best_in_category(category), SkillMatchType.RELATED, category)
            )
            value = 0.75
        # 3. TRANSFERABLE: candidate has a skill in an adjacent category
        elif category and (transferable_skill := _has_transferable(index, category)):
            matches.append(SkillMatch(canonical, transferable_skill, SkillMatchType.TRANSFERABLE, category))
            value = 0.5
        else:
            matches.append(SkillMatch(canonical, None, SkillMatchType.MISSING, category))
            value = 0.0

        if required or preferred:
            weight = 1.0 if key in required else 0.5
        else:
            weight = 1.0
        total_weight += weight
        weighted_score += value * weight

    score = 50.0 if not matches else (100.0 * weighted_score / total_weight if total_weight else 50.0)
    return SkillMatchResult(matches=matches, score=round(min(100.0, max(0.0, score)), 1))


def _has_transferable(index: _CandidateIndex, category: str) -> Optional[str]:
    for candidate_category, skills in index.categories.items():
        if categories_related(category, candidate_category):
            return skills[0]
    return None


def missing_must_haves(
    must_have_skills: Iterable[str],
    candidate_skills: Iterable[str],
    match_level: str = "related",
) -> List[str]:
    """Return the subset of must-have skills the candidate cannot satisfy."""
    index = _CandidateIndex.build(candidate_skills)
    missing: List[str] = []
    for raw in must_have_skills:
        if not str(raw).strip():
            continue
        canonical, category = resolve_skill(str(raw))
        key = canonical.lower()
        if key in index.canonical:
            continue
        if match_level == "related":
            if category and index.best_in_category(category):
                continue  # a related skill satisfies the must-have
            if category and _has_transferable(index, category):
                continue
        missing.append(canonical)
    return missing


def taxonomy_rows() -> List[Dict[str, object]]:
    """Rows for the ``skills`` SQLite table."""
    rows: List[Dict[str, object]] = []
    for category, skills in SKILL_TAXONOMY.items():
        for skill in sorted(skills):
            rows.append({"name": skill, "category": category, "is_alias": 0})
    for alias, canonical in SKILL_ALIASES.items():
        category = _CANONICAL_TO_CATEGORY.get(canonical.lower())
        if category:
            rows.append({"name": alias, "category": category, "is_alias": 1})
    return rows
