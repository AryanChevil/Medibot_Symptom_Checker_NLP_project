"""
symptom_core_v4.py

Core logic for:
- training model on Training.csv
- semantic symptom matching with sentence-transformers
- evidence-aware ranking + triage layer
- knowledge base lookup (description, precautions, severity)

Designed to be imported by a GUI (Streamlit) and a CLI.

DISCLAIMER: Educational demo only. Not medical advice.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Set

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore


def normalize_symptom(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def split_phrases(text: str) -> List[str]:
    t = text.lower()
    t = re.sub(r"\b(and|&|with)\b", ",", t)
    parts = [p.strip(" .;:!?\t\r\n") for p in t.split(",")]
    parts = [p for p in parts if p]
    return parts[:12]


def resolve_file(data_dir: Path, filename: str) -> Path:
    p1 = data_dir / filename
    p2 = data_dir / "Data" / filename
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Could not find {filename}. Looked in: {p1} and {p2}")


@dataclass
class KnowledgeBase:
    severity: Dict[str, int]
    description: Dict[str, str]
    precautions: Dict[str, List[str]]


def load_dataset(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = resolve_file(data_dir, "Training.csv")
    test_path = resolve_file(data_dir, "Testing.csv")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def load_knowledge_base(data_dir: Path) -> KnowledgeBase:
    sev_path = resolve_file(data_dir, "Symptom_severity.csv")
    desc_path = resolve_file(data_dir, "symptom_Description.csv")
    prec_path = resolve_file(data_dir, "symptom_precaution.csv")

    sev_df = pd.read_csv(sev_path, header=None, names=["symptom", "severity"])
    severity = {str(s).strip(): int(v) for s, v in sev_df.values if pd.notna(s) and pd.notna(v)}

    desc_df = pd.read_csv(desc_path, header=None, names=["disease", "description"])
    description = {str(d).strip(): str(t).strip() for d, t in desc_df.values if pd.notna(d) and pd.notna(t)}

    prec_df = pd.read_csv(prec_path, header=None)
    precautions: Dict[str, List[str]] = {}
    for row in prec_df.itertuples(index=False):
        disease = str(row[0]).strip()
        items = [str(x).strip() for x in row[1:] if pd.notna(x) and str(x).strip()]
        precautions[disease] = items

    return KnowledgeBase(severity=severity, description=description, precautions=precautions)


def build_model(name: str, random_state: int = 42):
    name = name.lower()
    if name in {"rf", "randomforest"}:
        return RandomForestClassifier(
            n_estimators=600,
            random_state=random_state,
            n_jobs=-1,
        )
    if name in {"dt", "tree", "decisiontree"}:
        return DecisionTreeClassifier(random_state=random_state)
    raise ValueError("Unknown model. Choose rf or dt.")


@dataclass
class TrainedSystem:
    model: object
    symptoms: List[str]
    reduced_by_disease: pd.DataFrame
    disease_symptom_sets: Dict[str, Set[str]]
    blocked_symptoms: Set[str]


def train_system(train_df: pd.DataFrame, model_name: str = "rf", do_eval: bool = False) -> TrainedSystem:
    if "prognosis" not in train_df.columns:
        raise ValueError("Training.csv must contain 'prognosis' column.")
    X = train_df.drop(columns=["prognosis"])
    y = train_df["prognosis"].astype(str)

    model = build_model(model_name)

    if do_eval:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy")
        _ = (cv_scores.mean(), cv_scores.std(), accuracy_score(y_val, model.fit(X_tr, y_tr).predict(X_val)))

    model.fit(X, y)

    reduced = train_df.groupby("prognosis").max()
    disease_symptom_sets: Dict[str, Set[str]] = {}
    for dis in reduced.index:
        row = reduced.loc[dis]
        disease_symptom_sets[str(dis)] = set(row[row == 1].index.tolist())

    blocked = {"extra_marital_contacts", "history_of_alcohol_consumption"}

    return TrainedSystem(
        model=model,
        symptoms=list(X.columns),
        reduced_by_disease=reduced,
        disease_symptom_sets=disease_symptom_sets,
        blocked_symptoms=blocked,
    )


def symptoms_to_vector(symptoms: Sequence[str], all_symptoms: Sequence[str]) -> np.ndarray:
    idx = {s: i for i, s in enumerate(all_symptoms)}
    v = np.zeros(len(all_symptoms), dtype=float)
    for s in symptoms:
        if s in idx:
            v[idx[s]] = 1.0
    return v


def evidence_scores(user_symptoms: Sequence[str], system: TrainedSystem) -> Dict[str, float]:
    user = set(user_symptoms)
    if not user:
        return {d: 0.0 for d in system.disease_symptom_sets.keys()}

    scores: Dict[str, float] = {}
    for disease, dset in system.disease_symptom_sets.items():
        inter = len(user & dset)
        if inter == 0:
            scores[disease] = 0.0
            continue

        coverage = inter / len(user)
        specificity = inter / max(len(dset), 1)

        if len(dset) <= 5 and inter < 2:
            scores[disease] = 0.0
            continue

        denom = coverage + specificity
        f1 = 0.0 if denom == 0 else 2 * coverage * specificity / denom

        if coverage < 0.5:
            f1 *= 0.6

        scores[disease] = float(f1)

    return scores


def rank_predictions(
    system: TrainedSystem,
    v: np.ndarray,
    user_symptoms: Sequence[str],
) -> List[Tuple[str, float, float, float, int, int]]:
    X1 = pd.DataFrame([v], columns=system.symptoms)
    model = system.model

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X1)[0]
        classes = list(model.classes_)
        proba_map = {classes[i]: float(proba[i]) for i in range(len(classes))}
    else:
        pred = str(model.predict(X1)[0])
        proba_map = {pred: 1.0}

    ev = evidence_scores(user_symptoms, system)
    user_set = set(user_symptoms)

    rows = []
    for disease in system.disease_symptom_sets.keys():
        p = proba_map.get(disease, 0.0)
        e = ev.get(disease, 0.0)
        combined = 0.35 * p + 0.65 * e
        dset = system.disease_symptom_sets[disease]
        inter = len(user_set & dset)
        rows.append((disease, float(combined), float(p), float(e), int(inter), int(len(dset))))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def severity_advice(symptoms: Sequence[str], days: int, kb: KnowledgeBase) -> str:
    if not symptoms:
        return "No symptoms provided."
    total = sum(kb.severity.get(s, 0) for s in symptoms)
    score = (total * max(days, 1)) / (len(symptoms) + 1)
    if score > 13:
        return "Based on severity+duration, consider consulting a doctor."
    return "Severity seems lower, but monitor symptoms and take precautions."


@dataclass
class SemanticMatcher:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    threshold: float = 0.55
    topn: int = 12

    def __post_init__(self) -> None:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        self.model = SentenceTransformer(self.model_name)
        self.symptoms: List[str] = []
        self.symptom_texts: List[str] = []
        self.emb = None

    def build(self, symptom_names: Sequence[str]) -> None:
        self.symptoms = list(symptom_names)
        self.symptom_texts = [s.replace("_", " ") for s in self.symptoms]
        self.emb = self.model.encode(self.symptom_texts, normalize_embeddings=True, show_progress_bar=False)

    def match(self, text: str) -> List[Tuple[str, float]]:
        if self.emb is None:
            raise RuntimeError("Matcher not built. Call build() first.")
        q = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = (self.emb @ q).astype(float)
        idx = np.argsort(sims)[::-1]
        out: List[Tuple[str, float]] = []
        for i in idx[: self.topn]:
            if sims[i] < self.threshold:
                break
            out.append((self.symptoms[int(i)], float(sims[i])))
        return out


PREFERENCE_RULES: List[Tuple[re.Pattern, List[str]]] = [
    (re.compile(r"\brunny\s*nose\b", re.I), ["runny_nose", "congestion"]),
    (re.compile(r"\bsore\s*throat\b", re.I), ["throat_irritation"]),
    (re.compile(r"\bthroat\b", re.I), ["throat_irritation"]),
    (re.compile(r"\bwhite\s*patch(es)?\b|\bpatch(es)?\b", re.I), ["patches_in_throat"]),
    (re.compile(r"\bbody\s*ache(s)?\b|\bbody\s*pain\b|\baches\b", re.I), ["muscle_pain", "joint_pain"]),
    (re.compile(r"\bfever\b", re.I), ["mild_fever", "high_fever"]),
    (re.compile(r"\bhigh\s*fever\b", re.I), ["high_fever", "mild_fever"]),
    (re.compile(r"\bchills?\b", re.I), ["chills"]),
    (re.compile(r"\bcough\b", re.I), ["cough"]),
    (re.compile(r"\bsneeze|sneezing\b", re.I), ["continuous_sneezing"]),
]

AVOID_AUTO_ADD: Set[str] = {"patches_in_throat", "extra_marital_contacts", "muscle_wasting"}


def explicitly_mentioned(symptom: str, raw_text: str) -> bool:
    t = raw_text.lower()
    if symptom == "patches_in_throat":
        return bool(re.search(r"white\s*patch|patch", t))
    if symptom == "extra_marital_contacts":
        return bool(re.search(r"extra\s*marital|unprotected\s*sex|multiple\s*partners", t))
    if symptom == "muscle_wasting":
        return bool(re.search(r"wasting|loss\s*of\s*muscle", t))
    return False


def recommend_symptoms_from_text(
    raw_text: str,
    matcher: SemanticMatcher,
    all_symptoms: Sequence[str],
    max_reco: int = 6
) -> Tuple[List[Tuple[str, float]], List[str]]:
    all_set = set(all_symptoms)
    top = matcher.match(raw_text)
    phrases = split_phrases(raw_text)

    reco: List[str] = []

    for ph in phrases:
        ph_matches = matcher.match(ph)
        if not ph_matches:
            continue

        preferred: Optional[str] = None
        for patt, prefs in PREFERENCE_RULES:
            if patt.search(ph):
                for p in prefs:
                    if p in all_set:
                        preferred = p
                        break
            if preferred:
                break

        if preferred:
            if preferred in AVOID_AUTO_ADD and not explicitly_mentioned(preferred, raw_text):
                continue
            if preferred not in reco:
                reco.append(preferred)
            continue

        best = ph_matches[0][0]
        if best in AVOID_AUTO_ADD and not explicitly_mentioned(best, raw_text):
            continue
        if best not in reco:
            reco.append(best)

    for s, _ in top:
        if len(reco) >= max_reco:
            break
        if s in reco:
            continue
        if s in AVOID_AUTO_ADD and not explicitly_mentioned(s, raw_text):
            continue
        reco.append(s)

    reco = [s for s in reco if s in all_set][:max_reco]
    return top, reco


RED_FLAG_SYMPTOMS: Set[str] = {
    "chest_pain",
    "breathlessness",
    "altered_sensorium",
    "coma",
    "blood_in_sputum",
    "bloody_stool",
    "fast_heart_rate",
    "yellowing_of_eyes",
}

MILD_URTI_SYMPTOMS: Set[str] = {
    "runny_nose",
    "throat_irritation",
    "continuous_sneezing",
    "cough",
    "congestion",
    "mild_fever",
    "chills",
    "headache",
    "fatigue",
    "muscle_pain",
    "joint_pain",
    "sinus_pressure",
    "phlegm",
}

COMMON_CONDITIONS_BASE = {"Common Cold", "Allergy", "Bronchial Asthma", "Pneumonia", "Gastroenteritis"}

SERIOUS_CONDITIONS_BASE = {
    "AIDS",
    "Tuberculosis",
    "Hepatitis B",
    "Hepatitis C",
    "Hepatitis D",
    "Hepatitis E",
    "Heart attack",
    "Paralysis (brain hemorrhage)",
    "Malaria",
    "Dengue",
    "Typhoid",
    "Jaundice",
}


def apply_triage(
    rows_sorted: List[Tuple[str, float, float, float, int, int]],
    user_symptoms: Sequence[str],
    days: int,
    system: TrainedSystem,
) -> Tuple[List[Tuple[str, float, float, float, float, int, int]], str]:
    if not rows_sorted:
        return [], ""

    user = set(user_symptoms)
    red_flags = sorted(list(user & RED_FLAG_SYMPTOMS))

    urti_ratio = (len(user & MILD_URTI_SYMPTOMS) / len(user)) if user else 0.0
    mild_urti_case = (urti_ratio >= 0.60 and len(red_flags) == 0 and days <= 7 and len(user) <= 10)

    top = rows_sorted[0]
    second = rows_sorted[1] if len(rows_sorted) > 1 else None
    top_combined, top_p, top_e = top[1], top[2], top[3]
    margin = (top_combined - second[1]) if second else 1.0
    low_conf = (top_p < 0.55 and top_e < 0.55) or (margin < 0.06) or (len(user) < 3)

    common_set = set(COMMON_CONDITIONS_BASE) & set(system.disease_symptom_sets.keys())
    serious_set = set(SERIOUS_CONDITIONS_BASE) & set(system.disease_symptom_sets.keys())

    adjusted: List[Tuple[str, float, float, float, float, int, int]] = []
    for disease, combined, p, e, inter, dcount in rows_sorted:
        rank_score = combined

        if mild_urti_case and low_conf:
            if disease in common_set:
                rank_score += 0.14
            if disease in serious_set and (inter < 2 or e < 0.40):
                rank_score -= 0.30

        if e >= 0.70:
            rank_score += 0.05

        adjusted.append((disease, float(rank_score), float(combined), float(p), float(e), int(inter), int(dcount)))

    adjusted.sort(key=lambda x: x[1], reverse=True)

    note_parts: List[str] = []
    if red_flags:
        note_parts.append(
            "⚠️ Red-flag symptom(s) detected: "
            + ", ".join(s.replace("_", " ") for s in red_flags)
            + ". If severe/sudden/worsening, seek urgent medical help."
        )
    elif mild_urti_case and low_conf:
        commons = [d for d, *_ in adjusted if d in common_set][:3]
        if commons:
            note_parts.append(
                "Triage note: symptoms look like a common upper-respiratory pattern (cold/allergy-like) "
                "and model confidence is low. Prioritizing common conditions first: "
                + ", ".join(commons)
                + "."
            )
        else:
            note_parts.append("Triage note: low confidence and non-specific symptoms. Add more symptoms.")
    elif low_conf:
        note_parts.append("Triage note: model confidence is low. Add more symptoms for better suggestions.")

    return adjusted, "\n\n".join(note_parts).strip()
