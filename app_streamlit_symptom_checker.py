from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
import streamlit as st

from symptom_core_v4 import (
    load_dataset,
    load_knowledge_base,
    train_system,
    SemanticMatcher,
    recommend_symptoms_from_text,
    symptoms_to_vector,
    rank_predictions,
    apply_triage,
    severity_advice,
)

# PDF generation (simple report)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors


# -----------------------------
# Page setup + light styling
# -----------------------------
st.set_page_config(page_title="Symptom Checker Prototype", page_icon="🩺", layout="wide")

st.markdown(
    """
<style>
.chip-wrap {display:flex; flex-wrap:wrap; gap:8px; margin-top:6px;}
.chip {
  display:inline-flex; align-items:center; gap:8px;
  padding:6px 10px; border-radius:999px;
  border:1px solid rgba(49, 51, 63, 0.2);
  background: rgba(49, 51, 63, 0.04);
  font-size: 13px;
}
.chip small {opacity:0.65;}
.block {padding:14px; border:1px solid rgba(49, 51, 63, 0.12); border-radius:14px;}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource
def load_resources(data_dir: str, model_name: str, sim_threshold: float):
    base = Path(data_dir).expanduser().resolve()
    train_df, _ = load_dataset(base)
    kb = load_knowledge_base(base)
    system = train_system(train_df, model_name=model_name, do_eval=False)

    matcher = SemanticMatcher(threshold=float(sim_threshold))
    matcher.build(system.symptoms)

    return system, kb, matcher


def init_state():
    st.session_state.setdefault("name", "")
    st.session_state.setdefault("days", 3)
    st.session_state.setdefault("symptoms", [])
    st.session_state.setdefault("chat", [])  # list[dict]: {role, content, meta?}
    st.session_state.setdefault("last_pred", None)
    st.session_state.setdefault("data_dir", str(Path(__file__).resolve().parent))
    st.session_state.setdefault("model_name", "rf")
    st.session_state.setdefault("sim_threshold", 0.55)


init_state()


# -----------------------------
# Confidence / uncertainty heuristics
# -----------------------------
def compute_confidence(adjusted_rows_top3: List[Tuple], total_symptoms: int) -> Dict[str, Any]:
    """
    adjusted_rows_top3 row format:
      (disease, rank_score, combined, model_prob, evidence, match_count, disease_symptom_count)
    """
    top = adjusted_rows_top3[0]
    second = adjusted_rows_top3[1] if len(adjusted_rows_top3) > 1 else None

    model_prob = float(top[3])
    evidence = float(top[4])
    margin = float(top[2] - second[2]) if second else 1.0

    # UI confidence in [0, 1]
    conf = 0.55 * evidence + 0.45 * model_prob
    conf = max(0.0, min(1.0, conf))

    uncertain = (
        total_symptoms < 3
        or (model_prob < 0.55 and evidence < 0.55)
        or margin < 0.05
        or conf < 0.55
    )

    return {
        "confidence": conf,
        "uncertain": uncertain,
        "model_prob": model_prob,
        "evidence": evidence,
        "margin": margin,
    }


def render_confidence_bar(info: Dict[str, Any]):
    conf = info["confidence"]
    st.markdown("**Confidence (heuristic)**")
    st.progress(conf)
    st.caption(
        f"Confidence={conf:.0%}  •  model={info['model_prob']:.0%}  •  evidence={info['evidence']:.2f}  •  margin={info['margin']:.3f}"
    )
    if info["uncertain"]:
        st.warning(
            "Uncertain mode: the system is not confident. Add more symptoms or re-check matches.\n\n"
            "For demos: show top-3 and avoid claiming a definitive diagnosis."
        )


# -----------------------------
# Report export (HTML + PDF)
# -----------------------------
def build_report_payload(bundle: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": bundle.get("name", ""),
        "days": bundle.get("days", 0),
        "symptoms": bundle.get("symptoms", []),
        "triage_note": bundle.get("triage_note", ""),
        "advice": bundle.get("advice", ""),
        "top3": bundle.get("top3", []),
        "top_disease": bundle.get("top_disease", ""),
        "description": bundle.get("description", ""),
        "precautions": bundle.get("precautions", []),
        "confidence": bundle.get("confidence_info", {}),
    }


def export_html(report: Dict[str, Any]) -> bytes:
    rows_html = "".join(
        f"<tr><td>{i+1}</td><td><b>{r['Disease']}</b></td><td>{r['Confidence']}</td><td>{r['Notes']}</td></tr>"
        for i, r in enumerate(report["top3"])
    )
    prec_html = "".join(f"<li>{p}</li>" for p in report["precautions"])
    sym_html = "".join(
        f"<span style='display:inline-block;padding:4px 10px;border:1px solid #ddd;border-radius:999px;margin:3px;'>{s}</span>"
        for s in report["symptoms"]
    )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Symptom Checker Report</title>
</head>
<body style="font-family:Arial, sans-serif; max-width: 920px; margin: 24px auto; line-height:1.45;">
  <h2>Symptom Checker Report (Educational Demo)</h2>
  <p style="opacity:0.8;">Generated: {report['generated_at']}</p>

  <h3>Patient Input</h3>
  <p><b>Name:</b> {report['name'] or '-'}<br/>
     <b>Days:</b> {report['days']}</p>

  <h4>Selected Symptoms</h4>
  <div>{sym_html or '<i>No symptoms</i>'}</div>

  <h3>Triage & Advisory</h3>
  <p><b>Triage note:</b> {report['triage_note'] or '-'}</p>
  <p><b>Advisory:</b> {report['advice']}</p>

  <h3>Top Predictions</h3>
  <table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse; width:100%;">
    <thead><tr><th>#</th><th>Disease</th><th>Confidence</th><th>Notes</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>

  <h3>Top Disease Details</h3>
  <p><b>{report['top_disease']}</b></p>
  <p>{report['description']}</p>

  <h4>Precautions</h4>
  <ol>{prec_html or '<li>-</li>'}</ol>

  <hr/>
  <p style="opacity:0.7; font-size: 12px;">
    Disclaimer: This report is generated by a student prototype and is NOT medical advice.
  </p>
</body>
</html>
"""
    return html.encode("utf-8")


def export_pdf(report: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Symptom Checker Report (Educational Demo)", styles["Title"]))
    story.append(Paragraph(f"Generated: {report['generated_at']}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Patient Input", styles["Heading2"]))
    story.append(Paragraph(f"Name: {report['name'] or '-'}", styles["Normal"]))
    story.append(Paragraph(f"Days with symptoms: {report['days']}", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Selected Symptoms", styles["Heading2"]))
    sym = ", ".join(report["symptoms"]) if report["symptoms"] else "-"
    story.append(Paragraph(sym, styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Triage & Advisory", styles["Heading2"]))
    story.append(Paragraph(f"Triage note: {report['triage_note'] or '-'}", styles["Normal"]))
    story.append(Paragraph(f"Advisory: {report['advice']}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Top Predictions", styles["Heading2"]))
    table_data = [["#", "Disease", "Confidence", "Notes"]]
    for i, r in enumerate(report["top3"], start=1):
        table_data.append([str(i), r["Disease"], r["Confidence"], r["Notes"]])

    tbl = Table(table_data, colWidths=[28, 190, 90, 230])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(tbl)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Top Disease Details", styles["Heading2"]))
    story.append(Paragraph(report["top_disease"], styles["Heading3"]))
    story.append(Paragraph(report["description"] or "-", styles["Normal"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Precautions", styles["Heading3"]))
    prec = report.get("precautions", [])
    if prec:
        for p in prec:
            story.append(Paragraph(f"• {p}", styles["Normal"]))
    else:
        story.append(Paragraph("-", styles["Normal"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Disclaimer: This report is generated by a student prototype and is NOT medical advice.", styles["Italic"]))
    doc.build(story)
    return buf.getvalue()


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🧩 Prototype Controls")

    data_dir = st.text_input("CSV folder (data-dir)", value=st.session_state["data_dir"])
    model_name = st.selectbox("Model", ["rf", "dt"], index=0 if st.session_state["model_name"] == "rf" else 1)
    sim_threshold = st.slider("Semantic match threshold", 0.45, 0.85, float(st.session_state["sim_threshold"]), 0.01)

    st.session_state["data_dir"] = data_dir
    st.session_state["model_name"] = model_name
    st.session_state["sim_threshold"] = sim_threshold

    st.markdown("---")
    st.caption("Run: streamlit run app_streamlit_symptom_checker_pro.py")

    st.markdown("---")
    st.subheader("Export")
    bundle = st.session_state.get("last_pred")
    if bundle:
        report = build_report_payload(bundle)
        st.download_button(
            "⬇️ Download HTML report",
            data=export_html(report),
            file_name="symptom_checker_report.html",
            mime="text/html",
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Download PDF report",
            data=export_pdf(report),
            file_name="symptom_checker_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.info("Run a prediction to enable export.")


# -----------------------------
# Load resources
# -----------------------------
try:
    system, kb, matcher = load_resources(data_dir, model_name, sim_threshold)
except Exception as e:
    st.error(f"Could not load resources: {e}")
    st.stop()


# -----------------------------
# Main layout
# -----------------------------
st.title("🩺 MediBot-Symptom Checker")
st.caption("Educational demo only — NOT medical advice.")

left, right = st.columns([1.15, 0.85], gap="large")

with right:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.subheader("Selected symptoms")

    name = st.text_input("Name", value=st.session_state["name"], placeholder="Optional")
    st.session_state["name"] = name

    days = st.number_input("Days with symptoms", min_value=0, max_value=365, value=int(st.session_state["days"]), step=1)
    st.session_state["days"] = int(days)

    symptoms: List[str] = st.session_state.get("symptoms", [])

    if symptoms:
        st.markdown(
            "<div class='chip-wrap'>"
            + "".join([f"<span class='chip'>{s}<small>symptom</small></span>" for s in symptoms])
            + "</div>",
            unsafe_allow_html=True,
        )
        st.caption("Remove any symptom below:")
        for s in symptoms:
            if st.button(f"✖ Remove: {s}", key=f"rm_{s}", use_container_width=True):
                st.session_state["symptoms"] = [x for x in symptoms if x != s]
                st.rerun()
    else:
        st.info("No symptoms selected yet.")

    st.markdown("---")
    st.subheader("Add symptom (autocomplete)")
    add_sym = st.selectbox("Search symptom", options=system.symptoms, index=0)
    if st.button("➕ Add selected symptom", use_container_width=True):
        cur = set(st.session_state.get("symptoms", []))
        cur.add(add_sym)
        st.session_state["symptoms"] = sorted(cur)
        st.success("Added.")
        st.rerun()

    st.markdown("---")
    predict = st.button("🧠 Predict", type="primary", use_container_width=True)
    clear_all = st.button("🧹 Clear all", use_container_width=True)

    if clear_all:
        st.session_state["symptoms"] = []
        st.session_state["chat"] = []
        st.session_state["last_pred"] = None
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

with left:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.subheader("Chat")
    st.caption("Type symptoms in natural language. The assistant will suggest symptoms; you can add them.")

    for m in st.session_state.get("chat", []):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            meta = m.get("meta")
            if meta and meta.get("type") == "matches":
                df = pd.DataFrame(meta["matches"], columns=["symptom", "similarity"])
                df["recommended"] = df["symptom"].isin(set(meta.get("recommended", []))).map(lambda x: "✅" if x else "")
                st.dataframe(df, use_container_width=True, hide_index=True)

                cols = st.columns([1, 1, 1])
                if cols[0].button("➕ Add recommended", key=f"add_reco_{meta['id']}", use_container_width=True):
                    cur = set(st.session_state.get("symptoms", []))
                    for s in meta.get("recommended", []):
                        cur.add(s)
                    st.session_state["symptoms"] = sorted(cur)
                    st.success("Added recommended symptoms.")
                    st.rerun()

                opts = [s for s, _ in meta["matches"]]
                default = [s for s in opts if s in set(meta.get("recommended", []))]
                pick = cols[1].multiselect("Pick symptoms", options=opts, default=default, key=f"pick_{meta['id']}")
                if cols[2].button("Add selected", key=f"add_sel_{meta['id']}", use_container_width=True):
                    cur = set(st.session_state.get("symptoms", []))
                    for s in pick:
                        cur.add(s)
                    st.session_state["symptoms"] = sorted(cur)
                    st.success("Added selected symptoms.")
                    st.rerun()

    user_text = st.chat_input("Describe your symptoms (e.g., runny nose, sneezing, mild fever)...")

    if user_text:
        st.session_state["chat"].append({"role": "user", "content": user_text})

        matches, reco = recommend_symptoms_from_text(user_text, matcher, system.symptoms, max_reco=6)

        if not matches:
            assistant_text = (
                "I couldn't confidently map that to known symptoms.\n\n"
                "Try rephrasing with simpler phrases like: **runny nose, cough, sore throat, fever**."
            )
            st.session_state["chat"].append({"role": "assistant", "content": assistant_text})
        else:
            assistant_text = (
                "Here are the closest symptom matches I found.\n\n"
                f"Recommended (phrase-aware): **{', '.join(reco)}**\n\n"
                "You can add the recommended symptoms or pick manually."
            )
            meta = {"type": "matches", "matches": matches, "recommended": reco, "id": len(st.session_state["chat"])}
            st.session_state["chat"].append({"role": "assistant", "content": assistant_text, "meta": meta})

        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


if predict:
    symptoms = st.session_state.get("symptoms", [])
    if not symptoms:
        st.error("Please add at least 1 symptom.")
        st.stop()

    v = symptoms_to_vector(symptoms, system.symptoms)
    base_rows = rank_predictions(system, v, symptoms)
    adjusted_rows, triage_note = apply_triage(base_rows, symptoms, int(days), system)

    top3 = adjusted_rows[:3]
    confidence_info = compute_confidence(top3, total_symptoms=len(symptoms))
    advice = severity_advice(symptoms, int(days), kb)

    top_disease = top3[0][0]
    bundle = {
        "name": st.session_state.get("name", ""),
        "days": int(days),
        "symptoms": list(symptoms),
        "triage_note": triage_note,
        "advice": advice,
        "top3": [
            {
                "Disease": d,
                "Confidence": f"{confidence_info['confidence']:.0%}",
                "Notes": f"model={p:.0%}, evidence={e:.2f}, matched={inter}/{len(symptoms)}",
            }
            for (d, _rank_score, _combined, p, e, inter, _dcount) in top3
        ],
        "top_disease": top_disease,
        "description": kb.description.get(top_disease, "No description available."),
        "precautions": kb.precautions.get(top_disease, []),
        "confidence_info": confidence_info,
    }
    st.session_state["last_pred"] = bundle

    st.markdown("---")
    st.subheader("Results")

    # 1) Show predicted disease FIRST (product-like)
    if confidence_info["uncertain"]:
        st.markdown(f"## ✅ Top suggestion (uncertain): **{top_disease}**")
        st.caption("Treat this as a possibility, not a diagnosis.")
    else:
        st.markdown(f"## ✅ Predicted disease: **{top_disease}**")

    st.markdown("### Description")
    st.write(bundle["description"])

    st.markdown("### Precautions")
    prec = bundle["precautions"]
    if prec:
        for i, ptxt in enumerate(prec, start=1):
            st.write(f"{i}. {ptxt}")
    else:
        st.write("No precaution list available.")

    st.markdown("---")

    # 2) User-facing triage + advisory next
    if triage_note:
        st.warning(triage_note)
    st.info(advice)

    # 3) Technical details BELOW (collapsed by default)
    with st.expander("Technical details (confidence + scores)"):
        render_confidence_bar(confidence_info)

        df_out = pd.DataFrame(
            [
                {
                    "Disease": d,
                    "RankScore": rank_score,
                    "BaseScore": combined,
                    "ModelProb": p,
                    "Evidence": e,
                    "Matches": f"{inter}/{len(symptoms)}",
                    "DiseaseSymptomCount": dcount,
                }
                for (d, rank_score, combined, p, e, inter, dcount) in top3
            ]
        )
        st.dataframe(df_out, use_container_width=True, hide_index=True)

        if confidence_info["uncertain"]:
            st.markdown(
                "**Uncertain mode:** confidence is low (few symptoms / weak evidence / small margin). "
                "Add more symptoms for a stronger, more stable suggestion."
            )

    with st.expander("Debug (optional)"):
        st.write("Selected symptoms:", symptoms)
        st.write("Confidence info:", confidence_info)

