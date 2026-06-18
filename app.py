import os
# Vercel-compatible matplotlib setup is mandatory.
os.environ["MPLCONFIGDIR"] = "/tmp"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import io
import json
import time
import zipfile
import hashlib
import uuid
import base64
from datetime import datetime
from typing import Optional, Dict, Any

from flask import Flask, render_template, request, jsonify, send_file, session
import pandas as pd
import fitz  # PyMuPDF
from google import genai
from google.genai import types
from werkzeug.utils import secure_filename

# ============================================================
# CONFIG
# ============================================================

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

GEMINI_MODEL = "gemini-2.5-flash"
# Do not hardcode API keys. Make sure GEMINI_API_KEY is in Vercel Environment Variables.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# In-memory session store (Note: For Vercel Serverless, instances may spin down. 
# In a true production app, use Redis/Postgres for state. We use in-memory to meet constraints without external DBs)
SESSIONS: Dict[str, Dict[str, Any]] = {}

def get_session_state(sid: str) -> dict:
    if sid not in SESSIONS:
        SESSIONS[sid] = {
            "project": {
                "title": "", "domain": "", "area": "", "journal": "IEEE",
                "page_count": 8, "keywords": "", "abstract_limit": 250,
            },
            "topic": {"idea": "", "evaluation": None},
            "papers": [], 
            "research_gap": None,
            "lit_review": {"author_wise": None, "thematic": None, "comparative": None, "table": None},
            "methodology": {
                "type": "Experimental", "dataset": "", "sample_size": "",
                "tools": "", "metrics": "", "generated": None,
            },
            "paper_sections": {
                "introduction": None, "literature_review": None, "methodology": None,
                "results": None, "discussion": None, "conclusion": None, "abstract": None,
            },
            "results_input": "",
            "authors": [],
            "figures": [],
            "datasets": {},
            "citations": [],
            "ai_cache": {},
            "export_buffer": None,
        }
    return SESSIONS[sid]

# ============================================================
# GEMINI CALL
# ============================================================

def _cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

def call_gemini(prompt: str, sid: str, max_tokens: int = 800, temperature: float = 0.4, use_cache: bool = True) -> Optional[str]:
    if not GEMINI_API_KEY:
        return None

    state = get_session_state(sid)
    key = _cache_key(prompt)
    if use_cache and key in state["ai_cache"]:
        return state["ai_cache"][key]

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        text = response.text.strip()
        if not text:
            return None
        if use_cache:
            state["ai_cache"][key] = text
        return text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return None

def call_gemini_json(prompt: str, sid: str, max_tokens: int = 700) -> Optional[dict]:
    raw = call_gemini(prompt + "\n\nRespond ONLY with valid JSON. No markdown, no preamble.", sid, max_tokens=max_tokens)
    if not raw:
        return None
    cleaned = raw.strip().strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
        return None

# ============================================================
# HELPERS
# ============================================================

def extract_pdf_info(file_bytes: bytes, filename: str, sid: str) -> dict:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = ""
    for i, page in enumerate(doc):
        full_text += page.get_text()
        if i >= 4:
            break
    doc.close()
    snippet = full_text[:3500]
    prompt = (
        "Extract from this academic paper excerpt: title, authors (comma separated), "
        "and abstract (max 80 words). If unclear, say 'Not found'.\n"
        f"TEXT:\n{snippet}\n\n"
        'Respond as JSON: {"title": "", "authors": "", "abstract": ""}'
    )
    result = call_gemini_json(prompt, sid, max_tokens=400)
    if not result:
        result = {"title": filename, "authors": "Not found", "abstract": "Not found"}
    return {
        "name": filename,
        "title": result.get("title", filename),
        "authors": result.get("authors", "Not found"),
        "abstract": result.get("abstract", "Not found"),
        "full_text": full_text,
    }

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html", gemini_configured=bool(GEMINI_API_KEY))

@app.route("/api/state", methods=["GET"])
def get_state():
    sid = request.headers.get("X-Session-ID")
    if not sid:
        return jsonify({"error": "No session ID"}), 400
    state = get_session_state(sid)
    # Filter datasets and large buffers before sending
    safe_state = {k: v for k, v in state.items() if k not in ["datasets", "ai_cache", "export_buffer"]}
    # For papers, remove full_text to save bandwidth
    safe_state["papers"] = [{"name": p["name"], "title": p["title"], "authors": p["authors"], "abstract": p["abstract"]} for p in state["papers"]]
    safe_state["datasets_keys"] = list(state.get("datasets", {}).keys())
    return jsonify(safe_state)

@app.route("/api/project", methods=["POST"])
def update_project():
    sid = request.headers.get("X-Session-ID")
    data = request.json
    state = get_session_state(sid)
    state["project"].update(data)
    return jsonify({"status": "ok"})

@app.route("/api/topic/evaluate", methods=["POST"])
def evaluate_topic():
    sid = request.headers.get("X-Session-ID")
    data = request.json
    idea = data.get("idea", "")
    state = get_session_state(sid)
    state["topic"]["idea"] = idea

    prompt = (
        "You are an academic research advisor. Evaluate this research topic idea strictly "
        "based on what is written; do not assume unstated context.\n"
        f"TOPIC: {idea}\n\n"
        "Score each from 1-10 and give one-line reasoning. Also rate dataset availability "
        "(High/Medium/Low/Unknown) and give 3 short actionable suggestions.\n"
        'JSON format: {"clarity": int, "clarity_note": "", "novelty": int, "novelty_note": "", '
        '"feasibility": int, "feasibility_note": "", "dataset_availability": "", '
        '"suggestions": ["", "", ""]}'
    )
    result = call_gemini_json(prompt, sid, max_tokens=500)
    if result:
        state["topic"]["evaluation"] = result
        return jsonify({"status": "ok", "evaluation": result})
    return jsonify({"error": "Failed to evaluate"}), 500

@app.route("/api/papers/upload", methods=["POST"])
def upload_papers():
    sid = request.headers.get("X-Session-ID")
    state = get_session_state(sid)
    files = request.files.getlist("files")
    existing_names = {p["name"] for p in state["papers"]}
    
    for f in files:
        if f.filename not in existing_names:
            info = extract_pdf_info(f.read(), f.filename, sid)
            state["papers"].append(info)
            
    return jsonify({"status": "ok"})

@app.route("/api/papers/<int:idx>", methods=["DELETE"])
def remove_paper(idx):
    sid = request.headers.get("X-Session-ID")
    state = get_session_state(sid)
    if 0 <= idx < len(state["papers"]):
        state["papers"].pop(idx)
    return jsonify({"status": "ok"})

@app.route("/api/gap/analyze", methods=["POST"])
def analyze_gap():
    sid = request.headers.get("X-Session-ID")
    state = get_session_state(sid)
    papers = state["papers"]
    if not papers:
        return jsonify({"error": "No papers"}), 400

    context = "\n\n".join(f"Paper {i+1} - {p['title']}: {p['abstract']}" for i, p in enumerate(papers))
    prompt = (
        "You are analyzing ONLY the following uploaded paper abstracts. Do not invent or "
        "reference any study not listed below. If information is insufficient, explicitly "
        "say so for that field.\n\n"
        f"{context}\n\n"
        "Identify: research_trends, common_limitations, future_work, potential_gaps. "
        "Each as a list of short bullet strings grounded strictly in the text above.\n"
        'JSON: {"research_trends": [], "common_limitations": [], "future_work": [], '
        '"potential_gaps": []}'
    )
    result = call_gemini_json(prompt, sid, max_tokens=700)
    if result:
        state["research_gap"] = result
        return jsonify({"status": "ok", "gap": result})
    return jsonify({"error": "Analysis failed"}), 500

@app.route("/api/litreview/generate", methods=["POST"])
def generate_litreview():
    sid = request.headers.get("X-Session-ID")
    data = request.json
    rtype = data.get("type")
    state = get_session_state(sid)
    papers = state["papers"]
    context = "\n\n".join(f"{i+1}. {p['title']} by {p['authors']}: {p['abstract']}" for i, p in enumerate(papers))

    if rtype == "author_wise":
        prompt = f"Based ONLY on these abstracts:\n{context}\n\nWrite a concise author-wise review (2-3 sentences per paper, grouped by author). Do not add information beyond what is given."
        res = call_gemini(prompt, sid, max_tokens=900)
        if res: state["lit_review"]["author_wise"] = res
    elif rtype == "thematic":
        prompt = f"Based ONLY on these abstracts:\n{context}\n\nGroup the papers into 2-4 themes based on shared methods or focus, and summarize each theme in 2-3 sentences. Stay strictly within the given content."
        res = call_gemini(prompt, sid, max_tokens=900)
        if res: state["lit_review"]["thematic"] = res
    elif rtype == "comparative":
        prompt = f"Based ONLY on these abstracts:\n{context}\n\nWrite a comparative summary highlighting similarities and differences in approach across papers. Be factual and concise."
        res = call_gemini(prompt, sid, max_tokens=700)
        if res: state["lit_review"]["comparative"] = res
    elif rtype == "table":
        prompt = (
            f"Based ONLY on these abstracts:\n{context}\n\nExtract for each paper: author, "
            "method, dataset, findings, limitations. If a field is not mentioned, write "
            "'Not specified'. Do not fabricate.\n"
            'JSON list: [{"author": "", "method": "", "dataset": "", "findings": "", "limitations": ""}]'
        )
        res = call_gemini_json(prompt, sid, max_tokens=900)
        if res: state["lit_review"]["table"] = res

    return jsonify({"status": "ok", "lit_review": state["lit_review"]})

@app.route("/api/methodology", methods=["POST"])
def generate_methodology():
    sid = request.headers.get("X-Session-ID")
    data = request.json
    state = get_session_state(sid)
    m = state["methodology"]
    m.update(data)
    
    prompt = (
        "Organize the following researcher-provided inputs into a formal methodology "
        "write-up with sections: Research Design, Data Collection, Processing, Analysis, "
        "Evaluation. Use ONLY the inputs given. If a field is empty or missing, write "
        "'Not specified by researcher' for that part instead of inventing details.\n"
        f"Research Type: {m['type']}\nDataset: {m['dataset'] or 'Not specified'}\n"
        f"Sample Size: {m['sample_size'] or 'Not specified'}\n"
        f"Tools: {m['tools'] or 'Not specified'}\nMetrics: {m['metrics'] or 'Not specified'}"
    )
    res = call_gemini(prompt, sid, max_tokens=700)
    if res:
        m["generated"] = res
        return jsonify({"status": "ok", "generated": res})
    return jsonify({"error": "Failed"}), 500

@app.route("/api/paper_builder/results_input", methods=["POST"])
def update_results_input():
    sid = request.headers.get("X-Session-ID")
    state = get_session_state(sid)
    state["results_input"] = request.json.get("results_input", "")
    return jsonify({"status": "ok"})

@app.route("/api/paper_builder/generate", methods=["POST"])
def generate_section():
    sid = request.headers.get("X-Session-ID")
    section_key = request.json.get("section")
    state = get_session_state(sid)
    
    proj = state["project"]
    gap = state["research_gap"]
    m = state["methodology"]
    parts = [
        f"Title: {proj['title']}", f"Domain: {proj['domain']} / {proj['area']}",
        f"Keywords: {proj['keywords']}",
    ]
    if gap:
        parts.append(f"Identified gaps: {gap.get('potential_gaps', [])}")
    if m["generated"]:
        parts.append(f"Methodology summary: {m['generated'][:600]}")
    context = "\n".join(parts)

    prompt = ""
    if section_key == "abstract":
        full_text = "\n".join(f"{k}: {v}" for k, v in state["paper_sections"].items() if v and k != "abstract")
        prompt = (
            f"Write an academic abstract (max {proj['abstract_limit']} words) summarizing "
            f"the following paper sections. Use ONLY the information given; do not add "
            f"new claims.\n{full_text[:4000]}"
        )
    elif section_key == "results":
        prompt = (
            "Write a formal Results section based STRICTLY on the following "
            "researcher-provided data. Do not invent any numbers, accuracy values, or "
            f"outcomes not stated.\nCONTEXT: {context}\n"
            f"RESULTS DATA: {state['results_input']}"
        )
    elif section_key == "literature_review" and state["lit_review"]["thematic"]:
        prompt = f"Rewrite the following into a formal Literature Review section for an academic paper:\n{state['lit_review']['thematic']}"
    elif section_key == "methodology" and state["methodology"]["generated"]:
        prompt = f"Format the following into a formal Methodology section for an academic paper:\n{state['methodology']['generated']}"
    else:
        prompt = (
            f"Write the {section_key.replace('_', ' ').title()} section of an academic paper. "
            f"Context:\n{context}\nBe formal, concise, and avoid fabricating data, "
            "statistics, or citations not provided."
        )

    res = call_gemini(prompt, sid, max_tokens=900)
    if res:
        state["paper_sections"][section_key] = res
        return jsonify({"status": "ok", "content": res})
    return jsonify({"error": "Failed"}), 500

@app.route("/api/paper_builder/update", methods=["POST"])
def update_section():
    sid = request.headers.get("X-Session-ID")
    data = request.json
    state = get_session_state(sid)
    state["paper_sections"][data["section"]] = data["content"]
    return jsonify({"status": "ok"})

@app.route("/api/authors", methods=["POST"])
def add_author():
    sid = request.headers.get("X-Session-ID")
    state = get_session_state(sid)
    state["authors"].append(request.json)
    return jsonify({"status": "ok"})

@app.route("/api/authors/<int:idx>", methods=["DELETE"])
def remove_author(idx):
    sid = request.headers.get("X-Session-ID")
    state = get_session_state(sid)
    if 0 <= idx < len(state["authors"]):
        state["authors"].pop(idx)
    return jsonify({"status": "ok"})

@app.route("/api/figures/csv", methods=["POST"])
def upload_csv():
    sid = request.headers.get("X-Session-ID")
    state = get_session_state(sid)
    file = request.files.get("file")
    if file:
        df = pd.read_csv(file)
        state["datasets"][file.filename] = df
        return jsonify({"status": "ok", "columns": df.columns.tolist()})
    return jsonify({"error": "No file"}), 400

@app.route("/api/figures/chart", methods=["POST"])
def generate_chart():
    sid = request.headers.get("X-Session-ID")
    state = get_session_state(sid)
    data = request.json
    ds_name = data.get("dataset")
    chart_type = data.get("type")
    x_col = data.get("x")
    y_col = data.get("y")
    
    if ds_name not in state["datasets"]:
        return jsonify({"error": "Dataset not found"}), 404
        
    df = state["datasets"][ds_name]
    
    fig, ax = plt.subplots()
    try:
        if chart_type == "Bar Chart":
            ax.bar(df[x_col], df[y_col])
        elif chart_type == "Pie Chart":
            ax.pie(df[y_col], labels=df[x_col], autopct='%1.1f%%')
        elif chart_type == "Line Chart":
            ax.plot(df[x_col], df[y_col])
        else:
            ax.scatter(df[x_col], df[y_col])
            
        ax.set_title(f"{chart_type}: {y_col} vs {x_col}")
        if chart_type != "Pie Chart":
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            plt.xticks(rotation=45)
            
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        
        fig_obj = {"type": chart_type, "x": x_col, "y": y_col, "dataset": ds_name, "image_b64": img_b64}
        state["figures"].append(fig_obj)
        return jsonify({"status": "ok", "figure": fig_obj})
    except Exception as e:
        plt.close(fig)
        return jsonify({"error": str(e)}), 500

@app.route("/api/figures/image", methods=["POST"])
def upload_image():
    sid = request.headers.get("X-Session-ID")
    state = get_session_state(sid)
    files = request.files.getlist("files")
    
    added = []
    for img in files:
        img_bytes = img.read()
        b64 = base64.b64encode(img_bytes).decode('utf-8')
        meta = {"name": img.filename, "size_kb": round(len(img_bytes) / 1024, 1)}
        fig_obj = {"type": "image", "meta": meta, "image_b64": b64}
        state["figures"].append(fig_obj)
        added.append(fig_obj)
        
    return jsonify({"status": "ok", "figures": added})

@app.route("/api/citations", methods=["POST"])
def add_citation():
    sid = request.headers.get("X-Session-ID")
    state = get_session_state(sid)
    state["citations"].append(request.json)
    return jsonify({"status": "ok"})

@app.route("/api/citations/<int:idx>", methods=["DELETE"])
def remove_citation(idx):
    sid = request.headers.get("X-Session-ID")
    state = get_session_state(sid)
    if 0 <= idx < len(state["citations"]):
        state["citations"].pop(idx)
    return jsonify({"status": "ok"})

# ============================================================
# EXPORT GENERATION LOGIC
# ============================================================

def escape_latex(text: str) -> str:
    if not text:
        return ""
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def build_latex(state) -> str:
    proj = state["project"]
    sections = state["paper_sections"]
    authors = state["authors"]

    author_block = " \\and ".join(
        f"{escape_latex(a['name'])}\\\\{escape_latex(a['institute'])}" for a in authors
    ) or "Author Name\\\\Institute"

    body = []
    order = ["abstract", "introduction", "literature_review", "methodology", "results",
              "discussion", "conclusion"]
    titles = {
        "abstract": None, "introduction": "Introduction", "literature_review": "Literature Review",
        "methodology": "Methodology", "results": "Results", "discussion": "Discussion",
        "conclusion": "Conclusion",
    }
    abstract_tex = ""
    for key in order:
        content = sections.get(key)
        if not content:
            continue
        if key == "abstract":
            abstract_tex = escape_latex(content)
        else:
            body.append(f"\\section{{{titles[key]}}}\n{escape_latex(content)}\n")

    citation_keys = [f"ref{i+1}" for i in range(len(state["citations"]))]

    return f"""\\documentclass[conference]{{IEEEtran}}
\\usepackage{{cite}}
\\begin{{document}}
\\title{{{escape_latex(proj['title'] or 'Untitled Paper')}}}
\\author{{{author_block}}}
\\maketitle

\\begin{{abstract}}
{abstract_tex or 'Abstract not yet generated.'}
\\end{{abstract}}

\\begin{{IEEEkeywords}}
{escape_latex(proj['keywords'])}
\\end{{IEEEkeywords}}

{''.join(body)}

\\bibliographystyle{{IEEEtran}}
\\bibliography{{references}}
\\end{{document}}
"""

def build_bibtex(state) -> str:
    entries = []
    for i, c in enumerate(state["citations"]):
        key = f"ref{i+1}"
        entries.append(
            f"@article{{{key},\n"
            f"  author = {{{c.get('authors', 'Unknown')}}},\n"
            f"  title = {{{c.get('title', 'Untitled')}}},\n"
            f"  year = {{{c.get('year', 'n.d.')}}},\n"
            f"  journal = {{{c.get('source', '')}}}\n"
            f"}}\n"
        )
    return "\n".join(entries) if entries else "% No citations added yet\n"

def build_metadata(state) -> dict:
    return {
        "project": state["project"],
        "authors": state["authors"],
        "topic_evaluation": state["topic"]["evaluation"],
        "num_reference_papers": len(state["papers"]),
        "num_citations": len(state["citations"]),
        "sections_completed": [k for k, v in state["paper_sections"].items() if v],
        "exported_at": datetime.utcnow().isoformat() + "Z",
    }

@app.route("/api/export/generate", methods=["POST"])
def generate_export():
    sid = request.headers.get("X-Session-ID")
    state = get_session_state(sid)
    
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("paper.tex", build_latex(state))
        zf.writestr("references.bib", build_bibtex(state))
        zf.writestr("metadata.json", json.dumps(build_metadata(state), indent=2, default=str))
    
    state["export_buffer"] = buf.getvalue()
    return jsonify({"status": "ok", "latex_preview": build_latex(state)})

@app.route("/api/export/download", methods=["GET"])
def download_export():
    sid = request.args.get("sid")
    if not sid:
        return "Missing session ID", 400
    state = get_session_state(sid)
    if not state.get("export_buffer"):
        return "No export generated", 404
        
    title = state['project']['title'] or 'research_paper'
    filename = f"{title.replace(' ', '_')}.zip"
    
    return send_file(
        io.BytesIO(state["export_buffer"]),
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)