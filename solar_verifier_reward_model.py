
from flask import Flask, request, jsonify
import re

app = Flask(__name__)

ALLOWED_LABELS = [
    "CRACK", "CLEAN PANEL", "FINGER", "BLACK CORE", "THICK LINE",
    "HORIZONTAL DISLOCATION", "VERTICAL DISLOCATION", "SHORT CIRCUIT"
]
NORMALIZE = {
    "CLEAN": "CLEAN PANEL","CLEAN-PANEL":"CLEAN PANEL","CLEAN_PANEL":"CLEAN PANEL","CLEAN PANEL.":"CLEAN PANEL","CLEANPANEL":"CLEAN PANEL",
    "FINGER DEFECT":"FINGER","FINGERS":"FINGER","CRACKS":"CRACK","STAR CRACK":"CRACK",
    "BLACK_CORE":"BLACK CORE","BLACK SPOT":"BLACK CORE","BLACK POINT":"BLACK CORE",
    "SHORT-CIRCUIT":"SHORT CIRCUIT","SHORTCIRCUIT":"SHORT CIRCUIT",
    "THICK_LINE":"THICK LINE","HORIZONTAL DIISLOCATION":"HORIZONTAL DISLOCATION",
    "VERTICAL DIISLOCATION":"VERTICAL DISLOCATION","DELAMINATION":"CLEAN PANEL",
    "MICROCRACKS":"CRACK","SHADINGS":"CLEAN PANEL","HOT SPOT":"BLACK CORE",
    "SOILING":"CLEAN PANEL","POTENTIAL INDUCED DEGRADATION (PID)":"CLEAN PANEL",
    "SHUNT DEFECT":"SHORT CIRCUIT",
}

STEP_RE = re.compile(r"\bstep\s*([1-7])\b", re.I)
PROB_RE = re.compile(r"-\s*[A-Za-z_ ()/\-]+:\s*\d{1,3}%")

def canon_label(s: str) -> str:
    if not s: return ""
    t = str(s).strip().upper().replace("-", " ").replace("_"," ")
    t = " ".join(t.split())
    return NORMALIZE.get(t, t)

def extract_defect_from_output(text: str) -> str:
    if not text: return ""
    # try structured "**Defect Type**: xxx"
    for line in text.splitlines():
        L = line.strip()
        if L.lower().startswith("- **defect type**"):
            parts = L.split(":", 1)
            target = parts[1] if len(parts) > 1 else ""
            lab = canon_label(target)
            break
    else:
        lab = canon_label(text)
    # snap to allowed labels by prefix
    for lbl in sorted(ALLOWED_LABELS, key=len, reverse=True):
        if lab.startswith(lbl): 
            return lbl
    return lab

def has_all_think_steps(s: str):
    seen = set(int(m.group(1)) for m in STEP_RE.finditer(s))
    return len(seen)

def has_probabilities(s: str) -> bool:
    # at least 3 " - Label: xx% " lines
    return len(PROB_RE.findall(s)) >= 3

def derive_response(query, prompt, response):
    if response: 
        return response
    if query and prompt and isinstance(query, str) and isinstance(prompt, str) and query.startswith(prompt):
        return query[len(prompt):]
    return response or query or ""

@app.route("/get_reward", methods=["POST"])
def get_reward():
    data = request.json or {}

    # Accept both single and batched
    queries  = data.get("query")   or data.get("queries")
    prompts  = data.get("prompt")  or data.get("prompts")
    labels   = data.get("label")   or data.get("labels")
    responses= data.get("response") or data.get("responses")

    # Normalize to lists
    def to_list(x):
        if x is None: return []
        return x if isinstance(x, list) else [x]

    queries, prompts, labels, responses = map(to_list, (queries, prompts, labels, responses))
    n = max(len(queries), len(prompts), len(labels), len(responses))
    # pad
    pad = lambda L: (L + [""]* (n - len(L))) if len(L) < n else L
    queries, prompts, labels, responses = map(pad, (queries, prompts, labels, responses))

    rewards, scores, logs = [], [], {"details": []}

    for q, p, lab_raw, resp_raw in zip(queries, prompts, labels, responses):
        resp = derive_response(q, p, resp_raw)

        pred = extract_defect_from_output(resp)
        # IMPORTANT: labels may be full formatted answers — extract too:
        gold = extract_defect_from_output(lab_raw) if lab_raw else ""

        correct_cls = int(pred in ALLOWED_LABELS and gold in ALLOWED_LABELS and pred == gold)
        base = 1.0 if correct_cls else -1.0

        nsteps = has_all_think_steps(resp)
        structure_bonus = (nsteps/7.0) * 0.5
        prob_bonus = 0.3 if has_probabilities(resp) else 0.0
        missing_penalty = -0.5 if (nsteps < 4 or ("<think>" not in resp or "<answer>" not in resp)) else 0.0

        r = base + structure_bonus + prob_bonus + missing_penalty
        # clamp to [-1, 1] for stability; map to [0,1] score for filtering
        r = max(-1.0, min(1.0, r))
        s = (r + 1.0) / 2.0

        rewards.append(r)
        scores.append(s)
        logs["details"].append({
            "pred": pred, "gold": gold,
            "correct_cls": bool(correct_cls),
            "n_steps": nsteps,
            "structure_bonus": structure_bonus,
            "prob_bonus": prob_bonus,
            "missing_penalty": missing_penalty
        })

    return jsonify({"rewards": rewards, "scores": scores, "extra_logs": logs})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
