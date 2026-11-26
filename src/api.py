"""Retention Copilot: landing, sign-in, upload -> risk results -> Arabic assistant -> report."""
import io, os
import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from . import db, scoring, assistant, report, agents
from .churn_prep import load_raw, RENAME
from . import generic
from fastapi.responses import JSONResponse

from fastapi.staticfiles import StaticFiles
app = FastAPI(title="Retention Copilot")
app.mount("/static", StaticFiles(directory="static"), name="static")
STATE = {}  # user_id -> scored rows (in-memory per process)
UPLOADS = {}  # user_id -> (dataframe, filename) waiting for the column-mapping step

class Cred(BaseModel):
    email: str
    password: str

class Q(BaseModel):
    question: str

def page(name):
    return HTMLResponse(open(f"static/{name}.html", encoding="utf-8").read())

@app.on_event("startup")
def startup():
    os.makedirs("results", exist_ok=True); db.init()

def me(req: Request):
    u = db.user_for(req.cookies.get("session"))
    if not u: raise HTTPException(401, "sign in first")
    return u

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/")
def landing(): return page("landing")
@app.get("/signin")
def signin_page(): return page("signin")
@app.get("/app")
def app_page(): return page("app")

@app.post("/api/signup")
def signup(c: Cred, resp: Response):
    if len(c.password) < 6 or "@" not in c.email: raise HTTPException(400, "Use a valid email and a password of 6+ characters")
    if not db.signup(c.email.lower(), c.password): raise HTTPException(409, "This email already has an account")
    tok = db.login(c.email.lower(), c.password)
    resp.set_cookie("session", tok, httponly=True, samesite="lax"); return {"ok": True}

@app.post("/api/login")
def login(c: Cred, resp: Response):
    tok = db.login(c.email.lower(), c.password)
    if not tok: raise HTTPException(401, "Wrong email or password")
    resp.set_cookie("session", tok, httponly=True, samesite="lax"); return {"ok": True}

@app.get("/api/me")
def whoami(req: Request): return me(req)

def _finish(u, rows, s, name, generic_mode=False):
    s["revenue_at_risk"] = round(sum(r["value"] for r in rows if r["band"] == "High"), 0)
    s["hist"] = [sum(1 for r in rows if min(int(r["risk"] * 10), 9) == b) for b in range(10)]
    s["tenure_risk"] = []
    for lo, hi, lab in [(0, 6, "0-6 months"), (7, 18, "7-18 months"), (19, 30, "19-30 months"), (31, 999, "31+ months")]:
        g = [r["risk"] for r in rows if lo <= r["months"] <= hi]
        s["tenure_risk"].append({"label": lab, "n": len(g), "avg": round(sum(g) / len(g), 3) if g else 0})
    STATE[u["id"]] = {"rows": rows, "summary": s, "generic": generic_mode}
    db.save_run(u["id"], name, {k: s[k] for k in ("total", "high", "medium", "low")})
    return s

def _process(u, df, name):
    if len(df) > 20000: raise HTTPException(413, "Max 20,000 customers per file")
    try:
        sc = scoring.score_dataframe(df.reset_index(drop=True))
        rows = [{"id": int(i) + 1, "risk": float(r.risk), "band": str(r.risk_band),
                 "reasons": list(r.reasons), "actions": list(r.actions),
                 "complaints": int(r.complaints) if "complaints" in sc else 0,
                 "months": int(r.subscription_length) if "subscription_length" in sc else 0,
                 "value": float(r.customer_value) if "customer_value" in sc else 0.0}
                for i, r in sc.iterrows()]
    except (ValueError, TypeError, KeyError) as e:
        raise HTTPException(400, f"The columns are there but some values could not be read as numbers ({e}). Check for text or empty cells.")
    return _finish(u, rows, scoring.summary(sc), name)

class Mapping(BaseModel):
    target: str
    id: str | None = None
    tenure: str | None = None
    value: str | None = None

@app.exception_handler(Exception)
async def any_error(request: Request, exc: Exception):
    return JSONResponse({"detail": "Something went wrong on our side while processing this file. Try the sample, or a smaller file."}, status_code=500)

@app.post("/api/upload")
async def upload(req: Request, file: UploadFile):
    u = me(req)
    raw = await file.read()
    if len(raw) > 80_000_000:
        raise HTTPException(413, "This file is larger than 80 MB. Remove columns you do not need and try again.")
    if not raw.strip():
        raise HTTPException(400, "This file is empty.")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "This does not look like a CSV file (comma-separated, first row = column names).")
    df.columns = [str(c).strip() for c in df.columns]
    if len(df) == 0 or len(df.columns) < 2:
        raise HTTPException(400, "The file needs a header row, at least 2 columns and some customers.")
    if set(RENAME) <= set(df.columns):
        return _process(u, df.rename(columns=RENAME), file.filename)
    UPLOADS[u["id"]] = (df, file.filename)
    try:
        guess = generic.suggest(df)
    except generic.DataError as e:
        raise HTTPException(400, str(e))
    return {"needs_mapping": True, "filename": file.filename, "rows": int(len(df)),
            "columns": list(df.columns), "suggest": guess}

@app.post("/api/train")
def train(m: Mapping, req: Request):
    u = me(req)
    if u["id"] not in UPLOADS:
        raise HTTPException(400, "Upload your file first.")
    df, name = UPLOADS[u["id"]]
    try:
        sc, met = generic.train_and_score(df, m.target, m.id, m.tenure, m.value)
    except generic.DataError as e:
        raise HTTPException(400, str(e))
    rows = [{"id": r.cust, "risk": float(r.risk), "band": str(r.risk_band), "reasons": list(r.reasons),
             "actions": list(r.actions), "complaints": 0, "months": int(r.months), "value": float(r.value)}
            for r in sc.itertuples()]
    band = sc["risk_band"].value_counts().to_dict()
    from collections import Counter
    c = Counter(x for rs in sc["reasons"] for x in rs)
    s = {"total": int(len(sc)), "high": int(band.get("High", 0)), "medium": int(band.get("Medium", 0)),
         "low": int(band.get("Low", 0)), "avg_risk": round(float(sc["risk"].mean()), 4),
         "top_reasons": c.most_common(5), "model": met, "has_value": bool(m.value), "has_tenure": bool(m.tenure)}
    return _finish(u, rows, s, name, generic_mode=True)

@app.post("/api/sample")
def sample(req: Request):
    u = me(req)
    return _process(u, load_raw("data/sample_customers.csv"), "Sample: 3,150 real telecom customers")

@app.get("/api/results")
def results(req: Request, band: str = "", limit: int = 60):
    u = me(req); st = STATE.get(u["id"])
    if not st: raise HTTPException(404, "Upload a customer file first")
    rows = sorted(st["rows"], key=lambda r: -r["risk"])
    if band: rows = [r for r in rows if r["band"] == band]
    return {"summary": st["summary"], "customers": rows[:limit], "history": db.runs(u["id"])}

@app.post("/api/assistant")
def ask(q: Q, req: Request):
    u = me(req); st = STATE.get(u["id"])
    return {"answer": assistant.answer(q.question, st["rows"] if st else [])}

@app.get("/api/report")
def get_report(req: Request):
    u = me(req); st = STATE.get(u["id"])
    if not st: raise HTTPException(404, "Upload a customer file first")
    top = sorted(st["rows"], key=lambda r: -r["risk"])[:15]
    path = f"results/report_{u['id']}.docx"
    report.build(path, st["summary"], top)
    return FileResponse(path, filename="retention-report.docx")


@app.post("/api/campaign")
def campaign(req: Request):
    u = me(req); st = STATE.get(u["id"])
    if not st: raise HTTPException(404, "Upload a customer file first")
    st["campaign"] = agents.run(st["rows"])
    return st["campaign"]

@app.get("/api/analytics")
def analytics(req: Request):
    u = me(req); st = STATE.get(u["id"])
    if not st: raise HTTPException(404, "Upload a customer file first")
    rows = st["rows"]
    if st.get("generic"):
        return {"by_complaint": [], "by_value": [],
                "value_at_risk_by_band": [{"band": b, "value": round(sum(r["value"] for r in rows if r["band"] == b))} for b in ("High", "Medium", "Low")],
                "dl_vs_ml": None, "nlp": None, "model": st["summary"].get("model")}
    def grp(f, labels):
        out = []
        for lab, fn in labels:
            g = [r for r in rows if fn(r)]
            out.append({"label": lab, "n": len(g), "avg_risk": round(sum(r["risk"] for r in g) / len(g), 3) if g else 0})
        return out
    return {
        "by_complaint": grp(None, [("No complaint", lambda r: r["complaints"] == 0), ("Filed a complaint", lambda r: r["complaints"] > 0)]),
        "by_value": grp(None, [("Value under 60", lambda r: r["value"] < 60), ("60 to 200", lambda r: 60 <= r["value"] < 200), ("200 and above", lambda r: r["value"] >= 200)]),
        "value_at_risk_by_band": [{"band": b, "value": round(sum(r["value"] for r in rows if r["band"] == b))} for b in ("High", "Medium", "Low")],
        "dl_vs_ml": __import__("json").load(open("results/dl_metrics.json")) if os.path.exists("results/dl_metrics.json") else None,
        "nlp": __import__("json").load(open("results/nlp_metrics.json")) if os.path.exists("results/nlp_metrics.json") else None,
    }

@app.post("/api/feedback")
def feedback(body: dict, req: Request):
    me(req)
    from . import nlp
    texts = [t.strip() for t in str(body.get("text", "")).split("\n") if t.strip()][:20]
    if not texts: raise HTTPException(400, "Paste at least one comment")
    return {"results": nlp.analyze(texts)}

@app.post("/api/site")
async def site(req: Request, file: UploadFile):
    me(req)
    from . import vision
    return vision.classify(await file.read())
