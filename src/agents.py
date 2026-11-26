"""Specialized multi-agent retention campaign. Each agent has one job and leaves a trace.
Segmenter (KMeans, unsupervised ML) -> Strategist (rules) -> Copywriter (LLM, template fallback)
-> Critic (checks the draft) -> Planner (assembles the campaign). All outputs come from the user's own results."""
import time
import numpy as np
from sklearn.cluster import KMeans
from . import genai

def run(rows, k=3, max_customers=300):
    trace, t0 = [], time.time()
    def log(agent, msg): trace.append({"agent": agent, "step": msg, "t": round(time.time() - t0, 1)})
    risky = sorted([r for r in rows if r["band"] == "High"], key=lambda r: -r["risk"])[:max_customers]
    if len(risky) < k:
        return {"campaigns": [], "trace": [{"agent": "Segmenter", "step": "not enough high-risk customers", "t": 0}], "seconds": 0}
    # Segmenter: group high-risk customers by their reason pattern + value + tenure
    all_reasons = sorted({x for r in risky for x in r["reasons"]})
    X = np.array([[x in r["reasons"] for x in all_reasons] + [r["value"] / 100.0, r["months"] / 40.0] for r in risky], dtype=float)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    log("Segmenter", f"grouped {len(risky)} high-risk customers into {k} segments with KMeans")
    campaigns = []
    for c in range(k):
        mem = [r for r, l in zip(risky, km.labels_) if l == c]
        cnt = {}
        for r in mem:
            for x in r["reasons"]: cnt[x] = cnt.get(x, 0) + 1
        top = sorted(cnt, key=cnt.get, reverse=True)[:2]
        acts = {}
        for r in mem:
            for a in r["actions"]: acts[a] = acts.get(a, 0) + 1
        action = max(acts, key=acts.get)
        log("Strategist", f"segment {c+1}: {len(mem)} customers, main reasons: {', '.join(top)}")
        msg = genai.draft(top)
        log("Copywriter", f"segment {c+1}: drafted Arabic message ({msg['source']})")
        issues = []
        if not genai._ok(msg["text"]): issues.append("failed length/language/no-promise check")
        if msg.get("llm_rejected"):
            log("Critic", f"segment {c+1}: rejected the LLM draft (placeholder, odd phrasing or no polite tone), used the safe template")
        elif msg["source"] == "template":
            log("Critic", f"segment {c+1}: template message checked (LLM switched off)")
        elif issues:
            msg = {"text": genai.template(top), "source": "template"}; log("Critic", f"segment {c+1}: rejected draft ({issues[0]}), used safe template")
        else:
            log("Critic", f"segment {c+1}: draft passed checks (Arabic, 40-450 chars, no discount or % promised)")
        campaigns.append({"segment": c + 1, "customers": len(mem), "avg_risk": round(float(np.mean([r["risk"] for r in mem])), 3),
                          "value_at_risk": round(sum(r["value"] for r in mem), 0), "reasons": top, "action": action,
                          "message": msg["text"], "message_source": msg["source"], "sample_ids": [r["id"] for r in mem[:8]]})
    log("Planner", "assembled campaign plan ordered by value at risk")
    campaigns.sort(key=lambda x: -x["value_at_risk"])
    return {"campaigns": campaigns, "trace": trace, "seconds": round(time.time() - t0, 1)}
