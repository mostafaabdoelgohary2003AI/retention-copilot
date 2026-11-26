"""GenAI: draft a short Arabic retention message with a small open LLM (Qwen2.5-0.5B-Instruct, runs on CPU).
A template fallback is used when the model is not available or its output fails the checks."""
import os, re
_tok = _lm = None
MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ENABLE = os.environ.get("ENABLE_LLM", "1") == "1"
REASON_AR = {
    "filed a complaint": "تقديمك شكوى مؤخراً",
    "experiencing frequent call failures": "انقطاع المكالمات المتكرر",
    "barely using the service": "قلة استخدامك للخدمة",
    "a new customer still in the fragile first months": "أنك عميل جديد معنا",
    "on a high tariff relative to usage": "أن باقتك قد لا تناسب استخدامك",
    "low and falling account value": "الرغبة في تقديم عرض يناسبك",
    "usage frequency is dropping": "انخفاض استخدامك للخدمة",
}
def _load():
    global _tok, _lm
    if _lm is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        _tok = AutoTokenizer.from_pretrained(MODEL)
        _lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    return _tok, _lm

def template(reasons):
    why = "، ".join(REASON_AR.get(r, "") for r in reasons[:2] if REASON_AR.get(r)) or "رغبتنا في خدمتك بشكل أفضل"
    return f"عزيزي العميل، نقدّر وجودك معنا. لاحظنا {why}، ونود التواصل معك لحل أي مشكلة وتقديم المساعدة. فريق خدمة العملاء بانتظار اتصالك."

def _ok(t):
    ar = len(re.findall(r"[\u0600-\u06FF]", t))
    polite = re.search(r"نقدر|نأسف|نعتذر|نود|شكرا|شكراً|عزيزي", t)
    bad = re.search(r"\d+\s*%|خصم|مجاني|[\[\]{}]|أنا |يمكنك تقديم|[A-Za-z]{3,}", t)
    return bool(polite) and not bad and 40 <= len(t) <= 450 and ar / max(len(t), 1) > 0.6

def _hosted(msgs, max_tokens=400):
    """OpenAI-compatible chat call (default: Gemini API free tier). Key only from env LLM_API_KEY."""
    import json, urllib.request
    key = os.environ.get("LLM_API_KEY")
    if not key: return None
    base = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
    model = os.environ.get("LLM_HOSTED_MODEL", "gemini-3.1-flash-lite")
    body = json.dumps({"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0.4}).encode()
    req = urllib.request.Request(base + "/chat/completions", body, {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=40))
    return d["choices"][0]["message"]["content"].strip(), model

def draft(reasons):
    """returns {text, source: 'llm-hosted'|'llm'|'template'}.
    Order: hosted LLM (if LLM_API_KEY set) -> local Qwen (if ENABLE_LLM=1) -> template. Every draft goes through the Critic checks."""
    why = "؛ ".join(REASON_AR.get(r, r) for r in reasons[:2])
    sys_p = "أنت موظف خدمة عملاء محترف. اكتب بالعربية الفصحى البسيطة رسالة قصيرة (ثلاث جمل) مهذبة لعميل قد يغادر. لا تذكر أي خصم أو نسبة أو وعد مالي ولا تستخدم أقواساً أو أسماء مكانية."
    rejected = False
    try:
        h = _hosted([{"role": "system", "content": sys_p}, {"role": "user", "content": f"اكتب رسالة لعميل. السبب: {why}."}])
        if h:
            if _ok(h[0]): return {"text": h[0], "source": "llm-hosted", "model": h[1]}
            rejected = True
    except Exception:
        pass
    if ENABLE:
        try:
            import torch
            tok, lm = _load()
            msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": f"اكتب رسالة لعميل. السبب: {why}."}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            with torch.no_grad():
                out = lm.generate(**ids, max_new_tokens=110, do_sample=False, repetition_penalty=1.1)
            t = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            if _ok(t): return {"text": t, "source": "llm"}
            rejected = True
        except Exception:
            pass
    return {"text": template(reasons), "source": "template", "llm_rejected": rejected}
