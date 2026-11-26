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

def draft(reasons):
    """returns {text, source: 'llm'|'template'}"""
    if ENABLE:
        try:
            import torch
            tok, lm = _load()
            why = "؛ ".join(REASON_AR.get(r, r) for r in reasons[:2])
            msgs = [{"role": "system", "content": "أنت موظف خدمة عملاء محترف. اكتب بالعربية الفصحى البسيطة رسالة قصيرة (ثلاث جمل) مهذبة لعميل قد يغادر. لا تذكر أي خصم أو نسبة أو وعد مالي."},
                    {"role": "user", "content": f"اكتب رسالة لعميل. السبب: {why}."}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            with torch.no_grad():
                out = lm.generate(**ids, max_new_tokens=110, do_sample=False, repetition_penalty=1.1)
            t = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            if _ok(t): return {"text": t, "source": "llm"}
            return {"text": template(reasons), "source": "template", "llm_rejected": True}
        except Exception:
            pass
    return {"text": template(reasons), "source": "template", "llm_rejected": False}
