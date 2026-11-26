"""Arabic customer-feedback analysis.
Sentiment: CAMeLBERT-MSA sentiment model (public, Hugging Face), used as-is.
Topics: simple keyword rules (labeled as rules in the UI)."""
import re
_pipe = None
MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment"

TOPICS = {
    "الأسعار": ["سعر", "اسعار", "أسعار", "غالي", "رخيص", "تكلفة", "فاتورة"],
    "جودة الخدمة": ["خدمة", "جودة", "سيء", "سيئة", "ممتاز", "رائع", "تعامل"],
    "الشبكة والتوصيل": ["شبكة", "انقطاع", "بطيء", "بطيئ", "تأخير", "توصيل", "تاخير"],
    "الدعم": ["دعم", "شكوى", "شكوي", "موظف", "اتصال", "رد"],
}

def _load():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        _pipe = pipeline("text-classification", model=MODEL, top_k=None, truncation=True, max_length=128)
    return _pipe

def topics(text):
    return [t for t, ws in TOPICS.items() if any(w in text for w in ws)]

def sentiment(texts):
    """returns list of {label, score, scores}"""
    out = []
    for r in _load()(list(texts), batch_size=8):
        sc = {x["label"]: round(x["score"], 4) for x in r}
        lab = max(sc, key=sc.get)
        out.append({"label": lab, "score": sc[lab], "scores": sc})
    return out

def analyze(texts):
    sent = sentiment(texts)
    return [{"text": t, **s, "topics": topics(t)} for t, s in zip(texts, sent)]
