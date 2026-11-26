"""Arabic assistant over the user's own scored data.
Honest: rule-based intent matching + templated Arabic answers from real numbers.
No LLM is called; nothing is generated that is not computed from the data."""
import re
from collections import Counter

REASON_AR = {
    "filed a complaint": "قدّم شكوى",
    "experiencing frequent call failures": "يعاني من انقطاع متكرر في المكالمات",
    "barely using the service": "يستخدم الخدمة قليلاً جداً",
    "a new customer still in the fragile first months": "عميل جديد في الأشهر الأولى الحساسة",
    "on a high tariff relative to usage": "باقته مرتفعة مقارنة باستخدامه",
    "low and falling account value": "قيمة حسابه منخفضة",
    "usage frequency is dropping": "تكرار استخدامه في تراجع",
    "no single dominant driver - a combination of small signals": "لا يوجد سبب واحد واضح، بل عدة إشارات صغيرة",
}
ACTION_AR = {
    "Call them personally within 48 hours and resolve the open issue.": "اتصل به شخصياً خلال 48 ساعة وحل مشكلته",
    "Escalate to network quality team; offer a service credit.": "حوّل الحالة لفريق جودة الشبكة وقدّم له رصيداً تعويضياً",
    "Send a re-engagement offer matched to their usage pattern.": "أرسل له عرضاً يناسب نمط استخدامه",
    "Enroll in the onboarding care program.": "ضمّه إلى برنامج رعاية العملاء الجدد",
    "Offer a plan-review to right-size their tariff.": "اعرض عليه مراجعة الباقة لتناسب استخدامه",
    "Assign a retention agent before the next billing cycle.": "كلّف موظف احتفاظ قبل الفاتورة القادمة",
    "Trigger the win-back campaign with a usage-based perk.": "فعّل حملة استعادة مع ميزة مرتبطة بالاستخدام",
    "include in the standard quarterly check-in": "ضمّه لمتابعة الربع العادية",
}

def _n(t):
    t = re.sub(r"[\u064B-\u0652\u0640]", "", t)
    t = re.sub(r"[أإآ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    return t

def answer(q, scored):
    """scored: list of dict rows with id,risk,band,reasons,actions"""
    if not scored:
        return "ارفع ملف العملاء أولاً حتى أستطيع الإجابة."
    t = _n(q)
    n = len(scored)
    high = [r for r in scored if r["band"] == "High"]
    med = [r for r in scored if r["band"] == "Medium"]
    if any(k in t for k in ["كم", "عدد"]) and any(k in t for k in ["خطر", "يغادر", "خطير", "عالي"]):
        return f"من أصل {n} عميل، يوجد {len(high)} عميل في خطر مرتفع ({len(high)/n:.0%}) و{len(med)} في خطر متوسط."
    if any(k in t for k in ["اكثر", "اعلي", "اخطر", "اول"]) and "عميل" in t:
        top = sorted(scored, key=lambda r: -r["risk"])[:5]
        lines = [f"العميل رقم {r['id']}: احتمال المغادرة {r['risk']:.0%}" for r in top]
        return "أكثر 5 عملاء في خطر:\n" + "\n".join(lines)
    if any(k in t for k in ["لماذا", "سبب", "اسباب", "ليش"]):
        c = Counter(r2 for r in high for r2 in r["reasons"])
        if not c:
            return "لا يوجد عملاء في خطر مرتفع في هذا الملف."
        lines = [f"- {REASON_AR.get(k, k)}: {v} عميل" for k, v in c.most_common(4)]
        return "أهم أسباب الخطر بين العملاء عاليي الخطورة:\n" + "\n".join(lines)
    if any(k in t for k in ["ماذا افعل", "اجراء", "توصي", "اعمل", "افعل", "نصيح"]):
        c = Counter(a for r in high for a in r["actions"])
        if not c:
            return "لا توجد إجراءات عاجلة، الوضع مستقر."
        lines = [f"- {ACTION_AR.get(k, k)} ({v} عميل)" for k, v in c.most_common(4)]
        return "الإجراءات الأكثر أولوية:\n" + "\n".join(lines)
    m = re.search(r"(\d+)", t)
    if m and "عميل" in t:
        i = int(m.group(1))
        r = next((x for x in scored if x["id"] == i), None)
        if r:
            rs = "، ".join(REASON_AR.get(x, x) for x in r["reasons"])
            ac = "، ".join(ACTION_AR.get(x, x) for x in r["actions"])
            return f"العميل {i}: احتمال المغادرة {r['risk']:.0%} ({r['band']}).\nالأسباب: {rs}.\nالمقترح: {ac}."
        return f"لم أجد عميلاً برقم {i}."
    return "يمكنني الإجابة عن: كم عميلاً في خطر؟ من أكثر العملاء خطراً؟ لماذا يغادر العملاء؟ ماذا أفعل؟ أو اسأل عن عميل برقمه."
