import os
import pytest
os.environ["SQLITE_PATH"] = "results/test_app.db"
if os.path.exists("results/test_app.db"): os.remove("results/test_app.db")
from fastapi.testclient import TestClient
from src.api import app
from src import assistant

def test_journey():
    with TestClient(app) as c:
        assert c.get("/").status_code == 200
        assert c.get("/api/results").status_code == 401
        assert c.post("/api/signup", json={"email": "a@b.co", "password": "secret1"}).status_code == 200
        r = c.post("/api/sample"); assert r.status_code == 200
        s = r.json(); assert s["total"] == 3150 and s["high"] > 0
        res = c.get("/api/results?band=High").json()
        assert all(x["band"] == "High" for x in res["customers"])
        assert "عميل" in c.post("/api/assistant", json={"question": "كم عميل في خطر؟"}).json()["answer"]
        assert c.get("/api/report").status_code == 200

def test_bad_login():
    with TestClient(app) as c:
        assert c.post("/api/login", json={"email": "x@y.z", "password": "nope123"}).status_code == 401

def test_assistant_without_data():
    assert "ارفع" in assistant.answer("كم؟", [])

def test_ai_modules():
    os.environ["ENABLE_LLM"] = "0"
    from src import genai
    genai.ENABLE = False
    import pandas as pd
    with TestClient(app) as c:
        c.post("/api/signup", json={"email": "ai@b.co", "password": "secret1"})
        c.post("/api/sample")
        a = c.get("/api/analytics").json()
        assert a["by_complaint"][1]["avg_risk"] > a["by_complaint"][0]["avg_risk"]
        r = c.post("/api/campaign").json()
        assert len(r["campaigns"]) == 3 and all(x["message_source"] == "template" for x in r["campaigns"])
        assert {t["agent"] for t in r["trace"]} == {"Segmenter", "Strategist", "Copywriter", "Critic", "Planner"}
        fb = c.post("/api/feedback", json={"text": "الشبكة ضعيفة والمكالمات تنقطع وهذا سيء"}).json()["results"][0]
        assert fb["label"] == "negative"

@pytest.mark.skipif(not os.path.exists("results/vision_resnet18_fp16.pt"),
                    reason="vision weights are not stored in git; train/export them with the sat-vision-service project")
def test_site_vision():
    with TestClient(app) as c:
        c.post("/api/signup", json={"email": "v@b.co", "password": "secret1"})
        c.post("/api/sample")
        from PIL import Image
        import io
        im = Image.new("RGB", (64, 64), (20, 90, 160)); b = io.BytesIO(); im.save(b, "PNG")
        s = c.post("/api/site", files={"file": ("t.png", b.getvalue(), "image/png")}).json()
        assert len(s["top"]) == 3 and "footfall" in s

