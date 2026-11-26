"""Tests for the general churn path.

Real data: the bundled UCI Iranian churn sample with its headers renamed, so the generic
path (not the Iranian path) has to train on it. The small frames in the error tests are
fixtures that only exercise error messages; no result is claimed from them."""
import io
import os
import pandas as pd
import pytest
os.environ["SQLITE_PATH"] = "results/test_generic.db"
if os.path.exists("results/test_generic.db"): os.remove("results/test_generic.db")
from fastapi.testclient import TestClient
from src.api import app
from src import generic

def real_generic_csv():
    df = pd.read_csv("data/sample_customers.csv")
    df.columns = [f"f{i}_" + c.strip().replace(" ", "_").lower() for i, c in enumerate(df.columns)]
    df = df.rename(columns={df.columns[-1]: "left_company"})
    df["left_company"] = df["left_company"].map({1: "Yes", 0: "No"})
    return df

def test_train_and_score_on_real_data():
    df = real_generic_csv()
    out, m = generic.train_and_score(df, "left_company", tenure="f2_subscription_length", value="f12_customer_value")
    assert len(out) == 3150 and out["risk"].between(0, 1).all()
    assert m["auc"] > 0.9 and m["accuracy"] > m["baseline_accuracy"]
    assert out.sort_values("risk").tail(100)["risk"].mean() > out["risk"].mean()

def test_suggest_finds_churn_column():
    assert generic.suggest(real_generic_csv())["target"] == "left_company"

def test_upload_mapping_train_flow():
    buf = io.BytesIO(real_generic_csv().to_csv(index=False).encode())
    with TestClient(app) as c:
        c.post("/api/signup", json={"email": "g@b.co", "password": "secret1"})
        r = c.post("/api/upload", files={"file": ("my.csv", buf, "text/csv")}).json()
        assert r["needs_mapping"] and r["suggest"]["target"] == "left_company" and r["rows"] == 3150
        s = c.post("/api/train", json={"target": "left_company"}); assert s.status_code == 200
        s = s.json(); assert s["total"] == 3150 and s["model"]["auc"] > 0.9
        assert c.get("/api/results?band=High").status_code == 200
        assert c.get("/api/analytics").status_code == 200
        assert c.post("/api/campaign").status_code == 200

def test_sample_path_still_uses_iranian_model():
    with TestClient(app) as c:
        c.post("/api/signup", json={"email": "s@b.co", "password": "secret1"})
        assert "model" not in c.post("/api/sample").json()
        raw = open("data/sample_customers.csv", "rb").read()
        r = c.post("/api/upload", files={"file": ("s.csv", raw, "text/csv")}).json()
        assert r["total"] == 3150 and "model" not in r

@pytest.mark.parametrize("df,target,msg", [
    (pd.DataFrame({"a": range(300), "churn": ["x"] * 300}), "churn", "exactly 2"),
    (pd.DataFrame({"a": range(300), "churn": ["Yes", "No", "maybe"] * 100}), "churn", "exactly 2"),
    (pd.DataFrame({"a": range(300), "churn": ["red", "blue"] * 150}), "churn", "which value means"),
    (pd.DataFrame({"a": range(50), "churn": [0, 1] * 25}), "churn", "at least 200"),
    (pd.DataFrame({"a": range(300), "churn": [0] * 295 + [1] * 5}), "churn", "smaller group"),
])
def test_clear_errors(df, target, msg):
    with pytest.raises(generic.DataError) as e:
        generic.train_and_score(df, target)
    assert msg in str(e.value)

def test_bad_files_get_json_errors():
    with TestClient(app) as c:
        c.post("/api/signup", json={"email": "e@b.co", "password": "secret1"})
        assert "empty" in c.post("/api/upload", files={"file": ("e.csv", b"", "text/csv")}).json()["detail"]
        r = c.post("/api/train", json={"target": "x"}); assert r.status_code == 400
        c.post("/api/upload", files={"file": ("a.csv", b"a,b\n1,2\n3,4\n", "text/csv")})
        r = c.post("/api/train", json={"target": "a"}); assert r.status_code == 400 and "detail" in r.json()


def test_large_file_is_sampled_not_rejected():
    big = pd.concat([real_generic_csv()] * 7, ignore_index=True)  # 22,050 rows: repeated real rows, only to test the size rule
    out, m = generic.train_and_score(big, "left_company")
    assert len(out) == generic.MAX_ROWS and m["rows"] == generic.MAX_ROWS
    assert any("random sample of 20,000" in w and "22,050" in w for w in m["warnings"])
    assert abs(m["churn_rate"] - 0.1571) < 0.002
