"""SQLite (local) / Postgres (DATABASE_URL) store: users, sessions, uploads."""
import hashlib, os, secrets, sqlite3, json
DB_URL = os.environ.get("DATABASE_URL", "")
SQLITE = os.environ.get("SQLITE_PATH", "results/app.db")
P = "%s" if DB_URL else "?"

def con():
    if DB_URL:
        import psycopg2
        return psycopg2.connect(DB_URL)
    os.makedirs("results", exist_ok=True)
    return sqlite3.connect(SQLITE)

def init():
    c = con(); cur = c.cursor()
    f = "migrations/001_init.sql" if DB_URL else "migrations/001_init.sqlite.sql"
    for stmt in open(f).read().split(";"):
        if stmt.strip():
            cur.execute(stmt)
    c.commit(); c.close()

def _hash(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 100_000).hex()

def signup(email, pw):
    c = con(); cur = c.cursor()
    salt = secrets.token_hex(8)
    try:
        cur.execute(f"INSERT INTO users (email, salt, pw_hash) VALUES ({P},{P},{P})", (email, salt, _hash(pw, salt)))
        c.commit()
    except Exception:
        c.rollback(); c.close(); return False
    c.close(); return True

def login(email, pw):
    c = con(); cur = c.cursor()
    cur.execute(f"SELECT id, salt, pw_hash FROM users WHERE email={P}", (email,))
    r = cur.fetchone()
    if not r or _hash(pw, r[1]) != r[2]:
        c.close(); return None
    tok = secrets.token_urlsafe(24)
    cur.execute(f"INSERT INTO sessions (token, user_id) VALUES ({P},{P})", (tok, r[0]))
    c.commit(); c.close(); return tok

def user_for(tok):
    if not tok: return None
    c = con(); cur = c.cursor()
    cur.execute(f"SELECT u.id, u.email FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token={P}", (tok,))
    r = cur.fetchone(); c.close()
    return {"id": r[0], "email": r[1]} if r else None

def save_run(user_id, name, summary):
    c = con(); cur = c.cursor()
    cur.execute(f"INSERT INTO runs (user_id, name, summary) VALUES ({P},{P},{P})", (user_id, name, json.dumps(summary)))
    c.commit(); c.close()

def runs(user_id):
    c = con(); cur = c.cursor()
    cur.execute(f"SELECT name, summary, created_at FROM runs WHERE user_id={P} ORDER BY id DESC LIMIT 10", (user_id,))
    rows = cur.fetchall(); c.close()
    return [{"name": r[0], "summary": json.loads(r[1]), "created_at": str(r[2])} for r in rows]
