"""
简易认证模块 — HMAC Token + SQLite (零外部依赖)
MVP 版本：邮箱 + 密码，无邮箱验证、无密码重置
"""
import os, sqlite3, hashlib, secrets, time, hmac, json, base64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "users.db")
SECRET = os.environ.get("JWT_SECRET", secrets.token_hex(32)).encode()
TOKEN_EXPIRE_HOURS = 720  # 30 天

# ── 简易 HMAC Token (不依赖 PyJWT) ──
def _make_token(payload: dict) -> str:
    payload["exp"] = int(time.time()) + TOKEN_EXPIRE_HOURS * 3600
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    sig = hmac.new(SECRET, body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"

def _verify_token(token: str) -> dict | None:
    try:
        body, sig = token.split(".", 1)
        expected = hmac.new(SECRET, body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(base64.urlsafe_b64decode(body.encode() + b"=" * (4 - len(body) % 4)))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None

# ── DB 初始化 ──
def _get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn

# ── 密码哈希 ──
def _hash_password(password: str, salt: str = "") -> tuple[str, str]:
    if not salt:
        salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
    return h.hex(), salt

# ── 注册 ──
def register_user(phone: str, password: str) -> dict:
    phone = phone.strip()
    if len(password) < 6:
        return {"error": "密码至少 6 位"}
    if not phone.isdigit() or len(phone) != 11 or not phone.startswith("1"):
        return {"error": "手机号格式不正确"}

    pw_hash, salt = _hash_password(password)
    conn = _get_db()
    try:
        conn.execute(
            "INSERT INTO users (phone, password_hash) VALUES (?, ?)",
            (phone, f"{salt}${pw_hash}"),
        )
        conn.commit()
        return {"ok": True, "phone": phone}
    except sqlite3.IntegrityError:
        return {"error": "该手机号已注册"}
    finally:
        conn.close()

# ── 登录 ──
def login_user(phone: str, password: str) -> dict:
    phone = phone.strip()
    conn = _get_db()
    row = conn.execute(
        "SELECT id, phone, password_hash FROM users WHERE phone = ?", (phone,)
    ).fetchone()
    conn.close()

    if not row:
        return {"error": "手机号或密码错误"}

    salt, stored_hash = row["password_hash"].split("$", 1)
    computed_hash, _ = _hash_password(password, salt)

    if computed_hash != stored_hash:
        return {"error": "手机号或密码错误"}

    token = _make_token({"user_id": row["id"], "phone": row["phone"]})
    return {"ok": True, "token": token, "phone": row["phone"], "user_id": row["id"]}

# ── 验证 Token ──
def verify_token(token: str) -> dict | None:
    payload = _verify_token(token)
    if payload:
        return {"user_id": payload["user_id"], "email": payload["email"]}
    return None
