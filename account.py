"""
用量管理 — 设备 ID + 次数 + HMAC 签名防伪 (零外部依赖)
"""
import os, sqlite3, json, time, hmac, hashlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "accounts.db")
FREE_USES = 2  # 新用户免费次数

# HMAC 密钥（生产环境用环境变量，开发期自动生成）
_SECRET = os.environ.get("DANCE_SECRET", "dev-secret-change-me").encode()


def sign_device(device_id: str) -> str:
    """给 device_id 签名，返回 token"""
    payload = f"{device_id}:{int(time.time())}"
    sig = hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()[:16]
    return f"{payload}:{sig}"


def verify_device(device_id: str, token: str) -> bool:
    """验证 token 是否合法（防伪造 + 防重放）"""
    try:
        parts = token.split(":")
        if len(parts) != 3:
            return False
        tid, ts, sig = parts
        if tid != device_id:
            return False
        # 时间戳超过 90 天拒绝（防止旧 token 复用）
        if abs(int(time.time()) - int(ts)) > 90 * 86400:
            return False
        expected = hmac.new(_SECRET, f"{tid}:{ts}".encode(), hashlib.sha256).hexdigest()[:16]
        return hmac.compare_digest(sig, expected)
    except Exception:
        return False


def _db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            device_id TEXT PRIMARY KEY,
            credits INTEGER NOT NULL DEFAULT 0,
            used INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS purchases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT NOT NULL,
            product_id TEXT NOT NULL,
            receipt TEXT,
            credits_added INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


def get_or_create(device_id: str) -> dict:
    """获取或创建设备账号，返回剩余次数等信息"""
    conn = _db()
    row = conn.execute("SELECT * FROM accounts WHERE device_id = ?", (device_id,)).fetchone()
    if not row:
        conn.execute(
            "INSERT INTO accounts (device_id, credits) VALUES (?, 0)",
            (device_id,),
        )
        conn.execute(
            "INSERT INTO purchases (device_id, product_id, credits_added) VALUES (?, ?, ?)",
            (device_id, "新用户赠送", FREE_USES),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM accounts WHERE device_id = ?", (device_id,)).fetchone()
    conn.close()
    return {
        "device_id": row["device_id"],
        "credits": row["credits"],
        "used": row["used"],
        "free_uses": FREE_USES,
        "created_at": row["created_at"],
    }


def deduct(device_id: str, task_id: str) -> dict:
    """扣一次用量（先扣免费额度再扣付费额度）"""
    conn = _db()
    row = conn.execute("SELECT * FROM accounts WHERE device_id = ?", (device_id,)).fetchone()
    if not row:
        conn.close()
        return {"error": "设备未注册"}

    used = row["used"]
    credits = row["credits"]

    if used < FREE_USES:
        used += 1
    elif credits > 0:
        credits -= 1
        used += 1
    else:
        conn.close()
        return {"error": "剩余次数不足，请充值"}

    conn.execute(
        "UPDATE accounts SET credits = ?, used = ?, updated_at = datetime('now') WHERE device_id = ?",
        (credits, used, device_id),
    )
    conn.execute(
        "INSERT INTO usage_log (device_id, task_id) VALUES (?, ?)",
        (device_id, task_id),
    )
    conn.commit()
    conn.close()
    return {"ok": True, "credits": credits, "used": used, "remaining": credits + max(0, FREE_USES - used)}


def get_history(device_id: str) -> dict:
    """返回充值 + 使用记录"""
    conn = _db()
    purchases = [dict(r) for r in conn.execute(
        "SELECT product_id, credits_added, created_at FROM purchases WHERE device_id = ? ORDER BY created_at DESC LIMIT 50",
        (device_id,),
    ).fetchall()]
    usage = [dict(r) for r in conn.execute(
        "SELECT task_id, created_at FROM usage_log WHERE device_id = ? ORDER BY created_at DESC LIMIT 50",
        (device_id,),
    ).fetchall()]
    conn.close()
    return {"purchases": purchases, "usage": usage}


def add_credits(device_id: str, amount: int, product_id: str = "", receipt: str = "") -> dict:
    """充值（Apple IAP 验证通过后调用）"""
    conn = _db()
    row = conn.execute("SELECT * FROM accounts WHERE device_id = ?", (device_id,)).fetchone()
    if not row:
        conn.execute(
            "INSERT INTO accounts (device_id, credits) VALUES (?, ?)",
            (device_id, amount),
        )
        credits = amount
        used = 0
    else:
        credits = row["credits"] + amount
        used = row["used"]
        conn.execute(
            "UPDATE accounts SET credits = ?, updated_at = datetime('now') WHERE device_id = ?",
            (credits, device_id),
        )

    conn.execute(
        "INSERT INTO purchases (device_id, product_id, receipt, credits_added) VALUES (?, ?, ?, ?)",
        (device_id, product_id, receipt, amount),
    )
    conn.commit()
    conn.close()
    return {"ok": True, "credits": credits, "used": used, "remaining": credits + max(0, FREE_USES - used)}
