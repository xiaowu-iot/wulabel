"""Wulabel 反馈上报服务。

职责：
- /feedback：验签 → 限流防刷 → 通过 SMTP（QQ 邮箱）把反馈发到作者邮箱。
  收件邮箱只在这里（服务端）出现，不进客户端。
- /update：检查更新。返回最新版本号和下载地址，顺带按匿名设备编号记录使用情况
  （每台设备每天一条，只记版本和系统，不记 IP），用 stats.py 查看统计。
"""

import json
import os
import re
import sqlite3
import time
import hmac
import base64
import smtplib
from email.message import EmailMessage
from pathlib import Path

import blake3
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


# —— 配置：优先环境变量，其次同目录下的 .env ——
def load_env() -> None:
    f = Path(__file__).with_name(".env")
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


load_env()

APP_SECRET = os.environ.get("APP_SECRET", "CHANGE_ME_shared_secret")
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.qq.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
MAIL_FROM = os.environ.get("MAIL_FROM", SMTP_USER)
FEEDBACK_TO = os.environ.get("FEEDBACK_TO", "")
DB_PATH = os.environ.get("DB_PATH", str(Path(__file__).with_name("feedback.db")))
# 最新版本信息：{"latest": "0.2.0", "url": "下载页地址", "notes": "新版本说明"}，发新版时改这个文件即可
LATEST_FILE = os.environ.get("LATEST_FILE", str(Path(__file__).with_name("latest.json")))

PER_DEVICE_HOUR = int(os.environ.get("PER_DEVICE_HOUR", "5"))
PER_DEVICE_DAY = int(os.environ.get("PER_DEVICE_DAY", "20"))
PER_IP_HOUR = int(os.environ.get("PER_IP_HOUR", "10"))
GLOBAL_DAY = int(os.environ.get("GLOBAL_DAY", "500"))

MAX_MESSAGE_CHARS = 2000
MAX_SHOT_BASE64 = 3_000_000
ALLOW_SKEW = 300

# 由共享密钥派生 32 字节 keyed-hash 密钥，和桌面端 feedback.rs 完全一致。
SIGN_KEY = blake3.blake3(APP_SECRET.encode("utf-8")).digest()


def _hex(data: bytes) -> str:
    return blake3.blake3(data).hexdigest()


def expected_sign(ts: int, device_id: str, message: str, screenshot: str) -> str:
    sign_input = (
        f"{ts}\n{device_id}\n"
        f"{_hex(message.encode('utf-8'))}\n{_hex(screenshot.encode('utf-8'))}"
    )
    return blake3.blake3(sign_input.encode("utf-8"), key=SIGN_KEY).hexdigest()


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS events (ts INTEGER, device TEXT, ip TEXT)")
    return conn


def err(msg: str, code: int = 400) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=code)


app = FastAPI(title="Wulabel feedback")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/feedback")
async def feedback(req: Request):
    try:
        data = await req.json()
    except Exception:
        return err("请求格式不对")

    message = (data.get("message") or "").strip()
    contact = (data.get("contact") or "").strip()
    app_version = str(data.get("app_version") or "")
    osname = str(data.get("os") or "")
    device_id = str(data.get("device_id") or "")
    screenshot = data.get("screenshot") or ""
    ts = data.get("ts")
    sign = str(data.get("sign") or "")

    if not message:
        return err("问题描述不能为空")
    if len(message) > MAX_MESSAGE_CHARS:
        return err("问题描述太长")
    if not isinstance(screenshot, str) or len(screenshot) > MAX_SHOT_BASE64:
        return err("截图太大")
    if not device_id or not sign or not isinstance(ts, int):
        return err("请求参数不完整")

    now = int(time.time())
    if abs(now - ts) > ALLOW_SKEW:
        return err("设备时间和服务器相差太大，请校准时间后重试")

    if not hmac.compare_digest(sign, expected_sign(ts, device_id, message, screenshot)):
        return err("签名校验失败", 403)

    xff = req.headers.get("x-forwarded-for") or ""
    ip = xff.split(",")[0].strip() or (req.client.host if req.client else "")

    conn = db()
    try:
        conn.execute("DELETE FROM events WHERE ts < ?", (now - 86400,))

        def count(where: str, *args) -> int:
            return conn.execute(f"SELECT COUNT(*) FROM events WHERE {where}", args).fetchone()[0]

        if count("device=? AND ts>?", device_id, now - 3600) >= PER_DEVICE_HOUR:
            return err("提交太频繁，请过一会儿再试", 429)
        if count("device=? AND ts>?", device_id, now - 86400) >= PER_DEVICE_DAY:
            return err("今天提交的反馈已经比较多了，明天再试吧", 429)
        if ip and count("ip=? AND ts>?", ip, now - 3600) >= PER_IP_HOUR:
            return err("提交太频繁，请过一会儿再试", 429)
        if count("ts>?", now - 86400) >= GLOBAL_DAY:
            return err("服务器繁忙，请稍后再试", 503)

        # 先占额度再发信，避免发信失败后被反复重试刷量。
        conn.execute("INSERT INTO events (ts, device, ip) VALUES (?,?,?)", (now, device_id, ip))
        conn.commit()
    finally:
        conn.close()

    try:
        send_mail(message, contact, app_version, osname, device_id, ip, screenshot)
    except Exception as e:  # noqa: BLE001
        return err(f"服务端发送失败：{e}", 500)

    return {"ok": True}


ID_RE = re.compile(r"^[0-9a-f]{8,64}$")
VER_RE = re.compile(r"^\d{1,4}(\.\d{1,4}){0,3}([-+.][0-9A-Za-z.]{1,20})?$")
OS_RE = re.compile(r"^[a-z0-9_]{1,16}$")


def usage_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS installs ("
        "device TEXT PRIMARY KEY, first_day TEXT, last_day TEXT, version TEXT, os TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS active ("
        "day TEXT, device TEXT, version TEXT, os TEXT, PRIMARY KEY (day, device))"
    )
    return conn


def today() -> str:
    # 按北京时间算「一天」
    return time.strftime("%Y-%m-%d", time.gmtime(time.time() + 8 * 3600))


def read_latest() -> dict:
    try:
        data = json.loads(Path(LATEST_FILE).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


@app.get("/update")
def update(req: Request):
    q = req.query_params
    version = (q.get("v") or "").strip().lstrip("v")[:32]
    osname = (q.get("os") or "").strip().lower()[:16]
    arch = (q.get("arch") or "").strip().lower()[:16]
    device = (q.get("id") or "").strip().lower()[:64]

    # 参数对得上才记，乱填的请求照样返回版本信息但不计数
    if ID_RE.match(device) and VER_RE.match(version) and OS_RE.match(osname) and (not arch or OS_RE.match(arch)):
        day = today()
        os_full = f"{osname} {arch}".strip()
        conn = usage_db()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO active (day, device, version, os) VALUES (?,?,?,?)",
                (day, device, version, os_full),
            )
            conn.execute(
                "INSERT INTO installs (device, first_day, last_day, version, os) VALUES (?,?,?,?,?) "
                "ON CONFLICT(device) DO UPDATE SET last_day=excluded.last_day, "
                "version=excluded.version, os=excluded.os",
                (device, day, day, version, os_full),
            )
            conn.commit()
        finally:
            conn.close()

    info = read_latest()
    return {
        "latest": str(info.get("latest") or version or ""),
        "url": str(info.get("url") or ""),
        "notes": str(info.get("notes") or ""),
    }


def send_mail(message, contact, app_version, osname, device_id, ip, screenshot) -> None:
    msg = EmailMessage()
    subject = message.replace("\n", " ")[:30]
    msg["Subject"] = f"[Wulabel 反馈] {subject}"
    msg["From"] = MAIL_FROM
    msg["To"] = FEEDBACK_TO
    msg.set_content(
        f"问题描述：\n{message}\n\n"
        f"联系方式：{contact or '（未填）'}\n"
        f"软件版本：{app_version}\n"
        f"操作系统：{osname}\n"
        f"设备标识：{device_id}\n"
        f"来源 IP：{ip}\n"
        f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    if screenshot:
        try:
            raw = base64.b64decode(screenshot)
            msg.add_attachment(raw, maintype="image", subtype="jpeg", filename="screenshot.jpg")
        except Exception:
            pass
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=20) as s:
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)
