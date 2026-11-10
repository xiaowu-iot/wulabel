#!/usr/bin/env python3
"""Wulabel 协作中转服务。

发起人电脑上的 Wulabel（主机）主动连到这里，外网的成员访问这里的地址，
中转把请求转给主机、把结果转回来。主机不需要公网 IP，也不用改路由器。
图片和标注只在内存里转发，不落盘。

    pip install aiohttp
    python wulabel_relay.py --port 8095 --secret 你自己定的密钥            # 自建：主机用这个密钥连
    python wulabel_relay.py --port 8095 --auth xuntong                      # 巡瞳官方中转：主机用巡瞳 API Key 连

外面用 nginx / caddy 加 HTTPS 反向代理到这个端口（要支持 WebSocket），见 README.md。

协议（每条一个 JSON 文本消息）：
  主机连 /_host?share=<共享编号>，带 Authorization: Bearer <密钥或 API Key>
  中转 → 主机：ready{url} / error{message} / req{id,method,path,headers,body} / ws-open{id,path} / ws-msg{id,data} / ws-close{id}
  主机 → 中转：res{id,status,headers,body} / ws-msg{id,data} / ws-close{id} / ping
  body 是 base64。成员访问 /s/<共享编号>/...
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
import time

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

SHARE_RE = re.compile(r"^[a-z0-9]{6,32}$")
FORWARD_HEADERS = ("authorization", "content-type", "accept", "if-none-match")
BACK_HEADERS = ("content-type", "cache-control")

log = logging.getLogger("wulabel-relay")

OFFLINE_PAGE = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>主机不在线 · Wulabel</title>
<style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#14161a;color:#d7dbe0;display:grid;place-items:center;height:100vh;margin:0}
div{max-width:420px;padding:24px;text-align:center;line-height:1.7}h2{margin:0 0 8px}p{color:#9aa1ab}</style></head>
<body><div><h2>发起人的 Wulabel 现在不在线</h2><p>可能还没开启协作、电脑休眠了或断网了。请联系发起人，确认他的 Wulabel 已开启「在线协作」，然后刷新这个页面。</p></div></body></html>"""


class Host:
    """一个连上来的主机（一个共享）"""

    def __init__(self, ws: web.WebSocketResponse, key_hash: str, share: str):
        self.ws = ws
        self.key_hash = key_hash
        self.share = share
        self.pending: dict[str, asyncio.Future] = {}
        self.clients: dict[str, web.WebSocketResponse] = {}
        self.lock = asyncio.Lock()
        self.since = time.time()

    async def send(self, obj: dict) -> None:
        async with self.lock:
            await self.ws.send_str(json.dumps(obj, ensure_ascii=False))


class Relay:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.hosts: dict[str, Host] = {}
        # 共享编号归属：同一个编号只能被同一个密钥 / API Key 使用，防止被别人顶掉
        self.owners: dict[str, str] = {}
        self.key_cache: dict[str, tuple[bool, float]] = {}
        self.http: ClientSession | None = None

    # ---------- 鉴权 ----------

    async def check(self, token: str) -> bool:
        a = self.args
        if a.auth == "none":
            return True
        if not token:
            return False
        if a.auth == "secret":
            return hmac.compare_digest(token.encode(), a.secret.encode())
        # 巡瞳：用 API Key 调平台的 /v1/models，能调通就是有效的 Key（缓存 10 分钟）
        h = hashlib.sha256(token.encode()).hexdigest()
        ok, exp = self.key_cache.get(h, (False, 0.0))
        if exp > time.time():
            return ok
        try:
            assert self.http is not None
            async with self.http.get(
                a.xuntong_url.rstrip("/") + "/v1/models",
                headers={"Authorization": f"Bearer {token}"},
                timeout=ClientTimeout(total=15),
            ) as r:
                ok = r.status == 200
        except Exception as e:  # 平台连不上时不缓存，下次再试
            log.warning("验证 API Key 失败：%s", e)
            return False
        self.key_cache[h] = (ok, time.time() + 600)
        return ok

    # ---------- 主机连接 ----------

    def public_base(self, request: web.Request) -> str:
        if self.args.public_url:
            return self.args.public_url.rstrip("/") + "/"
        proto = request.headers.get("X-Forwarded-Proto", request.scheme)
        host = request.headers.get("X-Forwarded-Host", request.host)
        prefix = request.headers.get("X-Forwarded-Prefix", "").strip("/")
        return f"{proto}://{host}/" + (prefix + "/" if prefix else "")

    async def host_handler(self, request: web.Request) -> web.StreamResponse:
        share = request.query.get("share", "")
        if not SHARE_RE.match(share):
            return web.json_response({"message": "共享编号格式不对"}, status=400)
        auth = request.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.startswith("Bearer ") else ""
        if not await self.check(token):
            return web.json_response({"message": "密钥或 API Key 不对"}, status=401)
        key_hash = hashlib.sha256(token.encode()).hexdigest()[:24]
        owner = self.owners.get(share)
        if owner and owner != key_hash:
            return web.json_response({"message": "这个共享编号已被别人使用，请在 Wulabel 里换一个"}, status=403)
        mine = [h for h in self.hosts.values() if h.key_hash == key_hash and h.share != share]
        if self.args.max_shares_per_key and len(mine) >= self.args.max_shares_per_key:
            return web.json_response({"message": f"同一个账号最多同时开 {self.args.max_shares_per_key} 个共享"}, status=429)

        ws = web.WebSocketResponse(heartbeat=30, max_msg_size=self.args.max_body_mb * 2 * 1024 * 1024)
        await ws.prepare(request)
        old = self.hosts.get(share)
        if old is not None:
            await old.ws.close()
        host = Host(ws, key_hash, share)
        self.hosts[share] = host
        self.owners[share] = key_hash
        url = self.public_base(request) + f"s/{share}/"
        log.info("主机上线 %s（当前 %d 个）", share, len(self.hosts))
        await host.send({"type": "ready", "url": url})
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except ValueError:
                    continue
                t, mid = data.get("type"), data.get("id", "")
                if t == "res":
                    fut = host.pending.pop(mid, None)
                    if fut is not None and not fut.done():
                        fut.set_result(data)
                elif t == "ws-msg":
                    c = host.clients.get(mid)
                    if c is not None and not c.closed:
                        try:
                            await c.send_str(data.get("data", ""))
                        except ConnectionError:
                            pass
                elif t == "ws-close":
                    c = host.clients.pop(mid, None)
                    if c is not None:
                        await c.close()
        finally:
            if self.hosts.get(share) is host:
                del self.hosts[share]
            for fut in host.pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("主机断开"))
            for c in list(host.clients.values()):
                await c.close()
            log.info("主机下线 %s（当前 %d 个）", share, len(self.hosts))
        return ws

    # ---------- 成员访问 ----------

    async def share_handler(self, request: web.Request) -> web.StreamResponse:
        share = request.match_info["share"]
        tail = request.match_info.get("tail", "")
        host = self.hosts.get(share)
        if host is None:
            if tail.startswith("api/"):
                return web.json_response({"error": "发起人的 Wulabel 现在不在线"}, status=503)
            return web.Response(text=OFFLINE_PAGE, content_type="text/html", status=503)
        path = "/" + tail + (("?" + request.query_string) if request.query_string else "")
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await self.tunnel_ws(request, host, path)

        body = await request.read()
        if len(body) > self.args.max_body_mb * 1024 * 1024:
            return web.json_response({"error": "请求太大"}, status=413)
        rid = secrets.token_hex(8)
        fut = asyncio.get_running_loop().create_future()
        host.pending[rid] = fut
        headers = {k.lower(): v for k, v in request.headers.items() if k.lower() in FORWARD_HEADERS}
        try:
            await host.send(
                {
                    "type": "req",
                    "id": rid,
                    "method": request.method,
                    "path": path,
                    "headers": headers,
                    "body": base64.b64encode(body).decode() if body else "",
                }
            )
            res = await asyncio.wait_for(fut, timeout=self.args.timeout)
        except asyncio.TimeoutError:
            host.pending.pop(rid, None)
            return web.json_response({"error": "主机处理超时"}, status=504)
        except ConnectionError:
            host.pending.pop(rid, None)
            return web.json_response({"error": "发起人的 Wulabel 断开了"}, status=503)
        out_body = base64.b64decode(res.get("body") or "")
        out_headers = {k: v for k, v in (res.get("headers") or {}).items() if k.lower() in BACK_HEADERS}
        return web.Response(status=int(res.get("status", 200)), body=out_body, headers=out_headers)

    async def tunnel_ws(self, request: web.Request, host: Host, path: str) -> web.StreamResponse:
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        cid = secrets.token_hex(8)
        host.clients[cid] = ws
        try:
            await host.send({"type": "ws-open", "id": cid, "path": path})
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await host.send({"type": "ws-msg", "id": cid, "data": msg.data})
        except ConnectionError:
            pass
        finally:
            host.clients.pop(cid, None)
            try:
                await host.send({"type": "ws-close", "id": cid})
            except Exception:
                pass
        return ws

    async def redirect_slash(self, request: web.Request) -> web.StreamResponse:
        raise web.HTTPFound(f"{request.match_info['share']}/")

    async def index(self, request: web.Request) -> web.StreamResponse:
        return web.json_response({"app": "wulabel-relay", "hosts": len(self.hosts)})

    async def on_startup(self, app: web.Application) -> None:
        self.http = ClientSession()

    async def on_cleanup(self, app: web.Application) -> None:
        if self.http is not None:
            await self.http.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Wulabel 协作中转服务")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址（默认只听本机，前面用 nginx 反代）")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--auth", choices=["secret", "xuntong", "none"], default="secret", help="主机怎么鉴权")
    ap.add_argument("--secret", default="", help="--auth secret 时主机要带的密钥")
    ap.add_argument("--xuntong-url", default="https://xuntongai.com", help="--auth xuntong 时验证 API Key 用的平台地址")
    ap.add_argument("--public-url", default="", help="对外的地址，比如 https://relay.example.com/；不填按请求头推断")
    ap.add_argument("--max-shares-per-key", type=int, default=3, help="同一个密钥 / API Key 最多同时开几个共享（0 不限）")
    ap.add_argument("--max-body-mb", type=int, default=50, help="单个请求最大多少 MB")
    ap.add_argument("--timeout", type=int, default=120, help="等主机响应的超时秒数")
    args = ap.parse_args()
    if args.auth == "secret" and len(args.secret) < 8:
        ap.error("自建中转请用 --secret 设置一个至少 8 位的密钥（Wulabel 里填同样的密钥）")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    relay = Relay(args)
    app = web.Application(client_max_size=args.max_body_mb * 1024 * 1024)
    app.on_startup.append(relay.on_startup)
    app.on_cleanup.append(relay.on_cleanup)
    app.router.add_get("/", relay.index)
    app.router.add_get("/_host", relay.host_handler)
    app.router.add_get("/s/{share}", relay.redirect_slash)
    app.router.add_route("*", "/s/{share}/{tail:.*}", relay.share_handler)
    log.info("Wulabel 中转服务 http://%s:%d（鉴权：%s）", args.host, args.port, args.auth)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
