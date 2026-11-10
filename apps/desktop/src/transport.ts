// 前端和后端的通道：在 Wulabel 软件里走 Tauri；在浏览器里（协作成员）走主机的 HTTP 接口 + WebSocket。
// 页面代码统一用这里的 invoke / listen，不直接用 @tauri-apps/api。
import { invoke as tauriInvoke } from "@tauri-apps/api/core";
import { listen as tauriListen, type UnlistenFn } from "@tauri-apps/api/event";

/** 在浏览器里打开（协作成员），不是 Wulabel 软件本身 */
export const isRemote = typeof window !== "undefined" && !("__TAURI_INTERNALS__" in window);

const TOKEN_KEY = "wulabel.token";

function readToken() {
  try {
    return sessionStorage.getItem(TOKEN_KEY) || localStorage.getItem(TOKEN_KEY) || "";
  } catch {
    return "";
  }
}

let token = isRemote ? readToken() : "";
let authLost: (() => void) | null = null;

export const getToken = () => token;

/** remember：关掉浏览器后还保持登录 */
export function setToken(t: string, remember: boolean) {
  token = t;
  try {
    sessionStorage.setItem(TOKEN_KEY, t);
    if (remember) localStorage.setItem(TOKEN_KEY, t);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* 存不了就只在内存里 */
  }
}

export function clearToken() {
  token = "";
  try {
    sessionStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* 忽略 */
  }
}

/** 登录失效（被停用、密码改了、主机重启）时回调，回到登录页 */
export function onAuthLost(f: (() => void) | null) {
  authLost = f;
}

export async function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  if (!isRemote) return tauriInvoke<T>(cmd, args);
  let r: Response;
  try {
    r = await fetch(`api/call/${cmd}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify(args ?? {}),
    });
  } catch {
    throw "连不上发起人的 Wulabel，请检查网络，或确认发起人还开着协作";
  }
  let j: unknown = null;
  try {
    j = await r.json();
  } catch {
    /* 不是 JSON */
  }
  const errOf = (fallback: string) => (j && typeof j === "object" && "error" in j ? String((j as { error: unknown }).error) : fallback);
  if (r.status === 401) {
    authLost?.();
    throw errOf("登录已失效，请重新登录");
  }
  if (!r.ok) throw errOf(r.status === 503 ? "发起人的 Wulabel 现在不在线" : `请求失败（${r.status}）`);
  return j as T;
}

/** 不需要登录的接口（登录页用） */
export async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  let j: unknown = null;
  try {
    j = await r.json();
  } catch {
    /* 不是 JSON */
  }
  if (!r.ok) {
    const e = j && typeof j === "object" && "error" in j ? String((j as { error: unknown }).error) : `请求失败（${r.status}）`;
    throw e;
  }
  return j as T;
}

// ---------- 实时事件 ----------

type Handler = (e: { payload: unknown }) => void;
const handlers = new Map<string, Set<Handler>>();

function dispatch(name: string, payload: unknown) {
  handlers.get(name)?.forEach((h) => {
    try {
      h({ payload });
    } catch (e) {
      console.error(e);
    }
  });
}

export function listen<T>(event: string, cb: (e: { payload: T }) => void): Promise<UnlistenFn> {
  if (!isRemote) return tauriListen<T>(event, cb);
  const h = cb as Handler;
  if (!handlers.has(event)) handlers.set(event, new Set());
  handlers.get(event)!.add(h);
  return Promise.resolve(() => {
    handlers.get(event)?.delete(h);
  });
}

let ws: WebSocket | null = null;
let liveWanted = false;
let retry: number | undefined;
let pinger: number | undefined;

/** 实时连接状态变化：connected / reconnecting（事件名 live-status） */
function setLive(state: string) {
  dispatch("live-status", state);
}

export function connectLive() {
  if (!isRemote || !token) return;
  liveWanted = true;
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  const url = new URL(`api/ws?t=${encodeURIComponent(token)}`, window.location.href);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  const sock = new WebSocket(url.toString());
  ws = sock;
  sock.onopen = () => setLive("connected");
  sock.onmessage = (m) => {
    try {
      const d = JSON.parse(String(m.data)) as { name: string; payload: unknown };
      dispatch(d.name, d.payload);
    } catch {
      /* 忽略 */
    }
  };
  sock.onclose = () => {
    if (ws === sock) ws = null;
    if (liveWanted && token) {
      setLive("reconnecting");
      window.clearTimeout(retry);
      retry = window.setTimeout(connectLive, 3000);
    }
  };
  window.clearInterval(pinger);
  pinger = window.setInterval(() => sendLive({ type: "ping" }), 25000);
}

export function disconnectLive() {
  liveWanted = false;
  window.clearTimeout(retry);
  window.clearInterval(pinger);
  ws?.close();
  ws = null;
}

export function sendLive(obj: unknown) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}
