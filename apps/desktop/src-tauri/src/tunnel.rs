//! 在线协作：主机主动连中转服务器（自建，或巡瞳官方中转），外网的成员访问中转的地址，
//! 中转把请求通过这条连接转给主机，主机在本地处理后回过去。主机不用公网 IP、不用改路由器。
//!
//! 协议（和 relay/wulabel_relay.py 对应，每条都是一个 JSON 文本消息）：
//!   中转 → 主机：ready{url} / error{message} / req{id,method,path,headers,body} / ws-open{id,path} / ws-msg{id,data} / ws-close{id}
//!   主机 → 中转：res{id,status,headers,body} / ws-msg{id,data} / ws-close{id} / ping
//! body 是 base64。

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::Router;
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tauri::Manager;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::Message as TMessage;
use tower::ServiceExt;

use crate::runtime::Runtime;
use crate::share::{self, Hub, RelayStatus};

/// 转发的一个请求最大多大（上传标注、图片都在这以内）
const MAX_BODY: usize = 64 * 1024 * 1024;

fn set_state(hub: &Hub, state: &str, url: &str, error: &str) {
    *hub.relay.lock() = RelayStatus { state: state.into(), url: url.into(), error: error.into() };
    hub.emit_status();
}

/// 中转地址和鉴权：自建中转用密钥，官方中转用巡瞳 API Key
fn endpoint(hub: &Hub) -> Result<(String, String), String> {
    let o = hub.opts.lock().clone();
    let (base, auth) = match o.mode.as_str() {
        "relay" => (o.relay_url.clone(), o.relay_secret.clone()),
        "xuntong" => (o.xuntong_relay_url.clone(), hub.app.state::<Runtime>().settings().ai_key.unwrap_or_default()),
        _ => return Err("没有开在线模式".into()),
    };
    let mut base = base.trim().to_string();
    if base.is_empty() {
        return Err("没有填中转服务器地址".into());
    }
    if let Some(rest) = base.strip_prefix("https://") {
        base = format!("wss://{rest}");
    } else if let Some(rest) = base.strip_prefix("http://") {
        base = format!("ws://{rest}");
    } else if !base.starts_with("ws://") && !base.starts_with("wss://") {
        base = format!("wss://{base}");
    }
    if !base.ends_with('/') {
        base.push('/');
    }
    Ok((format!("{base}_host?share={}", o.share_id), auth.trim().to_string()))
}

/// 开一个后台任务连中转，断了自动重连，直到停止共享
pub fn start(hub: Arc<Hub>) {
    tauri::async_runtime::spawn(async move {
        let mut closed = hub.closed_rx();
        let mut backoff = 2u64;
        loop {
            if *closed.borrow() {
                break;
            }
            set_state(&hub, "connecting", "", "");
            let began = Instant::now();
            let r = run_once(&hub).await;
            if *closed.borrow() {
                break;
            }
            // 连上过一阵子再断的，重连间隔从头算
            if began.elapsed() > Duration::from_secs(60) {
                backoff = 2;
            }
            let e = match r {
                Ok(()) => "和中转服务器的连接断开了".to_string(),
                Err(e) => e,
            };
            set_state(&hub, "error", "", &format!("{e}，{backoff} 秒后重连"));
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_secs(backoff)) => {}
                _ = closed.changed() => break,
            }
            backoff = (backoff * 2).min(60);
        }
    });
}

async fn run_once(hub: &Arc<Hub>) -> Result<(), String> {
    let (url, auth) = endpoint(hub)?;
    let mut req = url.as_str().into_client_request().map_err(|e| format!("中转地址不对：{e}"))?;
    if !auth.is_empty() {
        let v = HeaderValue::from_str(&format!("Bearer {auth}")).map_err(|_| "密钥里有不能用的字符".to_string())?;
        req.headers_mut().insert("Authorization", v);
    }
    let (ws, _) = tokio_tungstenite::connect_async(req).await.map_err(|e| format!("连不上中转服务器：{e}"))?;
    let (mut sink, mut stream) = ws.split();
    let (out_tx, mut out_rx) = mpsc::channel::<String>(256);
    let writer = tokio::spawn(async move {
        while let Some(s) = out_rx.recv().await {
            if sink.send(TMessage::Text(s.into())).await.is_err() {
                break;
            }
        }
    });
    let router = hub.router.lock().clone().ok_or_else(|| "协作服务没有启动".to_string())?;
    let mut tunnels: HashMap<String, mpsc::Sender<String>> = HashMap::new();
    let mut closed = hub.closed_rx();
    let result = loop {
        tokio::select! {
            m = stream.next() => match m {
                Some(Ok(TMessage::Text(t))) => {
                    if let Some(e) = handle(hub, &router, t.as_str(), &out_tx, &mut tunnels).await {
                        break Err(e);
                    }
                }
                Some(Ok(TMessage::Close(_))) | None => break Ok(()),
                Some(Err(e)) => break Err(format!("中转连接出错：{e}")),
                Some(Ok(_)) => {}
            },
            _ = closed.changed() => break Ok(()),
            _ = tokio::time::sleep(Duration::from_secs(25)) => {
                let _ = out_tx.send(json!({ "type": "ping" }).to_string()).await;
            }
        }
    };
    tunnels.clear();
    writer.abort();
    result
}

/// 处理中转发来的一条消息；返回 Some(错误) 表示要断开
async fn handle(
    hub: &Arc<Hub>,
    router: &Router,
    text: &str,
    out: &mpsc::Sender<String>,
    tunnels: &mut HashMap<String, mpsc::Sender<String>>,
) -> Option<String> {
    let v: Value = serde_json::from_str(text).ok()?;
    let id = v["id"].as_str().unwrap_or("").to_string();
    let kind = v["type"].as_str().unwrap_or("").to_string();
    match kind.as_str() {
        "ready" => set_state(hub, "online", v["url"].as_str().unwrap_or(""), ""),
        "error" => return Some(v["message"].as_str().unwrap_or("中转服务器拒绝了连接").to_string()),
        "req" => {
            let router = router.clone();
            let out = out.clone();
            tokio::spawn(async move {
                let res = forward(&router, &v).await;
                let _ = out.send(json!({ "type": "res", "id": id, "status": res.0, "headers": res.1, "body": res.2 }).to_string()).await;
            });
        }
        "ws-open" => {
            let path = v["path"].as_str().unwrap_or("");
            let token = path.split_once("t=").map(|(_, t)| t.split('&').next().unwrap_or("").to_string()).unwrap_or_default();
            let token = percent_decode(&token);
            let Some(m) = hub.member_of(&token) else {
                let _ = out.send(json!({ "type": "ws-close", "id": id }).to_string()).await;
                return None;
            };
            let (in_tx, mut in_rx) = mpsc::channel::<String>(64);
            tunnels.insert(id.clone(), in_tx);
            let (sess_tx, mut sess_rx) = mpsc::channel::<String>(128);
            let out2 = out.clone();
            let id2 = id.clone();
            // 会话要推给成员的消息，包成 ws-msg 发给中转
            tokio::spawn(async move {
                while let Some(s) = sess_rx.recv().await {
                    if out2.send(json!({ "type": "ws-msg", "id": id2, "data": s }).to_string()).await.is_err() {
                        break;
                    }
                }
                let _ = out2.send(json!({ "type": "ws-close", "id": id2 }).to_string()).await;
            });
            let incoming = futures_util::stream::poll_fn(move |cx| in_rx.poll_recv(cx));
            tokio::spawn(share::session(hub.clone(), m, Box::pin(incoming), sess_tx));
        }
        "ws-msg" => {
            if let Some(tx) = tunnels.get(&id) {
                let _ = tx.send(v["data"].as_str().unwrap_or("").to_string()).await;
            }
        }
        "ws-close" => {
            // 丢掉发送端，会话那边收到结束就退出
            tunnels.remove(&id);
        }
        _ => {}
    }
    None
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Some(n) = std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(n);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// 把中转转来的 HTTP 请求交给本地的路由处理，返回 (状态码, 响应头, base64 内容)
async fn forward(router: &Router, v: &Value) -> (u16, Value, String) {
    let fail = |code: u16, text: &str| (code, json!({ "content-type": "application/json" }), B64.encode(json!({ "error": text }).to_string()));
    let method = v["method"].as_str().unwrap_or("GET");
    let path = v["path"].as_str().unwrap_or("/");
    let body = match v["body"].as_str() {
        Some(b) if !b.is_empty() => match B64.decode(b) {
            Ok(b) => b,
            Err(_) => return fail(400, "请求内容不对"),
        },
        _ => Vec::new(),
    };
    let mut b = axum::http::Request::builder().method(method).uri(path);
    if let Some(h) = v["headers"].as_object() {
        for (k, val) in h {
            let k = k.to_ascii_lowercase();
            // 只带需要的头
            if matches!(k.as_str(), "authorization" | "content-type" | "accept" | "if-none-match") {
                if let Some(s) = val.as_str() {
                    b = b.header(k.as_str(), s);
                }
            }
        }
    }
    let req = match b.body(Body::from(body)) {
        Ok(r) => r,
        Err(_) => return fail(400, "请求格式不对"),
    };
    let resp = match router.clone().oneshot(req).await {
        Ok(r) => r,
        Err(_) => return fail(500, "处理请求失败"),
    };
    let status = resp.status().as_u16();
    let mut headers = serde_json::Map::new();
    for (k, val) in resp.headers() {
        if matches!(k.as_str(), "content-type" | "cache-control") {
            if let Ok(s) = val.to_str() {
                headers.insert(k.as_str().to_string(), json!(s));
            }
        }
    }
    match axum::body::to_bytes(resp.into_body(), MAX_BODY).await {
        Ok(bytes) => (status, Value::Object(headers), B64.encode(&bytes)),
        Err(_) => fail(500, "响应太大"),
    }
}
