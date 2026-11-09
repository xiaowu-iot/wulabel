//! 使用反馈上报：软件内把问题描述和截图发到作者邮箱。
//!
//! 客户端只负责收集 + 签名 + 发送到反馈服务，真正的发信和防刷都在服务端
//! （server/feedback/）。收件邮箱不在客户端出现，避免被逆向后直接拿去刷。

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tauri::Manager;

// ———————————————— 部署相关 ————————————————
// 反馈服务入口：官网子域 feedback.xiaowuiot.com，由服务器 nginx 反代到反馈后端
// （本机 127.0.0.1:8094）。自建部署改成你自己的地址即可。
const FEEDBACK_ENDPOINT: &str = "https://feedback.xiaowuiot.com/feedback";
// 客户端与服务端共享的签名密钥。源码开源，所以这里只放占位符；正式打包时用环境
// 变量 WULABEL_FEEDBACK_SECRET 注入真实密钥（要和服务端 .env 的 APP_SECRET 一致）：
//   PowerShell:  $env:WULABEL_FEEDBACK_SECRET="<密钥>"; npm run tauri build
// 签名只能挡“随手用脚本刷接口”这类低端滥用，真正兜底的是服务端限流和发信上限。
const FEEDBACK_APP_SECRET: &str = match option_env!("WULABEL_FEEDBACK_SECRET") {
    Some(s) => s,
    None => "wulabel-public-placeholder",
};

const MAX_MESSAGE_CHARS: usize = 2000;
const MAX_SHOT_BASE64: usize = 3_000_000; // 约 2.2MB 的 PNG/JPEG
const ALLOW_SKEW_SECS: u64 = 300;

#[derive(Deserialize)]
pub struct FeedbackInput {
    pub message: String,
    pub contact: Option<String>,
    /// 截图，dataURL（data:image/...;base64,xxx）或纯 base64，可空。
    pub screenshot: Option<String>,
}

#[derive(Serialize)]
struct FeedbackPayload {
    message: String,
    contact: String,
    app_version: String,
    os: String,
    device_id: String,
    screenshot: String,
    ts: u64,
    sign: String,
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// 把共享密钥派生成 32 字节的 keyed-hash 密钥（和服务端一致）。
fn sign_key() -> [u8; 32] {
    *blake3::hash(FEEDBACK_APP_SECRET.as_bytes()).as_bytes()
}

fn hex_hash(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

/// 对关键字段签名，防篡改 + 防重放（带时间戳）。服务端用同样算法重算比对。
fn sign(ts: u64, device_id: &str, message: &str, shot_b64: &str) -> String {
    let input = format!(
        "{ts}\n{device_id}\n{}\n{}",
        hex_hash(message.as_bytes()),
        hex_hash(shot_b64.as_bytes()),
    );
    blake3::keyed_hash(&sign_key(), input.as_bytes()).to_hex().to_string()
}

/// 读取或生成一个匿名设备标识，持久化到配置目录。只用于服务端按设备限流，
/// 不含任何真实身份信息。
fn device_id(app: &tauri::AppHandle) -> String {
    let Ok(dir) = app.path().app_config_dir() else {
        return "unknown".into();
    };
    let f = dir.join("device.id");
    if let Ok(s) = std::fs::read_to_string(&f) {
        let s = s.trim();
        if !s.is_empty() {
            return s.to_string();
        }
    }
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let seed = format!("{}-{}-{}", nanos, std::process::id(), dir.display());
    let id = blake3::hash(seed.as_bytes()).to_hex()[..24].to_string();
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(&f, &id);
    id
}

#[tauri::command]
pub async fn submit_feedback(app: tauri::AppHandle, input: FeedbackInput) -> Result<(), String> {
    let message = input.message.trim().to_string();
    if message.is_empty() {
        return Err("请先填写问题描述".into());
    }
    if message.chars().count() > MAX_MESSAGE_CHARS {
        return Err(format!("问题描述太长（最多 {MAX_MESSAGE_CHARS} 字）"));
    }

    // 去掉 dataURL 前缀，只留纯 base64，前后端对同一字符串签名/校验。
    let raw_shot = input.screenshot.unwrap_or_default();
    let shot = raw_shot.split_once(',').map(|(_, b)| b.to_string()).unwrap_or(raw_shot);
    if shot.len() > MAX_SHOT_BASE64 {
        return Err("截图太大，请去掉截图后再提交".into());
    }

    let contact = input.contact.unwrap_or_default().trim().to_string();
    let device = device_id(&app);
    let ts = now_secs();
    let app_version = env!("CARGO_PKG_VERSION").to_string();
    let os = format!("{} {}", std::env::consts::OS, std::env::consts::ARCH);
    let sign = sign(ts, &device, &message, &shot);

    let payload = FeedbackPayload {
        message,
        contact,
        app_version,
        os,
        device_id: device,
        screenshot: shot,
        ts,
        sign,
    };
    let body = serde_json::to_string(&payload).map_err(|e| e.to_string())?;

    // ureq 是阻塞调用，放到阻塞线程池，别卡住异步运行时。
    let result = tauri::async_runtime::spawn_blocking(move || {
        ureq::post(FEEDBACK_ENDPOINT)
            .timeout(Duration::from_secs(20))
            .set("Content-Type", "application/json; charset=utf-8")
            .send_string(&body)
    })
    .await
    .map_err(|e| e.to_string())?;

    match result {
        Ok(_) => Ok(()),
        Err(ureq::Error::Status(code, resp)) => {
            let text = resp.into_string().unwrap_or_default();
            let msg = serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_string))
                .unwrap_or_else(|| match code {
                    429 => "提交太频繁，请稍后再试".into(),
                    _ => format!("提交失败（{code}）"),
                });
            Err(msg)
        }
        Err(e) => Err(format!("网络连接失败，请检查网络后重试：{e}")),
    }
}

// 让 ALLOW_SKEW_SECS 在客户端也有意义：时间偏差过大时服务端会拒，这里仅做个本地提示的占位。
#[allow(dead_code)]
const _SKEW: u64 = ALLOW_SKEW_SECS;
