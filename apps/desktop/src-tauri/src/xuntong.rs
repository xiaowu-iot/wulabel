//! 和巡瞳 AI 平台对接（用户自己的 API Key）：
//! - 把训练好的模型上传成平台上的自定义模型：POST /v1/custom-models（multipart：file + meta）
//! - 查已上传的模型：GET /v1/custom-models
//! - 把平台上的告警图片拉回来标注：GET /v1/alarms，图片导入项目，识别框作为预标注（来源「巡瞳」）
//! 接口说明见 docs/巡瞳平台对接接口.md。

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::ai;
use crate::annotations;
use crate::error::{msg, AppResult};
use crate::images;
use crate::project::{Project, IMAGES_DIR};
use crate::runtime::Runtime;
use crate::training::{self, Predictor};

/// 在 JSON 里按几种可能的路径找一个数
fn find_f64(v: &serde_json::Value, paths: &[&[&str]]) -> Option<f64> {
    paths.iter().find_map(|p| {
        let mut cur = v;
        for k in *p {
            cur = cur.get(*k)?;
        }
        cur.as_f64()
    })
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UploadResult {
    pub id: Option<serde_json::Value>,
    pub code: String,
    /// pending 平台校验中 / ready 可以启用 / failed
    pub status: String,
    /// 平台上已启用（ready 之后要在控制台点启用）
    pub enabled: bool,
    pub message: String,
    /// 平台控制台地址，去那里配置、启用模型
    pub console_url: String,
}

/// 上传训练好的模型。还没导出 ONNX 的先导出
pub fn upload_model(p: &Project, rt: &Runtime, run_id: i64, name: &str, log: &(dyn Fn(String) + Sync)) -> AppResult<UploadResult> {
    let (base, key) = ai::config(rt)?;
    let run = training::get_run(p, run_id)?;
    if run.engine == training::EXTERNAL {
        return Err(msg("导入的外部模型不能上传，请上传在 Wulabel 里训练的模型"));
    }
    let task = p.meta.task.as_str();
    if task != "detect" {
        return Err(msg("巡瞳平台目前只支持目标检测模型"));
    }
    let dir = PathBuf::from(&run.dir);
    let onnx = dir.join("model.onnx");
    if !onnx.exists() {
        log("先导出 ONNX…".into());
        training::export_onnx(p, rt, run_id, log)?;
    }
    let meta_json: serde_json::Value = fs::read_to_string(dir.join("model.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let classes = meta_json["classes"].clone();
    if !classes.is_array() {
        return Err(msg("读不到模型的类别（model.json），请重新导出 ONNX"));
    }
    let eval = training::detail(p, run_id).ok().and_then(|d| d.eval).unwrap_or_default();
    let conf = find_f64(&eval, &[&["thresholds", "best", "conf"], &["val", "thresholds", "best", "conf"], &["test", "thresholds", "best", "conf"]]);
    let precision = find_f64(&eval, &[&["precision"], &["val", "precision"]]);
    let recall = find_f64(&eval, &[&["recall"], &["val", "recall"]]);
    let name = if name.trim().is_empty() { format!("{} {}", p.meta.name, run.name) } else { name.trim().to_string() };
    let meta = serde_json::json!({
        "name": name,
        "task": task,
        "format": if task == "classify" { "wulabel-cls" } else { "wulabel-det" },
        "classes": classes,
        "inputSize": run.config["imgSize"].as_u64(),
        "description": format!("Wulabel 训练（{}），{} {}", run.name, if task == "classify" { "准确率" } else { "mAP50" }, run.best_map50.map(|v| format!("{v:.3}")).unwrap_or_else(|| "—".into())),
        "source": "wulabel",
        "recommendedConf": conf.map(|c| c.clamp(0.05, 0.95)),
        "metrics": { "map50": run.best_map50, "precision": precision, "recall": recall },
    });
    let file = fs::read(&onnx).map_err(|e| msg(format!("读取模型文件失败：{e}")))?;
    let boundary = format!("----wulabel{}", crate::team::token(8));
    let mut body = Vec::with_capacity(file.len() + 4096);
    body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"meta\"\r\nContent-Type: application/json; charset=utf-8\r\n\r\n").as_bytes());
    body.extend_from_slice(meta.to_string().as_bytes());
    body.extend_from_slice(
        format!("\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"model.onnx\"\r\nContent-Type: application/octet-stream\r\n\r\n").as_bytes(),
    );
    body.extend_from_slice(&file);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    log(format!("上传模型（{:.1} MB）…", file.len() as f64 / 1048576.0));
    let resp = ureq::post(&format!("{base}/v1/custom-models"))
        .set("Authorization", &format!("Bearer {key}"))
        .set("Content-Type", &format!("multipart/form-data; boundary={boundary}"))
        .timeout(Duration::from_secs(600))
        .send_bytes(&body)
        .map_err(|e| match e {
            // 平台前面的 nginx 限制了上传大小，返回的是网页不是 JSON，单独说清楚
            ureq::Error::Status(413, _) => msg(format!(
                "模型文件 {:.1} MB，超过了巡瞳平台目前允许的上传大小。请联系平台放宽上传限制后再传",
                file.len() as f64 / 1048576.0
            )),
            e => ai::err_of(e),
        })?;
    let v = ai::parse(resp)?;
    Ok(UploadResult {
        id: v.get("id").cloned(),
        code: v["code"].as_str().unwrap_or("").to_string(),
        status: v["status"].as_str().unwrap_or("pending").to_string(),
        enabled: v["enabled"].as_bool().unwrap_or(false),
        message: v["message"].as_str().unwrap_or("").to_string(),
        console_url: format!("{base}/console"),
    })
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlatformModel {
    pub id: serde_json::Value,
    pub code: String,
    pub name: String,
    pub status: String,
    pub enabled: bool,
    pub message: String,
    pub created_at: Option<i64>,
}

pub fn list_models(rt: &Runtime) -> AppResult<Vec<PlatformModel>> {
    let (base, key) = ai::config(rt)?;
    let v = ai::get(&format!("{base}/v1/custom-models"), &key)?;
    let list = v.as_array().cloned().or_else(|| v["items"].as_array().cloned()).or_else(|| v["data"].as_array().cloned()).unwrap_or_default();
    Ok(list
        .iter()
        .map(|m| PlatformModel {
            id: m["id"].clone(),
            code: m["code"].as_str().unwrap_or("").to_string(),
            name: m["name"].as_str().unwrap_or("").to_string(),
            status: m["status"].as_str().unwrap_or("").to_string(),
            enabled: m["enabled"].as_bool().unwrap_or(false),
            message: m["message"].as_str().unwrap_or("").to_string(),
            created_at: m["createdAt"].as_i64(),
        })
        .collect())
}

/// 删除平台上的一个自定义模型（托管名额满了时先删旧版本）
pub fn delete_model(rt: &Runtime, id: &str) -> AppResult<()> {
    let (base, key) = ai::config(rt)?;
    let id: String = id.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect();
    let resp = ureq::delete(&format!("{base}/v1/custom-models/{id}"))
        .set("Authorization", &format!("Bearer {key}"))
        .timeout(Duration::from_secs(30))
        .call()
        .map_err(ai::err_of)?;
    ai::parse(resp)?;
    Ok(())
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PullOptions {
    /// 拉最近几天的
    pub days: u32,
    /// 只要这个模型的告警（模型编码），不填全部
    pub model: Option<String>,
    /// 最多拉多少张
    pub max: usize,
    /// 告警的识别框作为预标注
    pub with_boxes: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PullResult {
    pub fetched: usize,
    pub added: usize,
    pub duplicated: usize,
    pub failed: usize,
    /// 平台上没有原图的告警（平台加原图功能之前的，或超过 7 天图片已清理）
    pub no_image: usize,
    pub boxes: usize,
    pub folder: String,
}

/// 把平台上的告警图片拉回项目
pub fn pull_alarms(
    p: &Project,
    rt: &Runtime,
    pr: &Predictor,
    opts: &PullOptions,
    progress: &(dyn Fn(usize, usize) + Sync),
    log: &(dyn Fn(String) + Sync),
) -> AppResult<PullResult> {
    let (base, key) = ai::config(rt)?;
    let max = opts.max.clamp(1, 5000);
    let since = (chrono::Utc::now().timestamp() - opts.days.max(1) as i64 * 86400) * 1000;
    // 1. 翻页取告警记录
    let mut items: Vec<serde_json::Value> = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let mut url = format!("{base}/v1/alarms?since={since}&limit={}", (max - items.len()).min(200));
        if let Some(m) = opts.model.as_deref().filter(|s| !s.is_empty()) {
            url.push_str(&format!("&model={m}"));
        }
        if let Some(c) = &cursor {
            url.push_str(&format!("&cursor={c}"));
        }
        let v = ai::get(&url, &key)?;
        let page = v["items"].as_array().cloned().unwrap_or_default();
        let n = page.len();
        items.extend(page);
        cursor = v["nextCursor"].as_str().map(String::from);
        log(format!("已取到 {} 条告警记录…", items.len()));
        if n == 0 || cursor.is_none() || items.len() >= max {
            break;
        }
    }
    items.truncate(max);
    if items.is_empty() {
        return Err(msg("这段时间平台上没有告警记录（只有配置了推送通道的账号才会产生告警记录）"));
    }
    // 2. 下载图片到项目的 images/xuntong_日期/
    let folder = format!("xuntong_{}", chrono::Local::now().format("%Y%m%d_%H%M%S"));
    let dir = p.root.join(IMAGES_DIR).join(&folder);
    fs::create_dir_all(&dir)?;
    let total = items.len();
    let mut files: Vec<(PathBuf, usize)> = Vec::new();
    let mut failed = 0;
    let mut no_image = 0;
    for (i, it) in items.iter().enumerate() {
        progress(i, total * 2);
        let Some(url) = it["imageUrl"].as_str().filter(|s| !s.is_empty()) else {
            no_image += 1;
            continue;
        };
        let url = if url.starts_with("http") { url.to_string() } else { format!("{base}/{}", url.trim_start_matches('/')) };
        let id: String = it["id"].as_str().map(String::from).unwrap_or_else(|| it["id"].to_string()).chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect();
        let path = dir.join(format!("{}_{}.jpg", i + 1, if id.is_empty() { "alarm".to_string() } else { id }));
        let got = ureq::get(&url)
            .set("Authorization", &format!("Bearer {key}"))
            .timeout(Duration::from_secs(60))
            .call()
            .map_err(ai::err_of)
            .and_then(|r| {
                let mut buf = Vec::new();
                std::io::Read::read_to_end(&mut r.into_reader(), &mut buf).map_err(|e| msg(format!("下载告警图片失败：{e}")))?;
                Ok(buf)
            });
        match got {
            Ok(bytes) if !bytes.is_empty() => {
                fs::write(&path, bytes)?;
                files.push((path, i));
            }
            _ => failed += 1,
        }
    }
    // 3. 导入（文件已经在项目里，只登记）
    let paths: Vec<PathBuf> = files.iter().map(|(f, _)| f.clone()).collect();
    let (res, map) = images::import_batch(p, &paths, false, None, &|d, t| progress(total + d * total / t.max(1), total * 2))?;
    // 4. 识别框作为预标注
    let mut boxes = 0;
    if opts.with_boxes {
        for (path, i) in &files {
            let Some(&image_id) = map.get(path) else { continue };
            // 框是按平台的原图尺寸归一化的；下载的图就是原图，读不出尺寸时用平台给的宽高
            let (w, h) = match image::image_dimensions(path) {
                Ok((w, h)) => (w as f64, h as f64),
                Err(_) => match (items[*i]["width"].as_f64(), items[*i]["height"].as_f64()) {
                    (Some(w), Some(h)) => (w, h),
                    _ => continue,
                },
            };
            let found = ai::parse_detections(&items[*i]["detections"], w, h);
            if found.is_empty() {
                continue;
            }
            let pred = ai::to_pred(p, rt, pr, image_id, found, log)?;
            let anns = training::task_annotations(p, &pred, 0.0, "ai")?;
            boxes += anns.len();
            annotations::save_predictions(p, image_id, &anns)?;
        }
    }
    progress(total * 2, total * 2);
    Ok(PullResult {
        fetched: total,
        added: res.added,
        duplicated: res.duplicated,
        failed: failed + res.failed.len(),
        no_image,
        boxes,
        folder: format!("{IMAGES_DIR}/{folder}"),
    })
}

// ---------- 在 Wulabel 里登录巡瞳账号（设备授权：浏览器里登录、点允许，平台发一个 Key 给这台电脑） ----------

fn base_url(rt: &Runtime) -> String {
    rt.settings()
        .ai_url
        .filter(|u| !u.trim().is_empty())
        .unwrap_or_else(|| ai::DEFAULT_URL.to_string())
        .trim()
        .trim_end_matches('/')
        .to_string()
}

fn device_name() -> String {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "Wulabel".into())
}

/// 用系统默认浏览器打开网址
pub fn open_url(url: &str) -> AppResult<()> {
    if !(url.starts_with("https://") || url.starts_with("http://")) {
        return Err(msg("地址不对"));
    }
    #[cfg(target_os = "windows")]
    let r = crate::runtime::command("rundll32").args(["url.dll,FileProtocolHandler", url]).spawn();
    #[cfg(target_os = "macos")]
    let r = std::process::Command::new("open").arg(url).spawn();
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let r = std::process::Command::new("xdg-open").arg(url).spawn();
    r.map(|_| ()).map_err(|e| msg(format!("打不开浏览器：{e}")))
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeviceCode {
    pub device_code: String,
    pub user_code: String,
    pub verify_url: String,
    pub expires_in: u64,
    pub interval: u64,
}

/// 申请登录码，并在浏览器里打开巡瞳的确认页
pub fn login_start(rt: &Runtime) -> AppResult<DeviceCode> {
    let base = base_url(rt);
    let body = serde_json::json!({ "client": "wulabel", "deviceName": device_name(), "clientVersion": env!("CARGO_PKG_VERSION") });
    let resp = ureq::post(&format!("{base}/v1/device/code"))
        .set("Content-Type", "application/json; charset=utf-8")
        .timeout(Duration::from_secs(30))
        .send_string(&body.to_string())
        .map_err(|e| match e {
            ureq::Error::Status(404, _) => msg("巡瞳平台还不支持在 Wulabel 里登录（平台升级后就能用）。现在请先到巡瞳控制台创建 API Key，在下面的「手动填 API Key」里填"),
            e => ai::err_of(e),
        })?;
    let v = ai::parse(resp)?;
    let s = |k: &str| v[k].as_str().map(String::from).ok_or_else(|| msg(format!("巡瞳平台返回的登录信息缺少 {k}")));
    let code = DeviceCode {
        device_code: s("deviceCode")?,
        user_code: s("userCode")?,
        verify_url: s("verifyUrl")?,
        expires_in: v["expiresIn"].as_u64().unwrap_or(600),
        interval: v["interval"].as_u64().unwrap_or(3).max(1),
    };
    // 打不开浏览器也没关系，界面上会显示地址让用户自己打开
    let _ = open_url(&code.verify_url);
    Ok(code)
}

/// 问一次用户确认了没有。确认了就把发下来的 Key 存到设置里。
/// 返回 {status: pending / slow_down / expired / denied / ok, account?, keyName?}
pub fn login_poll(rt: &Runtime, device_code: &str) -> AppResult<serde_json::Value> {
    let base = base_url(rt);
    let r = ureq::post(&format!("{base}/v1/device/token"))
        .set("Content-Type", "application/json; charset=utf-8")
        .timeout(Duration::from_secs(30))
        .send_string(&serde_json::json!({ "deviceCode": device_code }).to_string());
    match r {
        Ok(resp) => {
            let v = ai::parse(resp)?;
            let key = v["apiKey"].as_str().filter(|k| !k.is_empty()).ok_or_else(|| msg("巡瞳平台没有返回 API Key"))?;
            let mut s = rt.settings();
            s.ai_key = Some(key.to_string());
            rt.save_settings(&s)?;
            Ok(serde_json::json!({ "status": "ok", "account": v["account"], "keyName": v["keyName"] }))
        }
        Err(ureq::Error::Status(400, resp)) => {
            let body: serde_json::Value = serde_json::from_str(&resp.into_string().unwrap_or_default()).unwrap_or_default();
            let code = body["code"].as_str().unwrap_or("");
            let status = match code {
                "authorization_pending" => "pending",
                "slow_down" => "slow_down",
                "expired_token" => "expired",
                "access_denied" => "denied",
                _ => return Err(msg(format!("登录失败：{}", body["message"].as_str().unwrap_or(code)))),
            };
            Ok(serde_json::json!({ "status": status }))
        }
        Err(e) => Err(ai::err_of(e)),
    }
}

/// 当前账号和额度（平台不支持时返回 null，界面只显示「已连接」）
pub fn account(rt: &Runtime) -> AppResult<serde_json::Value> {
    let (base, key) = ai::config(rt)?;
    match ureq::get(&format!("{base}/v1/account"))
        .set("Authorization", &format!("Bearer {key}"))
        .timeout(Duration::from_secs(20))
        .call()
    {
        Ok(resp) => ai::parse(resp),
        Err(ureq::Error::Status(404, _)) => Ok(serde_json::Value::Null),
        Err(e) => Err(ai::err_of(e)),
    }
}

/// 退出登录：让平台吊销这个 Key（失败也没关系），本机清掉
pub fn logout(rt: &Runtime) -> AppResult<()> {
    if let Ok((base, key)) = ai::config(rt) {
        let _ = ureq::delete(&format!("{base}/v1/device/token"))
            .set("Authorization", &format!("Bearer {key}"))
            .timeout(Duration::from_secs(15))
            .call();
    }
    let mut s = rt.settings();
    s.ai_key = None;
    rt.save_settings(&s)
}
