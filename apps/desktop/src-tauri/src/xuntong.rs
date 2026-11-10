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
    pub status: String,
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
    if task != "detect" && task != "classify" {
        return Err(msg("巡瞳平台目前只支持检测和分类模型"));
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
        "recommendedConf": conf,
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
        .map_err(ai::err_of)?;
    let v = ai::parse(resp)?;
    Ok(UploadResult {
        id: v.get("id").cloned(),
        code: v["code"].as_str().unwrap_or("").to_string(),
        status: v["status"].as_str().unwrap_or("pending").to_string(),
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
            created_at: m["createdAt"].as_i64(),
        })
        .collect())
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
        return Err(msg("这段时间平台上没有告警记录"));
    }
    // 2. 下载图片到项目的 images/xuntong_日期/
    let folder = format!("xuntong_{}", chrono::Local::now().format("%Y%m%d_%H%M%S"));
    let dir = p.root.join(IMAGES_DIR).join(&folder);
    fs::create_dir_all(&dir)?;
    let total = items.len();
    let mut files: Vec<(PathBuf, usize)> = Vec::new();
    let mut failed = 0;
    for (i, it) in items.iter().enumerate() {
        progress(i, total * 2);
        let Some(url) = it["imageUrl"].as_str() else {
            failed += 1;
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
            let (w, h) = match image::image_dimensions(path) {
                Ok((w, h)) => (w as f64, h as f64),
                Err(_) => continue,
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
        boxes,
        folder: format!("{IMAGES_DIR}/{folder}"),
    })
}
