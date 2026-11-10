//! 巡瞳 AI 平台辅助标注：用用户自己账号的 API Key 调识别接口，结果作为预标注（来源记为 ai）。
//!
//! - 专用模型：POST /v1/detect（image、model、minConfidence）→ detections[{label, name, confidence, box}]
//! - 大模型：POST /v1/analyze（image、scene）→ hazards[{category, description, box}]
//! - 可用模型：GET /v1/models
//! - 按项目类别识别：POST /v1/detect-classes（image、classes、instructions）→ 只返回项目里的类别
//! - 聊天定规则：POST /v1/chat/rules（classes、instructions、messages）→ reply、建议的规则、试识别结果
//! 位置框 box 是 [x1, y1, x2, y2]，按图片宽高 0–1000 归一化。每次识别会消耗用户的套餐额度。

use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::annotations;
use crate::error::{msg, AppError, AppResult};
use crate::project::Project;
use crate::runtime::Runtime;
use crate::training::{self, PredResult, Predictor, PrelabelResult, RawPred};

/// 巡瞳 AI 默认地址（可在设置里改，比如私有化部署）
pub const DEFAULT_URL: &str = "https://xuntongai.com";
/// 大模型通用识别，在模型列表里用这个编码表示
const LLM: &str = "llm";
/// 按项目自己的类别识别（只返回项目里有的类别）
pub const CLASSES: &str = "classes";
/// 项目设置里存整体识别规则的键
const RULES_KEY: &str = "ai_instructions";
/// 项目绑定的巡瞳平台规则编号（rs_xxx）：同步后巡瞳自己的识别、告警也按这套规则
const RULE_SET_KEY: &str = "xt_rule_set";
/// 平台的限制：类别 1–30 个、类别名 ≤30 字、英文提示词 ≤120、识别说明 ≤500、整体规则 ≤1000、每条聊天 ≤2000、最多 30 条
const MAX_CLASSES: usize = 30;
const MAX_NAME: usize = 30;
const MAX_INSTRUCTIONS: usize = 1000;
const MAX_MESSAGE: usize = 2000;
const MAX_MESSAGES: usize = 30;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AiModel {
    pub code: String,
    pub name: String,
    pub description: String,
}

pub(crate) fn config(rt: &Runtime) -> AppResult<(String, String)> {
    let s = rt.settings();
    let key = s.ai_key.unwrap_or_default().trim().to_string();
    if key.is_empty() {
        return Err(msg("还没有填巡瞳 AI 的 API Key。请到「设置 → 巡瞳 AI」填写（在巡瞳 AI 控制台的「API Key」页创建）"));
    }
    let url = s.ai_url.filter(|u| !u.trim().is_empty()).unwrap_or_else(|| DEFAULT_URL.to_string());
    Ok((url.trim().trim_end_matches('/').to_string(), key))
}

pub(crate) fn err_of(e: ureq::Error) -> AppError {
    match e {
        ureq::Error::Status(code, resp) => {
            let body = resp.into_string().unwrap_or_default();
            let detail = serde_json::from_str::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| {
                    ["message", "msg", "error"].iter().find_map(|k| v[*k].as_str().map(String::from))
                })
                .unwrap_or_else(|| body.chars().take(200).collect());
            match code {
                401 | 403 => msg(format!("巡瞳 AI 拒绝了请求（{code}）：{detail}。请到「设置」里检查 API Key 和 IP 白名单")),
                402 | 429 => msg(format!("巡瞳 AI 额度不足或调用太频繁：{detail}")),
                _ => msg(format!("巡瞳 AI 返回错误（{code}）：{detail}")),
            }
        }
        e => msg(format!("连不上巡瞳 AI：{e}")),
    }
}

pub(crate) fn parse(resp: ureq::Response) -> AppResult<serde_json::Value> {
    let text = resp.into_string().map_err(|e| msg(format!("读取巡瞳 AI 返回失败：{e}")))?;
    serde_json::from_str(&text).map_err(|_| msg("巡瞳 AI 返回的不是 JSON，请检查设置里的地址"))
}

pub(crate) fn get(url: &str, key: &str) -> AppResult<serde_json::Value> {
    let r = ureq::get(url).set("Authorization", &format!("Bearer {key}")).timeout(Duration::from_secs(30)).call();
    parse(r.map_err(err_of)?)
}

/// 发 JSON 请求（PUT 等），返回 JSON
fn send_json(method: &str, url: &str, key: &str, body: &serde_json::Value) -> AppResult<serde_json::Value> {
    let r = ureq::request(method, url)
        .set("Authorization", &format!("Bearer {key}"))
        .set("Content-Type", "application/json; charset=utf-8")
        .timeout(Duration::from_secs(60))
        .send_string(&body.to_string());
    parse(r.map_err(err_of)?)
}

fn cut(s: &str, n: usize) -> String {
    s.trim().chars().take(n).collect()
}

/// 规则换成平台接口要的 classes（检查数量、类别名长度，超长的说明截断）
fn platform_classes(r: &Rules) -> AppResult<Vec<serde_json::Value>> {
    if r.classes.is_empty() {
        return Err(msg("项目里还没有类别，先在右侧添加类别（可以写上识别说明）"));
    }
    if r.classes.len() > MAX_CLASSES {
        return Err(msg(format!("巡瞳按类别识别最多支持 {MAX_CLASSES} 个类别，项目里有 {} 个", r.classes.len())));
    }
    if let Some(c) = r.classes.iter().find(|c| c.name.chars().count() > MAX_NAME) {
        return Err(msg(format!("类别名「{}」超过 {MAX_NAME} 个字，巡瞳按类别识别用不了，请改短一些", c.name)));
    }
    Ok(r
        .classes
        .iter()
        .map(|c| serde_json::json!({ "name": c.name, "prompt": cut(&c.prompt, 120), "description": cut(&c.description, 500) }))
        .collect())
}

fn post(url: &str, key: &str, body: &serde_json::Value) -> AppResult<serde_json::Value> {
    let r = ureq::post(url)
        .set("Authorization", &format!("Bearer {key}"))
        .set("Content-Type", "application/json; charset=utf-8")
        .timeout(Duration::from_secs(120))
        .send_string(&body.to_string());
    parse(r.map_err(err_of)?)
}

/// 可用模型：平台上开通的专用模型 + 大模型通用识别
pub fn models(rt: &Runtime) -> AppResult<Vec<AiModel>> {
    let (base, key) = config(rt)?;
    let v = get(&format!("{base}/v1/models"), &key)?;
    let list = v.as_array().cloned().or_else(|| v["models"].as_array().cloned()).or_else(|| v["data"].as_array().cloned());
    let mut out: Vec<AiModel> = vec![AiModel {
        code: CLASSES.into(),
        name: "按项目类别识别（推荐）".into(),
        description: "只找项目里的类别，按每个类别的识别说明和整体规则来，不会多认别的东西".into(),
    }];
    let platform: Vec<AiModel> = list
        .unwrap_or_default()
        .iter()
        .filter_map(|m| {
            Some(AiModel {
                code: m["code"].as_str()?.to_string(),
                name: m["name"].as_str().unwrap_or_else(|| m["code"].as_str().unwrap_or("")).to_string(),
                description: m["description"].as_str().unwrap_or("").to_string(),
            })
        })
        .collect();
    out.extend(platform);
    out.push(AiModel {
        code: LLM.into(),
        name: "大模型通用识别".into(),
        description: "按安全隐患类别（消防、电气、环境卫生等）找出画面里的隐患位置，每张图计 1 张大模型额度".into(),
    });
    Ok(out)
}

/// 标准 base64 编码（带 = 补齐）
pub fn b64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

/// 图片按 EXIF 摆正后，长边不超过 2048 重新编码成 JPEG，返回 dataURL 和原图（摆正后）宽高
fn encode_image(path: &Path) -> AppResult<(String, f64, f64)> {
    let img = crate::images::decode(path)?;
    let (w, h) = (img.width() as f64, img.height() as f64);
    let img = if w.max(h) > 2048.0 { img.resize(2048, 2048, image::imageops::FilterType::Triangle) } else { img };
    let rgb = img.to_rgb8();
    let mut buf = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 85)
        .encode_image(&rgb)
        .map_err(|e| msg(format!("图片编码失败：{e}")))?;
    Ok((format!("data:image/jpeg;base64,{}", b64(&buf)), w, h))
}

/// 0–1000 归一化的 [x1, y1, x2, y2] 换成像素 [x, y, w, h]
fn to_px(b: &serde_json::Value, w: f64, h: f64) -> Option<[f64; 4]> {
    let v: Vec<f64> = b.as_array()?.iter().filter_map(|x| x.as_f64()).collect();
    if v.len() != 4 {
        return None;
    }
    let x0 = (v[0] / 1000.0 * w).clamp(0.0, w);
    let y0 = (v[1] / 1000.0 * h).clamp(0.0, h);
    let x1 = (v[2] / 1000.0 * w).clamp(0.0, w);
    let y1 = (v[3] / 1000.0 * h).clamp(0.0, h);
    let r = |x: f64| (x * 100.0).round() / 100.0;
    (x1 - x0 >= 2.0 && y1 - y0 >= 2.0).then(|| [r(x0), r(y0), r(x1 - x0), r(y1 - y0)])
}

/// 识别一张图，返回 (类别名, 分数, 像素框)。按项目类别识别时要传 rules
fn recognize(base: &str, key: &str, path: &Path, model: &str, conf: f64, rules: &Rules) -> AppResult<Vec<(String, f64, [f64; 4])>> {
    let (image, w, h) = encode_image(path)?;
    let mut out = Vec::new();
    if model == CLASSES {
        let body = serde_json::json!({
            "image": image,
            "classes": platform_classes(rules)?,
            "instructions": cut(&rules.instructions, MAX_INSTRUCTIONS),
            "minConfidence": conf,
        });
        let v = post(&format!("{base}/v1/detect-classes"), key, &body)?;
        out = parse_detections(&v["detections"], w, h);
        // 平台万一返回了清单外的类别，这里再挡一道
        out.retain(|(n, _, _)| rules.classes.iter().any(|c| &c.name == n));
    } else if model == LLM {
        let v = post(&format!("{base}/v1/analyze"), key, &serde_json::json!({ "image": image, "scene": "general" }))?;
        // 大模型没有逐条分数，按整体置信度给一个参考值
        let score = match v["confidence"].as_str() {
            Some("high") => 0.9,
            Some("low") => 0.5,
            _ => 0.7,
        };
        for hz in v["hazards"].as_array().into_iter().flatten() {
            let Some(b) = to_px(&hz["box"], w, h) else { continue };
            let name = hz["category"].as_str().filter(|s| !s.is_empty()).unwrap_or("隐患").to_string();
            out.push((name, score, b));
        }
    } else {
        let body = serde_json::json!({ "image": image, "model": model, "minConfidence": conf });
        let v = post(&format!("{base}/v1/detect"), key, &body)?;
        for d in v["detections"].as_array().into_iter().flatten() {
            let Some(b) = to_px(&d["box"], w, h) else { continue };
            let name = d["name"].as_str().or(d["label"].as_str()).unwrap_or("目标").to_string();
            out.push((name, d["confidence"].as_f64().unwrap_or(1.0), b));
        }
    }
    out.retain(|(_, s, _)| *s >= conf);
    Ok(out)
}

/// detections[{name, confidence, box}] 换成 (类别名, 分数, 像素框)
pub(crate) fn parse_detections(v: &serde_json::Value, w: f64, h: f64) -> Vec<(String, f64, [f64; 4])> {
    v.as_array()
        .into_iter()
        .flatten()
        .filter_map(|d| {
            let b = to_px(&d["box"], w, h)?;
            let name = d["name"].as_str().or(d["label"].as_str())?.to_string();
            Some((name, d["confidence"].as_f64().unwrap_or(1.0), b))
        })
        .collect()
}

/// 识别结果换成预测结果：类别名对应到项目类别（没有的自动新建）；分割项目用 SAM 把框变成轮廓
pub(crate) fn to_pred(
    p: &Project,
    rt: &Runtime,
    pr: &Predictor,
    image_id: i64,
    found: Vec<(String, f64, [f64; 4])>,
    log: &(dyn Fn(String) + Sync),
) -> AppResult<PredResult> {
    let mut names: Vec<String> = Vec::new();
    for (n, _, _) in &found {
        if !names.contains(n) {
            names.push(n.clone());
        }
    }
    let map = training::class_map(p, &names)?;
    let idx = |n: &str| names.iter().position(|x| x == n).unwrap_or(0);
    let mut raw = RawPred {
        boxes: found.iter().map(|(n, s, b)| [b[0], b[1], b[2], b[3], idx(n) as f64, *s]).collect(),
        ..Default::default()
    };
    if p.meta.task == "segment" && !raw.boxes.is_empty() {
        let bx: Vec<(usize, f64, [f64; 4])> = found.iter().map(|(n, s, b)| (idx(n), *s, *b)).collect();
        match crate::assist::boxes_to_polys(p, rt, pr, image_id, &bx, log) {
            Ok(polys) => raw.polys = polys,
            // SAM 用不了时退回矩形轮廓（task_annotations / 前端会把框转成矩形）
            Err(e) => log(format!("SAM 出轮廓失败，先用矩形：{e}")),
        }
    }
    Ok(training::to_result(raw, &map))
}

/// 用巡瞳 AI 识别一张图
pub fn detect(
    p: &Project,
    rt: &Runtime,
    pr: &Predictor,
    image_id: i64,
    model: &str,
    conf: f64,
    log: &(dyn Fn(String) + Sync),
) -> AppResult<PredResult> {
    let (base, key) = config(rt)?;
    let (stored, _) = crate::images::locate(p, image_id)?;
    let found = recognize(&base, &key, &p.resolve(&stored), model, conf, &rules(p)?)?;
    to_pred(p, rt, pr, image_id, found, log)
}

/// 用巡瞳 AI 批量预标注所有「未标注」且没有标注的图（逐张调用，每张消耗 1 次额度）
pub fn prelabel_all(
    p: &Project,
    rt: &Runtime,
    pr: &Predictor,
    model: &str,
    conf: f64,
    progress: &dyn Fn(usize, usize),
    log: &(dyn Fn(String) + Sync),
) -> AppResult<PrelabelResult> {
    let (base, key) = config(rt)?;
    // 专用模型（含自己上传的模型）走批量接口，并发发送，快很多
    if model != CLASSES && model != LLM {
        return prelabel_special(p, rt, pr, &base, &key, model, conf, progress, log);
    }
    let rules = rules(p)?;
    let items = training::prelabel_targets(p)?;
    let total = items.len();
    let mut result = PrelabelResult { images: 0, boxes: 0 };
    progress(0, total);
    for (n, (id, path)) in items.iter().enumerate() {
        let r = recognize(&base, &key, path, model, conf, &rules).and_then(|found| to_pred(p, rt, pr, *id, found, log));
        let pred = match r {
            Ok(v) => v,
            // 额度用完、Key 不对等错误后面的图也会失败，直接停下，已经标好的保留
            Err(e) => return Err(msg(format!("第 {} 张出错，已预标注 {} 张：{e}", n + 1, result.images))),
        };
        let anns = training::task_annotations(p, &pred, conf, "ai")?;
        if !anns.is_empty() {
            result.boxes += anns.len();
            result.images += 1;
            annotations::save_predictions(p, *id, &anns)?;
        }
        progress(n + 1, total);
    }
    Ok(result)
}

/// 一批几张、同时几批：平台每个 Key 每秒最多 20 次识别，批内逐张识别，2 批并发不会超；
/// 超了整批会失败（前面识别过的也白算），所以宁可保守
const BATCH: usize = 8;
const WORKERS: usize = 2;

/// 发一批到 /v1/detect/batch，429（太频繁）时退避重试
fn detect_batch(base: &str, key: &str, reqs: &[serde_json::Value]) -> AppResult<Vec<serde_json::Value>> {
    let body = serde_json::Value::Array(reqs.to_vec()).to_string();
    let mut wait = 2;
    for attempt in 0..5 {
        let r = ureq::post(&format!("{base}/v1/detect/batch"))
            .set("Authorization", &format!("Bearer {key}"))
            .set("Content-Type", "application/json; charset=utf-8")
            .timeout(Duration::from_secs(300))
            .send_string(&body);
        match r {
            Ok(resp) => {
                let v = parse(resp)?;
                return v.as_array().cloned().ok_or_else(|| msg("巡瞳 AI 批量识别返回的格式不对"));
            }
            Err(ureq::Error::Status(429, _)) if attempt < 4 => {
                std::thread::sleep(Duration::from_secs(wait));
                wait *= 2;
            }
            Err(e) => return Err(err_of(e)),
        }
    }
    Err(msg("巡瞳 AI 一直提示调用太频繁，请稍后再试"))
}

/// 专用模型批量预标注：每批 8 张走批量接口，2 批并发；图片编码在后台线程里做。
/// 结果回到当前线程统一保存（数据库、进度都在这里）
#[allow(clippy::too_many_arguments)]
fn prelabel_special(
    p: &Project,
    rt: &Runtime,
    pr: &Predictor,
    base: &str,
    key: &str,
    model: &str,
    conf: f64,
    progress: &dyn Fn(usize, usize),
    log: &(dyn Fn(String) + Sync),
) -> AppResult<PrelabelResult> {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let items = training::prelabel_targets(p)?;
    let total = items.len();
    let mut result = PrelabelResult { images: 0, boxes: 0 };
    progress(0, total);
    if total == 0 {
        return Ok(result);
    }
    let chunks: Vec<&[(i64, std::path::PathBuf)]> = items.chunks(BATCH).collect();
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    // 每批的结果：(图片 id, 原图宽, 高, detections)
    type Batch = AppResult<Vec<(i64, f64, f64, serde_json::Value)>>;
    let (tx, rx) = std::sync::mpsc::channel::<Batch>();
    let mut first_err: Option<String> = None;
    let mut done = 0;
    std::thread::scope(|s| {
        for _ in 0..WORKERS.min(chunks.len()) {
            let tx = tx.clone();
            let (chunks, next, stop) = (&chunks, &next, &stop);
            s.spawn(move || loop {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(chunk) = chunks.get(i) else { break };
                let r = (|| -> Batch {
                    let mut meta = Vec::new();
                    let mut reqs = Vec::new();
                    for (id, path) in chunk.iter() {
                        let (image, w, h) = encode_image(path)?;
                        reqs.push(serde_json::json!({ "image": image, "model": model, "minConfidence": conf }));
                        meta.push((*id, w, h));
                    }
                    let res = detect_batch(base, key, &reqs)?;
                    Ok(meta.into_iter().zip(res).map(|((id, w, h), v)| (id, w, h, v["detections"].clone())).collect())
                })();
                if tx.send(r).is_err() {
                    break;
                }
            });
        }
        drop(tx);
        for r in rx {
            match r {
                Ok(list) => {
                    for (id, w, h, dets) in list {
                        done += 1;
                        if first_err.is_some() {
                            continue;
                        }
                        let saved = (|| -> AppResult<usize> {
                            let found: Vec<_> = parse_detections(&dets, w, h).into_iter().filter(|(_, s, _)| *s >= conf).collect();
                            if found.is_empty() {
                                return Ok(0);
                            }
                            let pred = to_pred(p, rt, pr, id, found, log)?;
                            let anns = training::task_annotations(p, &pred, conf, "ai")?;
                            if !anns.is_empty() {
                                annotations::save_predictions(p, id, &anns)?;
                            }
                            Ok(anns.len())
                        })();
                        match saved {
                            Ok(0) => {}
                            Ok(n) => {
                                result.images += 1;
                                result.boxes += n;
                            }
                            Err(e) => {
                                first_err = Some(e.to_string());
                                stop.store(true, Ordering::Relaxed);
                            }
                        }
                    }
                    progress(done.min(total), total);
                }
                // 额度用完、Key 不对等：停下，已经标好的保留
                Err(e) => {
                    if first_err.is_none() {
                        first_err = Some(e.to_string());
                    }
                    stop.store(true, Ordering::Relaxed);
                }
            }
        }
    });
    if let Some(e) = first_err {
        return Err(msg(format!("批量预标注中途停下，已预标注 {} 张：{e}", result.images)));
    }
    log(format!("巡瞳 AI 批量预标注完成：{} 张图、{} 个目标", result.images, result.boxes));
    Ok(result)
}

// ---------- 识别规则、聊天定规则 ----------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassRule {
    #[serde(default)]
    pub id: i64,
    pub name: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub description: String,
}

/// 项目的识别规则：整体规则 + 每个类别的识别说明
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Rules {
    pub instructions: String,
    pub classes: Vec<ClassRule>,
    /// 绑定的巡瞳平台规则编号，没同步过是空
    #[serde(default)]
    pub rule_set: String,
}

pub fn rules(p: &Project) -> AppResult<Rules> {
    let instructions = crate::team::get_setting(p, RULES_KEY)?.unwrap_or_default();
    let classes = annotations::list_labels(p)?
        .into_iter()
        .map(|l| ClassRule { id: l.id, name: l.name, prompt: l.prompt, description: l.description })
        .collect();
    let rule_set = crate::team::get_setting(p, RULE_SET_KEY)?.unwrap_or_default();
    Ok(Rules { instructions, classes, rule_set })
}

/// 绑定的平台规则编号（只留安全字符）
fn bound_code(r: &Rules) -> String {
    r.rule_set.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect()
}

/// 把项目的规则同步到巡瞳平台：没绑定就新建一套并绑定，绑定了就更新。返回平台规则编号。
/// 新建时默认停用：平台上存着，Wulabel 用规则编号调用照样能用；要让巡瞳自己的识别（抓拍、告警）
/// 也按这套规则，由用户另外打开 set_platform_enabled。
/// 更新时只发名字、类别、整体规则，不碰启停和适用场景，免得覆盖用户在巡瞳控制台里的设置。
pub fn sync_rules(p: &Project, rt: &Runtime) -> AppResult<String> {
    let (base, key) = config(rt)?;
    let r = rules(p)?;
    let content = serde_json::json!({
        "name": cut(&format!("Wulabel · {}", p.meta.name), 60),
        "classes": platform_classes(&r)?,
        "instructions": cut(&r.instructions, MAX_INSTRUCTIONS),
    });
    let code = bound_code(&r);
    if !code.is_empty() {
        match send_json("PUT", &format!("{base}/v1/rule-sets/{code}"), &key, &content) {
            Ok(_) => return Ok(code),
            // 平台上被删了（或不是这个账号的）：重新建一套
            Err(e) if e.to_string().contains("404") => {}
            Err(e) => return Err(e),
        }
    }
    let mut body = content;
    body["enabled"] = serde_json::json!(false);
    body["scenes"] = serde_json::json!(["*"]);
    // 超过每个账号 20 套的限制时，平台的提示原样带出来
    let v = post(&format!("{base}/v1/rule-sets"), &key, &body)?;
    let code = v["code"].as_str().ok_or_else(|| msg("巡瞳平台没有返回规则编号"))?.to_string();
    crate::team::set_setting(p, RULE_SET_KEY, &code)?;
    Ok(code)
}

/// 平台上这套规则的当前状态（启用、适用场景）。用户可能在巡瞳控制台里改过，以平台为准
pub fn rule_set_info(p: &Project, rt: &Runtime) -> AppResult<serde_json::Value> {
    let code = bound_code(&rules(p)?);
    if code.is_empty() {
        return Ok(serde_json::Value::Null);
    }
    let (base, key) = config(rt)?;
    let v = get(&format!("{base}/v1/rule-sets/{code}"), &key)?;
    Ok(serde_json::json!({ "code": code, "enabled": v["enabled"].as_bool().unwrap_or(false), "scenes": v["scenes"], "name": v["name"] }))
}

/// 「平台识别也用这套规则」：只改启停（适用场景用户可以在巡瞳控制台里调）
pub fn set_platform_enabled(p: &Project, rt: &Runtime, enabled: bool) -> AppResult<()> {
    let code = bound_code(&rules(p)?);
    if code.is_empty() {
        return Err(msg("先同步到巡瞳平台"));
    }
    let (base, key) = config(rt)?;
    send_json("PUT", &format!("{base}/v1/rule-sets/{code}"), &key, &serde_json::json!({ "enabled": enabled }))?;
    Ok(())
}
/// 解除和平台规则的绑定（平台上的规则不删）
pub fn unbind_rules(p: &Project) -> AppResult<()> {
    crate::team::set_setting(p, RULE_SET_KEY, "")
}

/// 保存规则：instructions 为 None 不改；classes 按类别名改识别说明（不会新增、改名类别）
pub fn save_rules(p: &Project, instructions: Option<&str>, classes: &[ClassRule]) -> AppResult<()> {
    if let Some(s) = instructions {
        let s = cut(s, MAX_INSTRUCTIONS);
        crate::team::set_setting(p, RULES_KEY, &s)?;
    }
    let labels = annotations::list_labels(p)?;
    for c in classes {
        if let Some(l) = labels.iter().find(|l| l.name == c.name) {
            annotations::set_label_description(p, l.id, &c.description)?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatMsg {
    /// user / assistant
    pub role: String,
    pub content: String,
    /// 这条消息附带的图（项目里的图片 id）
    #[serde(default)]
    pub image_id: Option<i64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RulesPatch {
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub classes: Vec<ClassRule>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatReply {
    pub reply: String,
    /// AI 建议的规则更新（用户点「采用」才保存）
    pub rules: Option<RulesPatch>,
    /// 按新规则在附带的图上试识别的结果
    pub preview: Option<PredResult>,
    pub preview_image_id: Option<i64>,
}

/// 和 AI 聊怎么识别：带上当前规则和对话，AI 回答并建议规则；最后一条带图时顺便试识别
pub fn chat_rules(p: &Project, rt: &Runtime, pr: &Predictor, messages: &[ChatMsg], log: &(dyn Fn(String) + Sync)) -> AppResult<ChatReply> {
    let (base, key) = config(rt)?;
    let r = rules(p)?;
    // 平台最多收 30 条，最后一条必须是用户的
    let messages = &messages[messages.len().saturating_sub(MAX_MESSAGES)..];
    if messages.last().map(|m| m.role.as_str()) != Some("user") {
        return Err(msg("最后一条要是你说的话"));
    }
    // 只给最后一张附图编码（平台也只把最近一张图发给 AI，省流量）
    let last_img = messages.iter().rposition(|m| m.image_id.is_some());
    let mut size = None;
    let mut msgs = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        let mut v = serde_json::json!({ "role": m.role, "content": cut(&m.content, MAX_MESSAGE) });
        if Some(i) == last_img {
            if let Some(id) = m.image_id {
                let (stored, _) = crate::images::locate(p, id)?;
                let (image, w, h) = encode_image(&p.resolve(&stored))?;
                v["image"] = serde_json::json!(image);
                size = Some((id, w, h));
            }
        }
        msgs.push(v);
    }
    let body = serde_json::json!({
        "classes": platform_classes(&r)?,
        "instructions": cut(&r.instructions, MAX_INSTRUCTIONS),
        "messages": msgs,
    });
    let v = post(&format!("{base}/v1/chat/rules"), &key, &body)?;
    let reply = v["reply"].as_str().or(v["content"].as_str()).unwrap_or("").to_string();
    let mut patch: Option<RulesPatch> = serde_json::from_value(v["rules"].clone()).ok();
    if let Some(pt) = &mut patch {
        // 只接受已有类别的说明，AI 不能新增、改名类别
        pt.classes.retain(|c| r.classes.iter().any(|x| x.name == c.name));
        if pt.instructions.as_deref().is_some_and(|s| s.trim().is_empty()) {
            pt.instructions = None;
        }
        if pt.instructions.is_none() && pt.classes.is_empty() {
            patch = None;
        }
    }
    let mut preview = None;
    let mut preview_image_id = None;
    if let Some((id, w, h)) = size {
        let mut found = parse_detections(&v["detections"], w, h);
        found.retain(|(n, _, _)| r.classes.iter().any(|c| &c.name == n));
        if !found.is_empty() {
            preview = Some(to_pred(p, rt, pr, id, found, log)?);
            preview_image_id = Some(id);
        }
    }
    Ok(ChatReply { reply, rules: patch, preview, preview_image_id })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_standard() {
        assert_eq!(b64(b""), "");
        assert_eq!(b64(b"f"), "Zg==");
        assert_eq!(b64(b"fo"), "Zm8=");
        assert_eq!(b64(b"foo"), "Zm9v");
        assert_eq!(b64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn normalized_box_to_pixels() {
        let b = to_px(&serde_json::json!([100, 200, 600, 700]), 2000.0, 1000.0).unwrap();
        assert_eq!(b, [200.0, 200.0, 1000.0, 500.0]);
        assert!(to_px(&serde_json::json!([1, 2, 3]), 100.0, 100.0).is_none());
    }
}
