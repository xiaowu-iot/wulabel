//! 巡瞳 AI 平台辅助标注：用用户自己账号的 API Key 调识别接口，结果作为预标注（来源记为 ai）。
//!
//! - 专用模型：POST /v1/detect（image、model、minConfidence）→ detections[{label, name, confidence, box}]
//! - 大模型：POST /v1/analyze（image、scene）→ hazards[{category, description, box}]
//! - 可用模型：GET /v1/models
//! 位置框 box 是 [x1, y1, x2, y2]，按图片宽高 0–1000 归一化。每次识别会消耗用户的套餐额度。

use std::path::Path;
use std::time::Duration;

use serde::Serialize;

use crate::annotations;
use crate::error::{msg, AppError, AppResult};
use crate::project::Project;
use crate::runtime::Runtime;
use crate::training::{self, PredResult, Predictor, PrelabelResult, RawPred};

/// 巡瞳 AI 默认地址（可在设置里改，比如私有化部署）
pub const DEFAULT_URL: &str = "https://xuntongai.com";
/// 大模型通用识别，在模型列表里用这个编码表示
const LLM: &str = "llm";

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AiModel {
    pub code: String,
    pub name: String,
    pub description: String,
}

fn config(rt: &Runtime) -> AppResult<(String, String)> {
    let s = rt.settings();
    let key = s.ai_key.unwrap_or_default().trim().to_string();
    if key.is_empty() {
        return Err(msg("还没有填巡瞳 AI 的 API Key。请到「设置 → 巡瞳 AI」填写（在巡瞳 AI 控制台的「API Key」页创建）"));
    }
    let url = s.ai_url.filter(|u| !u.trim().is_empty()).unwrap_or_else(|| DEFAULT_URL.to_string());
    Ok((url.trim().trim_end_matches('/').to_string(), key))
}

fn err_of(e: ureq::Error) -> AppError {
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

fn parse(resp: ureq::Response) -> AppResult<serde_json::Value> {
    let text = resp.into_string().map_err(|e| msg(format!("读取巡瞳 AI 返回失败：{e}")))?;
    serde_json::from_str(&text).map_err(|_| msg("巡瞳 AI 返回的不是 JSON，请检查设置里的地址"))
}

fn get(url: &str, key: &str) -> AppResult<serde_json::Value> {
    let r = ureq::get(url).set("Authorization", &format!("Bearer {key}")).timeout(Duration::from_secs(30)).call();
    parse(r.map_err(err_of)?)
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
    let mut out: Vec<AiModel> = list
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

/// 识别一张图，返回 (类别名, 分数, 像素框)
fn recognize(base: &str, key: &str, path: &Path, model: &str, conf: f64) -> AppResult<Vec<(String, f64, [f64; 4])>> {
    let (image, w, h) = encode_image(path)?;
    let mut out = Vec::new();
    if model == LLM {
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

/// 识别结果换成预测结果：类别名对应到项目类别（没有的自动新建）；分割项目用 SAM 把框变成轮廓
fn to_pred(
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
    let found = recognize(&base, &key, &p.resolve(&stored), model, conf)?;
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
    let items = training::prelabel_targets(p)?;
    let total = items.len();
    let mut result = PrelabelResult { images: 0, boxes: 0 };
    progress(0, total);
    for (n, (id, path)) in items.iter().enumerate() {
        let r = recognize(&base, &key, path, model, conf).and_then(|found| to_pred(p, rt, pr, *id, found, log));
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
