//! 标注辅助：SAM 交互分割、文本提示自动标注。
//! 模型跑在常驻推理进程里（worker 的 serve），第一次用到时自动安装 transformers 并下载模型。

use std::path::PathBuf;
use std::sync::Mutex;

use serde::Serialize;

use crate::annotations;
use crate::error::{msg, AppResult};
use crate::project::Project;
use crate::runtime::Runtime;
use crate::training::{self, PredResult, Predictor, PrelabelResult, RawPred};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SamResult {
    /// 多边形 [x1, y1, ...]，原图像素坐标；没分出目标时为空
    pub points: Vec<f64>,
    /// 外接框 [x, y, w, h]
    #[serde(rename = "box")]
    pub bbox: Vec<f64>,
    pub score: f64,
}

/// 已经确认装好辅助组件的 Python（换了环境要重新检查）
static DEPS_OK: Mutex<Option<PathBuf>> = Mutex::new(None);

/// 确保训练环境里有 transformers（SAM、文本检测模型都靠它），没有就自动安装
pub fn ensure_deps(rt: &Runtime, log: &(dyn Fn(String) + Sync)) -> AppResult<()> {
    let py = rt.python().ok_or_else(|| msg("还没有训练环境，请先到「设置」里安装"))?;
    if DEPS_OK.lock().unwrap().as_ref() == Some(&py) {
        return Ok(());
    }
    rt.ensure_package("transformers", "transformers", log)?;
    // OWLv2 的图片预处理用到 scipy
    rt.ensure_package("scipy", "scipy", log)?;
    *DEPS_OK.lock().unwrap() = Some(py);
    Ok(())
}

fn nums(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_f64()).collect()).unwrap_or_default()
}

fn forward_log<'a>(log: &'a (dyn Fn(String) + Sync)) -> impl Fn(&serde_json::Value) + 'a {
    move |v: &serde_json::Value| {
        if v["type"] == "log" {
            if let Some(m) = v["msg"].as_str() {
                log(m.to_string());
            }
        }
    }
}

/// SAM 交互分割：points 是 [x, y, 1 包含 / 0 排除]，bbox 是 [x0, y0, x1, y1]，都是原图像素坐标
pub fn sam_segment(
    p: &Project,
    rt: &Runtime,
    pr: &Predictor,
    image_id: i64,
    points: &[[f64; 3]],
    bbox: Option<[f64; 4]>,
    log: &(dyn Fn(String) + Sync),
) -> AppResult<SamResult> {
    if points.is_empty() && bbox.is_none() {
        return Err(msg("请先在图上点一下目标，或者框选"));
    }
    ensure_deps(rt, log)?;
    let (stored, _) = crate::images::locate(p, image_id)?;
    let req = serde_json::json!({ "cmd": "sam", "path": p.resolve(&stored), "points": points, "box": bbox });
    let mut out: Option<serde_json::Value> = None;
    let fwd = forward_log(log);
    pr.call(rt, &req, &mut |v| {
        if v["type"] == "sam" {
            out = Some(v.clone());
        } else {
            fwd(v);
        }
    })?;
    let v = out.ok_or_else(|| msg("SAM 没有返回结果"))?;
    Ok(SamResult { points: nums(&v["points"]), bbox: nums(&v["box"]), score: v["score"].as_f64().unwrap_or(0.0) })
}

/// 常用中文类别名 → 英文（开放词汇检测模型只懂英文）。用户在类别里填了英文提示词时优先用用户的
fn zh_en(name: &str) -> Option<&'static str> {
    const DICT: [(&str, &str); 66] = [
        ("人", "person"), ("行人", "person"), ("人员", "person"), ("工人", "worker"), ("安全帽", "helmet"),
        ("头盔", "helmet"), ("未戴安全帽", "head without helmet"), ("反光衣", "reflective vest"), ("口罩", "face mask"),
        ("人脸", "face"), ("脸", "face"), ("手", "hand"), ("车", "car"), ("汽车", "car"), ("轿车", "car"),
        ("车辆", "vehicle"), ("卡车", "truck"), ("货车", "truck"), ("公交车", "bus"), ("大巴", "bus"),
        ("摩托车", "motorcycle"), ("电动车", "electric scooter"), ("自行车", "bicycle"), ("单车", "bicycle"),
        ("车牌", "license plate"), ("船", "boat"), ("飞机", "airplane"), ("火车", "train"), ("狗", "dog"),
        ("猫", "cat"), ("鸟", "bird"), ("马", "horse"), ("牛", "cow"), ("羊", "sheep"), ("火", "fire"),
        ("明火", "fire"), ("火焰", "flame"), ("烟", "smoke"), ("烟雾", "smoke"), ("香烟", "cigarette"),
        ("烟头", "cigarette butt"), ("垃圾", "garbage"), ("垃圾桶", "trash can"), ("纸箱", "cardboard box"),
        ("箱子", "box"), ("灭火器", "fire extinguisher"), ("消防栓", "fire hydrant"), ("手机", "cell phone"),
        ("瓶子", "bottle"), ("杯子", "cup"), ("椅子", "chair"), ("桌子", "table"), ("背包", "backpack"),
        ("包", "bag"), ("雨伞", "umbrella"), ("电脑", "computer"), ("笔记本电脑", "laptop"), ("电视", "tv"),
        ("红绿灯", "traffic light"), ("交通灯", "traffic light"), ("树", "tree"), ("门", "door"), ("窗户", "window"),
        ("梯子", "ladder"), ("裂缝", "crack"), ("积水", "puddle"),
    ];
    DICT.iter().find(|(zh, _)| *zh == name).map(|(_, en)| *en)
}

/// 各类别的英文提示词：[(类别序号, 英文)]，以及类别序号对应的类别 id
fn queries(p: &Project) -> AppResult<(Vec<(usize, String)>, Vec<i64>)> {
    let labels = annotations::list_labels(p)?;
    if labels.is_empty() {
        return Err(msg("先添加要标注的类别"));
    }
    let mut qs = Vec::new();
    let mut ids = Vec::new();
    let mut missing = Vec::new();
    for (i, l) in labels.iter().enumerate() {
        ids.push(l.id);
        let words: Vec<String> = if !l.prompt.trim().is_empty() {
            l.prompt
                .split(|c: char| matches!(c, ',' | '，' | ';' | '；'))
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        } else if l.name.is_ascii() {
            vec![l.name.replace(|c: char| c == '_' || c == '-', " ")]
        } else if let Some(en) = zh_en(l.name.trim()) {
            vec![en.to_string()]
        } else {
            missing.push(l.name.clone());
            Vec::new()
        };
        qs.extend(words.into_iter().map(|w| (i, w)));
    }
    if qs.is_empty() {
        return Err(msg(format!(
            "这些类别还没有英文提示词：{}。文本提示模型只懂英文，请在类别的编辑里填上，如 安全帽 → helmet",
            missing.join("、")
        )));
    }
    Ok((qs, ids))
}

fn text_request(qs: &[(usize, String)], conf: f64, seg: bool, items: &[(i64, PathBuf)]) -> serde_json::Value {
    serde_json::json!({
        "cmd": "text",
        "queries": qs.iter().map(|(i, w)| serde_json::json!([i, w])).collect::<Vec<_>>(),
        "conf": conf,
        "seg": seg,
        "images": items.iter().map(|(id, path)| serde_json::json!({ "id": id, "path": path })).collect::<Vec<_>>(),
    })
}

/// 文本提示：按类别名（英文提示词）在一张图里找目标；分割项目再用 SAM 出轮廓
pub fn text_detect(
    p: &Project,
    rt: &Runtime,
    pr: &Predictor,
    image_id: i64,
    conf: f64,
    log: &(dyn Fn(String) + Sync),
) -> AppResult<PredResult> {
    ensure_deps(rt, log)?;
    let (qs, ids) = queries(p)?;
    let (stored, _) = crate::images::locate(p, image_id)?;
    let req = text_request(&qs, conf, p.meta.task == "segment", &[(image_id, p.resolve(&stored))]);
    let mut raw = RawPred::default();
    let fwd = forward_log(log);
    pr.call(rt, &req, &mut |v| {
        if v["type"] == "pred" {
            raw = training::parse_pred(v);
        } else {
            fwd(v);
        }
    })?;
    Ok(training::to_result(raw, &ids))
}

/// 文本提示批量预标注：所有「未标注」且没有标注的图，结果来源记为 text
pub fn text_prelabel_all(
    p: &Project,
    rt: &Runtime,
    pr: &Predictor,
    conf: f64,
    progress: &dyn Fn(usize, usize),
    log: &(dyn Fn(String) + Sync),
) -> AppResult<PrelabelResult> {
    ensure_deps(rt, log)?;
    let (qs, ids) = queries(p)?;
    let seg = p.meta.task == "segment";
    let items = training::prelabel_targets(p)?;
    let total = items.len();
    let mut result = PrelabelResult { images: 0, boxes: 0 };
    progress(0, total);
    let mut done = 0;
    for chunk in items.chunks(8) {
        let mut got: Vec<(i64, RawPred)> = Vec::new();
        let fwd = forward_log(log);
        pr.call(rt, &text_request(&qs, conf, seg, chunk), &mut |v| {
            if v["type"] == "pred" {
                got.push((v["id"].as_i64().unwrap_or(-1), training::parse_pred(v)));
            } else {
                fwd(v);
            }
        })?;
        for (id, raw) in got {
            let anns = training::task_annotations(p, &training::to_result(raw, &ids), conf, "text")?;
            if !anns.is_empty() {
                result.boxes += anns.len();
                result.images += 1;
                annotations::save_predictions(p, id, &anns)?;
            }
        }
        done += chunk.len();
        progress(done, total);
    }
    Ok(result)
}

/// 把框（巡瞳 AI 等给的）用 SAM 转成分割轮廓。boxes：(类别序号, 分数, [x, y, w, h])；返回 RawPred.polys
pub fn boxes_to_polys(
    p: &Project,
    rt: &Runtime,
    pr: &Predictor,
    image_id: i64,
    boxes: &[(usize, f64, [f64; 4])],
    log: &(dyn Fn(String) + Sync),
) -> AppResult<Vec<(usize, f64, Vec<f64>)>> {
    if boxes.is_empty() {
        return Ok(Vec::new());
    }
    ensure_deps(rt, log)?;
    let (stored, _) = crate::images::locate(p, image_id)?;
    let req = serde_json::json!({
        "cmd": "sam_boxes",
        "images": [{
            "id": image_id,
            "path": p.resolve(&stored),
            "boxes": boxes.iter().map(|(c, s, b)| serde_json::json!([b[0], b[1], b[0] + b[2], b[1] + b[3], c, s])).collect::<Vec<_>>(),
        }],
    });
    let mut raw = RawPred::default();
    let fwd = forward_log(log);
    pr.call(rt, &req, &mut |v| {
        if v["type"] == "pred" {
            raw = training::parse_pred(v);
        } else {
            fwd(v);
        }
    })?;
    Ok(raw.polys)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictionary() {
        assert_eq!(zh_en("安全帽"), Some("helmet"));
        assert_eq!(zh_en("不存在的类别"), None);
    }
}
