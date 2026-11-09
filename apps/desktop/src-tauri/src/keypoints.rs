//! 关键点模板：每个关键点项目一套（点名 + 骨架连线），存在 keypoint_templates 表。
//!
//! 关键点标注（type = "keypoints"）的数据格式：
//!     {"box": [x, y, w, h], "kpts": [x1, y1, v1, x2, y2, v2, ...]}
//! 原图像素坐标；v：0 没标（看不到 / 不在图里）、1 标了但被遮挡、2 标了且可见（和 COCO 一致）。

use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::error::{msg, AppResult};
use crate::project::Project;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Template {
    pub name: String,
    /// 关键点名称，顺序就是标注时点击的顺序
    pub points: Vec<String>,
    /// 骨架连线，点的序号从 0 开始
    pub skeleton: Vec<[usize; 2]>,
}

/// COCO 人体 17 点
pub fn coco_person() -> Template {
    let points = [
        "鼻子", "左眼", "右眼", "左耳", "右耳", "左肩", "右肩", "左肘", "右肘", "左腕", "右腕", "左髋", "右髋", "左膝",
        "右膝", "左踝", "右踝",
    ];
    // COCO 官方骨架（1 起始）换成 0 起始
    let skeleton = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
        [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7],
    ];
    Template {
        name: "COCO 人体 17 点".into(),
        points: points.iter().map(|s| s.to_string()).collect(),
        skeleton: skeleton.iter().map(|[a, b]| [a - 1, b - 1]).collect(),
    }
}

/// 项目的关键点模板；还没有就写入默认的 COCO 人体 17 点
pub fn get(p: &Project) -> AppResult<Template> {
    let conn = p.conn.lock();
    let row: Option<(String, String, String)> = conn
        .query_row("SELECT name, points_json, skeleton_json FROM keypoint_templates ORDER BY id LIMIT 1", [], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .optional()?;
    if let Some((name, pts, sk)) = row {
        return Ok(Template {
            name,
            points: serde_json::from_str(&pts).unwrap_or_default(),
            skeleton: serde_json::from_str(&sk).unwrap_or_default(),
        });
    }
    let t = coco_person();
    conn.execute(
        "INSERT INTO keypoint_templates (name, points_json, skeleton_json) VALUES (?1, ?2, ?3)",
        params![t.name, serde_json::to_string(&t.points)?, serde_json::to_string(&t.skeleton)?],
    )?;
    Ok(t)
}

/// 已有关键点标注时不允许改点数（改了旧标注就对不上了）；只改名字和连线可以
pub fn save(p: &Project, t: &Template) -> AppResult<()> {
    let points: Vec<String> = t.points.iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    if points.is_empty() {
        return Err(msg("至少要有 1 个关键点"));
    }
    if points.len() > 100 {
        return Err(msg("关键点最多 100 个"));
    }
    if t.skeleton.iter().any(|[a, b]| *a >= points.len() || *b >= points.len() || a == b) {
        return Err(msg("骨架连线里有不存在的点"));
    }
    let old = get(p)?;
    let conn = p.conn.lock();
    if old.points.len() != points.len() {
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM annotations WHERE type = 'keypoints'", [], |r| r.get(0))?;
        if n > 0 {
            return Err(msg(format!(
                "已经有 {n} 个关键点标注，不能把点数从 {} 改成 {}。只能改点的名字和连线",
                old.points.len(),
                points.len()
            )));
        }
    }
    let name = if t.name.trim().is_empty() { "自定义".to_string() } else { t.name.trim().to_string() };
    conn.execute("DELETE FROM keypoint_templates", [])?;
    conn.execute(
        "INSERT INTO keypoint_templates (name, points_json, skeleton_json) VALUES (?1, ?2, ?3)",
        params![name, serde_json::to_string(&points)?, serde_json::to_string(&t.skeleton)?],
    )?;
    Ok(())
}

/// 左右翻转时关键点的对应关系（YOLO-pose 的 flip_idx）：名字里「左/右」「left/right」互换
pub fn flip_index(points: &[String]) -> Vec<usize> {
    let swap = |s: &str| -> String {
        let lower = s.to_lowercase();
        let pairs = [("左", "右"), ("left", "right"), ("l_", "r_"), ("_l", "_r")];
        for (a, b) in pairs {
            if lower.contains(a) {
                return lower.replacen(a, b, 1);
            }
            if lower.contains(b) {
                return lower.replacen(b, a, 1);
            }
        }
        lower
    };
    points
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let other = swap(name);
            points.iter().position(|q| q.to_lowercase() == other).unwrap_or(i)
        })
        .collect()
}

/// 校验并解析关键点标注数据：返回 (外接框, 关键点)
pub fn parse(data: &serde_json::Value) -> Option<([f64; 4], Vec<f64>)> {
    let b: Vec<f64> = serde_json::from_value(data.get("box")?.clone()).ok()?;
    let k: Vec<f64> = serde_json::from_value(data.get("kpts")?.clone()).ok()?;
    if b.len() != 4 || k.len() % 3 != 0 || b.iter().chain(k.iter()).any(|v| !v.is_finite()) {
        return None;
    }
    Some(([b[0], b[1], b[2], b[3]], k))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coco_flip() {
        let t = coco_person();
        let f = flip_index(&t.points);
        assert_eq!(f, vec![0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]);
        assert_eq!(t.skeleton.len(), 19);
        assert!(t.skeleton.iter().all(|[a, b]| *a < 17 && *b < 17));
    }

    #[test]
    fn parse_data() {
        let d = serde_json::json!({"box": [1, 2, 3, 4], "kpts": [1, 1, 2, 0, 0, 0]});
        let (b, k) = parse(&d).unwrap();
        assert_eq!(b, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(k.len(), 6);
        assert!(parse(&serde_json::json!({"box": [1, 2, 3], "kpts": []})).is_none());
    }
}
