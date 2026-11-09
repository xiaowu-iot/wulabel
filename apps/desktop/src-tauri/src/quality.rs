//! 图片质量筛查：找出模糊、过暗、过曝、对比度太低的图。纯 Rust 计算，不需要训练环境。
//! 在缩到长边 512 的灰度图上算：拉普拉斯方差（越小越糊）、平均亮度、过曝像素比例、亮度标准差（对比度）。

use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use serde::Serialize;

use crate::error::AppResult;
use crate::project::Project;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QualityItem {
    pub image_id: i64,
    /// blur 模糊 / dark 过暗 / bright 过曝 / flat 对比度低
    pub issues: Vec<String>,
    /// 清晰度（拉普拉斯方差），越大越清楚
    pub sharpness: f64,
    /// 平均亮度 0~255
    pub brightness: f64,
    /// 过曝像素比例
    pub overexposed: f64,
    /// 亮度标准差
    pub contrast: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QualityReport {
    pub checked: usize,
    /// 有问题的图，问题多、程度重的在前
    pub items: Vec<QualityItem>,
    pub failed: usize,
}

/// 判定阈值（在缩到 512 的灰度图上）
const BLUR: f64 = 60.0;
const DARK: f64 = 40.0;
const BRIGHT: f64 = 215.0;
const OVER: f64 = 0.25;
const FLAT: f64 = 18.0;

fn measure(path: &std::path::Path) -> Option<(f64, f64, f64, f64)> {
    let img = crate::images::decode(path).ok()?;
    let g = img.thumbnail(512, 512).to_luma8();
    let (w, h) = (g.width() as usize, g.height() as usize);
    if w < 3 || h < 3 {
        return None;
    }
    let px = g.as_raw();
    let n = (w * h) as f64;
    let mean = px.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = px.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let over = px.iter().filter(|&&v| v >= 250).count() as f64 / n;
    // 拉普拉斯（4 邻域）方差
    let mut lap = Vec::with_capacity((w - 2) * (h - 2));
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let c = px[y * w + x] as f64;
            let v = 4.0 * c - px[(y - 1) * w + x] as f64 - px[(y + 1) * w + x] as f64 - px[y * w + x - 1] as f64 - px[y * w + x + 1] as f64;
            lap.push(v);
        }
    }
    let lm = lap.iter().sum::<f64>() / lap.len() as f64;
    let lv = lap.iter().map(|v| (v - lm).powi(2)).sum::<f64>() / lap.len() as f64;
    Some((lv, mean, over, var.sqrt()))
}

/// 扫描全部图片（only_todo：只看未标注的）
pub fn scan(p: &Project, only_todo: bool, progress: &(dyn Fn(usize, usize) + Sync)) -> AppResult<QualityReport> {
    let rows: Vec<(i64, String)> = {
        let conn = p.conn.lock();
        let sql = if only_todo { "SELECT id, path FROM images WHERE status = 'todo' ORDER BY id" } else { "SELECT id, path FROM images ORDER BY id" };
        let mut stmt = conn.prepare(sql)?;
        let v = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_, _>>()?;
        v
    };
    let total = rows.len();
    let done = AtomicUsize::new(0);
    progress(0, total);
    let results: Vec<Option<QualityItem>> = rows
        .par_iter()
        .map(|(id, path)| {
            let r = measure(&p.resolve(path)).map(|(sharp, bright, over, contrast)| {
                let mut issues = Vec::new();
                if sharp < BLUR {
                    issues.push("blur".to_string());
                }
                if bright < DARK {
                    issues.push("dark".to_string());
                }
                if bright > BRIGHT || over > OVER {
                    issues.push("bright".to_string());
                }
                if contrast < FLAT {
                    issues.push("flat".to_string());
                }
                QualityItem { image_id: *id, issues, sharpness: sharp, brightness: bright, overexposed: over, contrast }
            });
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 20 == 0 || n == total {
                progress(n, total);
            }
            r
        })
        .collect();
    let failed = results.iter().filter(|r| r.is_none()).count();
    let mut items: Vec<QualityItem> = results.into_iter().flatten().filter(|q| !q.issues.is_empty()).collect();
    // 问题多的、模糊程度重的排前面
    items.sort_by(|a, b| b.issues.len().cmp(&a.issues.len()).then(a.sharpness.partial_cmp(&b.sharpness).unwrap_or(std::cmp::Ordering::Equal)));
    Ok(QualityReport { checked: total, items, failed })
}
