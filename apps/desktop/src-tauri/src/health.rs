//! 数据体检：规则检查（极小框、重复框、近似重复图片、类别不均衡）和模型检查（漏标、错标、框不准、多余框）

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::db;
use crate::error::AppResult;
use crate::images;
use crate::project::Project;

/// 小于这个边长（像素）的框算极小框
pub const TINY: f64 = 8.0;
/// 同类别两个框 IoU 超过这个值算重复框
const DUP_BOX_IOU: f64 = 0.85;
/// 感知哈希汉明距离不超过这个值算近似重复图片
const DUP_IMAGE_DIST: u32 = 4;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Issue {
    /// missing / wrong_class / loose_box / extra_box / tiny_box / dup_box / dup_image / no_label
    pub kind: String,
    pub image_id: i64,
    #[serde(default)]
    pub label_id: Option<i64>,
    #[serde(default)]
    pub pred_label_id: Option<i64>,
    /// 问题框 [x, y, w, h]
    #[serde(default)]
    pub bbox: Option<[f64; 4]>,
    #[serde(default)]
    pub score: Option<f64>,
    #[serde(default)]
    pub iou: Option<f64>,
    pub detail: String,
    /// 忽略用的标识，读取时填充
    #[serde(default)]
    pub key: String,
}

impl Issue {
    /// 用于「忽略」的稳定标识
    pub fn key(&self) -> String {
        let b = self.bbox.map(|b| format!("{:.0},{:.0},{:.0},{:.0}", b[0], b[1], b[2], b[3])).unwrap_or_default();
        format!("{}:{}:{}", self.kind, self.image_id, b)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassStat {
    pub label_id: i64,
    pub name: String,
    pub color: String,
    pub boxes: i64,
    pub images: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HealthReport {
    pub created_at: i64,
    pub images_checked: usize,
    pub boxes_checked: usize,
    /// 用于模型检查的训练记录
    pub run_id: Option<i64>,
    pub run_name: Option<String>,
    pub issues: Vec<Issue>,
    pub classes: Vec<ClassStat>,
    /// 框短边长度分布：(区间名, 数量)
    pub sizes: Vec<(String, i64)>,
    pub dup_groups: Vec<Vec<i64>>,
    pub suggestions: Vec<String>,
    /// 有问题的图片的宽高，前端在缩略图上画框用
    #[serde(default)]
    pub dims: HashMap<i64, (u32, u32)>,
}

pub struct Gt {
    pub image_id: i64,
    pub label_id: Option<i64>,
    pub bbox: [f64; 4],
}

pub fn iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let ix = (a[0] + a[2]).min(b[0] + b[2]) - a[0].max(b[0]);
    let iy = (a[1] + a[3]).min(b[1] + b[3]) - a[1].max(b[1]);
    if ix <= 0.0 || iy <= 0.0 {
        return 0.0;
    }
    let inter = ix * iy;
    inter / (a[2] * a[3] + b[2] * b[3] - inter)
}

/// 已标注、已审核图片上的所有目标框：矩形框；分割多边形、关键点目标按外接框算
fn load_gts(p: &Project) -> AppResult<(Vec<i64>, Vec<Gt>)> {
    let conn = p.conn.lock();
    let ids: Vec<i64> = conn
        .prepare("SELECT id FROM images WHERE status IN ('labeled', 'reviewed') ORDER BY id")?
        .query_map([], |r| r.get(0))?
        .collect::<Result<_, _>>()?;
    let mut stmt = conn.prepare(
        "SELECT a.image_id, a.label_id, a.type, a.data_json, i.width, i.height
         FROM annotations a JOIN images i ON i.id = a.image_id
         WHERE a.type IN ('bbox', 'polygon', 'keypoints', 'tag') AND i.status IN ('labeled', 'reviewed')
         ORDER BY a.image_id, a.id",
    )?;
    let gts = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, Option<i64>>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, f64>(4)?,
                r.get::<_, f64>(5)?,
            ))
        })?
        .filter_map(|r| r.ok())
        .filter_map(|(image_id, label_id, kind, d, w, h)| {
            let value: serde_json::Value = serde_json::from_str(&d).ok()?;
            let bbox = match kind.as_str() {
                // 分类标签按整图一个框算，这样模型检查能直接查出「类别可能标错」
                "tag" => [0.0, 0.0, w, h],
                "keypoints" => crate::keypoints::parse(&value)?.0,
                "polygon" => {
                    let v: Vec<f64> = serde_json::from_value(value).ok()?;
                    if v.len() < 6 {
                        return None;
                    }
                    let xs = v.iter().step_by(2);
                    let ys = v.iter().skip(1).step_by(2);
                    let (x0, x1) = xs.fold((f64::MAX, f64::MIN), |(a, b), x| (a.min(*x), b.max(*x)));
                    let (y0, y1) = ys.fold((f64::MAX, f64::MIN), |(a, b), y| (a.min(*y), b.max(*y)));
                    [x0, y0, x1 - x0, y1 - y0]
                }
                _ => {
                    let v: Vec<f64> = serde_json::from_value(value).ok()?;
                    if v.len() != 4 {
                        return None;
                    }
                    [v[0], v[1], v[2], v[3]]
                }
            };
            Some(Gt { image_id, label_id, bbox })
        })
        .collect();
    Ok((ids, gts))
}

fn class_stats(p: &Project) -> AppResult<Vec<ClassStat>> {
    let conn = p.conn.lock();
    let mut stmt = conn.prepare(
        "SELECT l.id, l.name, l.color,
                (SELECT COUNT(*) FROM annotations a JOIN images i ON i.id = a.image_id
                 WHERE a.label_id = l.id AND i.status IN ('labeled', 'reviewed')),
                (SELECT COUNT(DISTINCT a.image_id) FROM annotations a JOIN images i ON i.id = a.image_id
                 WHERE a.label_id = l.id AND i.status IN ('labeled', 'reviewed'))
         FROM labels l ORDER BY l.sort, l.id",
    )?;
    let rows = stmt
        .query_map([], |r| {
            Ok(ClassStat { label_id: r.get(0)?, name: r.get(1)?, color: r.get(2)?, boxes: r.get(3)?, images: r.get(4)? })
        })?
        .collect::<Result<_, _>>()?;
    Ok(rows)
}

const SIZE_BUCKETS: [(f64, &str); 7] =
    [(8.0, "<8"), (16.0, "8–16"), (32.0, "16–32"), (64.0, "32–64"), (128.0, "64–128"), (256.0, "128–256"), (f64::MAX, "≥256")];

/// 规则检查：不需要模型
pub fn rule_issues(gts: &[Gt]) -> (Vec<Issue>, Vec<(String, i64)>) {
    let mut issues = Vec::new();
    let mut sizes = vec![0i64; SIZE_BUCKETS.len()];
    let mut by_image: HashMap<i64, Vec<&Gt>> = HashMap::new();
    for g in gts {
        by_image.entry(g.image_id).or_default().push(g);
        let short = g.bbox[2].min(g.bbox[3]);
        let i = SIZE_BUCKETS.iter().position(|(lim, _)| short < *lim).unwrap_or(SIZE_BUCKETS.len() - 1);
        sizes[i] += 1;
        if short < TINY {
            issues.push(Issue {
                kind: "tiny_box".into(),
                image_id: g.image_id,
                label_id: g.label_id,
                pred_label_id: None,
                bbox: Some(g.bbox),
                score: None,
                iou: None,
                detail: format!("{:.0}×{:.0} 像素，模型很难学到", g.bbox[2], g.bbox[3]),
                key: String::new(),
            });
        }
        if g.label_id.is_none() {
            issues.push(Issue {
                kind: "no_label".into(),
                image_id: g.image_id,
                label_id: None,
                pred_label_id: None,
                bbox: Some(g.bbox),
                score: None,
                iou: None,
                detail: "这个框没有类别，不会用于训练".into(),
                key: String::new(),
            });
        }
    }
    let mut ids: Vec<_> = by_image.keys().copied().collect();
    ids.sort();
    for id in ids {
        let list = &by_image[&id];
        for i in 0..list.len() {
            for j in i + 1..list.len() {
                if list[i].label_id == list[j].label_id {
                    let v = iou(&list[i].bbox, &list[j].bbox);
                    if v > DUP_BOX_IOU {
                        issues.push(Issue {
                            kind: "dup_box".into(),
                            image_id: id,
                            label_id: list[j].label_id,
                            pred_label_id: None,
                            bbox: Some(list[j].bbox),
                            score: None,
                            iou: Some(v),
                            detail: format!("和另一个同类框重叠 {:.0}%，可能重复标了", v * 100.0),
                            key: String::new(),
                        });
                    }
                }
            }
        }
    }
    let sizes = SIZE_BUCKETS.iter().zip(sizes).map(|((_, n), c)| (n.to_string(), c)).collect();
    (issues, sizes)
}

/// 差值哈希（dHash）：缩成 9×8 灰度图，比较左右相邻像素
fn dhash(path: &std::path::Path) -> Option<u64> {
    let img = image::open(path).ok()?.to_luma8();
    let small = image::imageops::resize(&img, 9, 8, image::imageops::FilterType::Triangle);
    let mut h = 0u64;
    for y in 0..8 {
        for x in 0..8 {
            h = (h << 1) | (small.get_pixel(x, y)[0] > small.get_pixel(x + 1, y)[0]) as u64;
        }
    }
    Some(h)
}

/// 近似重复图片分组（只看未跳过的图）
pub fn duplicate_groups(p: &Project, progress: &(dyn Fn(usize, usize) + Sync)) -> AppResult<Vec<Vec<i64>>> {
    let rows: Vec<(i64, String, String)> = {
        let conn = p.conn.lock();
        let mut stmt = conn.prepare("SELECT id, path, hash FROM images WHERE status != 'skipped' ORDER BY id")?;
        let v = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.collect::<Result<_, _>>()?;
        v
    };
    let total = rows.len();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let hashes: Vec<(i64, u64)> = rows
        .par_iter()
        .filter_map(|(id, path, hash)| {
            let r = images::ensure_thumb(p, path, hash).ok().and_then(|t| dhash(&t)).map(|h| (*id, h));
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n % 50 == 0 || n == total {
                progress(n, total);
            }
            r
        })
        .collect();
    // 并查集
    let n = hashes.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for i in 0..n {
        for j in i + 1..n {
            if (hashes[i].1 ^ hashes[j].1).count_ones() <= DUP_IMAGE_DIST {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                if a != b {
                    parent[b] = a;
                }
            }
        }
    }
    let mut groups: HashMap<usize, Vec<i64>> = HashMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(hashes[i].0);
    }
    let mut out: Vec<Vec<i64>> = groups.into_values().filter(|g| g.len() > 1).collect();
    for g in &mut out {
        g.sort();
    }
    out.sort_by_key(|g| (std::cmp::Reverse(g.len()), g[0]));
    Ok(out)
}

/// 模型预测（像素坐标 xywh、项目类别 id、分数）
pub struct Pred {
    pub label_id: i64,
    pub bbox: [f64; 4],
    pub score: f64,
}

/// 模型检查：用预测结果和标注对比
pub fn model_issues(gts: &[Gt], preds: &HashMap<i64, Vec<Pred>>, name_of: &dyn Fn(Option<i64>) -> String) -> Vec<Issue> {
    const SURE: f64 = 0.5; // 模型认为「很确定」的分数
    let mut by_image: HashMap<i64, Vec<&Gt>> = HashMap::new();
    for g in gts {
        by_image.entry(g.image_id).or_default().push(g);
    }
    let mut ids: Vec<i64> = preds.keys().copied().collect();
    ids.sort();
    let mut issues = Vec::new();
    for id in ids {
        let ps = &preds[&id];
        let gs = by_image.get(&id).cloned().unwrap_or_default();
        let mut wrong_seen = HashSet::new();
        for pr in ps.iter().filter(|p| p.score >= SURE) {
            let best = gs
                .iter()
                .enumerate()
                .map(|(i, g)| (i, iou(&pr.bbox, &g.bbox)))
                .max_by(|a, b| a.1.total_cmp(&b.1));
            match best {
                None => issues.push(missing(id, pr, name_of)),
                Some((_, v)) if v < 0.3 => issues.push(missing(id, pr, name_of)),
                Some((gi, v)) => {
                    let g = gs[gi];
                    if g.label_id != Some(pr.label_id) {
                        if v >= 0.5 && wrong_seen.insert(gi) {
                            issues.push(Issue {
                                kind: "wrong_class".into(),
                                image_id: id,
                                label_id: g.label_id,
                                pred_label_id: Some(pr.label_id),
                                bbox: Some(g.bbox),
                                score: Some(pr.score),
                                iou: Some(v),
                                detail: format!("标的是「{}」，模型认为是「{}」（{:.2}）", name_of(g.label_id), name_of(Some(pr.label_id)), pr.score),
                                key: String::new(),
                            });
                        }
                    } else if v < 0.6 && pr.score >= 0.7 {
                        issues.push(Issue {
                            kind: "loose_box".into(),
                            image_id: id,
                            label_id: g.label_id,
                            pred_label_id: Some(pr.label_id),
                            bbox: Some(g.bbox),
                            score: Some(pr.score),
                            iou: Some(v),
                            detail: format!("和模型预测的框只重合 {:.0}%，框可能画偏或画得太大太小", v * 100.0),
                            key: String::new(),
                        });
                    }
                }
            }
        }
        // 标了但模型在那里什么都没看到
        for g in &gs {
            let seen = ps.iter().filter(|p| p.score >= 0.05).map(|p| iou(&p.bbox, &g.bbox)).fold(0.0, f64::max);
            if seen < 0.1 {
                issues.push(Issue {
                    kind: "extra_box".into(),
                    image_id: id,
                    label_id: g.label_id,
                    pred_label_id: None,
                    bbox: Some(g.bbox),
                    score: None,
                    iou: None,
                    detail: format!("标了「{}」，但模型在这里什么都没看到，可能标错了", name_of(g.label_id)),
                    key: String::new(),
                });
            }
        }
    }
    issues
}

fn missing(image_id: i64, p: &Pred, name_of: &dyn Fn(Option<i64>) -> String) -> Issue {
    Issue {
        kind: "missing".into(),
        image_id,
        label_id: None,
        pred_label_id: Some(p.label_id),
        bbox: Some(p.bbox),
        score: Some(p.score),
        iou: None,
        detail: format!("模型认为这里有「{}」（{:.2}），但没有标注", name_of(Some(p.label_id)), p.score),
        key: String::new(),
    }
}

fn suggestions(r: &HealthReport) -> Vec<String> {
    let mut out = Vec::new();
    let count = |k: &str| r.issues.iter().filter(|i| i.kind == k).count();
    let max = r.classes.iter().map(|c| c.boxes).max().unwrap_or(0);
    for c in &r.classes {
        if max >= 20 && c.boxes * 5 < max {
            out.push(format!(
                "「{}」只有 {} 个框，最多的类别有 {} 个。样本太少模型容易漏检，建议再标 {} 个以上。",
                c.name,
                c.boxes,
                max,
                (max / 3 - c.boxes).max(20)
            ));
        } else if c.boxes > 0 && c.boxes < 30 {
            out.push(format!("「{}」只有 {} 个框，建议每个类别至少 100 个框再训练。", c.name, c.boxes));
        }
    }
    let tiny = count("tiny_box");
    if tiny > 0 {
        out.push(format!("{tiny} 个框短边小于 {TINY:.0} 像素。远处的小目标建议在训练时用更大的输入尺寸（960 或 1280），实在看不清的可以删掉。"));
    }
    let dup: usize = r.dup_groups.iter().map(|g| g.len() - 1).sum();
    if dup > 0 {
        out.push(format!("有 {} 组近似重复的图片（多出 {dup} 张）。重复图片分到验证集会让评估结果虚高，建议每组只留一张。", r.dup_groups.len()));
    }
    let (m, w) = (count("missing"), count("wrong_class"));
    if m + w > 0 {
        out.push(format!("模型发现 {m} 处疑似漏标、{w} 处疑似标错类别。修正后重新训练，效果通常会明显提升。"));
    }
    if r.run_id.is_none() && r.images_checked > 0 {
        out.push("训练一个模型后再体检，可以自动找出漏标、错标和框画偏的地方。".into());
    }
    out
}

fn report_path(p: &Project) -> PathBuf {
    p.root.join(".cache").join("health.json")
}

fn ignore_path(p: &Project) -> PathBuf {
    p.root.join("health_ignore.json")
}

fn load_ignored(p: &Project) -> HashSet<String> {
    fs::read_to_string(ignore_path(p)).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
}

pub fn ignore(p: &Project, keys: &[String]) -> AppResult<()> {
    let mut set = load_ignored(p);
    set.extend(keys.iter().cloned());
    let mut v: Vec<_> = set.into_iter().collect();
    v.sort();
    fs::write(ignore_path(p), serde_json::to_string_pretty(&v)?)?;
    Ok(())
}

/// 读上次体检结果，去掉已忽略的和已删除图片的问题
pub fn load(p: &Project) -> AppResult<Option<HealthReport>> {
    let Ok(text) = fs::read_to_string(report_path(p)) else { return Ok(None) };
    let Ok(mut r) = serde_json::from_str::<HealthReport>(&text) else { return Ok(None) };
    let ignored = load_ignored(p);
    let alive: HashSet<i64> = {
        let conn = p.conn.lock();
        let mut stmt = conn.prepare("SELECT id FROM images")?;
        let v = stmt.query_map([], |row| row.get(0))?.collect::<Result<_, _>>()?;
        v
    };
    r.issues.retain(|i| alive.contains(&i.image_id) && !ignored.contains(&i.key()));
    for i in &mut r.issues {
        i.key = i.key();
    }
    r.dup_groups = r
        .dup_groups
        .into_iter()
        .map(|g| g.into_iter().filter(|id| alive.contains(id)).collect::<Vec<_>>())
        .filter(|g| g.len() > 1)
        .collect();
    Ok(Some(r))
}

/// 运行体检。predict 不为空时做模型检查（传入每张图的预测结果）。
pub fn run(
    p: &Project,
    run: Option<(i64, String)>,
    predict: Option<&dyn Fn(&[i64]) -> AppResult<HashMap<i64, Vec<Pred>>>>,
    progress: &(dyn Fn(usize, usize) + Sync),
) -> AppResult<HealthReport> {
    let (ids, gts) = load_gts(p)?;
    let (mut issues, sizes) = rule_issues(&gts);
    let classes = class_stats(p)?;
    let names: HashMap<i64, String> = classes.iter().map(|c| (c.label_id, c.name.clone())).collect();
    let name_of = |id: Option<i64>| id.and_then(|i| names.get(&i).cloned()).unwrap_or_else(|| "未分类".into());
    let dup_groups = duplicate_groups(p, progress)?;
    if let Some(f) = predict {
        let preds = f(&ids)?;
        issues.extend(model_issues(&gts, &preds, &name_of));
    }
    let mut r = HealthReport {
        created_at: db::now(),
        images_checked: ids.len(),
        boxes_checked: gts.len(),
        run_id: run.as_ref().map(|r| r.0),
        run_name: run.map(|r| r.1),
        issues,
        classes,
        sizes,
        dup_groups,
        suggestions: Vec::new(),
        dims: HashMap::new(),
    };
    r.suggestions = suggestions(&r);
    {
        let conn = p.conn.lock();
        let mut stmt = conn.prepare("SELECT width, height FROM images WHERE id = ?1")?;
        for id in r.issues.iter().map(|i| i.image_id).collect::<HashSet<_>>() {
            if let Ok(d) = stmt.query_row([id], |row| Ok((row.get(0)?, row.get(1)?))) {
                r.dims.insert(id, d);
            }
        }
    }
    fs::create_dir_all(p.root.join(".cache"))?;
    fs::write(report_path(p), serde_json::to_string(&r)?)?;
    Ok(load(p)?.unwrap_or(r))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotations;
    use crate::dataset::tests::{bbox, temp};
    use image::{Rgb, RgbImage};

    fn gt(image_id: i64, label: i64, b: [f64; 4]) -> Gt {
        Gt { image_id, label_id: Some(label), bbox: b }
    }

    #[test]
    fn rules() {
        let gts = vec![
            gt(1, 1, [0.0, 0.0, 5.0, 40.0]),      // 极小框
            gt(1, 1, [10.0, 10.0, 100.0, 100.0]), // 和下一个重复
            gt(1, 1, [12.0, 11.0, 100.0, 100.0]),
            gt(1, 2, [10.0, 10.0, 100.0, 100.0]), // 不同类别不算重复
            Gt { image_id: 2, label_id: None, bbox: [0.0, 0.0, 50.0, 50.0] },
        ];
        let (issues, sizes) = rule_issues(&gts);
        let kinds: Vec<&str> = issues.iter().map(|i| i.kind.as_str()).collect();
        assert_eq!(kinds.iter().filter(|k| **k == "tiny_box").count(), 1);
        assert_eq!(kinds.iter().filter(|k| **k == "dup_box").count(), 1);
        assert_eq!(kinds.iter().filter(|k| **k == "no_label").count(), 1);
        assert_eq!(sizes[0], ("<8".to_string(), 1));
        assert_eq!(sizes.iter().map(|s| s.1).sum::<i64>(), 5);
    }

    #[test]
    fn model_compare() {
        let gts = vec![
            gt(1, 1, [0.0, 0.0, 50.0, 50.0]),     // 模型同意
            gt(1, 1, [100.0, 0.0, 50.0, 50.0]),   // 模型说是类别 2
            gt(1, 1, [0.0, 100.0, 50.0, 50.0]),   // 模型框偏了
            gt(1, 2, [300.0, 300.0, 40.0, 40.0]), // 模型什么都没看到
        ];
        let p = |l: i64, b: [f64; 4], s: f64| Pred { label_id: l, bbox: b, score: s };
        let preds = HashMap::from([(
            1,
            vec![
                p(1, [1.0, 1.0, 50.0, 50.0], 0.95),
                p(2, [100.0, 0.0, 50.0, 50.0], 0.9),
                p(1, [10.0, 108.0, 50.0, 50.0], 0.9), // 重合约 50%
                p(1, [200.0, 200.0, 30.0, 30.0], 0.8), // 漏标
                p(1, [400.0, 0.0, 30.0, 30.0], 0.3),   // 分数低，不算
            ],
        )]);
        let issues = model_issues(&gts, &preds, &|id| format!("c{}", id.unwrap_or(0)));
        let kinds: Vec<&str> = issues.iter().map(|i| i.kind.as_str()).collect();
        assert_eq!(kinds, vec!["wrong_class", "loose_box", "missing", "extra_box"], "{issues:#?}");
        assert_eq!(issues[0].pred_label_id, Some(2));
        assert_eq!(issues[2].bbox, Some([200.0, 200.0, 30.0, 30.0]));
    }

    #[test]
    fn full_run_with_duplicates_and_ignore() {
        let base = temp("health");
        let src = base.join("src");
        fs::create_dir_all(&src).unwrap();
        // a 和 a2 几乎一样（只差一个像素），b 完全不同
        let mut a = RgbImage::from_fn(64, 48, |x, y| Rgb([(x * 4) as u8, (y * 5) as u8, 100]));
        a.save(src.join("a.png")).unwrap();
        a.put_pixel(0, 0, Rgb([255, 255, 255]));
        a.save(src.join("a2.png")).unwrap();
        RgbImage::from_fn(64, 48, |x, y| Rgb([((63 - x) * 4) as u8, 0, (y * 5) as u8])).save(src.join("b.png")).unwrap();
        let p = Project::create(&base, "p", "detect").unwrap();
        images::import(&p, &images::collect(&[src]), true, &|_, _| {}).unwrap();
        let l = annotations::add_label(&p, "x", "#f00").unwrap();
        annotations::save(&p, 1, &[bbox(l.id, 0.0, 0.0, 4.0, 4.0)]).unwrap();

        let r = run(&p, None, None, &|_, _| {}).unwrap();
        assert_eq!(r.dup_groups, vec![vec![1, 2]]);
        assert_eq!(r.issues.len(), 1, "一个极小框");
        assert!(!r.suggestions.is_empty());

        ignore(&p, &[r.issues[0].key()]).unwrap();
        assert!(load(&p).unwrap().unwrap().issues.is_empty(), "忽略后不再出现");

        // 带模型检查：模型在图 1 看到一个没标的目标
        let predict = |ids: &[i64]| -> AppResult<HashMap<i64, Vec<Pred>>> {
            assert_eq!(ids, &[1]);
            Ok(HashMap::from([(1, vec![Pred { label_id: l.id, bbox: [20.0, 20.0, 20.0, 20.0], score: 0.9 }])]))
        };
        let r2 = run(&p, Some((7, "run".into())), Some(&predict), &|_, _| {}).unwrap();
        assert!(r2.issues.iter().any(|i| i.kind == "missing"));
        assert_eq!(r2.run_id, Some(7));
        drop(p);
        let _ = fs::remove_dir_all(base);
    }
}
