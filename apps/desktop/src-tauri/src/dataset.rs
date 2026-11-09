//! 数据集划分与导出（YOLO / COCO / VOC）

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::params;
use serde::{Deserialize, Serialize};

use crate::error::{msg, AppResult};
use crate::project::Project;

pub const SPLITS: [&str; 3] = ["train", "val", "test"];

/// 能参与训练的图：已标注、已审核（未标注和跳过的不算）
const TRAINABLE: &str = "status IN ('labeled', 'reviewed')";

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SplitSummary {
    pub train: i64,
    pub val: i64,
    pub test: i64,
    pub unset: i64,
}

pub fn split_summary(p: &Project) -> AppResult<SplitSummary> {
    let conn = p.conn.lock();
    let mut s = SplitSummary { train: 0, val: 0, test: 0, unset: 0 };
    let mut stmt = conn.prepare(&format!("SELECT split, COUNT(*) FROM images WHERE {TRAINABLE} GROUP BY split"))?;
    for row in stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))? {
        let (k, n) = row?;
        match k.as_str() {
            "train" => s.train = n,
            "val" => s.val = n,
            "test" => s.test = n,
            _ => s.unset += n,
        }
    }
    Ok(s)
}

fn check_ratios(r: [f64; 3]) -> AppResult<[f64; 3]> {
    if r.iter().any(|x| !x.is_finite() || *x < 0.0) || r[0] <= 0.0 {
        return Err(msg("划分比例不对：训练集必须大于 0"));
    }
    let sum: f64 = r.iter().sum();
    Ok([r[0] / sum, r[1] / sum, r[2] / sum])
}

/// 给可训练的图分配训练/验证/测试集。all=false 时只分配还没划分的图（已划分的保持不变，保证可复现）。
/// 按图片哈希排序打乱，同样的数据每次结果一样。
pub fn assign_splits(p: &Project, ratios: [f64; 3], all: bool) -> AppResult<SplitSummary> {
    let r = check_ratios(ratios)?;
    {
        let mut conn = p.conn.lock();
        let tx = conn.transaction()?;
        let cond = if all { TRAINABLE.to_string() } else { format!("{TRAINABLE} AND split = 'unset'") };
        let ids: Vec<i64> = {
            let mut stmt = tx.prepare(&format!("SELECT id FROM images WHERE {cond} ORDER BY hash"))?;
            let v = stmt.query_map([], |row| row.get(0))?.collect::<Result<_, _>>()?;
            v
        };
        let n = ids.len();
        let n_val = (n as f64 * r[1]).round() as usize;
        let n_test = (n as f64 * r[2]).round() as usize;
        // 有验证比例时至少给 1 张，否则训练时没法评估
        let n_val = if r[1] > 0.0 && n >= 2 { n_val.max(1) } else { n_val };
        let n_test = n_test.min(n.saturating_sub(n_val + 1));
        {
            let mut up = tx.prepare("UPDATE images SET split = ?2 WHERE id = ?1")?;
            for (i, id) in ids.iter().enumerate() {
                let split = if i < n_val { "val" } else if i < n_val + n_test { "test" } else { "train" };
                up.execute(params![id, split])?;
            }
        }
        tx.commit()?;
    }
    split_summary(p)
}

// ---------- 导出 ----------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Yolo,
    Coco,
    Voc,
    /// 图像分类文件夹：train/类别/图片.jpg（ImageNet / YOLO-cls 通用布局）
    Cls,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportOptions {
    pub format: Format,
    pub out_dir: PathBuf,
    /// 只导出已审核的图
    #[serde(default)]
    pub only_reviewed: bool,
    /// 还没划分的图按这个比例分配
    #[serde(default = "default_ratios")]
    pub ratios: [f64; 3],
}

fn default_ratios() -> [f64; 3] {
    [0.8, 0.2, 0.0]
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportResult {
    pub out_dir: String,
    pub images: usize,
    pub boxes: usize,
    /// 分割多边形数量
    pub polys: usize,
    /// 图像分类：有类别标签的图片数
    pub tags: usize,
    /// 关键点目标数量
    pub kpts: usize,
    pub train: usize,
    pub val: usize,
    pub test: usize,
    pub classes: Vec<String>,
}

pub struct ExportImage {
    pub id: i64,
    pub src: PathBuf,
    pub file_name: String,
    pub width: u32,
    pub height: u32,
    pub split: String,
    /// (类别序号, x, y, w, h)
    pub boxes: Vec<(usize, f64, f64, f64, f64)>,
    /// 分割多边形：(类别序号, [x1, y1, x2, y2, ...])，原图像素坐标
    pub polys: Vec<(usize, Vec<f64>)>,
    /// 图像分类：整图的类别序号（单标签，取第一个）
    pub tag: Option<usize>,
    /// 关键点目标：(类别序号, 外接框 [x, y, w, h], [x1, y1, v1, ...])
    pub kpts: Vec<(usize, [f64; 4], Vec<f64>)>,
}

impl ExportResult {
    /// 全部标注数（框、多边形、关键点目标、分类标签合计）
    pub fn total(&self) -> usize {
        self.boxes + self.polys + self.tags + self.kpts
    }
}

/// 类别名做文件夹名：去掉文件系统不允许的字符
fn safe_dir(name: &str) -> String {
    let s: String = name.chars().map(|c| if r#"\/:*?"<>|"#.contains(c) || c.is_control() { '_' } else { c }).collect();
    let s = s.trim().trim_end_matches('.').to_string();
    if s.is_empty() { "_".into() } else { s }
}

/// 多边形外接框 (x, y, w, h)
fn poly_bbox(pts: &[f64]) -> (f64, f64, f64, f64) {
    let xs = pts.iter().step_by(2);
    let ys = pts.iter().skip(1).step_by(2);
    let x0 = xs.clone().cloned().fold(f64::MAX, f64::min);
    let x1 = xs.cloned().fold(f64::MIN, f64::max);
    let y0 = ys.clone().cloned().fold(f64::MAX, f64::min);
    let y1 = ys.cloned().fold(f64::MIN, f64::max);
    (x0, y0, x1 - x0, y1 - y0)
}

/// 多边形面积（鞋带公式）
fn poly_area(pts: &[f64]) -> f64 {
    let n = pts.len() / 2;
    let mut a = 0.0;
    for i in 0..n {
        let (x1, y1) = (pts[2 * i], pts[2 * i + 1]);
        let j = (i + 1) % n;
        let (x2, y2) = (pts[2 * j], pts[2 * j + 1]);
        a += x1 * y2 - x2 * y1;
    }
    a.abs() / 2.0
}

/// 把矩形框转成 4 点多边形（YOLO-seg 混合导出时用）
fn box_to_poly(x: f64, y: f64, w: f64, h: f64) -> Vec<f64> {
    vec![x, y, x + w, y, x + w, y + h, x, y + h]
}

/// 读出要导出的图和框。类别序号按类别列表顺序从 0 开始。
pub fn collect(p: &Project, only_reviewed: bool) -> AppResult<(Vec<String>, Vec<ExportImage>)> {
    let conn = p.conn.lock();
    let mut labels: Vec<(i64, String)> = Vec::new();
    {
        let mut stmt = conn.prepare("SELECT id, name FROM labels ORDER BY sort, id")?;
        for row in stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))? {
            labels.push(row?);
        }
    }
    if labels.is_empty() {
        return Err(msg("还没有类别，先在标注页添加类别并标注"));
    }
    let index: HashMap<i64, usize> = labels.iter().enumerate().map(|(i, (id, _))| (*id, i)).collect();
    let cond = if only_reviewed { "status = 'reviewed'" } else { TRAINABLE };
    let mut images = Vec::new();
    {
        let mut stmt = conn.prepare(&format!("SELECT id, path, width, height, split FROM images WHERE {cond} ORDER BY id"))?;
        let rows = stmt.query_map([], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get(2)?, r.get(3)?, r.get::<_, String>(4)?))
        })?;
        for row in rows {
            let (id, path, width, height, split) = row?;
            let src = p.resolve(&path);
            let file_name = src.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| format!("{id}.jpg"));
            images.push(ExportImage {
                id,
                src,
                file_name,
                width,
                height,
                split,
                boxes: Vec::new(),
                polys: Vec::new(),
                tag: None,
                kpts: Vec::new(),
            });
        }
    }
    if images.is_empty() {
        return Err(msg(if only_reviewed { "没有已审核的图片" } else { "没有已标注的图片" }));
    }
    let pos: HashMap<i64, usize> = images.iter().enumerate().map(|(i, im)| (im.id, i)).collect();
    let mut stmt = conn.prepare(
        "SELECT image_id, label_id, type, data_json FROM annotations
         WHERE type IN ('bbox', 'polygon', 'tag', 'keypoints') ORDER BY id",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok((r.get::<_, i64>(0)?, r.get::<_, Option<i64>>(1)?, r.get::<_, String>(2)?, r.get::<_, String>(3)?))
    })?;
    for row in rows {
        let (image_id, label_id, kind, data) = row?;
        let (Some(&i), Some(label_id)) = (pos.get(&image_id), label_id) else { continue };
        let Some(&cls) = index.get(&label_id) else { continue };
        push_annotation(&mut images[i], cls, &kind, &data);
    }
    dedupe_names(&mut images);
    Ok((labels.into_iter().map(|(_, n)| n).collect(), images))
}

/// 把一条标注（类型 + data_json）放进导出图片
pub fn push_annotation(im: &mut ExportImage, cls: usize, kind: &str, data: &str) {
    match kind {
        "tag" => {
            if im.tag.is_none() {
                im.tag = Some(cls);
            }
        }
        "keypoints" => {
            let value: serde_json::Value = serde_json::from_str(data).unwrap_or_default();
            if let Some((b, k)) = crate::keypoints::parse(&value) {
                if b[2] > 0.0 && b[3] > 0.0 {
                    im.kpts.push((cls, b, k));
                }
            }
        }
        "bbox" => {
            let v: Vec<f64> = serde_json::from_str(data).unwrap_or_default();
            if v.len() == 4 && v[2] > 0.0 && v[3] > 0.0 {
                im.boxes.push((cls, v[0], v[1], v[2], v[3]));
            }
        }
        "polygon" => {
            let v: Vec<f64> = serde_json::from_str(data).unwrap_or_default();
            if v.len() >= 6 && v.len() % 2 == 0 {
                im.polys.push((cls, v));
            }
        }
        _ => {}
    }
}

/// 不同文件夹里的同名图片，导出时加上 id 区分
pub fn dedupe_names(images: &mut [ExportImage]) {
    let mut seen = HashSet::new();
    for im in images.iter_mut() {
        if !seen.insert(im.file_name.to_lowercase()) {
            im.file_name = format!("{}_{}", im.id, im.file_name);
            seen.insert(im.file_name.to_lowercase());
        }
    }
}

/// 优先硬链接（不占额外空间），跨盘时退回复制
fn link_or_copy(src: &Path, dst: &Path) -> AppResult<()> {
    if fs::hard_link(src, dst).is_err() {
        fs::copy(src, dst).map_err(|e| msg(format!("复制图片失败 {}：{e}", src.display())))?;
    }
    Ok(())
}

fn prepare_out(out: &Path) -> AppResult<()> {
    if out.exists() && fs::read_dir(out)?.next().is_some() {
        return Err(msg(format!("导出目录不为空：{}", out.display())));
    }
    fs::create_dir_all(out)?;
    Ok(())
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn yaml_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

fn check_format(p: &Project, f: Format) -> AppResult<()> {
    let classify = p.meta.task == "classify";
    if classify && f != Format::Cls {
        return Err(msg("图像分类项目请导出为「分类文件夹」格式"));
    }
    if !classify && f == Format::Cls {
        return Err(msg("「分类文件夹」格式只适合图像分类项目"));
    }
    Ok(())
}

/// 导出当前数据
pub fn export(p: &Project, opts: &ExportOptions, progress: &dyn Fn(usize, usize)) -> AppResult<ExportResult> {
    check_format(p, opts.format)?;
    // 还没划分的图先按比例划分，并记到库里，下次导出结果一致
    assign_splits(p, opts.ratios, false)?;
    let (classes, images) = collect(p, opts.only_reviewed)?;
    write(p, opts, classes, images, progress)
}

/// 导出某个数据集版本（划分用版本里记下的，不重新分配）
pub fn export_version(p: &Project, version_id: i64, opts: &ExportOptions, progress: &dyn Fn(usize, usize)) -> AppResult<ExportResult> {
    check_format(p, opts.format)?;
    let (classes, images) = crate::versions::collect(p, version_id, opts.only_reviewed)?;
    write(p, opts, classes, images, progress)
}

fn write(
    p: &Project,
    opts: &ExportOptions,
    classes: Vec<String>,
    mut images: Vec<ExportImage>,
    progress: &dyn Fn(usize, usize),
) -> AppResult<ExportResult> {
    let classify = p.meta.task == "classify";
    if classify {
        // 没打类别标签的图对分类没有用
        images.retain(|im| im.tag.is_some());
        if images.is_empty() {
            return Err(msg("还没有给图片打类别标签"));
        }
    }
    let out = &opts.out_dir;
    prepare_out(out)?;
    let total = images.len();
    let mut result = ExportResult {
        out_dir: out.to_string_lossy().into_owned(),
        images: total,
        boxes: images.iter().map(|i| i.boxes.len()).sum(),
        polys: images.iter().map(|i| i.polys.len()).sum(),
        tags: images.iter().filter(|i| i.tag.is_some()).count(),
        kpts: images.iter().map(|i| i.kpts.len()).sum(),
        train: 0,
        val: 0,
        test: 0,
        classes: classes.clone(),
    };
    for im in &images {
        match im.split.as_str() {
            "val" => result.val += 1,
            "test" => result.test += 1,
            _ => result.train += 1,
        }
    }
    // 只要有多边形，YOLO 就按分割格式导出（矩形框转成四点多边形），避免同一数据集里混两种格式
    let seg_mode = result.polys > 0;
    // 关键点项目：YOLO 按 YOLO-pose 格式导出，COCO 带 keypoints
    let template = if p.meta.task == "pose" { Some(crate::keypoints::get(p)?) } else { None };
    let nk = template.as_ref().map(|t| t.points.len()).unwrap_or(0);
    let split_of = |im: &ExportImage| if SPLITS.contains(&im.split.as_str()) { im.split.clone() } else { "train".into() };
    let used: Vec<&str> = SPLITS.iter().copied().filter(|s| images.iter().any(|im| split_of(im) == *s)).collect();

    match opts.format {
        Format::Yolo => {
            for s in &used {
                fs::create_dir_all(out.join("images").join(s))?;
                fs::create_dir_all(out.join("labels").join(s))?;
            }
            for (n, im) in images.iter().enumerate() {
                let s = split_of(im);
                link_or_copy(&im.src, &out.join("images").join(&s).join(&im.file_name))?;
                let (w, h) = (im.width as f64, im.height as f64);
                let norm_poly = |pts: &[f64]| -> String {
                    pts.chunks(2).map(|p| format!("{:.6} {:.6}", p[0] / w, p[1] / h)).collect::<Vec<_>>().join(" ")
                };
                let lines: Vec<String> = if template.is_some() {
                    // YOLO-pose：类别 cx cy w h 然后每个点 x y v（归一化）；没有关键点的框补 0
                    let mut ls = Vec::new();
                    let line = |c: usize, b: [f64; 4], k: &[f64]| -> String {
                        let mut s = format!(
                            "{c} {:.6} {:.6} {:.6} {:.6}",
                            (b[0] + b[2] / 2.0) / w,
                            (b[1] + b[3] / 2.0) / h,
                            b[2] / w,
                            b[3] / h
                        );
                        for i in 0..nk {
                            let (x, y, v) = match k.get(i * 3..i * 3 + 3) {
                                Some(t) if t[2] > 0.0 => (t[0] / w, t[1] / h, t[2]),
                                _ => (0.0, 0.0, 0.0),
                            };
                            s += &format!(" {x:.6} {y:.6} {v:.0}");
                        }
                        s
                    };
                    for (c, b, k) in &im.kpts {
                        ls.push(line(*c, *b, k));
                    }
                    for (c, x, y, bw, bh) in &im.boxes {
                        ls.push(line(*c, [*x, *y, *bw, *bh], &[]));
                    }
                    ls
                } else if seg_mode {
                    let mut ls = Vec::new();
                    for (c, pts) in &im.polys {
                        ls.push(format!("{c} {}", norm_poly(pts)));
                    }
                    for (c, x, y, bw, bh) in &im.boxes {
                        ls.push(format!("{c} {}", norm_poly(&box_to_poly(*x, *y, *bw, *bh))));
                    }
                    ls
                } else {
                    im.boxes
                        .iter()
                        .map(|(c, x, y, bw, bh)| {
                            format!("{c} {:.6} {:.6} {:.6} {:.6}", (x + bw / 2.0) / w, (y + bh / 2.0) / h, bw / w, bh / h)
                        })
                        .collect()
                };
                let stem = Path::new(&im.file_name).file_stem().unwrap().to_string_lossy().into_owned();
                fs::write(out.join("labels").join(&s).join(format!("{stem}.txt")), lines.join("\n"))?;
                progress(n + 1, total);
            }
            let mut yaml = String::from("# 由 xlabel 导出\npath: .\n");
            for s in &used {
                yaml += &format!("{s}: images/{s}\n");
            }
            if let Some(t) = &template {
                let flip: Vec<String> = crate::keypoints::flip_index(&t.points).iter().map(|i| i.to_string()).collect();
                yaml += &format!("kpt_shape: [{nk}, 3]\nflip_idx: [{}]\n", flip.join(", "));
                yaml += &format!(
                    "# 关键点顺序：{}\n",
                    t.points.iter().enumerate().map(|(i, n)| format!("{i}={n}")).collect::<Vec<_>>().join("，")
                );
            }
            yaml += &format!("nc: {}\nnames:\n", classes.len());
            for (i, c) in classes.iter().enumerate() {
                yaml += &format!("  {i}: {}\n", yaml_quote(c));
            }
            fs::write(out.join("data.yaml"), yaml)?;
            fs::write(out.join("classes.txt"), classes.join("\n"))?;
        }
        Format::Coco => {
            fs::create_dir_all(out.join("annotations"))?;
            let categories: Vec<_> = classes
                .iter()
                .enumerate()
                .map(|(i, c)| match &template {
                    // COCO 关键点：类别里带点名和骨架（骨架序号从 1 开始）
                    Some(t) => serde_json::json!({
                        "id": i + 1, "name": c, "supercategory": "none",
                        "keypoints": t.points,
                        "skeleton": t.skeleton.iter().map(|[a, b]| [a + 1, b + 1]).collect::<Vec<_>>(),
                    }),
                    None => serde_json::json!({ "id": i + 1, "name": c, "supercategory": "none" }),
                })
                .collect();
            let mut n = 0;
            let mut ann_id = 1;
            for s in &used {
                fs::create_dir_all(out.join("images").join(s))?;
                let mut imgs = Vec::new();
                let mut anns = Vec::new();
                for im in images.iter().filter(|im| split_of(im) == *s) {
                    link_or_copy(&im.src, &out.join("images").join(s).join(&im.file_name))?;
                    imgs.push(serde_json::json!({
                        "id": im.id, "file_name": im.file_name, "width": im.width, "height": im.height
                    }));
                    for (c, x, y, bw, bh) in &im.boxes {
                        anns.push(serde_json::json!({
                            "id": ann_id, "image_id": im.id, "category_id": c + 1,
                            "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0
                        }));
                        ann_id += 1;
                    }
                    for (c, pts) in &im.polys {
                        let (bx, by, bw, bh) = poly_bbox(pts);
                        anns.push(serde_json::json!({
                            "id": ann_id, "image_id": im.id, "category_id": c + 1,
                            "bbox": [bx, by, bw, bh], "area": poly_area(pts), "iscrowd": 0,
                            "segmentation": [pts]
                        }));
                        ann_id += 1;
                    }
                    for (c, b, k) in &im.kpts {
                        // 点数统一成模板点数，缺的补 0
                        let mut kp: Vec<f64> = k.clone();
                        kp.resize(nk.max(k.len() / 3) * 3, 0.0);
                        let visible = kp.chunks(3).filter(|t| t[2] > 0.0).count();
                        anns.push(serde_json::json!({
                            "id": ann_id, "image_id": im.id, "category_id": c + 1,
                            "bbox": b, "area": b[2] * b[3], "iscrowd": 0,
                            "keypoints": kp, "num_keypoints": visible
                        }));
                        ann_id += 1;
                    }
                    n += 1;
                    progress(n, total);
                }
                let doc = serde_json::json!({
                    "info": { "description": "由 xlabel 导出" },
                    "images": imgs, "annotations": anns, "categories": categories
                });
                fs::write(out.join("annotations").join(format!("instances_{s}.json")), serde_json::to_string(&doc)?)?;
            }
        }
        Format::Voc => {
            for d in ["JPEGImages", "Annotations", "ImageSets/Main"] {
                fs::create_dir_all(out.join(d))?;
            }
            let mut sets: HashMap<String, Vec<String>> = HashMap::new();
            for (n, im) in images.iter().enumerate() {
                link_or_copy(&im.src, &out.join("JPEGImages").join(&im.file_name))?;
                let stem = Path::new(&im.file_name).file_stem().unwrap().to_string_lossy().into_owned();
                let mut xml = format!(
                    "<annotation>\n  <folder>JPEGImages</folder>\n  <filename>{}</filename>\n  <size>\n    <width>{}</width>\n    <height>{}</height>\n    <depth>3</depth>\n  </size>\n  <segmented>0</segmented>\n",
                    xml_escape(&im.file_name), im.width, im.height
                );
                let mut voc_obj = |cls: usize, x: f64, y: f64, bw: f64, bh: f64| {
                    xml += &format!(
                        "  <object>\n    <name>{}</name>\n    <pose>Unspecified</pose>\n    <truncated>0</truncated>\n    <difficult>0</difficult>\n    <bndbox>\n      <xmin>{:.0}</xmin>\n      <ymin>{:.0}</ymin>\n      <xmax>{:.0}</xmax>\n      <ymax>{:.0}</ymax>\n    </bndbox>\n  </object>\n",
                        xml_escape(&classes[cls]), x, y, x + bw, y + bh
                    );
                };
                for (c, x, y, bw, bh) in &im.boxes {
                    voc_obj(*c, *x, *y, *bw, *bh);
                }
                // VOC 不支持多边形和关键点，用外接框导出
                for (c, pts) in &im.polys {
                    let (bx, by, bw, bh) = poly_bbox(pts);
                    voc_obj(*c, bx, by, bw, bh);
                }
                for (c, b, _) in &im.kpts {
                    voc_obj(*c, b[0], b[1], b[2], b[3]);
                }
                xml += "</annotation>\n";
                fs::write(out.join("Annotations").join(format!("{stem}.xml")), xml)?;
                sets.entry(split_of(im)).or_default().push(stem);
                progress(n + 1, total);
            }
            for (s, names) in sets {
                fs::write(out.join("ImageSets/Main").join(format!("{s}.txt")), names.join("\n"))?;
            }
            fs::write(out.join("labels.txt"), classes.join("\n"))?;
        }
        Format::Cls => {
            // 分类文件夹：{split}/{类别}/{图片}；另附 annotations/{split}.json 清单（带图片 id，训练和评估用）
            let dirs: Vec<String> = {
                // 类别名做文件夹名，去掉非法字符后撞名的加序号
                let mut seen = HashSet::new();
                classes
                    .iter()
                    .map(|c| {
                        let base = safe_dir(c);
                        let mut name = base.clone();
                        let mut i = 2;
                        while !seen.insert(name.to_lowercase()) {
                            name = format!("{base}_{i}");
                            i += 1;
                        }
                        name
                    })
                    .collect()
            };
            fs::create_dir_all(out.join("annotations"))?;
            let mut n = 0;
            for s in &used {
                let mut list = Vec::new();
                for im in images.iter().filter(|im| split_of(im) == *s) {
                    let Some(c) = im.tag else { continue };
                    let d = out.join(s).join(&dirs[c]);
                    fs::create_dir_all(&d)?;
                    link_or_copy(&im.src, &d.join(&im.file_name))?;
                    list.push(serde_json::json!({
                        "id": im.id, "file": format!("{s}/{}/{}", dirs[c], im.file_name), "label": c
                    }));
                    n += 1;
                    progress(n, total);
                }
                let doc = serde_json::json!({ "classes": classes, "images": list });
                fs::write(out.join("annotations").join(format!("{s}.json")), serde_json::to_string(&doc)?)?;
            }
            fs::write(out.join("classes.txt"), classes.join("\n"))?;
        }
    }
    Ok(result)
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::annotations::{self, Annotation};
    use crate::images;
    use image::{Rgb, RgbImage};

    pub fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("xlabel-ds-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    pub fn bbox(label: i64, x: f64, y: f64, w: f64, h: f64) -> Annotation {
        Annotation { label_id: Some(label), kind: "bbox".into(), data: serde_json::json!([x, y, w, h]), score: None, source: "manual".into() }
    }

    /// 10 张 100x50 的图，前 8 张有标注
    pub fn sample(base: &Path) -> Project {
        let src = base.join("src");
        fs::create_dir_all(&src).unwrap();
        for i in 0..10u8 {
            RgbImage::from_pixel(100, 50, Rgb([i * 20, 0, 0])).save(src.join(format!("im{i}.jpg"))).unwrap();
        }
        let p = Project::create(base, "p", "detect").unwrap();
        images::import(&p, &images::collect(&[src]), true, &|_, _| {}).unwrap();
        let car = annotations::add_label(&p, "car", "#f00").unwrap();
        let person = annotations::add_label(&p, "person & <kid>", "#0f0").unwrap();
        for id in 1..=8 {
            annotations::save(&p, id, &[bbox(car.id, 10.0, 5.0, 20.0, 10.0), bbox(person.id, 50.0, 10.0, 30.0, 40.0)]).unwrap();
        }
        p
    }

    #[test]
    fn split_is_stable_and_partial() {
        let base = temp("split");
        let p = sample(&base);
        let s = assign_splits(&p, [0.75, 0.25, 0.0], false).unwrap();
        assert_eq!((s.train, s.val, s.test, s.unset), (6, 2, 0, 0));
        // 再标两张：只给新图分配，旧的不变
        let before: Vec<String> = (1..=8).map(|i| split_of_id(&p, i)).collect();
        annotations::save(&p, 9, &[bbox(1, 0.0, 0.0, 5.0, 5.0)]).unwrap();
        let s2 = assign_splits(&p, [0.5, 0.5, 0.0], false).unwrap();
        assert_eq!(s2.unset, 0);
        let after: Vec<String> = (1..=8).map(|i| split_of_id(&p, i)).collect();
        assert_eq!(before, after);
        assert!(assign_splits(&p, [0.0, 1.0, 0.0], true).is_err());
        drop(p);
        let _ = fs::remove_dir_all(base);
    }

    fn split_of_id(p: &Project, id: i64) -> String {
        p.conn.lock().query_row("SELECT split FROM images WHERE id = ?1", [id], |r| r.get(0)).unwrap()
    }

    #[test]
    fn export_three_formats() {
        let base = temp("export");
        let p = sample(&base);
        for fmt in [Format::Yolo, Format::Coco, Format::Voc] {
            let out = base.join(format!("{fmt:?}"));
            let r = export(&p, &ExportOptions { format: fmt, out_dir: out.clone(), only_reviewed: false, ratios: [0.75, 0.25, 0.0] }, &|_, _| {}).unwrap();
            assert_eq!((r.images, r.boxes, r.train, r.val), (8, 16, 6, 2));
            match fmt {
                Format::Yolo => {
                    let yaml = fs::read_to_string(out.join("data.yaml")).unwrap();
                    assert!(yaml.contains("nc: 2") && yaml.contains("val: images/val"));
                    let train: Vec<_> = fs::read_dir(out.join("labels/train")).unwrap().collect();
                    assert_eq!(train.len(), 6);
                    let first = fs::read_dir(out.join("labels/train")).unwrap().next().unwrap().unwrap().path();
                    let txt = fs::read_to_string(first).unwrap();
                    // car: 中心 (20,10)/(100,50) = 0.2,0.2；宽高 0.2,0.2
                    assert!(txt.lines().next().unwrap().starts_with("0 0.200000 0.200000 0.200000 0.200000"), "{txt}");
                }
                Format::Coco => {
                    let v: serde_json::Value = serde_json::from_str(&fs::read_to_string(out.join("annotations/instances_train.json")).unwrap()).unwrap();
                    assert_eq!(v["images"].as_array().unwrap().len(), 6);
                    assert_eq!(v["annotations"].as_array().unwrap().len(), 12);
                    assert_eq!(v["categories"][1]["name"], "person & <kid>");
                    assert_eq!(fs::read_dir(out.join("images/val")).unwrap().count(), 2);
                }
                Format::Voc => {
                    assert_eq!(fs::read_dir(out.join("Annotations")).unwrap().count(), 8);
                    let xml = fs::read_to_string(fs::read_dir(out.join("Annotations")).unwrap().next().unwrap().unwrap().path()).unwrap();
                    assert!(xml.contains("<name>person &amp; &lt;kid&gt;</name>") && xml.contains("<xmax>30</xmax>"));
                    assert!(out.join("ImageSets/Main/val.txt").exists());
                }
                Format::Cls => unreachable!(),
            }
            // 目录不为空时拒绝导出
            assert!(export(&p, &ExportOptions { format: fmt, out_dir: out, only_reviewed: false, ratios: [1.0, 0.0, 0.0] }, &|_, _| {}).is_err());
        }
        assert!(collect(&p, true).is_err(), "没有已审核的图");
        drop(p);
        let _ = fs::remove_dir_all(base);
    }
}
