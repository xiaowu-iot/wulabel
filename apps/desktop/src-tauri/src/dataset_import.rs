//! 导入带标注的数据集：自动识别 YOLO / COCO / VOC

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::params;
use serde::Serialize;

use crate::annotations::{self, Annotation};
use crate::error::{msg, AppResult};
use crate::images;
use crate::project::Project;

const PALETTE: [&str; 10] = ["#3d8bfd", "#e5534b", "#2fb67c", "#e7a33e", "#9b7bf2", "#14b8a6", "#f472b6", "#a3e635", "#f97316", "#60a5fa"];

#[derive(Debug, Clone, PartialEq)]
enum Coords {
    /// 归一化中心点坐标（YOLO），宽高要等图片导入后再换算
    Norm(f64, f64, f64, f64),
    /// 像素坐标 x, y, w, h
    Abs(f64, f64, f64, f64),
}

#[derive(Debug, Clone)]
struct Item {
    src: PathBuf,
    split: Option<String>,
    boxes: Vec<(String, Coords)>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetImportResult {
    pub format: String,
    pub images: usize,
    pub added: usize,
    pub duplicated: usize,
    pub boxes: usize,
    pub labels_added: Vec<String>,
    /// 已有标注的重复图片，不覆盖原标注
    pub kept_existing: usize,
    pub failed: Vec<String>,
}

fn all_images(dir: &Path) -> Vec<PathBuf> {
    images::collect(&[dir.to_path_buf()])
}

fn by_name(files: &[PathBuf]) -> HashMap<String, PathBuf> {
    files
        .iter()
        .filter_map(|f| Some((f.file_name()?.to_string_lossy().to_lowercase(), f.clone())))
        .collect()
}

fn split_from_path(p: &Path) -> Option<String> {
    p.components().rev().skip(1).find_map(|c| {
        let s = c.as_os_str().to_string_lossy().to_lowercase();
        match s.as_str() {
            "train" | "training" => Some("train".into()),
            "val" | "valid" | "validation" => Some("val".into()),
            "test" | "testing" => Some("test".into()),
            _ => None,
        }
    })
}

// ---------- YOLO ----------

/// 读 data.yaml 里的 names，支持 `names: [a, b]`、列表和 `0: a` 三种写法
fn yaml_names(text: &str) -> Vec<String> {
    let unquote = |s: &str| s.trim().trim_matches(|c| c == '\'' || c == '"').to_string();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let t = line.trim_start();
        let Some(rest) = t.strip_prefix("names:") else { continue };
        let rest = rest.trim();
        if rest.starts_with('[') {
            return rest.trim_matches(|c| c == '[' || c == ']').split(',').map(unquote).filter(|s| !s.is_empty()).collect();
        }
        let mut out: Vec<(usize, String)> = Vec::new();
        while let Some(next) = lines.peek() {
            let n = next.trim();
            if n.is_empty() || n.starts_with('#') {
                lines.next();
                continue;
            }
            if !next.starts_with(' ') && !next.starts_with('\t') && !n.starts_with('-') {
                break;
            }
            if let Some(v) = n.strip_prefix('-') {
                out.push((out.len(), unquote(v)));
            } else if let Some((k, v)) = n.split_once(':') {
                if let Ok(i) = k.trim().parse() {
                    out.push((i, unquote(v)));
                }
            }
            lines.next();
        }
        out.sort_by_key(|x| x.0);
        return out.into_iter().map(|x| x.1).collect();
    }
    Vec::new()
}

fn yolo_classes(dir: &Path) -> Vec<String> {
    for f in ["data.yaml", "dataset.yaml", "data.yml"] {
        if let Ok(t) = fs::read_to_string(dir.join(f)) {
            let n = yaml_names(&t);
            if !n.is_empty() {
                return n;
            }
        }
    }
    for f in ["classes.txt", "obj.names", "labels.txt"] {
        if let Ok(t) = fs::read_to_string(dir.join(f)) {
            return t.lines().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        }
    }
    Vec::new()
}

/// images/a/b.jpg 对应 labels/a/b.txt；没有 labels 目录时找同目录的 b.txt
fn yolo_label_file(img: &Path) -> Option<PathBuf> {
    let mut parts: Vec<_> = img.components().map(|c| c.as_os_str().to_os_string()).collect();
    if let Some(i) = parts.iter().rposition(|c| c.to_string_lossy().eq_ignore_ascii_case("images")) {
        parts[i] = "labels".into();
        let p: PathBuf = parts.iter().collect::<PathBuf>().with_extension("txt");
        if p.exists() {
            return Some(p);
        }
    }
    let same = img.with_extension("txt");
    same.exists().then_some(same)
}

fn read_yolo(dir: &Path) -> AppResult<Vec<Item>> {
    let classes = yolo_classes(dir);
    let mut items = Vec::new();
    for img in all_images(dir) {
        let mut boxes = Vec::new();
        if let Some(lf) = yolo_label_file(&img) {
            for line in fs::read_to_string(&lf)?.lines() {
                let v: Vec<f64> = line.split_whitespace().filter_map(|s| s.parse().ok()).collect();
                if v.len() < 5 {
                    continue;
                }
                let cls = v[0] as usize;
                let name = classes.get(cls).cloned().unwrap_or_else(|| format!("class_{cls}"));
                let c = if v.len() == 5 {
                    Coords::Norm(v[1], v[2], v[3], v[4])
                } else {
                    // 分割格式：多边形点，取外接框
                    let xs: Vec<f64> = v[1..].iter().step_by(2).copied().collect();
                    let ys: Vec<f64> = v[2..].iter().step_by(2).copied().collect();
                    let (x0, x1) = (xs.iter().cloned().fold(f64::MAX, f64::min), xs.iter().cloned().fold(f64::MIN, f64::max));
                    let (y0, y1) = (ys.iter().cloned().fold(f64::MAX, f64::min), ys.iter().cloned().fold(f64::MIN, f64::max));
                    Coords::Norm((x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0)
                };
                boxes.push((name, c));
            }
        }
        items.push(Item { split: split_from_path(&img), src: img, boxes });
    }
    Ok(items)
}

// ---------- COCO ----------

fn find_coco_jsons(dir: &Path) -> Vec<PathBuf> {
    walkdir::WalkDir::new(dir)
        .max_depth(3)
        .into_iter()
        .flatten()
        .filter(|e| e.file_type().is_file() && e.path().extension().is_some_and(|x| x.eq_ignore_ascii_case("json")))
        .filter(|e| {
            fs::read_to_string(e.path())
                .ok()
                .map(|t| t.contains("\"images\"") && t.contains("\"annotations\"") && t.contains("\"categories\""))
                .unwrap_or(false)
        })
        .map(|e| e.into_path())
        .collect()
}

fn read_coco(dir: &Path, jsons: &[PathBuf]) -> AppResult<Vec<Item>> {
    let files = all_images(dir);
    let names = by_name(&files);
    let mut items: Vec<Item> = Vec::new();
    for jf in jsons {
        let v: serde_json::Value = serde_json::from_str(&fs::read_to_string(jf)?)?;
        let stem = jf.file_stem().unwrap_or_default().to_string_lossy().to_lowercase();
        let split = ["train", "val", "test"].iter().find(|s| stem.contains(*s)).map(|s| s.to_string());
        let cats: HashMap<i64, String> = v["categories"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|c| Some((c["id"].as_i64()?, c["name"].as_str()?.to_string())))
            .collect();
        let mut by_id: HashMap<i64, usize> = HashMap::new();
        for im in v["images"].as_array().into_iter().flatten() {
            let (Some(id), Some(fname)) = (im["id"].as_i64(), im["file_name"].as_str()) else { continue };
            let base = Path::new(fname).file_name().map(|s| s.to_string_lossy().to_lowercase()).unwrap_or_default();
            let found = [dir.join(fname), jf.parent().unwrap_or(dir).join(fname)]
                .into_iter()
                .find(|p| p.is_file())
                .or_else(|| names.get(&base).cloned());
            if let Some(src) = found {
                by_id.insert(id, items.len());
                items.push(Item { split: split.clone().or_else(|| split_from_path(&src)), src, boxes: Vec::new() });
            }
        }
        for a in v["annotations"].as_array().into_iter().flatten() {
            let (Some(img), Some(cat)) = (a["image_id"].as_i64(), a["category_id"].as_i64()) else { continue };
            let Some(&i) = by_id.get(&img) else { continue };
            let b: Vec<f64> = a["bbox"].as_array().into_iter().flatten().filter_map(|x| x.as_f64()).collect();
            if b.len() == 4 {
                let name = cats.get(&cat).cloned().unwrap_or_else(|| format!("class_{cat}"));
                items[i].boxes.push((name, Coords::Abs(b[0], b[1], b[2], b[3])));
            }
        }
    }
    Ok(items)
}

// ---------- VOC ----------

fn find_voc_xmls(dir: &Path) -> Vec<PathBuf> {
    walkdir::WalkDir::new(dir)
        .into_iter()
        .flatten()
        .filter(|e| e.file_type().is_file() && e.path().extension().is_some_and(|x| x.eq_ignore_ascii_case("xml")))
        .map(|e| e.into_path())
        .collect()
}

fn read_voc(dir: &Path, xmls: &[PathBuf]) -> AppResult<Vec<Item>> {
    let files = all_images(dir);
    let names = by_name(&files);
    // ImageSets/Main/{train,val,test}.txt 里的划分
    let mut sets: HashMap<String, String> = HashMap::new();
    for s in ["train", "val", "test"] {
        for f in walkdir::WalkDir::new(dir).into_iter().flatten() {
            if f.file_name().to_string_lossy().eq_ignore_ascii_case(&format!("{s}.txt")) {
                for line in fs::read_to_string(f.path()).unwrap_or_default().lines() {
                    sets.insert(line.trim().to_lowercase(), s.to_string());
                }
            }
        }
    }
    let mut items = Vec::new();
    for xf in xmls {
        let text = fs::read_to_string(xf)?;
        let Ok(doc) = roxmltree::Document::parse(&text) else { continue };
        let root = doc.root_element();
        if !root.has_tag_name("annotation") {
            continue;
        }
        let child_text = |n: roxmltree::Node, tag: &str| {
            n.children().find(|c| c.has_tag_name(tag)).and_then(|c| c.text()).map(|s| s.trim().to_string())
        };
        let stem = xf.file_stem().unwrap_or_default().to_string_lossy().to_lowercase();
        let fname = child_text(root, "filename").unwrap_or_default().to_lowercase();
        let src = names.get(&fname).cloned().or_else(|| {
            images::IMAGE_EXTS.iter().find_map(|e| names.get(&format!("{stem}.{e}")).cloned())
        });
        let Some(src) = src else { continue };
        let mut boxes = Vec::new();
        for obj in root.children().filter(|c| c.has_tag_name("object")) {
            let Some(name) = child_text(obj, "name") else { continue };
            let Some(bb) = obj.children().find(|c| c.has_tag_name("bndbox")) else { continue };
            let num = |t: &str| child_text(bb, t).and_then(|s| s.parse::<f64>().ok());
            if let (Some(x0), Some(y0), Some(x1), Some(y1)) = (num("xmin"), num("ymin"), num("xmax"), num("ymax")) {
                boxes.push((name, Coords::Abs(x0, y0, x1 - x0, y1 - y0)));
            }
        }
        items.push(Item { split: sets.get(&stem).cloned().or_else(|| split_from_path(&src)), src, boxes });
    }
    Ok(items)
}

/// 识别格式并读出图片和框
fn read(dir: &Path) -> AppResult<(String, Vec<Item>)> {
    if !dir.is_dir() {
        return Err(msg("请选择数据集所在的文件夹"));
    }
    let jsons = find_coco_jsons(dir);
    if !jsons.is_empty() {
        return Ok(("COCO".into(), read_coco(dir, &jsons)?));
    }
    let xmls = find_voc_xmls(dir);
    if !xmls.is_empty() {
        let items = read_voc(dir, &xmls)?;
        if !items.is_empty() {
            return Ok(("VOC".into(), items));
        }
    }
    let items = read_yolo(dir)?;
    if items.iter().any(|i| !i.boxes.is_empty()) {
        return Ok(("YOLO".into(), items));
    }
    Err(msg("没有识别出数据集格式。支持 YOLO（images + labels + data.yaml）、COCO（json）、VOC（Annotations/*.xml）"))
}

pub fn import(p: &Project, dir: &Path, progress: &(dyn Fn(usize, usize) + Sync)) -> AppResult<DatasetImportResult> {
    let (format, items) = read(dir)?;
    if items.is_empty() {
        return Err(msg("数据集里没有找到图片"));
    }
    // 1. 导入图片（复制进项目）
    let srcs: Vec<PathBuf> = items.iter().map(|i| i.src.clone()).collect();
    let (r, ids) = images::import_mapped(p, &srcs, true, progress)?;

    // 2. 补齐类别
    let mut labels: HashMap<String, i64> =
        annotations::list_labels(p)?.into_iter().map(|l| (l.name, l.id)).collect();
    let mut labels_added = Vec::new();
    for it in &items {
        for (name, _) in &it.boxes {
            if !labels.contains_key(name) {
                let color = PALETTE[labels.len() % PALETTE.len()];
                let l = annotations::add_label(p, name, color)?;
                labels.insert(name.clone(), l.id);
                labels_added.push(name.clone());
            }
        }
    }

    // 3. 写标注：已经有标注的图不覆盖
    let mut boxes = 0;
    let mut kept_existing = 0;
    for it in &items {
        let Some(&id) = ids.get(&it.src) else { continue };
        let current = annotations::load(p, id)?;
        if !current.annotations.is_empty() {
            kept_existing += 1;
            continue;
        }
        let (w, h) = (current.width as f64, current.height as f64);
        let anns: Vec<Annotation> = it
            .boxes
            .iter()
            .filter_map(|(name, c)| {
                let (x, y, bw, bh) = match *c {
                    Coords::Norm(cx, cy, nw, nh) => ((cx - nw / 2.0) * w, (cy - nh / 2.0) * h, nw * w, nh * h),
                    Coords::Abs(x, y, bw, bh) => (x, y, bw, bh),
                };
                // 裁到图片范围内
                let x0 = x.clamp(0.0, w);
                let y0 = y.clamp(0.0, h);
                let x1 = (x + bw).clamp(0.0, w);
                let y1 = (y + bh).clamp(0.0, h);
                if x1 - x0 < 1.0 || y1 - y0 < 1.0 {
                    return None;
                }
                let r2 = |v: f64| (v * 100.0).round() / 100.0;
                Some(Annotation {
                    label_id: labels.get(name).copied(),
                    kind: "bbox".into(),
                    data: serde_json::json!([r2(x0), r2(y0), r2(x1 - x0), r2(y1 - y0)]),
                    score: None,
                    source: "manual".into(),
                })
            })
            .collect();
        if !anns.is_empty() {
            boxes += anns.len();
            annotations::save(p, id, &anns)?;
        }
        if let Some(s) = &it.split {
            p.conn.lock().execute("UPDATE images SET split = ?2 WHERE id = ?1 AND split = 'unset'", params![id, s])?;
        }
    }
    Ok(DatasetImportResult {
        format,
        images: items.len(),
        added: r.added,
        duplicated: r.duplicated,
        boxes,
        labels_added,
        kept_existing,
        failed: r.failed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{self, tests::{sample, temp}, ExportOptions, Format};

    #[test]
    fn yaml_names_variants() {
        assert_eq!(yaml_names("nc: 2\nnames: ['car', \"person\"]\n"), vec!["car", "person"]);
        assert_eq!(yaml_names("names:\n  - car\n  - bus\ntrain: x\n"), vec!["car", "bus"]);
        assert_eq!(yaml_names("names:\n  1: bus\n  0: 'car'\nnc: 2\n"), vec!["car", "bus"]);
    }

    /// 导出三种格式，再导入新项目，框的数量和位置要一致
    #[test]
    fn roundtrip_three_formats() {
        let base = temp("roundtrip");
        let src = sample(&base);
        for fmt in [Format::Yolo, Format::Coco, Format::Voc] {
            let out = base.join(format!("out-{fmt:?}"));
            dataset::export(&src, &ExportOptions { format: fmt, out_dir: out.clone(), only_reviewed: false, ratios: [0.75, 0.25, 0.0] }, &|_, _| {}).unwrap();
            let dst_dir = base.join(format!("dst-{fmt:?}"));
            fs::create_dir_all(&dst_dir).unwrap();
            let dst = Project::create(&dst_dir, "d", "detect").unwrap();
            let r = import(&dst, &out, &|_, _| {}).unwrap();
            assert_eq!(r.format, format!("{fmt:?}").to_uppercase());
            assert_eq!((r.images, r.added, r.boxes), (8, 8, 16), "{fmt:?}");
            assert_eq!(r.labels_added, vec!["car", "person & <kid>"]);
            let s = dataset::split_summary(&dst).unwrap();
            assert_eq!((s.train, s.val), (6, 2), "{fmt:?} 划分要保留");
            // 第一张图的 car 框还原到 (10,5,20,10)
            let anns = annotations::load(&dst, 1).unwrap().annotations;
            let car = anns.iter().find(|a| a.label_id == Some(1)).unwrap();
            let v: Vec<f64> = serde_json::from_value(car.data.clone()).unwrap();
            assert!((v[0] - 10.0).abs() < 0.5 && (v[2] - 20.0).abs() < 0.5, "{fmt:?} {v:?}");
            // 再导一次：图片重复，已有标注不覆盖
            let r2 = import(&dst, &out, &|_, _| {}).unwrap();
            assert_eq!((r2.added, r2.duplicated, r2.kept_existing, r2.boxes), (0, 8, 8, 0));
        }
        let empty = base.join("empty");
        fs::create_dir_all(&empty).unwrap();
        let p = Project::create(&base, "e", "detect").unwrap();
        assert!(import(&p, &empty, &|_, _| {}).is_err());
        let _ = fs::remove_dir_all(base);
    }
}
