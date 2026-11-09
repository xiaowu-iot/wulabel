//! 导入带标注的数据集：自动识别 YOLO（含分割、关键点）/ COCO（含分割、关键点）/ VOC / 分类文件夹

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

/// 分割多边形坐标：归一化点对（YOLO）或像素点对（COCO）
#[derive(Debug, Clone, PartialEq)]
enum PolyCoords {
    Norm(Vec<f64>),
    Abs(Vec<f64>),
}

#[derive(Debug, Clone)]
struct Item {
    src: PathBuf,
    split: Option<String>,
    boxes: Vec<(String, Coords)>,
    polys: Vec<(String, PolyCoords)>,
    /// 图像分类：整图类别
    tag: Option<String>,
    /// 关键点目标：框 + [x, y, v, ...]（Norm 时 x、y 是归一化坐标）
    kpts: Vec<(String, Coords, PolyCoords)>,
}

impl Item {
    fn new(src: PathBuf, split: Option<String>) -> Item {
        Item { src, split, boxes: Vec::new(), polys: Vec::new(), tag: None, kpts: Vec::new() }
    }

    fn has_any(&self) -> bool {
        !self.boxes.is_empty() || !self.polys.is_empty() || self.tag.is_some() || !self.kpts.is_empty()
    }
}

/// 读出来的关键点点名（COCO 类别里的 keypoints，或 YOLO 的 kpt_shape 推出的点数）
#[derive(Debug, Clone, Default)]
struct KptInfo {
    names: Vec<String>,
    skeleton: Vec<[usize; 2]>,
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

/// data.yaml 里的 kpt_shape: [点数, 维度]（YOLO-pose），维度 2 = 只有 x y，3 = x y v
fn yolo_kpt_shape(dir: &Path) -> Option<(usize, usize)> {
    for f in ["data.yaml", "dataset.yaml", "data.yml"] {
        let Ok(t) = fs::read_to_string(dir.join(f)) else { continue };
        for line in t.lines() {
            let Some(rest) = line.trim_start().strip_prefix("kpt_shape:") else { continue };
            let nums: Vec<usize> =
                rest.split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()).filter_map(|s| s.parse().ok()).collect();
            if nums.len() >= 2 && nums[0] > 0 && (nums[1] == 2 || nums[1] == 3) {
                return Some((nums[0], nums[1]));
            }
        }
    }
    None
}

fn read_yolo(dir: &Path) -> AppResult<(Vec<Item>, Option<KptInfo>)> {
    let classes = yolo_classes(dir);
    let kshape = yolo_kpt_shape(dir);
    let mut items = Vec::new();
    for img in all_images(dir) {
        let mut it = Item::new(img.clone(), split_from_path(&img));
        if let Some(lf) = yolo_label_file(&img) {
            for line in fs::read_to_string(&lf)?.lines() {
                let v: Vec<f64> = line.split_whitespace().filter_map(|s| s.parse().ok()).collect();
                if v.len() < 5 {
                    continue;
                }
                let cls = v[0] as usize;
                let name = classes.get(cls).cloned().unwrap_or_else(|| format!("class_{cls}"));
                if let Some((k, d)) = kshape {
                    // YOLO-pose：class cx cy w h 然后 k 个点（x y [v]）
                    if v.len() == 5 + k * d {
                        let mut pts = Vec::with_capacity(k * 3);
                        for i in 0..k {
                            let p = &v[5 + i * d..5 + i * d + d];
                            let vis = if d == 3 { p[2] } else if p[0] > 0.0 || p[1] > 0.0 { 2.0 } else { 0.0 };
                            pts.extend([p[0], p[1], vis]);
                        }
                        it.kpts.push((name, Coords::Norm(v[1], v[2], v[3], v[4]), PolyCoords::Norm(pts)));
                    }
                    continue;
                }
                if v.len() == 5 {
                    it.boxes.push((name, Coords::Norm(v[1], v[2], v[3], v[4])));
                } else if v.len() >= 7 && v.len() % 2 == 1 {
                    // 分割格式：class 后面是多边形点，原样保留
                    it.polys.push((name, PolyCoords::Norm(v[1..].to_vec())));
                }
            }
        }
        items.push(it);
    }
    let info = kshape.map(|(k, _)| KptInfo { names: (1..=k).map(|i| format!("点{i}")).collect(), skeleton: Vec::new() });
    Ok((items, info))
}

// ---------- 分类文件夹 ----------

fn is_split_name(s: &str) -> bool {
    matches!(s.to_lowercase().as_str(), "train" | "training" | "val" | "valid" | "validation" | "test" | "testing")
}

fn subdirs(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(dir)
        .map(|rd| rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    v.sort();
    v
}

/// 分类文件夹：根目录/类别/图片，或 根目录/train|val|test/类别/图片
fn read_cls(dir: &Path) -> Vec<Item> {
    let mut items = Vec::new();
    let top = subdirs(dir);
    let split_layout = !top.is_empty() && top.iter().all(|d| d.file_name().is_some_and(|n| is_split_name(&n.to_string_lossy())));
    let groups: Vec<(Option<String>, PathBuf)> = if split_layout {
        top.iter()
            .flat_map(|s| {
                let split = split_from_path(&s.join("x"));
                subdirs(s).into_iter().map(move |c| (split.clone(), c))
            })
            .collect()
    } else {
        top.into_iter().map(|c| (None, c)).collect()
    };
    for (split, class_dir) in groups {
        let Some(name) = class_dir.file_name().map(|n| n.to_string_lossy().trim().to_string()) else { continue };
        if name.is_empty() || name.starts_with('.') || name.eq_ignore_ascii_case("annotations") {
            continue;
        }
        for img in all_images(&class_dir) {
            let mut it = Item::new(img, split.clone());
            it.tag = Some(name.clone());
            items.push(it);
        }
    }
    items
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

fn read_coco(dir: &Path, jsons: &[PathBuf]) -> AppResult<(Vec<Item>, Option<KptInfo>)> {
    let files = all_images(dir);
    let names = by_name(&files);
    let mut items: Vec<Item> = Vec::new();
    let mut kinfo: Option<KptInfo> = None;
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
        // COCO 关键点：类别里带 keypoints（点名）和 skeleton（从 1 开始）
        if kinfo.is_none() {
            for c in v["categories"].as_array().into_iter().flatten() {
                let pts: Vec<String> =
                    c["keypoints"].as_array().into_iter().flatten().filter_map(|x| x.as_str().map(String::from)).collect();
                if pts.is_empty() {
                    continue;
                }
                let skeleton = c["skeleton"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|e| {
                        let a = e.as_array()?;
                        let (x, y) = (a.first()?.as_u64()? as usize, a.get(1)?.as_u64()? as usize);
                        (x >= 1 && y >= 1 && x <= pts.len() && y <= pts.len()).then_some([x - 1, y - 1])
                    })
                    .collect();
                kinfo = Some(KptInfo { names: pts, skeleton });
                break;
            }
        }
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
                let sp = split.clone().or_else(|| split_from_path(&src));
                items.push(Item::new(src, sp));
            }
        }
        for a in v["annotations"].as_array().into_iter().flatten() {
            let (Some(img), Some(cat)) = (a["image_id"].as_i64(), a["category_id"].as_i64()) else { continue };
            let Some(&i) = by_id.get(&img) else { continue };
            let name = cats.get(&cat).cloned().unwrap_or_else(|| format!("class_{cat}"));
            // 关键点目标
            let kp: Vec<f64> = a["keypoints"].as_array().into_iter().flatten().filter_map(|x| x.as_f64()).collect();
            if !kp.is_empty() && kp.len() % 3 == 0 {
                let b: Vec<f64> = a["bbox"].as_array().into_iter().flatten().filter_map(|x| x.as_f64()).collect();
                let bbox = if b.len() == 4 {
                    Coords::Abs(b[0], b[1], b[2], b[3])
                } else {
                    // 没有框就用可见点的外接框
                    let vis: Vec<&[f64]> = kp.chunks(3).filter(|t| t[2] > 0.0).collect();
                    if vis.is_empty() {
                        continue;
                    }
                    let (x0, x1) = vis.iter().fold((f64::MAX, f64::MIN), |(lo, hi), t| (lo.min(t[0]), hi.max(t[0])));
                    let (y0, y1) = vis.iter().fold((f64::MAX, f64::MIN), |(lo, hi), t| (lo.min(t[1]), hi.max(t[1])));
                    Coords::Abs(x0, y0, (x1 - x0).max(1.0), (y1 - y0).max(1.0))
                };
                items[i].kpts.push((name, bbox, PolyCoords::Abs(kp)));
                continue;
            }
            // 有多边形分割就按多边形导入，否则用检测框
            let seg = a["segmentation"].as_array().and_then(|arr| arr.first()).and_then(|s| s.as_array());
            if let Some(poly) = seg {
                let pts: Vec<f64> = poly.iter().filter_map(|x| x.as_f64()).collect();
                if pts.len() >= 6 && pts.len() % 2 == 0 {
                    items[i].polys.push((name, PolyCoords::Abs(pts)));
                    continue;
                }
            }
            let b: Vec<f64> = a["bbox"].as_array().into_iter().flatten().filter_map(|x| x.as_f64()).collect();
            if b.len() == 4 {
                items[i].boxes.push((name, Coords::Abs(b[0], b[1], b[2], b[3])));
            }
        }
    }
    Ok((items, kinfo))
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
        let sp = sets.get(&stem).cloned().or_else(|| split_from_path(&src));
        let mut it = Item::new(src, sp);
        it.boxes = boxes;
        items.push(it);
    }
    Ok(items)
}

/// 识别格式并读出图片和标注。分类项目优先按「分类文件夹」识别
fn read(dir: &Path, task: &str) -> AppResult<(String, Vec<Item>, Option<KptInfo>)> {
    if !dir.is_dir() {
        return Err(msg("请选择数据集所在的文件夹"));
    }
    if task == "classify" {
        let items = read_cls(dir);
        if !items.is_empty() {
            return Ok(("分类文件夹".into(), items, None));
        }
        return Err(msg("没有识别出分类数据集。请选择这样的文件夹：每个类别一个子文件夹（猫/、狗/…），或者 train/猫/、val/猫/ 这样再分一层"));
    }
    let jsons = find_coco_jsons(dir);
    if !jsons.is_empty() {
        let (items, k) = read_coco(dir, &jsons)?;
        return Ok(("COCO".into(), items, k));
    }
    let xmls = find_voc_xmls(dir);
    if !xmls.is_empty() {
        let items = read_voc(dir, &xmls)?;
        if !items.is_empty() {
            return Ok(("VOC".into(), items, None));
        }
    }
    let (items, k) = read_yolo(dir)?;
    if items.iter().any(Item::has_any) {
        return Ok((if k.is_some() { "YOLO-pose" } else { "YOLO" }.into(), items, k));
    }
    Err(msg("没有识别出数据集格式。支持 YOLO（images + labels + data.yaml）、COCO（json）、VOC（Annotations/*.xml）"))
}

pub fn import(p: &Project, dir: &Path, progress: &(dyn Fn(usize, usize) + Sync)) -> AppResult<DatasetImportResult> {
    let (format, items, kinfo) = read(dir, &p.meta.task)?;
    if items.is_empty() {
        return Err(msg("数据集里没有找到图片"));
    }
    // 关键点：项目里还没有关键点标注时，模板换成数据集的点名（点数不同的标注没法用）
    let mut nk = None;
    if p.meta.task == "pose" && items.iter().any(|i| !i.kpts.is_empty()) {
        let cur = crate::keypoints::get(p)?;
        let ds_k = items.iter().flat_map(|i| i.kpts.iter()).map(|(_, _, k)| match k {
            PolyCoords::Norm(v) | PolyCoords::Abs(v) => v.len() / 3,
        });
        let k = ds_k.max().unwrap_or(0);
        if k != cur.points.len() {
            let has: i64 =
                p.conn.lock().query_row("SELECT COUNT(*) FROM annotations WHERE type = 'keypoints'", [], |r| r.get(0))?;
            if has > 0 {
                return Err(msg(format!(
                    "数据集每个目标 {k} 个关键点，项目模板是 {} 个点，而且项目里已经有关键点标注，没法合并",
                    cur.points.len()
                )));
            }
            let info = kinfo.clone().filter(|i| i.names.len() == k).unwrap_or(KptInfo {
                names: (1..=k).map(|i| format!("点{i}")).collect(),
                skeleton: Vec::new(),
            });
            crate::keypoints::save(
                p,
                &crate::keypoints::Template { name: format!("{format} 导入"), points: info.names, skeleton: info.skeleton },
            )?;
        } else if let Some(info) = kinfo.clone().filter(|i| i.names.len() == k && cur.points.iter().all(|n| n.starts_with('点'))) {
            // 点数相同、项目还是默认编号名时，用数据集里有意义的点名
            crate::keypoints::save(
                p,
                &crate::keypoints::Template { name: format!("{format} 导入"), points: info.names, skeleton: info.skeleton },
            )?;
        }
        nk = Some(k);
    }
    // 1. 导入图片（复制进项目）
    let srcs: Vec<PathBuf> = items.iter().map(|i| i.src.clone()).collect();
    let (r, ids) = images::import_mapped(p, &srcs, true, progress)?;

    // 2. 补齐类别
    let mut labels: HashMap<String, i64> =
        annotations::list_labels(p)?.into_iter().map(|l| (l.name, l.id)).collect();
    let mut labels_added = Vec::new();
    for it in &items {
        let names = it
            .boxes
            .iter()
            .map(|(n, _)| n)
            .chain(it.polys.iter().map(|(n, _)| n))
            .chain(it.kpts.iter().map(|(n, _, _)| n))
            .chain(it.tag.iter());
        for name in names {
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
        let r2 = |v: f64| (v * 100.0).round() / 100.0;
        let mut anns: Vec<Annotation> = it
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
                Some(Annotation {
                    label_id: labels.get(name).copied(),
                    kind: "bbox".into(),
                    data: serde_json::json!([r2(x0), r2(y0), r2(x1 - x0), r2(y1 - y0)]),
                    score: None,
                    source: "manual".into(),
                })
            })
            .collect();
        // 分割多边形
        for (name, pc) in &it.polys {
            let pts: Vec<f64> = match pc {
                PolyCoords::Norm(v) => v.iter().enumerate().map(|(k, val)| if k % 2 == 0 { val * w } else { val * h }).collect(),
                PolyCoords::Abs(v) => v.clone(),
            };
            let pts: Vec<f64> = pts
                .chunks(2)
                .filter(|p| p.len() == 2)
                .flat_map(|p| [r2(p[0].clamp(0.0, w)), r2(p[1].clamp(0.0, h))])
                .collect();
            if pts.len() >= 6 {
                anns.push(Annotation {
                    label_id: labels.get(name).copied(),
                    kind: "polygon".into(),
                    data: serde_json::json!(pts),
                    score: None,
                    source: "manual".into(),
                });
            }
        }
        // 关键点目标：框和点都换算成像素坐标、裁进图片，点数补齐到模板
        for (name, bc, kc) in &it.kpts {
            let (x, y, bw, bh) = match *bc {
                Coords::Norm(cx, cy, nw, nh) => ((cx - nw / 2.0) * w, (cy - nh / 2.0) * h, nw * w, nh * h),
                Coords::Abs(x, y, bw, bh) => (x, y, bw, bh),
            };
            let (x0, y0) = (x.clamp(0.0, w), y.clamp(0.0, h));
            let (x1, y1) = ((x + bw).clamp(0.0, w), (y + bh).clamp(0.0, h));
            if x1 - x0 < 1.0 || y1 - y0 < 1.0 {
                continue;
            }
            if p.meta.task != "pose" {
                // 不是关键点项目：只要目标框
                anns.push(Annotation {
                    label_id: labels.get(name).copied(),
                    kind: "bbox".into(),
                    data: serde_json::json!([r2(x0), r2(y0), r2(x1 - x0), r2(y1 - y0)]),
                    score: None,
                    source: "manual".into(),
                });
                continue;
            }
            let (raw, norm) = match kc {
                PolyCoords::Norm(v) => (v, true),
                PolyCoords::Abs(v) => (v, false),
            };
            let mut kp: Vec<f64> = Vec::with_capacity(raw.len());
            for t in raw.chunks(3).filter(|t| t.len() == 3) {
                let (px, py) = if norm { (t[0] * w, t[1] * h) } else { (t[0], t[1]) };
                let v = t[2].round().clamp(0.0, 2.0);
                if v > 0.0 {
                    kp.extend([r2(px.clamp(0.0, w)), r2(py.clamp(0.0, h)), v]);
                } else {
                    kp.extend([0.0, 0.0, 0.0]);
                }
            }
            if let Some(n) = nk {
                kp.resize(n * 3, 0.0);
            }
            anns.push(Annotation {
                label_id: labels.get(name).copied(),
                kind: "keypoints".into(),
                data: serde_json::json!({ "box": [r2(x0), r2(y0), r2(x1 - x0), r2(y1 - y0)], "kpts": kp }),
                score: None,
                source: "manual".into(),
            });
        }
        // 图像分类标签
        if let Some(name) = &it.tag {
            anns.push(Annotation {
                label_id: labels.get(name).copied(),
                kind: "tag".into(),
                data: serde_json::json!([]),
                score: None,
                source: "manual".into(),
            });
        }
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
            dataset::export(&src, &ExportOptions { format: fmt, out_dir: out.clone(), only_reviewed: false, ratios: [0.75, 0.25, 0.0], copy_files: false }, &|_, _| {}).unwrap();
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
