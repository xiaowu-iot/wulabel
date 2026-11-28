//! 批量识别的截图「自带原图和识别结果」：画了框的 JPEG 后面接上不画框的原图和一段 JSON。
//! 看图软件只显示前面画了框的图；把截图导入 Wulabel 时用原图入库，识别结果写成预标注（虚线），
//! 在模型认错的基础上直接改，再训练。
//!
//! 文件结构：[画了框的 JPEG][原图 JPEG][JSON][尾巴 16 字节：b"WLSHOT01" + 原图长度 u32 LE + JSON 长度 u32 LE]
//! 原图长度为 0 表示前面那张就是原图（截图没画框）。

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::annotations::{self, Annotation};
use crate::error::AppResult;
use crate::images::{self, ImportResult};
use crate::project::Project;

const MAGIC: &[u8; 8] = b"WLSHOT01";
const TAIL: u64 = 16;

/// 一个识别结果：类别、分数，检测类有框，分割有轮廓；分类只有类别
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ShotItem {
    pub label: String,
    #[serde(default)]
    pub score: f64,
    #[serde(default, rename = "box", skip_serializing_if = "Option::is_none")]
    pub bbox: Option<[f64; 4]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub poly: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ShotInfo {
    /// 识别用的模型的任务：detect / segment / pose / classify / multilabel
    pub task: String,
    pub width: u32,
    pub height: u32,
    /// 截图来自哪个视频、第几秒
    #[serde(default)]
    pub video: String,
    #[serde(default)]
    pub time: f64,
    #[serde(default)]
    pub items: Vec<ShotItem>,
}

/// 在截图文件后面接上原图（None 表示截图本身就是原图）和识别结果
pub fn append(path: &Path, clean: Option<&[u8]>, info: &ShotInfo) -> std::io::Result<()> {
    let json = serde_json::to_vec(info).map_err(std::io::Error::other)?;
    let clean = clean.unwrap_or(&[]);
    let mut f = OpenOptions::new().append(true).open(path)?;
    f.write_all(clean)?;
    f.write_all(&json)?;
    f.write_all(MAGIC)?;
    f.write_all(&(clean.len() as u32).to_le_bytes())?;
    f.write_all(&(json.len() as u32).to_le_bytes())?;
    Ok(())
}

pub struct Shot {
    /// 不画框的原图（JPEG 字节）
    pub clean: Vec<u8>,
    pub info: ShotInfo,
}

/// 读出截图里的原图和识别结果；不是 Wulabel 批量识别的截图（或者被别的软件另存过、尾巴没了）返回 None
pub fn read(path: &Path) -> Option<Shot> {
    let mut f = File::open(path).ok()?;
    let len = f.metadata().ok()?.len();
    if len < TAIL {
        return None;
    }
    f.seek(SeekFrom::End(-(TAIL as i64))).ok()?;
    let mut tail = [0u8; TAIL as usize];
    f.read_exact(&mut tail).ok()?;
    if &tail[..8] != MAGIC {
        return None;
    }
    let clean_len = u64::from(u32::from_le_bytes(tail[8..12].try_into().ok()?));
    let json_len = u64::from(u32::from_le_bytes(tail[12..16].try_into().ok()?));
    let json_at = len.checked_sub(TAIL + json_len)?;
    let clean_at = json_at.checked_sub(clean_len)?;
    f.seek(SeekFrom::Start(json_at)).ok()?;
    let mut json = vec![0u8; json_len as usize];
    f.read_exact(&mut json).ok()?;
    let info: ShotInfo = serde_json::from_slice(&json).ok()?;
    // 原图长度为 0：前面那张就是原图
    let (start, n) = if clean_len > 0 { (clean_at, clean_len) } else { (0, json_at) };
    if n == 0 {
        return None;
    }
    f.seek(SeekFrom::Start(start)).ok()?;
    let mut clean = vec![0u8; n as usize];
    f.read_exact(&mut clean).ok()?;
    Some(Shot { clean, info })
}

/// 识别结果按项目的任务换成预标注（来源 model，图仍是未标注，等人确认）。
/// id_of：类别名 → 项目类别 id。关键点项目截图里没有关键点，不写
fn convert(task: &str, info: &ShotInfo, id_of: &dyn Fn(&str) -> Option<i64>) -> Vec<Annotation> {
    let (w, h) = (f64::from(info.width), f64::from(info.height));
    let clamp_box = |b: [f64; 4]| -> Option<[f64; 4]> {
        let (x0, y0) = (b[0].clamp(0.0, w), b[1].clamp(0.0, h));
        let (x1, y1) = ((b[0] + b[2]).clamp(0.0, w), (b[1] + b[3]).clamp(0.0, h));
        (x1 - x0 >= 1.0 && y1 - y0 >= 1.0).then_some([x0, y0, x1 - x0, y1 - y0])
    };
    let poly_box = |pts: &[f64]| -> Option<[f64; 4]> {
        let xy: Vec<&[f64]> = pts.chunks(2).filter(|c| c.len() == 2).collect();
        if xy.len() < 3 {
            return None;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for c in xy {
            x0 = x0.min(c[0]);
            x1 = x1.max(c[0]);
            y0 = y0.min(c[1]);
            y1 = y1.max(c[1]);
        }
        Some([x0, y0, x1 - x0, y1 - y0])
    };
    let ann = |label: &str, kind: &str, data: serde_json::Value, score: f64| {
        id_of(label).map(|id| Annotation { label_id: Some(id), kind: kind.into(), data, score: Some(score), source: "model".into() })
    };
    let items = &info.items;
    match task {
        "classify" => items
            .iter()
            .max_by(|a, b| a.score.total_cmp(&b.score))
            .and_then(|it| ann(&it.label, "tag", serde_json::json!([]), it.score))
            .into_iter()
            .collect(),
        "multilabel" => items.iter().filter_map(|it| ann(&it.label, "tag", serde_json::json!([]), it.score)).collect(),
        "segment" => items
            .iter()
            .filter_map(|it| {
                let pts: Vec<f64> = match (&it.poly, it.bbox) {
                    (Some(p), _) if p.len() >= 6 => {
                        p.chunks(2).filter(|c| c.len() == 2).flat_map(|c| [c[0].clamp(0.0, w), c[1].clamp(0.0, h)]).collect()
                    }
                    // 只有框：先当成矩形轮廓，人再细改
                    (_, Some(b)) => {
                        let b = clamp_box(b)?;
                        vec![b[0], b[1], b[0] + b[2], b[1], b[0] + b[2], b[1] + b[3], b[0], b[1] + b[3]]
                    }
                    _ => return None,
                };
                ann(&it.label, "polygon", serde_json::json!(pts), it.score)
            })
            .collect(),
        "detect" => items
            .iter()
            .filter_map(|it| {
                let b = clamp_box(it.bbox.or_else(|| it.poly.as_deref().and_then(|pts| poly_box(pts)))?)?;
                ann(&it.label, "bbox", serde_json::json!(b), it.score)
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// 识别结果换成当前项目的预标注：类别按名字对上项目类别，项目里没有的自动新建
pub fn to_annotations(p: &Project, info: &ShotInfo) -> AppResult<Vec<Annotation>> {
    let mut names: Vec<String> = Vec::new();
    for it in &info.items {
        if !names.contains(&it.label) {
            names.push(it.label.clone());
        }
    }
    if names.is_empty() {
        return Ok(Vec::new());
    }
    let ids = crate::training::class_map(p, &names)?;
    let id_of = |n: &str| names.iter().position(|x| x == n).and_then(|i| ids.get(i).copied());
    Ok(convert(&p.meta.task, info, &id_of))
}

fn merge(into: &mut ImportResult, r: ImportResult) {
    into.added += r.added;
    into.duplicated += r.duplicated;
    into.failed.extend(r.failed);
    into.stopped += r.stopped;
    into.prelabeled += r.prelabeled;
}

/// 导入图片（数据页导入、批量识别「加进项目去标」）：认出批量识别的截图，用里面的原图入库、
/// 识别结果写成预标注；其他图照常导入。两部分共用一个导入批次，「撤销上次导入」一起撤掉。
/// 可以用 images::STOP_IMPORT 中途停止
pub fn import(p: &Project, files: &[PathBuf], copy: bool, progress: &(dyn Fn(usize, usize) + Sync)) -> AppResult<ImportResult> {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let tmp = std::env::temp_dir().join(format!("wulabel-shots-{}-{nanos}", std::process::id()));
    let mut plain: Vec<PathBuf> = Vec::new();
    let mut shots: Vec<(PathBuf, ShotInfo)> = Vec::new();
    let mut used: HashSet<String> = HashSet::new();
    for f in files {
        let Some(shot) = read(f) else {
            plain.push(f.clone());
            continue;
        };
        // 原图写到临时文件夹（文件名照旧，重名的加序号），复制进项目后删掉
        let name = f.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "shot.jpg".into());
        let name = if used.insert(name.clone()) { name } else { format!("{}_{name}", shots.len()) };
        fs::create_dir_all(&tmp)?;
        let dst = tmp.join(&name);
        fs::write(&dst, &shot.clean)?;
        shots.push((dst, shot.info));
    }
    if shots.is_empty() {
        return images::import_batch_stoppable(p, files, copy, None, progress).map(|r| r.0);
    }
    let batch = images::next_batch(p)?;
    let (n1, n2) = (plain.len() * 2, shots.len() * 2);
    let total = n1 + n2;
    let mut result = ImportResult { added: 0, duplicated: 0, failed: Vec::new(), stopped: 0, prelabeled: 0 };
    // 立即调用的闭包：出错时也要删掉临时文件夹
    let r = (|| -> AppResult<()> {
        if !plain.is_empty() {
            let (r, _) = images::import_batch_stoppable(p, &plain, copy, Some(batch), &|d, _| progress(d, total))?;
            merge(&mut result, r);
        }
        // 原图在临时文件夹里，必须复制进项目
        let shot_files: Vec<PathBuf> = shots.iter().map(|s| s.0.clone()).collect();
        let (r, map) = images::import_batch_stoppable(p, &shot_files, true, Some(batch), &|d, _| progress(n1 + d, total))?;
        merge(&mut result, r);
        for (path, info) in &shots {
            if let Some(&id) = map.get(path) {
                let anns = to_annotations(p, info)?;
                if !anns.is_empty() && annotations::save_predictions(p, id, &anns)? {
                    result.prelabeled += 1;
                }
            }
        }
        Ok(())
    })();
    let _ = fs::remove_dir_all(&tmp);
    r.map(|_| result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info() -> ShotInfo {
        ShotInfo {
            task: "segment".into(),
            width: 100,
            height: 80,
            video: "SF1/v.mp4".into(),
            time: 12.0,
            items: vec![
                ShotItem { label: "扇叶损坏".into(), score: 0.93, bbox: Some([10.0, 10.0, 30.0, 20.0]), poly: Some(vec![10.0, 10.0, 40.0, 10.0, 40.0, 30.0]) },
                ShotItem { label: "缺失配件".into(), score: 0.7, bbox: Some([90.0, 70.0, 30.0, 30.0]), poly: None },
                ShotItem { label: "全新商品".into(), score: 0.88, bbox: None, poly: None },
            ],
        }
    }

    #[test]
    fn append_and_read_back() {
        let dir = std::env::temp_dir().join(format!("wulabel-shotfile-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        // 画了框的截图后面接原图
        let drawn = dir.join("a.jpg");
        fs::write(&drawn, b"DRAWN-JPEG").unwrap();
        append(&drawn, Some(b"CLEAN-JPEG"), &info()).unwrap();
        let s = read(&drawn).unwrap();
        assert_eq!(s.clean, b"CLEAN-JPEG");
        assert_eq!(s.info, info());
        // 没画框的截图：原图就是前面那张
        let plain = dir.join("b.jpg");
        fs::write(&plain, b"PLAIN-JPEG").unwrap();
        append(&plain, None, &info()).unwrap();
        assert_eq!(read(&plain).unwrap().clean, b"PLAIN-JPEG");
        // 普通图片、被另存过（尾巴没了）的截图：不认
        let other = dir.join("c.jpg");
        fs::write(&other, b"just a normal jpeg file").unwrap();
        assert!(read(&other).is_none());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn import_uses_clean_frame_and_prelabels() {
        use image::{Rgb, RgbImage};
        let base = std::env::temp_dir().join(format!("wulabel-shotimport-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        // 原图是蓝的，画了框的截图是红的：导入后项目里应该是蓝的那张
        let clean = base.join("clean.jpg");
        RgbImage::from_pixel(100, 80, Rgb([0, 0, 200])).save(&clean).unwrap();
        let shot = base.join("扇叶损坏_00m12s_0.93.jpg");
        RgbImage::from_pixel(100, 80, Rgb([200, 0, 0])).save(&shot).unwrap();
        let mut i = info();
        i.task = "detect".into();
        append(&shot, Some(&fs::read(&clean).unwrap()), &i).unwrap();
        let normal = base.join("normal.png");
        RgbImage::from_pixel(60, 40, Rgb([0, 200, 0])).save(&normal).unwrap();

        let p = Project::create(&base, "p", "detect").unwrap();
        let r = import(&p, &[shot.clone(), normal.clone()], true, &|_, _| {}).unwrap();
        assert_eq!((r.added, r.prelabeled), (2, 1));
        let page = images::list(&p, None, None, None, 0, 10).unwrap();
        assert_eq!(page.total, 2);
        // 截图那张：类别自动新建，框是模型预标注，图仍是未标注
        let id = page.items.iter().find(|x| x.name.contains("扇叶损坏")).unwrap().id;
        let a = annotations::load(&p, id).unwrap();
        assert_eq!(a.status, "todo");
        assert_eq!(a.annotations.len(), 2);
        assert!(a.annotations.iter().all(|x| x.source == "model" && x.kind == "bbox"));
        // 入库的是原图（蓝），不是画了框的（红）
        let path: String = p.conn.lock().query_row("SELECT path FROM images WHERE id = ?1", [id], |r| r.get(0)).unwrap();
        let stored = p.resolve(&path);
        let px = image::open(stored).unwrap().to_rgb8().get_pixel(50, 40).0;
        assert!(px[2] > 150 && px[0] < 60, "{px:?}");
        drop(p);
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn converts_by_project_task() {
        let i = info();
        let id_of = |n: &str| match n {
            "扇叶损坏" => Some(1),
            "缺失配件" => Some(2),
            "全新商品" => Some(3),
            _ => None,
        };
        // 检测：有框用框，超出图的部分裁掉；没框的分类结果不要
        let d = convert("detect", &i, &id_of);
        assert_eq!(d.len(), 2);
        assert_eq!(d[0].kind, "bbox");
        assert_eq!(d[0].data, serde_json::json!([10.0, 10.0, 30.0, 20.0]));
        assert_eq!(d[1].data, serde_json::json!([90.0, 70.0, 10.0, 10.0]));
        assert!(d.iter().all(|a| a.source == "model"));
        // 分割：有轮廓用轮廓，只有框的当矩形
        let s = convert("segment", &i, &id_of);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].kind, "polygon");
        assert_eq!(s[1].data.as_array().unwrap().len(), 8);
        // 分类取最高分的一个，多标签全要
        let c = convert("classify", &i, &id_of);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].label_id, Some(1));
        assert_eq!(convert("multilabel", &i, &id_of).len(), 3);
        // 关键点项目不写
        assert!(convert("pose", &i, &id_of).is_empty());
    }
}
