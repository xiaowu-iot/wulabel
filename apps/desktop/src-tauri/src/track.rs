//! 视频标注：关键帧插值、逐帧跟踪。同一个文件夹里的图按文件名排序，当作视频的帧序列
//! （视频抽帧、视频流采集出来的图就是这样存的）。
//!
//! 「关键帧」是有人工标注（source = manual）的框 / 关键点的帧。插值、跟踪生成的标注 source 是
//! interp / track，只写在关键帧之间的帧上：重新插值、重新跟踪时会替换掉这些帧上非人工的框，人工标的不动。

use std::cmp::Ordering;
use std::fs;

use rusqlite::OptionalExtension;
use serde::Serialize;

use crate::annotations::{self, Annotation};
use crate::db;
use crate::error::{msg, AppResult};
use crate::evaltools::worker_events;
use crate::keypoints;
use crate::project::Project;
use crate::runtime::Runtime;

const KINDS: [&str; 2] = ["bbox", "keypoints"];
/// 一次最多往后跟踪多少帧
pub const MAX_TRACK: usize = 500;

fn parent(path: &str) -> &str {
    match path.rfind(|c| c == '/' || c == '\\') {
        Some(i) => &path[..i],
        None => "",
    }
}

/// 这张图所在的帧序列：(图片 id, 存储路径)，按文件名排序；以及它在序列里的位置
fn sequence(p: &Project, image_id: i64) -> AppResult<(Vec<(i64, String)>, usize)> {
    let conn = p.conn.lock();
    let path: String = conn
        .query_row("SELECT path FROM images WHERE id = ?1", [image_id], |r| r.get(0))
        .optional()?
        .ok_or_else(|| msg(format!("图片不存在：{image_id}")))?;
    let dir = parent(&path).to_string();
    let mut stmt = conn.prepare("SELECT id, path FROM images")?;
    let mut v: Vec<(String, i64)> = stmt
        .query_map([], |r| Ok((r.get::<_, String>(1)?, r.get::<_, i64>(0)?)))?
        .flatten()
        .filter(|(pth, _)| parent(pth) == dir)
        .collect();
    v.sort();
    let seq: Vec<(i64, String)> = v.into_iter().map(|(pth, id)| (id, pth)).collect();
    let pos = seq.iter().position(|(id, _)| *id == image_id).ok_or_else(|| msg("图片不存在"))?;
    Ok((seq, pos))
}

#[derive(Debug, Clone)]
struct Obj {
    label: Option<i64>,
    kind: String,
    b: [f64; 4],
    kpts: Vec<f64>,
}

fn obj(a: &Annotation) -> Option<Obj> {
    match a.kind.as_str() {
        "bbox" => {
            let v: Vec<f64> = serde_json::from_value(a.data.clone()).ok()?;
            (v.len() == 4).then(|| Obj { label: a.label_id, kind: a.kind.clone(), b: [v[0], v[1], v[2], v[3]], kpts: Vec::new() })
        }
        "keypoints" => {
            let (b, k) = keypoints::parse(&a.data)?;
            Some(Obj { label: a.label_id, kind: a.kind.clone(), b, kpts: k })
        }
        _ => None,
    }
}

fn r1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

/// 裁到图片范围内；框基本出了画面就不要了
fn clamp(mut o: Obj, w: u32, h: u32) -> Option<Obj> {
    let (w, h) = (w as f64, h as f64);
    let x0 = o.b[0].clamp(0.0, w);
    let y0 = o.b[1].clamp(0.0, h);
    let x1 = (o.b[0] + o.b[2]).clamp(0.0, w);
    let y1 = (o.b[1] + o.b[3]).clamp(0.0, h);
    if x1 - x0 < 1.0 || y1 - y0 < 1.0 {
        return None;
    }
    o.b = [r1(x0), r1(y0), r1(x1 - x0), r1(y1 - y0)];
    for pt in o.kpts.chunks_mut(3) {
        if pt[2] > 0.0 && (pt[0] < 0.0 || pt[1] < 0.0 || pt[0] > w || pt[1] > h) {
            // 点出了画面：当作没标
            pt[0] = 0.0;
            pt[1] = 0.0;
            pt[2] = 0.0;
        } else {
            pt[0] = r1(pt[0]);
            pt[1] = r1(pt[1]);
        }
    }
    Some(o)
}

fn to_ann(o: &Obj, source: &str) -> Annotation {
    let data = if o.kind == "bbox" {
        serde_json::json!(o.b)
    } else {
        serde_json::json!({ "box": o.b, "kpts": o.kpts })
    };
    Annotation { label_id: o.label, kind: o.kind.clone(), data, score: None, source: source.into() }
}

fn iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let ix = (a[0] + a[2]).min(b[0] + b[2]) - a[0].max(b[0]);
    let iy = (a[1] + a[3]).min(b[1] + b[3]) - a[1].max(b[1]);
    if ix <= 0.0 || iy <= 0.0 {
        return 0.0;
    }
    let i = ix * iy;
    i / (a[2] * a[3] + b[2] * b[3] - i)
}

/// 两个关键帧上的目标一一对应：同类别、同类型，按重叠度 + 中心距离贪心配对；离得太远的不配
fn pair(a: &[Obj], b: &[Obj], w: u32, h: u32) -> Vec<(usize, usize)> {
    let diag = ((w as f64).powi(2) + (h as f64).powi(2)).sqrt().max(1.0);
    let mut cands = Vec::new();
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            if x.label != y.label || x.kind != y.kind || x.kpts.len() != y.kpts.len() {
                continue;
            }
            let o = iou(&x.b, &y.b);
            let dx = (x.b[0] + x.b[2] / 2.0) - (y.b[0] + y.b[2] / 2.0);
            let dy = (x.b[1] + x.b[3] / 2.0) - (y.b[1] + y.b[3] / 2.0);
            let d = (dx * dx + dy * dy).sqrt() / diag;
            if o == 0.0 && d > 0.35 {
                continue;
            }
            cands.push((1.0 - o + d, i, j));
        }
    }
    cands.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(Ordering::Equal));
    let (mut ua, mut ub) = (vec![false; a.len()], vec![false; b.len()]);
    let mut out = Vec::new();
    for (_, i, j) in cands {
        if !ua[i] && !ub[j] {
            ua[i] = true;
            ub[j] = true;
            out.push((i, j));
        }
    }
    out
}

fn lerp(a: &Obj, b: &Obj, t: f64) -> Obj {
    let l = |x: f64, y: f64| x + (y - x) * t;
    let bx = [l(a.b[0], b.b[0]), l(a.b[1], b.b[1]), l(a.b[2], b.b[2]), l(a.b[3], b.b[3])];
    let kpts = a
        .kpts
        .chunks(3)
        .zip(b.kpts.chunks(3))
        .flat_map(|(p, q)| {
            if p[2] > 0.0 && q[2] > 0.0 {
                [l(p[0], q[0]), l(p[1], q[1]), if t < 0.5 { p[2] } else { q[2] }]
            } else {
                // 只在一头标了的点，中间不知道在哪，不标
                [0.0, 0.0, 0.0]
            }
        })
        .collect();
    Obj { label: a.label, kind: a.kind.clone(), b: bx, kpts }
}

fn manual_obj(a: &Annotation) -> bool {
    KINDS.contains(&a.kind.as_str()) && a.source == "manual"
}

/// 帧上除了「非人工的框 / 关键点」以外的标注（插值、跟踪写之前先去掉旧的自动结果）
fn keep(anns: Vec<Annotation>) -> Vec<Annotation> {
    anns.into_iter().filter(|a| !KINDS.contains(&a.kind.as_str()) || a.source == "manual").collect()
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InterpResult {
    /// 上一个关键帧
    pub from_id: i64,
    /// 中间写了多少帧
    pub frames: usize,
    /// 插值的目标数
    pub objects: usize,
    /// 两头对不上的目标数（只在一头出现，没插值）
    pub unmatched: usize,
}

/// 从上一个关键帧插值到这一帧（这一帧要先标好）
pub fn interpolate(p: &Project, image_id: i64) -> AppResult<InterpResult> {
    let (seq, pos) = sequence(p, image_id)?;
    let cur = annotations::load(p, image_id)?;
    let b: Vec<Obj> = cur.annotations.iter().filter(|a| KINDS.contains(&a.kind.as_str())).filter_map(obj).collect();
    if b.is_empty() {
        return Err(msg("这一帧还没有框：先标好这一帧，再从上一个关键帧插值过来"));
    }
    let mut from = None;
    for i in (0..pos).rev() {
        let m = annotations::load(p, seq[i].0)?;
        if m.annotations.iter().any(manual_obj) {
            from = Some((i, m));
            break;
        }
    }
    let (ia, prev) = from.ok_or_else(|| msg("往前没找到关键帧：先在前面某一帧手动标好，再标这一帧，然后插值"))?;
    if pos - ia < 2 {
        return Err(msg("上一个关键帧就是前一帧，中间没有要插值的帧"));
    }
    let a: Vec<Obj> = prev.annotations.iter().filter(|x| KINDS.contains(&x.kind.as_str())).filter_map(obj).collect();
    let pairs = pair(&a, &b, cur.width, cur.height);
    if pairs.is_empty() {
        return Err(msg("两个关键帧上的目标对不上（要同一个类别、位置别差太远）"));
    }
    let n = pos - ia;
    for k in ia + 1..pos {
        let t = (k - ia) as f64 / n as f64;
        let m = annotations::load(p, seq[k].0)?;
        let mut anns = keep(m.annotations);
        for (i, j) in &pairs {
            if let Some(o) = clamp(lerp(&a[*i], &b[*j], t), m.width, m.height) {
                anns.push(to_ann(&o, "interp"));
            }
        }
        annotations::save(p, seq[k].0, &anns)?;
    }
    Ok(InterpResult {
        from_id: seq[ia].0,
        frames: n - 1,
        objects: pairs.len(),
        unmatched: a.len().max(b.len()) - pairs.len(),
    })
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrackResult {
    /// 写了多少帧
    pub frames: usize,
    pub objects: usize,
    /// 中途跟丢的目标数
    pub lost: usize,
    /// 碰到后面已经手动标过的帧停下
    pub hit_keyframe: bool,
    /// 跟踪到的最后一帧
    pub last_id: Option<i64>,
}

/// 从这一帧的框出发往后跟踪 count 帧。碰到手动标过的帧就停，不覆盖人工标注
pub fn track(
    p: &Project,
    rt: &Runtime,
    image_id: i64,
    count: usize,
    log: &(dyn Fn(String) + Sync),
    progress: &(dyn Fn(usize, usize) + Sync),
) -> AppResult<TrackResult> {
    let (seq, pos) = sequence(p, image_id)?;
    let cur = annotations::load(p, image_id)?;
    let objs: Vec<Obj> = cur.annotations.iter().filter(|a| KINDS.contains(&a.kind.as_str())).filter_map(obj).collect();
    if objs.is_empty() {
        return Err(msg("这一帧还没有框，先标好要跟踪的目标"));
    }
    let mut frames = Vec::new();
    let mut hit_keyframe = false;
    for k in pos + 1..(pos + 1 + count.clamp(1, MAX_TRACK)).min(seq.len()) {
        if annotations::load(p, seq[k].0)?.annotations.iter().any(manual_obj) {
            hit_keyframe = true;
            break;
        }
        frames.push(k);
    }
    if frames.is_empty() {
        return Err(msg(if hit_keyframe { "下一帧已经手动标过了" } else { "后面没有帧了（同一文件夹里按文件名排序）" }));
    }
    let paths: Vec<String> = std::iter::once(pos)
        .chain(frames.iter().copied())
        .map(|k| p.resolve(&seq[k].1).to_string_lossy().into_owned())
        .collect();
    let boxes: Vec<[f64; 4]> = objs.iter().map(|o| o.b).collect();
    let cfg_path = std::env::temp_dir().join(format!("wulabel-track-{}-{}.json", std::process::id(), db::now()));
    fs::write(&cfg_path, serde_json::to_string(&serde_json::json!({ "frames": paths, "boxes": boxes }))?)?;
    let cfg_arg = cfg_path.to_string_lossy().to_string();
    let total = frames.len();
    let mut results: Vec<Vec<Option<[f64; 4]>>> = Vec::new();
    let mut lost = 0;
    progress(0, total);
    let r = worker_events(rt, &["track", cfg_arg.as_str()], &mut |v| match v["type"].as_str() {
        Some("frame") => {
            let bs: Vec<Option<[f64; 4]>> = serde_json::from_value(v["boxes"].clone()).unwrap_or_default();
            results.push(bs);
            progress(results.len(), total);
        }
        Some("log") => log(v["msg"].as_str().unwrap_or("").to_string()),
        Some("done") => lost = v["lost"].as_u64().unwrap_or(0) as usize,
        _ => {}
    }, None);
    let _ = fs::remove_file(&cfg_path);
    r?;
    let mut written = 0;
    let mut last_id = None;
    for (i, bs) in results.iter().enumerate() {
        let Some(&k) = frames.get(i) else { break };
        let id = seq[k].0;
        let m = annotations::load(p, id)?;
        let mut anns = keep(m.annotations);
        let mut any = false;
        for (o, nb) in objs.iter().zip(bs.iter()) {
            let Some(nb) = nb else { continue };
            // 关键点跟着框平移、缩放
            let (cx0, cy0) = (o.b[0] + o.b[2] / 2.0, o.b[1] + o.b[3] / 2.0);
            let (cx1, cy1) = (nb[0] + nb[2] / 2.0, nb[1] + nb[3] / 2.0);
            let (sx, sy) = (nb[2] / o.b[2].max(1.0), nb[3] / o.b[3].max(1.0));
            let mut kpts = o.kpts.clone();
            for pt in kpts.chunks_mut(3) {
                if pt[2] > 0.0 {
                    pt[0] = cx1 + (pt[0] - cx0) * sx;
                    pt[1] = cy1 + (pt[1] - cy0) * sy;
                }
            }
            let moved = Obj { label: o.label, kind: o.kind.clone(), b: *nb, kpts };
            if let Some(c) = clamp(moved, m.width, m.height) {
                anns.push(to_ann(&c, "track"));
                any = true;
            }
        }
        // 全部跟丢就不再往后写
        if !any {
            break;
        }
        annotations::save(p, id, &anns)?;
        written += 1;
        last_id = Some(id);
    }
    Ok(TrackResult { frames: written, objects: objs.len(), lost, hit_keyframe, last_id })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn interpolate_between_keyframes() {
        let base = std::env::temp_dir().join(format!("xlabel-test-interp-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let src = base.join("frames");
        fs::create_dir_all(&src).unwrap();
        // 5 帧，颜色不同免得被当成重复图
        for i in 0..5u8 {
            RgbImage::from_pixel(100, 100, Rgb([i * 40, 10, 10])).save(src.join(format!("v_{i:06}.png"))).unwrap();
        }
        let p = Project::create(&base, "插值", "detect").unwrap();
        crate::images::import(&p, &crate::images::collect(&[src]), true, &|_, _| {}).unwrap();
        let lid = annotations::add_label(&p, "车", "#ff0000").unwrap().id;
        let (seq, _) = sequence(&p, 1).unwrap();
        let ids: Vec<i64> = seq.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids.len(), 5);
        let bx = |x: f64| Annotation { label_id: Some(lid), kind: "bbox".into(), data: serde_json::json!([x, 10.0, 20.0, 20.0]), score: None, source: "manual".into() };
        annotations::save(&p, ids[0], &[bx(0.0)]).unwrap();
        annotations::save(&p, ids[4], &[bx(40.0)]).unwrap();

        let r = interpolate(&p, ids[4]).unwrap();
        assert_eq!((r.from_id, r.frames, r.objects, r.unmatched), (ids[0], 3, 1, 0));
        let mid = annotations::load(&p, ids[2]).unwrap().annotations;
        assert_eq!(mid.len(), 1);
        assert_eq!(mid[0].source, "interp");
        assert_eq!(mid[0].data, serde_json::json!([20.0, 10.0, 20.0, 20.0]));

        // 再插一次：替换旧的插值结果，不会叠加
        interpolate(&p, ids[4]).unwrap();
        assert_eq!(annotations::load(&p, ids[1]).unwrap().annotations.len(), 1);

        // 只有一个关键帧时报错
        assert!(interpolate(&p, ids[0]).is_err());
        let _ = fs::remove_dir_all(&base);
    }
}
