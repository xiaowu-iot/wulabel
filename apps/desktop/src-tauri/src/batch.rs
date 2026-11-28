//! 批量识别一个文件夹：用训练好的模型把文件夹里的图片和视频都看一遍，
//! 结果导出成表格（Excel 能直接打开），也可以按识别出的类别把文件复制到各自的文件夹。
//! 视频每隔几秒抽一帧来识别，只看一遍；识别出的每个类别存一张分数最高的截图，
//! 按视频所在的子文件夹（比如按单号分的）放好。

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex};

use serde::{Deserialize, Serialize};

use crate::batch_report::{self, Record};
use crate::dataset::safe_dir;
use crate::error::{msg, AppResult};
use crate::project::Project;
use crate::runtime::Runtime;
use crate::t;
use crate::training::{self, Predictor, RawPred};
use crate::video::{self, Extractor};

static STOP: AtomicBool = AtomicBool::new(false);
/// 视频抽帧的 ffmpeg，点停止时杀掉
static EXTRACTOR: LazyLock<Extractor> = LazyLock::new(Extractor::default);

/// 最近一次批量识别的视频截图。界面上「复核」宫格按序号取图（xl://…/review/{序号}），前端不能借此读任意路径
static REVIEW: LazyLock<Mutex<Vec<PathBuf>>> = LazyLock::new(Default::default);

/// 复核宫格第 i 张截图的路径
pub fn review_file(i: usize) -> Option<PathBuf> {
    REVIEW.lock().ok()?.get(i).cloned()
}

/// 让正在跑的批量识别在这一批结束后停下（已经识别的照样写进表格）
pub fn stop() {
    STOP.store(true, Ordering::Relaxed);
    EXTRACTOR.stop();
}

fn default_every() -> f64 {
    1.0
}

fn default_min_frames() -> usize {
    1
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchOptions {
    pub run_id: i64,
    /// 要识别的文件夹（包括子文件夹）
    pub input: String,
    /// 结果放到这里
    pub output: String,
    pub conf: f64,
    /// 按类别把图片、视频复制到各自的文件夹
    pub sort: bool,
    /// 多标签分类：每个类别单独的阈值（类别名 → 阈值），没写的类别用 conf
    #[serde(default)]
    pub class_conf: HashMap<String, f64>,
    /// 视频每隔几秒看一帧
    #[serde(default = "default_every")]
    pub video_every: f64,
    /// 视频里一个类别至少几帧检出才算有
    #[serde(default = "default_min_frames")]
    pub video_min_frames: usize,
    /// 视频截图放回视频所在的文件夹（子文件夹「识别截图」），不放结果文件夹
    #[serde(default)]
    pub shots_beside: bool,
    /// 截图上画框和类别；关掉就存原图。不管画不画，截图文件里都带着原图和识别结果（见 shotfile）
    #[serde(default = "yes")]
    pub draw_boxes: bool,
    /// 跳过上次已经识别过、之后没改过的文件（结果文件夹里记着），表格里沿用上次的结果
    #[serde(default)]
    pub skip_done: bool,
}

/// 最高分低于「阈值 + 这个数」的结果单独列出来：刚过阈值的最容易认错，最值得人看一眼
const LOW_MARGIN: f64 = 0.15;

fn yes() -> bool {
    true
}

/// 结果文件夹里放一个这个文件，下次识别时整个文件夹跳过（结果可以放在要识别的文件夹里面）
const MARK: &str = ".wulabel-result";

fn mark_dir(dir: &Path) {
    if fs::create_dir_all(dir).is_ok() && !dir.join(MARK).exists() {
        let _ = fs::write(dir.join(MARK), b"Wulabel batch recognition results\n");
    }
}

/// 文件在不在以前的结果文件夹里（从文件往上找到要识别的文件夹为止，不含它本身）
fn in_result_dir(input: &Path, file: &Path, cache: &mut HashMap<PathBuf, bool>) -> bool {
    let mut p = file.parent();
    while let Some(d) = p {
        if d == input || !d.starts_with(input) {
            break;
        }
        let hit = *cache.entry(d.to_path_buf()).or_insert_with(|| d.join(MARK).exists());
        if hit {
            return true;
        }
        p = d.parent();
    }
    false
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchResult {
    pub images: usize,
    pub videos: usize,
    /// 视频一共看了多少帧
    pub frames: usize,
    /// 读不了、识别出错的图片和视频
    pub failed: usize,
    /// 表格文件
    pub csv: String,
    pub output: String,
    /// 每个类别的文件数（按个数从多到少），最后是没有结果的
    pub counts: Vec<(String, usize)>,
    /// 拿不准（检测是没有目标）的图，可以加进项目去标
    pub uncertain: Vec<String>,
    pub stopped: bool,
    /// 视频截图的路径，「加进项目去标」用（导入时用里面的原图，识别结果当预标注）
    pub shots: Vec<String>,
    /// 什么都没识别出来的视频（相对要识别的文件夹）：可能有模型没学过的新情况
    pub unrecognized: Vec<String>,
    /// 这些视频里均匀取的几帧原图，「加进项目去标」用
    pub unrecognized_frames: Vec<String>,
    /// 上次识别过、没改过，这次跳过的文件数
    pub skipped: usize,
    /// 分数偏低（刚过阈值）的图片和视频截图：最容易认错，优先复核
    pub low: Vec<String>,
    /// 按单号汇总的表格、网页报告
    pub summary: String,
    pub report: String,
}

/// 没识别出东西的视频，均匀取几帧（第 1/4、1/2、3/4 处）
fn sample_frames(n: usize) -> Vec<usize> {
    let mut ks: Vec<usize> = [n / 4, n / 2, n * 3 / 4].into_iter().filter(|&k| k < n).collect();
    ks.dedup();
    ks
}

/// 一张图识别出的类别：(类别序号, 最高分, 个数)。分类每个类别个数是 1；视频的个数是检出的帧数
pub type Hits = Vec<(usize, f64, usize)>;

/// 表格的一行：一张图或一个视频；hits 是 None 表示识别出错
pub struct Row {
    pub file: PathBuf,
    pub hits: Option<Hits>,
    pub video: bool,
    /// 视频每个识别出的类别存的截图：(类别序号, 截图路径)
    pub shots: Vec<(usize, PathBuf)>,
}

/// 把一张图的预测结果按任务整理成类别列表（按分数从高到低）。
/// 单标签分类只取最高的一个；多标签分类取分数够的全部（class_conf 是按类别序号的单独阈值，可以为空）；
/// 检测类数每个类别有几个目标
pub fn summarize(raw: &RawPred, task: &str, conf: f64, class_conf: &[f64]) -> Hits {
    let mut hits: Hits = Vec::new();
    match task {
        "classify" => {
            if let Some(&(c, s)) = raw.tags.iter().max_by(|a, b| a.1.total_cmp(&b.1)) {
                if s >= conf {
                    hits.push((c, s, 1));
                }
            }
        }
        "multilabel" => {
            let thr = |c: usize| class_conf.get(c).copied().unwrap_or(conf);
            hits = raw.tags.iter().filter(|t| t.1 >= thr(t.0)).map(|&(c, s)| (c, s, 1)).collect();
        }
        _ => {
            let objs = raw
                .boxes
                .iter()
                .map(|b| (b[4] as usize, b[5]))
                .chain(raw.polys.iter().map(|p| (p.0, p.1)))
                .chain(raw.poses.iter().map(|p| (p.0, p.1)));
            let mut by: BTreeMap<usize, (f64, usize)> = BTreeMap::new();
            for (c, s) in objs.filter(|o| o.1 >= conf) {
                let e = by.entry(c).or_insert((0.0, 0));
                e.0 = e.0.max(s);
                e.1 += 1;
            }
            hits = by.into_iter().map(|(c, (s, n))| (c, s, n)).collect();
        }
    }
    hits.sort_by(|a, b| b.1.total_cmp(&a.1));
    hits
}

/// 视频每一帧的结果汇总成整段视频的结果：一个类别至少 min_frames 帧检出才算有。
/// 返回 (类别序号, 最高分, 检出帧数)，以及每个类别分数最高的是第几帧
pub fn aggregate(frames: &[Hits], min_frames: usize) -> (Hits, HashMap<usize, usize>) {
    let mut by: BTreeMap<usize, (f64, usize, usize)> = BTreeMap::new();
    for (k, hits) in frames.iter().enumerate() {
        for &(c, s, _) in hits {
            let e = by.entry(c).or_insert((f64::MIN, 0, k));
            e.1 += 1;
            if s > e.0 {
                e.0 = s;
                e.2 = k;
            }
        }
    }
    let mut hits: Hits = Vec::new();
    let mut best = HashMap::new();
    for (c, (s, n, k)) in by {
        if n >= min_frames.max(1) {
            hits.push((c, s, n));
            best.insert(c, k);
        }
    }
    hits.sort_by(|a, b| b.1.total_cmp(&a.1));
    (hits, best)
}

/// 子文件夹里的图拍平成一个文件名：a/b/c.jpg → a_b_c.jpg
fn flat_name(input: &Path, file: &Path) -> String {
    let rel = file.strip_prefix(input).unwrap_or(file);
    let parts: Vec<String> = rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
    parts.join("_")
}

/// 文件所在的子文件夹（相对要识别的文件夹，用 / 分隔），在最外层是空的
fn rel_folder(input: &Path, file: &Path) -> String {
    let rel = file.strip_prefix(input).unwrap_or(file);
    rel.parent().map(|p| p.to_string_lossy().replace('\\', "/")).unwrap_or_default()
}

fn stem(file: &Path) -> String {
    file.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

/// 视频截图放的文件夹：视频在子文件夹里（比如按单号分的）就用同样的子文件夹，在最外层就用视频文件名
fn shot_dir(input: &Path, out: &Path, file: &Path) -> PathBuf {
    let rel = file.strip_prefix(input).unwrap_or(file);
    match rel.parent().filter(|p| !p.as_os_str().is_empty()) {
        Some(p) => p.components().fold(out.to_path_buf(), |d, c| d.join(safe_dir(&c.as_os_str().to_string_lossy()))),
        None => out.join(safe_dir(&stem(file))),
    }
}

/// 秒数写成 01m05s，放在截图文件名里
fn mmss(sec: f64) -> String {
    let s = sec.max(0.0).round() as u64;
    format!("{:02}m{:02}s", s / 60, s % 60)
}

/// 截图上要画的东西：检测类是每个目标的框（分割是轮廓）、类别、分数、颜色；分类是这一帧识别出的类别（写在左上角）
fn shot_items(
    frame: Option<&(Hits, RawPred)>,
    task: &str,
    detect: bool,
    conf: f64,
    name_of: &dyn Fn(usize) -> String,
    colors: &HashMap<String, String>,
) -> Vec<serde_json::Value> {
    let Some((hits, raw)) = frame else { return Vec::new() };
    let item = |c: usize, s: f64| {
        let name = name_of(c);
        serde_json::json!({ "label": name, "score": s, "color": colors.get(&name) })
    };
    if !detect {
        return hits.iter().map(|&(c, s, _)| item(c, s)).collect();
    }
    if task == "segment" && !raw.polys.is_empty() {
        return raw
            .polys
            .iter()
            .filter(|p| p.1 >= conf)
            .map(|(c, s, pts)| {
                let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
                for xy in pts.chunks(2).filter(|xy| xy.len() == 2) {
                    x0 = x0.min(xy[0]);
                    x1 = x1.max(xy[0]);
                    y0 = y0.min(xy[1]);
                    y1 = y1.max(xy[1]);
                }
                let mut v = item(*c, *s);
                v["box"] = serde_json::json!([x0, y0, x1 - x0, y1 - y0]);
                v["poly"] = serde_json::json!(pts);
                v
            })
            .collect();
    }
    raw.boxes
        .iter()
        .filter(|b| b[5] >= conf)
        .map(|b| {
            let mut v = item(b[4] as usize, b[5]);
            v["box"] = serde_json::json!([b[0], b[1], b[2], b[3]]);
            v
        })
        .collect()
}

/// 存一张视频截图（推理进程画不了时的备用）：检测类模型画上框（红色），分类模型直接复制
fn save_shot(src: &Path, dst: &Path, boxes: &[[f64; 4]]) -> AppResult<()> {
    if boxes.is_empty() {
        fs::copy(src, dst)?;
        return Ok(());
    }
    let mut img = image::open(src).map_err(|e| msg(e.to_string()))?.to_rgb8();
    let (w, h) = img.dimensions();
    let th = i64::from((w.max(h) / 400).max(2));
    let (wi, hi) = (i64::from(w), i64::from(h));
    let red = image::Rgb([230u8, 40, 40]);
    for b in boxes {
        let (x0, y0) = (b[0].round() as i64, b[1].round() as i64);
        let (x1, y1) = ((b[0] + b[2]).round() as i64, (b[1] + b[3]).round() as i64);
        for y in y0.max(0)..=y1.min(hi - 1) {
            for x in x0.max(0)..=x1.min(wi - 1) {
                if x - x0 < th || x1 - x < th || y - y0 < th || y1 - y < th {
                    img.put_pixel(x as u32, y as u32, red);
                }
            }
        }
    }
    img.save_with_format(dst, image::ImageFormat::Jpeg).map_err(|e| msg(e.to_string()))
}

/// 按类别复制图片和视频，整理出每个文件的记录（表格、汇总、网页报告由 batch_report 写）
pub fn write_outputs(
    input: &Path,
    out: &Path,
    classes: &[String],
    rows: &[Row],
    sort: bool,
    detect: bool,
) -> AppResult<(Vec<Record>, Vec<(String, usize)>)> {
    fs::create_dir_all(out)?;
    let name_of = |c: usize| classes.get(c).cloned().unwrap_or_else(|| c.to_string());
    let none = if detect { t!("没有目标") } else { t!("拿不准") };
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut empty = 0;
    let mut records = Vec::new();
    for row in rows {
        let file = &row.file;
        let (size, mtime) = batch_report::stamp(file);
        records.push(Record {
            rel: file.strip_prefix(input).unwrap_or(file).to_string_lossy().replace('\\', "/"),
            folder: rel_folder(input, file),
            video: row.video,
            size,
            mtime,
            hits: row.hits.as_ref().map(|h| h.iter().map(|&(c, s, n)| (name_of(c), s, n)).collect()),
            shots: row.shots.iter().map(|s| s.1.to_string_lossy().into_owned()).collect(),
            path: file.to_string_lossy().into_owned(),
        });
        let Some(h) = &row.hits else { continue };
        if h.is_empty() {
            empty += 1;
        }
        for &(c, _, _) in h {
            *counts.entry(name_of(c)).or_default() += 1;
        }
        if sort && row.video {
            // 视频复制的是每个类别的截图，名字前面加上单号（所在子文件夹，最外层的用视频名）
            let folder = rel_folder(input, file).replace('/', "_");
            let tag = if folder.is_empty() { stem(file) } else { folder };
            for (c, shot) in &row.shots {
                let dir = out.join(safe_dir(&name_of(*c)));
                fs::create_dir_all(&dir)?;
                let name = shot.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                let _ = fs::copy(shot, dir.join(format!("{}_{name}", safe_dir(&tag))));
            }
        } else if sort {
            let dirs: Vec<String> = if h.is_empty() { vec![none.clone()] } else { h.iter().map(|&(c, _, _)| name_of(c)).collect() };
            for d in dirs {
                let dir = out.join(safe_dir(&d));
                fs::create_dir_all(&dir)?;
                fs::copy(file, dir.join(flat_name(input, file)))
                    .map_err(|e| msg(t!("复制图片失败 {path}：{e}", path = file.display(), e = e)))?;
            }
        }
    }
    let mut list: Vec<(String, usize)> = counts.into_iter().collect();
    list.sort_by(|a, b| b.1.cmp(&a.1));
    if empty > 0 {
        list.push((none, empty));
    }
    Ok((records, list))
}

/// 识别用到的东西：模型、阈值，以及模型返回的类别名
struct Ctx<'a> {
    rt: &'a Runtime,
    pr: &'a Predictor,
    weights: PathBuf,
    task: String,
    req_conf: f64,
    opts: &'a BatchOptions,
    classes: Vec<String>,
}

impl Ctx<'_> {
    /// 一批图片送去识别，返回每张的类别汇总和原始结果（没拿到结果的是 None）；点了停止就只返回识别过的部分
    fn predict(&mut self, paths: &[PathBuf], on_done: &mut dyn FnMut(usize)) -> AppResult<Vec<Option<(Hits, RawPred)>>> {
        let items: Vec<(i64, PathBuf)> = paths.iter().enumerate().map(|(i, f)| (i as i64, f.clone())).collect();
        let mut out: Vec<Option<(Hits, RawPred)>> = Vec::new();
        for chunk in items.chunks(16) {
            if STOP.load(Ordering::Relaxed) {
                break;
            }
            let mut got: Vec<(i64, RawPred)> = Vec::new();
            self.classes = self.pr.request(self.rt, &self.weights, chunk, self.req_conf, &mut |id, r| got.push((id, r)))?;
            let cc: Vec<f64> = self.classes.iter().map(|n| self.opts.class_conf.get(n).copied().unwrap_or(self.opts.conf)).collect();
            out.resize_with(out.len() + chunk.len(), || None);
            for (id, raw) in got {
                if let Some(slot) = out.get_mut(id as usize) {
                    *slot = Some((summarize(&raw, &self.task, self.opts.conf, &cc), raw));
                }
            }
            on_done(chunk.len());
        }
        Ok(out)
    }
}

pub fn run(p: &Project, rt: &Runtime, pr: &Predictor, opts: &BatchOptions, progress: &(dyn Fn(usize, usize) + Sync)) -> AppResult<BatchResult> {
    STOP.store(false, Ordering::Relaxed);
    EXTRACTOR.reset();
    let input = PathBuf::from(&opts.input);
    let out = PathBuf::from(&opts.output);
    if !input.is_dir() {
        return Err(msg(t!("请选择要识别的文件夹")));
    }
    if opts.output.trim().is_empty() {
        return Err(msg(t!("请选择结果放在哪个文件夹")));
    }
    // 结果文件夹可以放在要识别的文件夹里面：做上记号，下次识别时整个跳过
    let canon = |p: &Path| fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    fs::create_dir_all(&out)?;
    if canon(&out) == canon(&input) {
        return Err(msg(t!("结果文件夹不能就是要识别的文件夹，请另选一个（可以是它里面的子文件夹）")));
    }
    mark_dir(&out);
    let mut skip_cache = HashMap::new();
    let mut files = crate::images::collect(&[input.clone()]);
    files.retain(|f| !in_result_dir(&input, f, &mut skip_cache));
    let mut videos = video::collect(&[input.clone()]);
    videos.retain(|f| !in_result_dir(&input, f, &mut skip_cache));
    // 跳过上次识别过、之后没改过的文件（识别出错的照样重来）
    let prev = if opts.skip_done { batch_report::load(&out) } else { batch_report::Saved::default() };
    let done_before: HashMap<&str, &Record> = prev.records.iter().filter(|r| r.hits.is_some()).map(|r| (r.rel.as_str(), r)).collect();
    let is_done = |f: &PathBuf| {
        let rel = f.strip_prefix(&input).unwrap_or(f).to_string_lossy().replace('\\', "/");
        done_before.get(rel.as_str()).is_some_and(|r| (r.size, r.mtime) == batch_report::stamp(f))
    };
    let before = files.len() + videos.len();
    files.retain(|f| !is_done(f));
    videos.retain(|f| !is_done(f));
    let skipped = before - files.len() - videos.len();
    // 项目里类别的颜色，截图上画框用
    let colors: HashMap<String, String> =
        crate::annotations::list_labels(p).map(|ls| ls.into_iter().map(|l| (l.name, l.color)).collect()).unwrap_or_default();
    if files.is_empty() && videos.is_empty() && skipped == 0 {
        return Err(msg(t!("这个文件夹里没有图片或视频")));
    }
    let task = p.meta.task.as_str();
    let detect = !crate::project::is_classify(task);
    let mut ctx = Ctx {
        rt,
        pr,
        weights: training::weights_of(p, opts.run_id)?,
        task: task.to_string(),
        // 分类模型返回前几名，由这里按阈值取舍；检测类让推理进程直接按阈值过滤
        req_conf: if detect { opts.conf } else { 0.01 },
        opts,
        classes: Vec::new(),
    };
    let every = if opts.video_every > 0.0 { opts.video_every } else { 1.0 };
    let fps = 1.0 / every;
    // 视频要用 ffmpeg 抽帧（第一次用会下载），先估算每个视频的帧数给进度条用
    let ffmpeg = if videos.is_empty() {
        None
    } else {
        Some(rt.ffmpeg(&|_| {}, &|_f, done, total, speed| {
            crate::dlprogress::forward(&serde_json::json!({
                "type": "download", "name": "ffmpeg", "done": done, "total": total, "speed": speed,
                "finished": total > 0 && done >= total,
            }))
        })?)
    };
    let est: Vec<usize> = videos
        .iter()
        .map(|v| ffmpeg.as_ref().and_then(|f| video::duration(f, v)).map(|d| ((d * fps).ceil() as usize).max(1)).unwrap_or(1))
        .collect();
    let total = files.len() + est.iter().sum::<usize>();
    let mut done = 0usize;
    progress(0, total);

    // 1. 图片
    let res = ctx.predict(&files, &mut |n| {
        done += n;
        progress(done, total);
    })?;
    let mut rows: Vec<Row> = files.iter().zip(res).map(|(f, r)| Row { file: f.clone(), hits: r.map(|x| x.0), video: false, shots: Vec::new() }).collect();
    let images = rows.len();
    let mut stopped = STOP.load(Ordering::Relaxed);

    // 2. 视频：抽帧到临时文件夹，一帧帧识别，汇总成一行；每个类别存一张分数最高的截图
    let mut frames = 0usize;
    let mut video_count = 0usize;
    let mut unrecognized: Vec<String> = Vec::new();
    let mut unrecognized_frames: Vec<String> = Vec::new();
    if let Some(ff) = ffmpeg.as_ref().filter(|_| !stopped) {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let tmp = std::env::temp_dir().join(format!("wulabel-batch-{}-{nanos}", std::process::id()));
        // 同一个文件夹里不止一个视频时，截图名前面加上视频名
        // 截图放结果文件夹（按子文件夹），或者放回视频所在文件夹的「识别截图」里
        let beside = t!("识别截图");
        let dir_of = |v: &Path| {
            if opts.shots_beside {
                v.parent().unwrap_or(&input).join(safe_dir(&beside))
            } else {
                shot_dir(&input, &out, v)
            }
        };
        let mut per_dir: HashMap<PathBuf, usize> = HashMap::new();
        for v in &videos {
            *per_dir.entry(dir_of(v)).or_default() += 1;
        }
        for (i, v) in videos.iter().enumerate() {
            if STOP.load(Ordering::Relaxed) {
                stopped = true;
                break;
            }
            let dir = tmp.join(format!("v{i}"));
            fs::create_dir_all(&dir)?;
            let base = done;
            let mut paths = match video::extract_one(ff, v, &dir, i, fps, 0, &EXTRACTOR, &|_| {}) {
                Ok(_) => crate::images::collect(&[dir.clone()]),
                Err(_) => Vec::new(),
            };
            paths.sort();
            let res = if paths.is_empty() || STOP.load(Ordering::Relaxed) {
                Vec::new()
            } else {
                ctx.predict(&paths, &mut |n| {
                    done += n;
                    progress(done.min(total), total);
                })?
            };
            if STOP.load(Ordering::Relaxed) {
                stopped = true;
                let _ = fs::remove_dir_all(&dir);
                break;
            }
            // 估算的帧数和实际不一样时，进度按估算的算，免得超过 100%
            done = base + est[i];
            progress(done.min(total), total);
            let mut shots: Vec<(usize, PathBuf)> = Vec::new();
            let hits = if res.is_empty() {
                None
            } else {
                frames += res.len();
                let per_frame: Vec<Hits> = res.iter().map(|r| r.as_ref().map(|x| x.0.clone()).unwrap_or_default()).collect();
                let (hits, best) = aggregate(&per_frame, opts.video_min_frames);
                let sd = dir_of(v);
                let prefix = if per_dir.get(&sd).copied().unwrap_or(0) > 1 { format!("{}_", stem(v)) } else { String::new() };
                if !hits.is_empty() {
                    if opts.shots_beside {
                        mark_dir(&sd);
                    } else {
                        fs::create_dir_all(&sd)?;
                    }
                }
                let name_of = |c: usize| ctx.classes.get(c).cloned().unwrap_or_else(|| c.to_string());
                for &(c, s, _) in &hits {
                    let k = best[&c];
                    let dst = sd.join(format!("{prefix}{}_{}_{s:.2}.jpg", safe_dir(&name_of(c)), mmss(k as f64 * every)));
                    let items = shot_items(res[k].as_ref(), task, detect, opts.conf, &name_of, &colors);
                    let _ = fs::remove_file(&dst);
                    // 像标注页一样画框、写类别和分数（推理进程画，能显示中文）；画不了就只画框。不画框就存原图
                    let drawn = opts.draw_boxes && {
                        let req = serde_json::json!({ "cmd": "draw", "src": &paths[k], "dst": &dst, "items": &items });
                        pr.call(rt, &req, &mut |_| {}).is_ok() && dst.exists()
                    };
                    if !opts.draw_boxes {
                        let _ = fs::copy(&paths[k], &dst);
                    } else if !drawn {
                        let boxes: Vec<[f64; 4]> = match (&res[k], detect) {
                            (Some((_, raw)), true) => {
                                raw.boxes.iter().filter(|b| b[5] >= opts.conf).map(|b| [b[0], b[1], b[2], b[3]]).collect()
                            }
                            _ => Vec::new(),
                        };
                        // 截图存不了不影响表格
                        let _ = save_shot(&paths[k], &dst, &boxes);
                    }
                    if dst.exists() {
                        // 截图文件后面接上原图和识别结果：拖回 Wulabel 时用原图入库，识别结果当预标注
                        let (width, height) = image::image_dimensions(&paths[k]).unwrap_or((0, 0));
                        let info = crate::shotfile::ShotInfo {
                            task: task.to_string(),
                            width,
                            height,
                            video: v.strip_prefix(&input).unwrap_or(v).to_string_lossy().replace('\\', "/"),
                            time: k as f64 * every,
                            items: items.iter().filter_map(|it| serde_json::from_value(it.clone()).ok()).collect(),
                        };
                        // 画了框的后面接原图；没画框的截图本身就是原图
                        let clean = if opts.draw_boxes { fs::read(&paths[k]).ok() } else { None };
                        let _ = crate::shotfile::append(&dst, clean.as_deref(), &info);
                        shots.push((c, dst));
                    }
                }
                if hits.is_empty() {
                    // 什么都没识别出来：可能是模型没学过的新情况。取几帧原图放进「没有目标 / 拿不准」文件夹，方便加回项目去标
                    let none_dir = out.join(safe_dir(&if detect { t!("没有目标") } else { t!("拿不准") }));
                    let folder = rel_folder(&input, v).replace('/', "_");
                    let tag = if folder.is_empty() { stem(v) } else { format!("{folder}_{}", stem(v)) };
                    unrecognized.push(v.strip_prefix(&input).unwrap_or(v).to_string_lossy().replace('\\', "/"));
                    for k in sample_frames(paths.len()) {
                        if fs::create_dir_all(&none_dir).is_ok() {
                            let dst = none_dir.join(format!("{}_{}.jpg", safe_dir(&tag), mmss(k as f64 * every)));
                            if fs::copy(&paths[k], &dst).is_ok() {
                                unrecognized_frames.push(dst.to_string_lossy().into_owned());
                            }
                        }
                    }
                }
                Some(hits)
            };
            let _ = fs::remove_dir_all(&dir);
            rows.push(Row { file: v.clone(), hits, video: true, shots });
            video_count += 1;
        }
        let _ = fs::remove_dir_all(&tmp);
    }

    let failed = rows.iter().filter(|r| r.hits.is_none()).count();
    let uncertain = rows
        .iter()
        .filter(|r| !r.video && r.hits.as_ref().is_some_and(|h| h.is_empty()))
        .map(|r| r.file.to_string_lossy().into_owned())
        .collect();
    // 分数偏低（刚过阈值）的：图片本身、视频的截图
    let low_line = (opts.conf + LOW_MARGIN).min(0.95);
    let low: Vec<String> = rows
        .iter()
        .filter(|r| r.hits.as_ref().and_then(|h| h.first()).is_some_and(|x| x.1 < low_line))
        .flat_map(|r| {
            if r.video {
                r.shots.iter().map(|s| s.1.to_string_lossy().into_owned()).collect::<Vec<_>>()
            } else {
                vec![r.file.to_string_lossy().into_owned()]
            }
        })
        .collect();
    if let Ok(mut r) = REVIEW.lock() {
        *r = rows.iter().flat_map(|r| r.shots.iter().map(|s| s.1.clone())).collect();
    }
    let (records, counts) = write_outputs(&input, &out, &ctx.classes, &rows, opts.sort, detect)?;
    // 表格、汇总、网页报告：这次识别的 + 上次识别过、这次跳过的
    let classes = if ctx.classes.is_empty() { prev.classes.clone() } else { ctx.classes.clone() };
    let mut all: Vec<Record> = Vec::new();
    if opts.skip_done {
        let fresh: std::collections::HashSet<&str> = records.iter().map(|r| r.rel.as_str()).collect();
        all.extend(prev.records.iter().filter(|r| !fresh.contains(r.rel.as_str())).cloned());
    }
    all.extend(records);
    all.sort_by(|a, b| a.rel.cmp(&b.rel));
    let written = batch_report::write_all(&out, &classes, &all, detect)?;
    let csv = written.csv;
    Ok(BatchResult {
        images,
        videos: video_count,
        frames,
        failed,
        csv: csv.to_string_lossy().into_owned(),
        output: out.to_string_lossy().into_owned(),
        counts,
        uncertain,
        stopped,
        shots: rows.iter().flat_map(|r| r.shots.iter().map(|s| s.1.to_string_lossy().into_owned())).collect(),
        unrecognized,
        unrecognized_frames,
        skipped,
        low,
        summary: written.summary.to_string_lossy().into_owned(),
        report: written.report.to_string_lossy().into_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::tests::temp;

    fn raw(tags: Vec<(usize, f64)>, boxes: Vec<[f64; 6]>) -> RawPred {
        RawPred { boxes, polys: Vec::new(), tags, poses: Vec::new() }
    }

    #[test]
    fn summarize_by_task() {
        let r = raw(vec![(1, 0.7), (0, 0.6), (2, 0.1)], Vec::new());
        assert_eq!(summarize(&r, "classify", 0.5, &[]), vec![(1, 0.7, 1)]);
        assert_eq!(summarize(&r, "classify", 0.8, &[]), vec![]);
        assert_eq!(summarize(&r, "multilabel", 0.5, &[]), vec![(1, 0.7, 1), (0, 0.6, 1)]);
        // 每个类别单独的阈值：类别 0 要 0.65，类别 1 要 0.5
        assert_eq!(summarize(&r, "multilabel", 0.9, &[0.65, 0.5]), vec![(1, 0.7, 1)]);
        let d = raw(Vec::new(), vec![[0.0, 0.0, 1.0, 1.0, 0.0, 0.9], [0.0, 0.0, 1.0, 1.0, 0.0, 0.6], [0.0, 0.0, 1.0, 1.0, 1.0, 0.3]]);
        assert_eq!(summarize(&d, "detect", 0.5, &[]), vec![(0, 0.9, 2)]);
    }

    #[test]
    fn aggregates_video_frames() {
        let frames: Vec<Hits> = vec![vec![(0, 0.6, 1)], vec![], vec![(0, 0.9, 1), (1, 0.7, 1)], vec![(0, 0.8, 1)]];
        let (hits, best) = aggregate(&frames, 1);
        assert_eq!(hits, vec![(0, 0.9, 3), (1, 0.7, 1)]);
        assert_eq!(best[&0], 2);
        assert_eq!(best[&1], 2);
        // 至少 2 帧才算：类别 1 只出现 1 帧，不算
        let (hits, best) = aggregate(&frames, 2);
        assert_eq!(hits, vec![(0, 0.9, 3)]);
        assert!(!best.contains_key(&1));
    }

    #[test]
    fn shot_dirs_follow_subfolders() {
        let input = Path::new("in");
        let out = Path::new("out");
        assert_eq!(shot_dir(input, out, Path::new("in/SF512/a.mp4")), out.join("SF512"));
        assert_eq!(shot_dir(input, out, Path::new("in/a/b/c.mp4")), out.join("a").join("b"));
        assert_eq!(shot_dir(input, out, Path::new("in/c.mp4")), out.join("c"));
        assert_eq!(mmss(65.4), "01m05s");
        assert_eq!(rel_folder(input, Path::new("in/SF512/a.mp4")), "SF512");
        assert_eq!(rel_folder(input, Path::new("in/a.jpg")), "");
    }

    #[test]
    fn samples_frames_evenly() {
        assert_eq!(sample_frames(0), Vec::<usize>::new());
        assert_eq!(sample_frames(1), vec![0]);
        assert_eq!(sample_frames(2), vec![0, 1]);
        assert_eq!(sample_frames(30), vec![7, 15, 22]);
    }

    #[test]
    fn skips_result_folders() {
        let base = temp("batch-skip");
        let input = base.join("in");
        let res = input.join("结果");
        let shots = input.join("SF1").join("识别截图");
        fs::create_dir_all(res.join("扇叶损坏")).unwrap();
        fs::create_dir_all(&shots).unwrap();
        mark_dir(&res);
        mark_dir(&shots);
        let mut cache = HashMap::new();
        assert!(in_result_dir(&input, &res.join("扇叶损坏").join("a.jpg"), &mut cache));
        assert!(in_result_dir(&input, &shots.join("x.jpg"), &mut cache));
        assert!(!in_result_dir(&input, &input.join("SF1").join("v.mp4"), &mut cache));
        assert!(!in_result_dir(&input, &input.join("a.jpg"), &mut cache));
        // 要识别的文件夹本身有记号也不算（只看它里面的子文件夹）
        mark_dir(&input);
        assert!(!in_result_dir(&input, &input.join("a.jpg"), &mut HashMap::new()));
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn writes_csv_and_sorts() {
        let base = temp("batch");
        let input = base.join("in");
        fs::create_dir_all(input.join("sub")).unwrap();
        let a = input.join("a.jpg");
        let b = input.join("sub").join("b.jpg");
        let c = input.join("c.jpg");
        let v = input.join("SF1").join("v.mp4");
        fs::create_dir_all(input.join("SF1")).unwrap();
        for f in [&a, &b, &c, &v] {
            fs::write(f, b"x").unwrap();
        }
        let classes = vec!["扇叶损坏".to_string(), "灰尘, 污渍".to_string()];
        let shot = base.join("shots").join("扇叶损坏_00m12s_0.80.jpg");
        fs::create_dir_all(shot.parent().unwrap()).unwrap();
        fs::write(&shot, b"x").unwrap();
        let row = |file: &PathBuf, hits: Option<Hits>, video: bool, shots: Vec<(usize, PathBuf)>| Row { file: file.clone(), hits, video, shots };
        let rows = vec![
            row(&a, Some(vec![(0, 0.9, 1), (1, 0.7, 1)]), false, vec![]),
            row(&b, Some(vec![]), false, vec![]),
            row(&c, None, false, vec![]),
            row(&v, Some(vec![(0, 0.8, 3)]), true, vec![(0, shot.clone())]),
        ];
        let out = base.join("out");
        let (records, counts) = write_outputs(&input, &out, &classes, &rows, true, false).unwrap();
        assert_eq!(records.len(), 4);
        assert_eq!(records[3].folder, "SF1");
        assert_eq!(records[3].hits, Some(vec![("扇叶损坏".to_string(), 0.8, 3)]));
        assert_eq!(records[3].shots.len(), 1);
        let csv = batch_report::write_all(&out, &classes, &records, false).unwrap().csv;
        let text = fs::read_to_string(&csv).unwrap();
        let lines: Vec<&str> = text.trim_start_matches('\u{feff}').lines().collect();
        assert_eq!(lines.len(), 5);
        // 类别名里有逗号要加引号
        assert!(lines[0].ends_with(",扇叶损坏,\"灰尘, 污渍\""));
        assert!(lines[1].starts_with("a.jpg,,"));
        assert!(lines[2].starts_with("sub/b.jpg,sub,"));
        // 视频一行：所在文件夹（单号）、每个类别检出几帧
        assert!(lines[4].starts_with("SF1/v.mp4,SF1,"));
        assert!(lines[4].contains("3"));
        // 两个类别的图复制到两个文件夹，拿不准的放一起，子文件夹的图名字拍平
        assert!(out.join("扇叶损坏").join("a.jpg").exists());
        // 视频复制的是截图（名字前面加单号），不复制视频本身
        assert!(out.join("扇叶损坏").join("SF1_扇叶损坏_00m12s_0.80.jpg").exists());
        assert!(!out.join("扇叶损坏").join("SF1_v.mp4").exists());
        assert!(out.join(safe_dir("灰尘, 污渍")).join("a.jpg").exists());
        assert!(out.join(safe_dir(&t!("拿不准"))).join("sub_b.jpg").exists());
        assert_eq!(counts.len(), 3);
        assert_eq!(counts[0], ("扇叶损坏".to_string(), 2));
        let _ = fs::remove_dir_all(base);
    }
}
