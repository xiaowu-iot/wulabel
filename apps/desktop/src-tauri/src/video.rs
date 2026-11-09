//! 视频抽帧：用 ffmpeg 按间隔把视频抽成图片，再走图片导入流程

use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::error::{msg, AppResult};
use crate::images;
use crate::project::{Project, IMAGES_DIR};
use crate::runtime::{command, Runtime};

pub const VIDEO_EXTS: [&str; 9] = ["mp4", "mov", "mkv", "avi", "webm", "flv", "wmv", "m4v", "ts"];

fn is_video(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| VIDEO_EXTS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// 展开用户选的文件和文件夹（文件夹递归查找视频）
pub fn collect(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for p in paths {
        if p.is_dir() {
            for e in walkdir::WalkDir::new(p).follow_links(true).into_iter().flatten() {
                if e.file_type().is_file() && is_video(e.path()) {
                    out.push(e.into_path());
                }
            }
        } else if p.is_file() && is_video(p) {
            out.push(p.clone());
        }
    }
    out.sort();
    out.dedup();
    out
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtractResult {
    pub videos: usize,
    pub frames: usize,
    pub added: usize,
    pub duplicated: usize,
    pub failed: Vec<String>,
    /// 视频流采集的帧保存在哪个文件夹（绝对路径）；本地视频抽帧为 None
    pub dir: Option<String>,
}

/// 进度回调：阶段（download / extract / import）、已完成、总数、速度（字节/秒，仅下载阶段）、说明
type Progress<'a> = &'a (dyn Fn(&str, u64, u64, u64, &str) + Sync);

fn video_name(p: &Path) -> String {
    p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

/// 抽帧并导入。every：每隔多少秒抽 1 帧；max_frames：每个视频最多抽多少帧，0 不限。
/// 抽出的帧一律复制进项目（临时文件导入后删除，不能只引用）。
pub fn extract(
    project: &Project,
    rt: &Runtime,
    paths: &[PathBuf],
    every: f64,
    max_frames: u64,
    progress: Progress,
    log: &(dyn Fn(String) + Sync),
) -> AppResult<ExtractResult> {
    let videos = collect(paths);
    if videos.is_empty() {
        return Err(msg("没有找到视频（支持 mp4、mov、mkv、avi、webm 等）"));
    }
    let fps = if every > 0.0 { 1.0 / every } else { 1.0 };

    // 1. 准备 ffmpeg（没有就下载，带进度）
    let ffmpeg = rt.ffmpeg(
        log,
        &|_f, d, t, s| progress("download", d, t, s, "正在下载 ffmpeg（首次使用，约 90MB）"),
    )?;

    // 2. 估算总帧数（用于进度条），再逐个视频抽帧到临时目录
    let frame_dir =
        std::env::temp_dir().join(format!("xlabel-frames-{}-{}", std::process::id(), now_nanos()));
    fs::create_dir_all(&frame_dir)?;

    let mut total_est: u64 = 0;
    for v in &videos {
        if let Some(d) = duration(&ffmpeg, v) {
            let n = (d * fps).ceil() as u64;
            total_est += if max_frames > 0 { n.min(max_frames) } else { n };
        }
    }

    let base = AtomicU64::new(0);
    for (i, v) in videos.iter().enumerate() {
        let label = format!("{}（{}/{}）", video_name(v), i + 1, videos.len());
        progress("extract", base.load(Ordering::Relaxed), total_est, 0, &label);
        let produced = extract_one(&ffmpeg, v, &frame_dir, i, fps, max_frames, &|frames| {
            progress("extract", base.load(Ordering::Relaxed) + frames, total_est, 0, &label);
        });
        match produced {
            Ok(n) => {
                base.fetch_add(n, Ordering::Relaxed);
            }
            Err(e) => {
                let _ = fs::remove_dir_all(&frame_dir);
                return Err(e);
            }
        }
    }

    // 3. 把抽出的帧当图片导入（内容去重、生成缩略图、写库）
    let frames = images::collect(&[frame_dir.clone()]);
    let frame_count = frames.len();
    if frames.is_empty() {
        let _ = fs::remove_dir_all(&frame_dir);
        return Err(msg("没有抽到任何帧，请确认视频能正常播放"));
    }
    let imp = images::import(project, &frames, true, &|done, total| {
        progress("import", done as u64, total as u64, 0, "正在导入抽出的帧");
    });
    let _ = fs::remove_dir_all(&frame_dir);
    let imp = imp?;

    Ok(ExtractResult {
        videos: videos.len(),
        frames: frame_count,
        added: imp.added,
        duplicated: imp.duplicated,
        failed: imp.failed,
        dir: None,
    })
}

fn now_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// 抽一个视频：输出到 out_dir/v{idx}_%06d.jpg，边抽边回调当前帧数，返回实际产出帧数
fn extract_one(
    ffmpeg: &Path,
    video: &Path,
    out_dir: &Path,
    idx: usize,
    fps: f64,
    max_frames: u64,
    on_frame: &(dyn Fn(u64) + Sync),
) -> AppResult<u64> {
    let prefix = format!("v{idx}_");
    let pattern = out_dir.join(format!("{prefix}%06d.jpg"));
    let mut c = command(ffmpeg);
    c.args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i"]).arg(video);
    c.args(["-vf", &format!("fps={fps}"), "-q:v", "2"]);
    if max_frames > 0 {
        c.args(["-frames:v", &max_frames.to_string()]);
    }
    c.args(["-progress", "pipe:1", "-nostats"]).arg(&pattern);
    c.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = c.spawn().map_err(|e| msg(format!("无法启动 ffmpeg：{e}")))?;
    let out = child.stdout.take().unwrap();
    let err = child.stderr.take().unwrap();
    let err_buf = std::sync::Mutex::new(String::new());
    // 另开线程排空 stderr，避免 stderr 写满管道时 ffmpeg 阻塞
    std::thread::scope(|s| {
        s.spawn(|| {
            let mut buf = String::new();
            let _ = BufReader::new(err).read_to_string(&mut buf);
            *err_buf.lock().unwrap() = buf;
        });
        // -progress 会持续输出 frame=N 这样的行
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            if let Some(n) = line.trim().strip_prefix("frame=") {
                if let Ok(v) = n.trim().parse::<u64>() {
                    on_frame(v);
                }
            }
        }
    });
    let status = child.wait()?;
    if !status.success() {
        let err = err_buf.lock().unwrap();
        let tail: Vec<&str> = err.lines().filter(|l| !l.trim().is_empty()).collect();
        let tail = tail[tail.len().saturating_sub(4)..].join("\n");
        return Err(msg(format!("抽帧失败（{}）：{tail}", video_name(video))));
    }
    // 以实际落盘的文件数为准
    let produced = fs::read_dir(out_dir)?
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with(&prefix))
        .count() as u64;
    Ok(produced)
}

// ---------- 视频流（直播 / 回放 URL）抽帧 ----------

/// 探测视频流得到的信息
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    /// 回放流的总时长（秒）；直播流为 None
    pub duration_secs: Option<f64>,
    pub is_live: bool,
    pub codec: String,
}

/// 正在进行的流抽帧任务，用来支持中途停止（停止后已抽的帧照常保留）
#[derive(Default)]
pub struct Extractor {
    child: Mutex<Option<Arc<Mutex<Child>>>>,
}

impl Extractor {
    /// 停止当前采集：杀掉 ffmpeg，已落盘的帧会在随后照常导入
    pub fn stop(&self) {
        if let Some(c) = self.child.lock().unwrap().take() {
            let _ = c.lock().unwrap().kill();
        }
    }

    fn set(&self, c: Option<Arc<Mutex<Child>>>) {
        *self.child.lock().unwrap() = c;
    }
}

/// 地址里的账号密码脱敏：rtsp://user:pass@host → rtsp://***@host
fn mask_url(url: &str) -> String {
    if let Some(i) = url.find("://") {
        let (scheme, rest) = url.split_at(i + 3);
        if let Some(at) = rest.find('@') {
            return format!("{scheme}***@{}", &rest[at + 1..]);
        }
    }
    url.to_string()
}

fn is_stream_url(url: &str) -> bool {
    url.contains("://")
}

/// RTSP 默认走 TCP（摄像头用 UDP 常丢包花屏）
fn stream_input_args(c: &mut std::process::Command, url: &str, rw_timeout_us: u64) {
    if url.starts_with("rtsp") {
        c.args(["-rtsp_transport", "tcp"]);
    }
    // 多少微秒没数据就判定断开，避免地址不通时一直卡着
    c.args(["-rw_timeout", &rw_timeout_us.to_string()]);
}

/// 从 ffmpeg stderr 里解析分辨率、编码、帧率、时长
fn parse_stream_info(text: &str) -> StreamInfo {
    let mut info = StreamInfo::default();
    if let Some(i) = text.find("Duration:") {
        let s = text[i + "Duration:".len()..].split(',').next().unwrap_or("").trim();
        if !s.starts_with("N/A") {
            let p: Vec<&str> = s.split(':').collect();
            if p.len() == 3 {
                if let (Ok(h), Ok(m), Ok(sec)) =
                    (p[0].trim().parse::<f64>(), p[1].parse::<f64>(), p[2].parse::<f64>())
                {
                    info.duration_secs = Some(h * 3600.0 + m * 60.0 + sec);
                }
            }
        }
    }
    info.is_live = info.duration_secs.is_none();
    if let Some(i) = text.find("Video:") {
        let line = text[i..].lines().next().unwrap_or("");
        let after = line["Video:".len()..].trim();
        info.codec = after.split(|c| c == ' ' || c == ',' || c == '(').next().unwrap_or("").trim().to_string();
        let toks: Vec<&str> = line.split(|c| c == ' ' || c == ',').map(|t| t.trim()).filter(|t| !t.is_empty()).collect();
        for t in &toks {
            if info.width == 0 {
                if let Some((w, h)) = t.split_once('x') {
                    if let (Ok(w), Ok(h)) = (w.parse::<u32>(), h.parse::<u32>()) {
                        if w >= 16 && h >= 16 {
                            info.width = w;
                            info.height = h;
                        }
                    }
                }
            }
        }
        for (k, t) in toks.iter().enumerate() {
            if *t == "fps" && k > 0 {
                if let Ok(f) = toks[k - 1].parse::<f64>() {
                    info.fps = f;
                }
            }
        }
    }
    info
}

/// 测试连接：探测视频流信息（是否可达、分辨率、直播还是回放）。最多等 12 秒。
pub fn probe_stream(rt: &Runtime, url: &str, log: &(dyn Fn(String) + Sync), progress: Progress) -> AppResult<StreamInfo> {
    let url = url.trim();
    if !is_stream_url(url) {
        return Err(msg("请填写视频流地址（rtsp:// 或 http(s):// 开头）"));
    }
    let ffmpeg = rt.ffmpeg(log, &|_f, d, t, s| progress("download", d, t, s, "正在下载 ffmpeg（首次使用，约 90MB）"))?;
    let mut c = command(&ffmpeg);
    c.args(["-hide_banner"]);
    stream_input_args(&mut c, url, 10_000_000);
    c.arg("-i").arg(url);
    c.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped());
    let mut child = c.spawn().map_err(|e| msg(format!("无法启动 ffmpeg：{e}")))?;
    let err = child.stderr.take().unwrap();
    let buf = Arc::new(Mutex::new(String::new()));
    let b2 = buf.clone();
    let t = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = BufReader::new(err).read_to_string(&mut s);
        *b2.lock().unwrap() = s;
    });
    let mut timed_out = true;
    for _ in 0..24 {
        if matches!(child.try_wait(), Ok(Some(_))) {
            timed_out = false;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    if timed_out {
        let _ = child.kill();
    }
    let _ = child.wait();
    let _ = t.join();
    let text = buf.lock().unwrap().clone();
    if !text.contains("Video:") {
        if timed_out {
            return Err(msg("连接超时，确认地址正确、网络能连上"));
        }
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        let tail = lines[lines.len().saturating_sub(3)..].join("；");
        return Err(msg(format!("读不到视频流：{}", if tail.is_empty() { "地址可能不对".to_string() } else { tail })));
    }
    Ok(parse_stream_info(&text))
}

/// 采集文件夹 / 文件名前缀：地址里的主机名 + 开始时间，例如 119.3.176.149_20261110_153012
fn stream_tag(url: &str) -> String {
    let rest = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let authority = rest.split('/').next().unwrap_or("");
    let host_port = authority.rsplit_once('@').map(|(_, h)| h).unwrap_or(authority); // 去掉账号密码
    let host: String = host_port
        .split(':')
        .next()
        .unwrap_or("")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' { c } else { '-' })
        .collect();
    let host = if host.is_empty() { "stream".to_string() } else { host };
    format!("{host}_{}", chrono::Local::now().format("%Y%m%d_%H%M%S"))
}

/// 从视频流抽帧：直播流用「最多帧数 / 最长时长」做上限，回放流读到结尾为止。
/// 帧直接写进项目 images/stream_<主机>_<时间>/，边采边入库（约 1.5 秒一次），
/// 新入库的图片 id 通过 on_frames 通知前端实时显示。整次采集共用一个导入批次。
/// dedup=true 时用 mpdecimate 丢掉几乎相同的帧。可被 Extractor::stop 中途停止，已抽的帧保留。
#[allow(clippy::too_many_arguments)]
pub fn extract_stream(
    project: &Project,
    rt: &Runtime,
    extractor: &Extractor,
    url: &str,
    every: f64,
    max_frames: u64,
    max_minutes: f64,
    dedup: bool,
    progress: Progress,
    on_frames: &(dyn Fn(&[i64], &str) + Sync),
    log: &(dyn Fn(String) + Sync),
) -> AppResult<ExtractResult> {
    let url = url.trim();
    if !is_stream_url(url) {
        return Err(msg("请填写视频流地址（rtsp:// 或 http(s):// 开头）"));
    }
    let fps = if every > 0.0 { 1.0 / every } else { 1.0 };
    let ffmpeg = rt.ffmpeg(log, &|_f, d, t, s| progress("download", d, t, s, "正在下载 ffmpeg（首次使用，约 90MB）"))?;

    // 一次采集一个文件夹，放在项目 images/ 下，名字带来源和时间，方便找
    let tag = stream_tag(url);
    let folder = format!("stream_{tag}");
    let rel_dir = format!("{IMAGES_DIR}/{folder}");
    let cap_dir = project.root.join(IMAGES_DIR).join(&folder);
    fs::create_dir_all(&cap_dir)?;
    let dir_str = cap_dir.to_string_lossy().into_owned();
    let pattern = cap_dir.join(format!("{tag}_%06d.jpg"));
    let batch = images::next_batch(project)?;
    let masked = mask_url(url);
    // 直播没有总帧数，用上限当进度分母（上限为 0 时进度条不显示百分比）
    let total = max_frames;
    progress("extract", 0, total, 0, &masked);

    let mut c = command(&ffmpeg);
    c.args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y"]);
    stream_input_args(&mut c, url, 30_000_000);
    c.arg("-i").arg(url);
    let vf = if dedup { format!("fps={fps},mpdecimate") } else { format!("fps={fps}") };
    c.args(["-vf", &vf, "-q:v", "2"]);
    if dedup {
        // mpdecimate 丢帧后要变帧率输出，否则会把丢掉的帧补成重复帧
        c.args(["-vsync", "vfr"]);
    }
    if max_frames > 0 {
        c.args(["-frames:v", &max_frames.to_string()]);
    }
    if max_minutes > 0.0 {
        c.args(["-t", &format!("{:.3}", max_minutes * 60.0)]);
    }
    c.args(["-progress", "pipe:1", "-nostats"]).arg(&pattern);
    c.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = c.spawn().map_err(|e| {
        let _ = fs::remove_dir_all(&cap_dir);
        msg(format!("无法启动 ffmpeg：{e}"))
    })?;
    let out = child.stdout.take().unwrap();
    let err = child.stderr.take().unwrap();
    let child = Arc::new(Mutex::new(child));
    extractor.set(Some(child.clone()));

    let mut res = ExtractResult {
        videos: 1,
        frames: 0,
        added: 0,
        duplicated: 0,
        failed: Vec::new(),
        dir: Some(dir_str.clone()),
    };
    let mut done_upto = 0usize; // 按文件名排序，前 done_upto 张已经处理过
    // 把已经写完的帧导入项目（ffmpeg 正在写的最后一张先不碰），新入库的图通知前端。
    // 帧本来就在项目文件夹里，所以按「只引用」方式入库，不再复制一份。
    let mut sweep = |all: bool| {
        let mut files: Vec<PathBuf> = match fs::read_dir(&cap_dir) {
            Ok(rd) => rd
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().map(|e| e.eq_ignore_ascii_case("jpg")).unwrap_or(false))
                .collect(),
            Err(_) => return,
        };
        files.sort();
        let ready = if all { files.len() } else { files.len().saturating_sub(1) };
        if ready <= done_upto {
            return;
        }
        let new_files = files[done_upto..ready].to_vec();
        done_upto = ready;
        let max_before: i64 =
            project.conn.lock().query_row("SELECT COALESCE(MAX(id), 0) FROM images", [], |r| r.get(0)).unwrap_or(0);
        match images::import_batch(project, &new_files, false, Some(batch), &|_, _| {}) {
            Ok((r, map)) => {
                res.added += r.added;
                res.duplicated += r.duplicated;
                res.failed.extend(r.failed);
                let mut ids: Vec<i64> = map.into_values().filter(|id| *id > max_before).collect();
                ids.sort_unstable();
                ids.dedup();
                if !ids.is_empty() {
                    on_frames(&ids, &dir_str);
                }
            }
            Err(e) => res.failed.push(e.to_string()),
        }
    };

    let err_buf = Mutex::new(String::new());
    let mut last_sweep = Instant::now();
    std::thread::scope(|sc| {
        sc.spawn(|| {
            let mut buf = String::new();
            let _ = BufReader::new(err).read_to_string(&mut buf);
            *err_buf.lock().unwrap() = buf;
        });
        // -progress 大约每 0.5 秒输出一次，顺便定时把新帧入库
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            if let Some(n) = line.trim().strip_prefix("frame=") {
                if let Ok(v) = n.trim().parse::<u64>() {
                    progress("extract", v, total, 0, &masked);
                }
            }
            if last_sweep.elapsed() >= Duration::from_millis(1500) {
                sweep(false);
                last_sweep = Instant::now();
            }
        }
    });
    let _ = child.lock().unwrap().wait();
    extractor.set(None);
    // 收尾：ffmpeg 已退出，最后一张也写完了。停止 / 断流导致的非零退出码不算失败
    sweep(true);
    let frame_count = done_upto;

    // 重复帧、没写完的残帧没有入库，从文件夹里删掉；文件夹空了也删掉
    let kept: HashSet<String> = {
        let conn = project.conn.lock();
        let mut stmt = conn.prepare("SELECT path FROM images WHERE path LIKE ?1")?;
        let rows = stmt.query_map([format!("{rel_dir}/%")], |r| r.get::<_, String>(0))?;
        let set: HashSet<String> = rows.flatten().collect();
        set
    };
    if let Ok(rd) = fs::read_dir(&cap_dir) {
        for e in rd.flatten() {
            let rel = format!("{rel_dir}/{}", e.file_name().to_string_lossy());
            if !kept.contains(&rel) {
                let _ = fs::remove_file(e.path());
            }
        }
    }
    if fs::read_dir(&cap_dir).map(|mut d| d.next().is_none()).unwrap_or(true) {
        let _ = fs::remove_dir(&cap_dir);
        res.dir = None;
    }

    if frame_count == 0 {
        let e = err_buf.lock().unwrap();
        let lines: Vec<&str> = e.lines().filter(|l| !l.trim().is_empty()).collect();
        let tail = lines[lines.len().saturating_sub(4)..].join("\n");
        return Err(msg(if tail.is_empty() {
            "没抽到任何帧，确认视频流地址能正常播放".to_string()
        } else {
            format!("采集失败：{tail}")
        }));
    }
    res.frames = frame_count;
    Ok(res)
}

/// 用 `ffmpeg -i` 读视频时长（秒），从 stderr 的 "Duration: HH:MM:SS.ms" 里解析
fn duration(ffmpeg: &Path, video: &Path) -> Option<f64> {
    let out = command(ffmpeg).args(["-hide_banner", "-i"]).arg(video).output().ok()?;
    let text = String::from_utf8_lossy(&out.stderr);
    let i = text.find("Duration:")?;
    let s = text[i + "Duration:".len()..].split(',').next()?.trim();
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let h: f64 = parts[0].trim().parse().ok()?;
    let m: f64 = parts[1].parse().ok()?;
    let sec: f64 = parts[2].parse().ok()?;
    Some(h * 3600.0 + m * 60.0 + sec)
}
