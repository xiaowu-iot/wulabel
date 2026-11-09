//! 视频抽帧：用 ffmpeg 按间隔把视频抽成图片，再走图片导入流程

use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;

use crate::error::{msg, AppResult};
use crate::images;
use crate::project::Project;
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
