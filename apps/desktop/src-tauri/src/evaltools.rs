//! 外部模型导入、用现在的数据重新评估、误报测试。都是一次性的 worker 任务。

use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{ChildStdin, Stdio};
use std::sync::Mutex;

use rusqlite::params;
use serde::{Deserialize, Serialize};

use crate::dataset::{self, ExportOptions, Format};
use crate::db;
use crate::error::{msg, AppResult};
use crate::project::Project;
use crate::runtime::Runtime;
use crate::training::{self, EXTERNAL};

/// 跑一个一次性的 worker 命令：stdout 每行 JSON 事件交给 on_event；出错时带上 worker 报的错误信息。
/// on_stdin 拿到子进程的标准输入（误报测试用来发停止命令）
pub fn worker_events(
    rt: &Runtime,
    args: &[&str],
    on_event: &mut dyn FnMut(&serde_json::Value),
    on_stdin: Option<&dyn Fn(Option<ChildStdin>)>,
) -> AppResult<()> {
    let mut c = rt.worker(args)?;
    c.stdin(if on_stdin.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = c.spawn().map_err(|e| msg(format!("无法启动 worker：{e}")))?;
    if let Some(f) = on_stdin {
        f(child.stdin.take());
    }
    let out = child.stdout.take().unwrap();
    let err = child.stderr.take().unwrap();
    let tail = Mutex::new(String::new());
    let mut error: Option<String> = None;
    std::thread::scope(|s| {
        s.spawn(|| {
            let mut buf = String::new();
            let _ = BufReader::new(err).read_to_string(&mut buf);
            *tail.lock().unwrap() = buf;
        });
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line.trim()) else { continue };
            if v["type"] == "error" {
                error = v["msg"].as_str().map(String::from);
            }
            on_event(&v);
        }
    });
    if let Some(f) = on_stdin {
        f(None);
    }
    let status = child.wait()?;
    if let Some(e) = error {
        return Err(msg(e));
    }
    if !status.success() {
        let t = tail.lock().unwrap();
        let lines: Vec<&str> = t.lines().filter(|l| !l.trim().is_empty()).collect();
        return Err(msg(format!("worker 异常退出：{}", lines[lines.len().saturating_sub(4)..].join("\n"))));
    }
    Ok(())
}

// ---------- 外部 ONNX 模型 ----------

/// 识别 ONNX 模型的格式、类别名、输入尺寸（worker 的 onnx-info 原样返回）
pub fn inspect_onnx(rt: &Runtime, path: &str) -> AppResult<serde_json::Value> {
    if !Path::new(path).is_file() {
        return Err(msg("模型文件不存在"));
    }
    let mut info = None;
    worker_events(rt, &["onnx-info", path], &mut |v| {
        if v["type"] == "onnx" {
            info = Some(v.clone());
        }
    }, None)?;
    info.ok_or_else(|| msg("读不出模型信息"))
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExternalModel {
    /// 显示名
    pub name: String,
    /// yolov8 / yolov5 / yolo_e2e / yolox / rtdetr / detr / wulabel
    pub kind: String,
    /// 模型的类别名（按模型里的顺序）；可以改成项目里的类别名来对应
    pub classes: Vec<String>,
    pub img_size: u32,
    /// 只用这些类别（模型类别序号），为空表示全用
    #[serde(default)]
    pub use_classes: Option<Vec<usize>>,
}

/// 导入外部模型：复制到 runs/ 下，记一条训练记录（engine = external），之后就能像训练出的模型一样用来预标注、体检、评估
pub fn import_onnx(p: &Project, path: &str, m: &ExternalModel) -> AppResult<i64> {
    if p.meta.task != "detect" {
        return Err(msg("外部模型目前只支持目标检测项目"));
    }
    if m.classes.is_empty() || m.classes.iter().any(|c| c.trim().is_empty()) {
        return Err(msg("请填写模型的类别名（和模型输出的顺序一致）"));
    }
    const KINDS: [&str; 7] = ["yolov8", "yolov5", "yolo_e2e", "yolox", "rtdetr", "detr", "wulabel"];
    if !KINDS.contains(&m.kind.as_str()) {
        return Err(msg(format!("不支持的模型格式：{}", m.kind)));
    }
    let base = format!("外部_{}", chrono::Local::now().format("%Y%m%d_%H%M%S"));
    let mut name = base.clone();
    let mut i = 1;
    while p.root.join("runs").join(&name).exists() {
        name = format!("{base}_{i}");
        i += 1;
    }
    let dir = p.root.join("runs").join(&name);
    fs::create_dir_all(&dir)?;
    fs::copy(path, dir.join("model.onnx")).map_err(|e| msg(format!("复制模型失败：{e}")))?;
    write_info(&dir, m, path)?;
    let config = serde_json::json!({
        "model": "external", "displayName": m.name.trim(), "kind": m.kind, "classes": m.classes,
        "imgSize": m.img_size, "source": path,
    });
    let conn = p.conn.lock();
    conn.execute(
        "INSERT INTO runs (dir, engine, task, status, config_json, created_at) VALUES (?1, ?2, ?3, 'done', ?4, ?5)",
        params![name, EXTERNAL, p.meta.task, config.to_string(), db::now()],
    )?;
    Ok(conn.last_insert_rowid())
}

fn write_info(dir: &Path, m: &ExternalModel, source: &str) -> AppResult<()> {
    let info = serde_json::json!({
        "kind": m.kind, "classes": m.classes.iter().map(|c| c.trim()).collect::<Vec<_>>(),
        "imgSize": m.img_size, "use": m.use_classes, "source": source,
    });
    fs::write(dir.join("model.json"), serde_json::to_string_pretty(&info)?)?;
    Ok(())
}

/// 改外部模型的类别名 / 要用的类别
pub fn update_external(p: &Project, run_id: i64, m: &ExternalModel) -> AppResult<()> {
    let run = training::get_run(p, run_id)?;
    if run.engine != EXTERNAL {
        return Err(msg("只有导入的外部模型可以改类别"));
    }
    let dir = PathBuf::from(&run.dir);
    write_info(&dir, m, run.config["source"].as_str().unwrap_or(""))?;
    let mut config = run.config.clone();
    config["classes"] = serde_json::json!(m.classes);
    config["displayName"] = serde_json::json!(m.name.trim());
    config["kind"] = serde_json::json!(m.kind);
    p.conn.lock().execute("UPDATE runs SET config_json = ?2 WHERE id = ?1", params![run_id, config.to_string()])?;
    Ok(())
}

// ---------- 重新评估 ----------

/// 用现在的数据（已标注、已审核的验证集 / 测试集）重新评估一个模型，结果覆盖这次训练的 eval.json
pub fn evaluate_run(p: &Project, rt: &Runtime, run_id: i64, log: &(dyn Fn(String) + Sync)) -> AppResult<f64> {
    let run = training::get_run(p, run_id)?;
    let weights = training::weights_of(p, run_id)?;
    let dir = PathBuf::from(&run.dir);
    let ds = dir.join("eval_dataset");
    let _ = fs::remove_dir_all(&ds);
    let format = if p.meta.task == "classify" { Format::Cls } else { Format::Coco };
    log("导出当前数据…".into());
    let res = dataset::export(p, &ExportOptions { format, out_dir: ds.clone(), only_reviewed: false, ratios: [0.8, 0.2, 0.0] }, &|_, _| {})?;
    let cfg = serde_json::json!({ "dataset": ds, "out": dir, "weights": weights, "classes": res.classes });
    let cfg_path = dir.join("eval_config.json");
    fs::write(&cfg_path, serde_json::to_string_pretty(&cfg)?)?;
    let cfg_arg = cfg_path.to_string_lossy().to_string();
    let mut map50 = 0.0;
    let r = worker_events(rt, &["evaluate", cfg_arg.as_str()], &mut |v| match v["type"].as_str() {
        Some("log") => log(v["msg"].as_str().unwrap_or("").to_string()),
        Some("progress") => log(format!("评估中 {}/{}", v["done"], v["total"])),
        Some("done") => map50 = v["map50"].as_f64().unwrap_or(0.0),
        _ => {}
    }, None);
    let _ = fs::remove_dir_all(&ds);
    r?;
    // 外部模型没有训练过程的指标，用评估结果当「最好成绩」
    if run.engine == EXTERNAL {
        p.conn.lock().execute("UPDATE runs SET best_metric = ?2 WHERE id = ?1", params![run_id, map50])?;
    }
    Ok(map50)
}

// ---------- 误报测试 ----------

/// 正在进行的误报测试（同一时间一个），用来发停止命令
#[derive(Default)]
pub struct FpTest {
    stdin: Mutex<Option<ChildStdin>>,
}

impl FpTest {
    pub fn stop(&self) {
        if let Some(s) = self.stdin.lock().unwrap().as_mut() {
            let _ = writeln!(s, "{{\"cmd\":\"stop\"}}");
            let _ = s.flush();
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FpConfig {
    pub run_id: i64,
    pub source: String,
    pub conf: f64,
    #[serde(default = "two")]
    pub fps: f64,
    #[serde(default)]
    pub max_seconds: f64,
    #[serde(default = "one")]
    pub consecutive: u32,
    #[serde(default = "ten")]
    pub cooldown: f64,
}

fn two() -> f64 {
    2.0
}
fn one() -> u32 {
    1
}
fn ten() -> f64 {
    10.0
}

/// 跑误报测试，事件（open / progress / alarm）交给 sink，返回最后的统计。误报帧存在项目 .cache/fptest/<时间>/
pub fn fp_test(
    p: &Project,
    rt: &Runtime,
    job: &FpTest,
    cfg: &FpConfig,
    sink: &(dyn Fn(&serde_json::Value) + Sync),
) -> AppResult<serde_json::Value> {
    let source = cfg.source.trim();
    if source.is_empty() {
        return Err(msg("请选择视频文件或填写视频流地址"));
    }
    if !source.contains("://") && !Path::new(source).is_file() {
        return Err(msg("视频文件不存在"));
    }
    let weights = training::weights_of(p, cfg.run_id)?;
    let out = p.root.join(".cache").join("fptest").join(chrono::Local::now().format("%Y%m%d_%H%M%S").to_string());
    fs::create_dir_all(&out)?;
    let wcfg = serde_json::json!({
        "source": source, "weights": weights, "conf": cfg.conf, "fps": cfg.fps, "max_seconds": cfg.max_seconds,
        "consecutive": cfg.consecutive, "cooldown": cfg.cooldown, "out": out,
    });
    let cfg_path = out.join("config.json");
    fs::write(&cfg_path, serde_json::to_string(&wcfg)?)?;
    let cfg_arg = cfg_path.to_string_lossy().to_string();
    let mut summary = None;
    let set_stdin = |s: Option<ChildStdin>| *job.stdin.lock().unwrap() = s;
    worker_events(rt, &["fptest", cfg_arg.as_str()], &mut |v| {
        if v["type"] == "done" {
            summary = Some(v.clone());
        } else {
            sink(v);
        }
    }, Some(&set_stdin))?;
    let mut s = summary.ok_or_else(|| msg("误报测试没有返回结果"))?;
    s["dir"] = serde_json::json!(out.to_string_lossy());
    Ok(s)
}

/// 把误报帧加进数据集当负样本：导入后直接标成「已审核」（没有标注 = 这张图里没有目标），训练时模型会学着不报
pub fn import_negatives(p: &Project, paths: &[String]) -> AppResult<usize> {
    let files: Vec<PathBuf> = paths.iter().map(PathBuf::from).filter(|f| f.is_file()).collect();
    if files.is_empty() {
        return Err(msg("没有可加入的图片"));
    }
    let (_, ids) = crate::images::import_mapped(p, &files, true, &|_, _| {})?;
    let conn = p.conn.lock();
    let mut n = 0;
    for id in ids.values() {
        // 已经有标注的图（重复导入）不改
        n += conn.execute(
            "UPDATE images SET status = 'reviewed'
             WHERE id = ?1 AND status = 'todo' AND NOT EXISTS (SELECT 1 FROM annotations a WHERE a.image_id = images.id)",
            [id],
        )?;
    }
    Ok(n)
}
