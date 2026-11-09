//! 训练任务与推理服务

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Stdio};
use std::sync::Arc;

use parking_lot::Mutex;
use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::annotations::{self, Annotation};
use crate::dataset::{self, ExportOptions, Format};
use crate::db;
use crate::error::{msg, AppResult};
use crate::project::Project;
use crate::runtime::Runtime;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainConfig {
    pub model: String,
    pub epochs: u32,
    pub img_size: u32,
    pub batch: u32,
    #[serde(default = "default_lr")]
    pub lr: f64,
    #[serde(default = "default_device")]
    pub device: String,
    #[serde(default)]
    pub only_reviewed: bool,
    #[serde(default = "default_ratios")]
    pub ratios: [f64; 3],
}

fn default_lr() -> f64 {
    0.01
}
fn default_device() -> String {
    "auto".into()
}
fn default_ratios() -> [f64; 3] {
    [0.8, 0.2, 0.0]
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunInfo {
    pub id: i64,
    pub name: String,
    pub dir: String,
    pub engine: String,
    pub status: String,
    pub config: serde_json::Value,
    pub best_map50: Option<f64>,
    pub created_at: i64,
    pub has_weights: bool,
    pub has_onnx: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunDetail {
    pub run: RunInfo,
    pub metrics: Vec<serde_json::Value>,
    pub eval: Option<serde_json::Value>,
    pub log: Vec<String>,
    pub error: Option<String>,
}

/// 当前正在训练的任务（同一时间只允许一个）
#[derive(Default)]
pub struct Jobs {
    current: Mutex<Option<(i64, PathBuf, Arc<Mutex<Child>>)>>,
}

impl Jobs {
    pub fn running(&self) -> Option<i64> {
        self.current.lock().as_ref().map(|c| c.0)
    }

    pub fn kill_all(&self) {
        if let Some((_, _, child)) = self.current.lock().as_ref() {
            let _ = child.lock().kill();
        }
    }
}

fn run_dir(p: &Project, name: &str) -> PathBuf {
    p.root.join("runs").join(name)
}

fn row_to_info(p: &Project, r: &rusqlite::Row) -> rusqlite::Result<RunInfo> {
    let dir: String = r.get(1)?;
    let abs = run_dir(p, &dir);
    let config: String = r.get(4)?;
    Ok(RunInfo {
        id: r.get(0)?,
        name: dir.clone(),
        dir: abs.to_string_lossy().into_owned(),
        engine: r.get(2)?,
        status: r.get(3)?,
        config: serde_json::from_str(&config).unwrap_or(serde_json::Value::Null),
        best_map50: r.get(5)?,
        created_at: r.get(6)?,
        has_weights: abs.join("best.pt").exists(),
        has_onnx: abs.join("model.onnx").exists(),
    })
}

const RUN_COLS: &str = "id, dir, engine, status, config_json, best_metric, created_at";

pub fn list_runs(p: &Project) -> AppResult<Vec<RunInfo>> {
    let conn = p.conn.lock();
    let mut stmt = conn.prepare(&format!("SELECT {RUN_COLS} FROM runs ORDER BY id DESC"))?;
    let rows = stmt.query_map([], |r| row_to_info(p, r))?.collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

pub fn get_run(p: &Project, id: i64) -> AppResult<RunInfo> {
    let conn = p.conn.lock();
    conn.query_row(&format!("SELECT {RUN_COLS} FROM runs WHERE id = ?1"), [id], |r| row_to_info(p, r))
        .optional()?
        .ok_or_else(|| msg("训练记录不存在"))
}

fn set_status(p: &Project, id: i64, status: &str) {
    let _ = p.conn.lock().execute("UPDATE runs SET status = ?2 WHERE id = ?1", params![id, status]);
}

/// 打开项目时，上次异常退出没收尾的「训练中」记录改成「已中断」
pub fn recover(p: &Project, jobs: &Jobs) {
    let running = jobs.running();
    let _ = p.conn.lock().execute(
        "UPDATE runs SET status = 'interrupted' WHERE status = 'running' AND id != ?1",
        [running.unwrap_or(-1)],
    );
}

pub fn detail(p: &Project, id: i64) -> AppResult<RunDetail> {
    let run = get_run(p, id)?;
    let dir = PathBuf::from(&run.dir);
    let metrics = fs::read_to_string(dir.join("metrics.jsonl"))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    let eval = fs::read_to_string(dir.join("eval.json")).ok().and_then(|s| serde_json::from_str(&s).ok());
    let log_text = fs::read_to_string(dir.join("log.txt")).unwrap_or_default();
    let lines: Vec<String> = log_text.lines().map(String::from).collect();
    let log = lines[lines.len().saturating_sub(300)..].to_vec();
    let error = fs::read_to_string(dir.join("error.txt")).ok();
    Ok(RunDetail { run, metrics, eval, log, error })
}

pub fn delete_run(p: &Project, jobs: &Jobs, id: i64) -> AppResult<()> {
    if jobs.running() == Some(id) {
        return Err(msg("训练还在进行，先停止再删除"));
    }
    let run = get_run(p, id)?;
    let _ = fs::remove_dir_all(&run.dir);
    p.conn.lock().execute("DELETE FROM runs WHERE id = ?1", [id])?;
    Ok(())
}

fn now_name() -> String {
    chrono::Local::now().format("%Y%m%d_%H%M%S").to_string()
}

/// 训练过程中的事件，发给前端
pub type EventSink = Arc<dyn Fn(i64, serde_json::Value) + Send + Sync>;

/// 准备数据集并启动训练进程，立即返回训练记录 id
pub fn start(p: Arc<Project>, rt: &Runtime, jobs: Arc<Jobs>, cfg: TrainConfig, sink: EventSink) -> AppResult<i64> {
    if jobs.running().is_some() {
        return Err(msg("已经有训练在进行中"));
    }
    if cfg.epochs == 0 || cfg.batch == 0 || cfg.img_size < 128 {
        return Err(msg("训练参数不对"));
    }
    let mut cmd = rt.worker(&["train"])?; // 先确认环境存在
    let name = {
        let mut n = now_name();
        let mut i = 1;
        while run_dir(&p, &n).exists() {
            n = format!("{}_{i}", now_name());
            i += 1;
        }
        n
    };
    let dir = run_dir(&p, &name);
    let ds = dir.join("dataset");
    let res = dataset::export(
        &p,
        &ExportOptions { format: Format::Coco, out_dir: ds.clone(), only_reviewed: cfg.only_reviewed, ratios: cfg.ratios },
        &|_, _| {},
    )
    .inspect_err(|_| {
        let _ = fs::remove_dir_all(&dir);
    })?;
    if res.boxes + res.polys == 0 {
        let _ = fs::remove_dir_all(&dir);
        return Err(msg("没有任何标注，先标注一些图片再训练"));
    }
    let worker_cfg = serde_json::json!({
        "dataset": ds, "out": dir, "model": cfg.model, "classes": res.classes, "task": p.meta.task,
        "epochs": cfg.epochs, "img_size": cfg.img_size, "batch": cfg.batch, "lr": cfg.lr, "device": cfg.device,
    });
    fs::write(dir.join("config.json"), serde_json::to_string_pretty(&worker_cfg)?)?;
    let mut saved = serde_json::to_value(&cfg)?;
    saved["classes"] = serde_json::json!(res.classes);
    saved["images"] = serde_json::json!({ "train": res.train, "val": res.val, "test": res.test, "boxes": res.boxes, "polys": res.polys });
    let id = {
        let conn = p.conn.lock();
        conn.execute(
            "INSERT INTO runs (dir, engine, task, status, config_json, created_at) VALUES (?1, 'torchvision', ?2, 'running', ?3, ?4)",
            params![name, p.meta.task, saved.to_string(), db::now()],
        )?;
        conn.last_insert_rowid()
    };

    cmd.arg(dir.join("config.json")).current_dir(&dir).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            set_status(&p, id, "failed");
            let _ = fs::write(dir.join("error.txt"), format!("无法启动训练进程：{e}"));
            return Err(msg(format!("无法启动训练进程：{e}")));
        }
    };
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let child = Arc::new(Mutex::new(child));
    *jobs.current.lock() = Some((id, dir.clone(), child.clone()));

    let log_file = Arc::new(Mutex::new(fs::OpenOptions::new().create(true).append(true).open(dir.join("log.txt"))?));
    let stderr_tail = Arc::new(Mutex::new(Vec::<String>::new()));
    {
        let log_file = log_file.clone();
        let tail = stderr_tail.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).split(b'\n').map_while(Result::ok) {
                let line = String::from_utf8_lossy(&line);
                let line = line.trim_end().rsplit('\r').next().unwrap_or("").to_string();
                // 下载权重的百分比进度不记录
                if line.is_empty() || line.trim_end_matches('%').parse::<f64>().is_ok() {
                    continue;
                }
                let _ = writeln!(log_file.lock(), "{line}");
                let mut t = tail.lock();
                t.push(line);
                if t.len() > 20 {
                    t.remove(0);
                }
            }
        });
    }
    std::thread::spawn(move || {
        let mut final_status = None;
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let Ok(ev) = serde_json::from_str::<serde_json::Value>(&line) else {
                let _ = writeln!(log_file.lock(), "{line}");
                continue;
            };
            match ev["type"].as_str() {
                Some("log") => {
                    let _ = writeln!(log_file.lock(), "{}", ev["msg"].as_str().unwrap_or(""));
                }
                Some("metric") => {
                    // 记录被选为最好模型的那一轮的 mAP50
                    if let (Some(m), true) = (ev["map50"].as_f64(), ev["best"] == true) {
                        let _ = p.conn.lock().execute("UPDATE runs SET best_metric = ?2 WHERE id = ?1", params![id, m]);
                    }
                    let _ = writeln!(
                        log_file.lock(),
                        "第 {} 轮：loss {:.4}，mAP50 {:.3}",
                        ev["epoch"],
                        ev["loss"].as_f64().unwrap_or(f64::NAN),
                        ev["map50"].as_f64().unwrap_or(0.0)
                    );
                }
                Some("done") => {
                    final_status = Some(if ev["stopped"] == true { "stopped" } else { "done" });
                }
                Some("error") => {
                    let m = ev["msg"].as_str().unwrap_or("未知错误").to_string();
                    let _ = fs::write(dir.join("error.txt"), format!("{m}\n\n{}", ev["trace"].as_str().unwrap_or("")));
                    let _ = writeln!(log_file.lock(), "错误：{m}");
                    final_status = Some("failed");
                }
                _ => {}
            }
            sink(id, ev);
        }
        let code = child.lock().wait().ok();
        let status = final_status.unwrap_or_else(|| {
            // 进程直接退出（崩溃、显存不足等）
            let tail = stderr_tail.lock().join("\n");
            let _ = fs::write(dir.join("error.txt"), format!("训练进程异常退出（{code:?}）\n\n{tail}"));
            if dir.join("STOP").exists() { "stopped" } else { "failed" }
        });
        let _ = fs::remove_file(dir.join("STOP"));
        set_status(&p, id, status);
        *jobs.current.lock() = None;
        sink(id, serde_json::json!({ "type": "finished", "status": status }));
    });
    Ok(id)
}

/// 请求停止：写 STOP 文件让训练在当前迭代后停下，60 秒还没退出就强制结束
pub fn stop(jobs: &Jobs, id: i64) -> AppResult<()> {
    let cur = jobs.current.lock();
    let Some((rid, dir, child)) = cur.as_ref() else { return Err(msg("没有正在进行的训练")) };
    if *rid != id {
        return Err(msg("这个训练没有在进行"));
    }
    fs::write(dir.join("STOP"), "")?;
    let child = child.clone();
    std::thread::spawn(move || {
        for _ in 0..60 {
            std::thread::sleep(std::time::Duration::from_secs(1));
            if matches!(child.lock().try_wait(), Ok(Some(_))) {
                return;
            }
        }
        let _ = child.lock().kill();
    });
    Ok(())
}

/// 分割模型（Mask R-CNN）不支持当前的 ONNX / 部署流程，提前拦住给出清楚提示
pub fn ensure_exportable(p: &Project, id: i64) -> AppResult<()> {
    let run = get_run(p, id)?;
    if run.config["model"].as_str().map(is_seg_model).unwrap_or(false) {
        return Err(msg("分割模型暂不支持导出 ONNX / 生成部署包，可以在软件内直接用它做预标注"));
    }
    Ok(())
}

pub fn export_onnx(p: &Project, rt: &Runtime, id: i64, log: &(dyn Fn(String) + Sync)) -> AppResult<String> {
    ensure_exportable(p, id)?;
    let run = get_run(p, id)?;
    let dir = PathBuf::from(&run.dir);
    let w = dir.join("best.pt");
    if !w.exists() {
        return Err(msg("这次训练没有产出模型"));
    }
    let out = dir.join("model.onnx");
    let mut c = rt.worker(&["export"])?;
    c.arg(&w).arg(&out);
    crate::runtime::run_logged(c, &|line| {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(m) = v["msg"].as_str() {
                log(m.to_string());
            }
        }
    })?;
    if !out.exists() {
        return Err(msg("导出失败"));
    }
    Ok(out.to_string_lossy().into_owned())
}

// ---------- 推理服务（预标注） ----------

struct ServeProc {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

#[derive(Default)]
pub struct Predictor {
    proc: Mutex<Option<ServeProc>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PredBox {
    pub label_id: i64,
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PredPoly {
    pub label_id: i64,
    /// 多边形点 [x1, y1, x2, y2, ...]，原图像素坐标
    pub points: Vec<f64>,
    pub score: f64,
}

/// 预测结果：检测模型给框，分割模型给多边形
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PredResult {
    pub boxes: Vec<PredBox>,
    pub polys: Vec<PredPoly>,
}

/// 推理进程返回的一张图的原始结果
#[derive(Debug, Clone, Default)]
pub struct RawPred {
    /// [x, y, w, h, 类别序号, 分数]
    pub boxes: Vec<[f64; 6]>,
    /// (类别序号, 分数, [x1, y1, ...])
    pub polys: Vec<(usize, f64, Vec<f64>)>,
}

/// 分割模型是不是 Mask R-CNN（Rust 侧判断，和 worker 的 MODELS 对应）
fn is_seg_model(model: &str) -> bool {
    model.starts_with("maskrcnn")
}

impl Predictor {
    pub fn shutdown(&self) {
        if let Some(mut s) = self.proc.lock().take() {
            let _ = writeln!(s.stdin, "{{\"cmd\":\"quit\"}}");
            let _ = s.child.kill();
        }
    }

    /// 发一次预测请求，每张图的结果回调一次
    fn request(
        &self,
        rt: &Runtime,
        weights: &Path,
        items: &[(i64, PathBuf)],
        conf: f64,
        on_pred: &mut dyn FnMut(i64, RawPred),
    ) -> AppResult<Vec<String>> {
        let mut guard = self.proc.lock();
        let alive = guard.as_mut().map(|s| matches!(s.child.try_wait(), Ok(None))).unwrap_or(false);
        if !alive {
            let mut c = rt.worker(&["serve"])?;
            c.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
            let mut child = c.spawn().map_err(|e| msg(format!("无法启动推理进程：{e}")))?;
            let stdin = child.stdin.take().unwrap();
            let stdout = BufReader::new(child.stdout.take().unwrap());
            *guard = Some(ServeProc { child, stdin, stdout });
        }
        let s = guard.as_mut().unwrap();
        let req = serde_json::json!({
            "cmd": "predict", "weights": weights, "conf": conf,
            "images": items.iter().map(|(id, p)| serde_json::json!({ "id": id, "path": p })).collect::<Vec<_>>(),
        });
        if writeln!(s.stdin, "{req}").and_then(|_| s.stdin.flush()).is_err() {
            *guard = None;
            return Err(msg("推理进程已退出，请重试"));
        }
        let mut line = String::new();
        loop {
            line.clear();
            if s.stdout.read_line(&mut line)? == 0 {
                *guard = None;
                return Err(msg("推理进程意外退出"));
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line.trim()) else { continue };
            match v["type"].as_str() {
                Some("pred") => {
                    let boxes: Vec<[f64; 6]> = serde_json::from_value(v["boxes"].clone()).unwrap_or_default();
                    let mut polys = Vec::new();
                    if let Some(arr) = v["polys"].as_array() {
                        for pl in arr {
                            let Some(a) = pl.as_array() else { continue };
                            // [类别序号, 分数, x1, y1, ...]，至少 3 个点
                            if a.len() < 8 {
                                continue;
                            }
                            let cls = a[0].as_f64().unwrap_or(0.0) as usize;
                            let score = a[1].as_f64().unwrap_or(0.0);
                            let pts: Vec<f64> = a[2..].iter().filter_map(|x| x.as_f64()).collect();
                            if pts.len() >= 6 && pts.len() % 2 == 0 {
                                polys.push((cls, score, pts));
                            }
                        }
                    }
                    on_pred(v["id"].as_i64().unwrap_or(-1), RawPred { boxes, polys });
                }
                Some("done") => {
                    return Ok(serde_json::from_value(v["classes"].clone()).unwrap_or_default());
                }
                Some("error") => return Err(msg(v["msg"].as_str().unwrap_or("推理失败").to_string())),
                _ => {}
            }
        }
    }
}

/// 模型的类别名对应到项目的类别 id，项目里没有的自动新建
fn class_map(p: &Project, classes: &[String]) -> AppResult<Vec<i64>> {
    const PALETTE: [&str; 6] = ["#3d8bfd", "#e5534b", "#2fb67c", "#e7a33e", "#9b7bf2", "#14b8a6"];
    let mut have: HashMap<String, i64> = annotations::list_labels(p)?.into_iter().map(|l| (l.name, l.id)).collect();
    let mut out = Vec::new();
    for (i, c) in classes.iter().enumerate() {
        let id = match have.get(c) {
            Some(id) => *id,
            None => {
                let l = annotations::add_label(p, c, PALETTE[i % PALETTE.len()])?;
                have.insert(c.clone(), l.id);
                l.id
            }
        };
        out.push(id);
    }
    Ok(out)
}

fn weights_of(p: &Project, run_id: i64) -> AppResult<PathBuf> {
    let w = PathBuf::from(get_run(p, run_id)?.dir).join("best.pt");
    if !w.exists() {
        return Err(msg("这次训练没有产出模型"));
    }
    Ok(w)
}

fn to_boxes(raw: Vec<[f64; 6]>, map: &[i64]) -> Vec<PredBox> {
    raw.into_iter()
        .filter_map(|b| {
            let label_id = *map.get(b[4] as usize)?;
            Some(PredBox { label_id, x: b[0], y: b[1], w: b[2], h: b[3], score: b[5] })
        })
        .collect()
}

fn to_polys(raw: Vec<(usize, f64, Vec<f64>)>, map: &[i64]) -> Vec<PredPoly> {
    raw.into_iter()
        .filter_map(|(cls, score, points)| {
            let label_id = *map.get(cls)?;
            Some(PredPoly { label_id, points, score })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::tests::{bbox, temp};
    use crate::images;
    use image::{Rgb, RgbImage};

    /// 端到端：需要设置 XLABEL_TEST_PYTHON 指向装有 torch 的 Python，否则跳过。
    /// 可选 XLABEL_TEST_TORCH_HOME：已下载的预训练权重目录，避免重复下载。
    #[test]
    fn train_predict_export_end_to_end() {
        let Ok(py) = std::env::var("XLABEL_TEST_PYTHON") else {
            eprintln!("跳过：没有设置 XLABEL_TEST_PYTHON");
            return;
        };
        let base = temp("e2e");
        let src = base.join("src");
        fs::create_dir_all(&src).unwrap();
        // 20 张图：灰底上一个红方块
        for i in 0..20u32 {
            let mut im = RgbImage::from_pixel(200, 150, Rgb([120, 120, 120]));
            let (x0, y0) = (10 + (i * 7) % 120, 10 + (i * 11) % 80);
            for x in x0..x0 + 50 {
                for y in y0..y0 + 40 {
                    im.put_pixel(x, y, Rgb([220, 30, 30]));
                }
            }
            im.save(src.join(format!("{i:02}.jpg"))).unwrap();
        }
        let p = Arc::new(Project::create(&base, "p", "detect").unwrap());
        images::import(&p, &images::collect(&[src]), true, &|_, _| {}).unwrap();
        let red = annotations::add_label(&p, "红方块", "#f00").unwrap();
        for id in 1..=16i64 {
            let i = (id - 1) as u32;
            let (x0, y0) = (10 + (i * 7) % 120, 10 + (i * 11) % 80);
            annotations::save(&p, id, &[bbox(red.id, x0 as f64, y0 as f64, 50.0, 40.0)]).unwrap();
        }

        let data = base.join("appdata");
        if let Ok(th) = std::env::var("XLABEL_TEST_TORCH_HOME") {
            let dst = data.join("torch").join("hub").join("checkpoints");
            fs::create_dir_all(&dst).unwrap();
            for f in fs::read_dir(PathBuf::from(th).join("hub").join("checkpoints")).unwrap().flatten() {
                fs::copy(f.path(), dst.join(f.file_name())).unwrap();
            }
        }
        let worker = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../worker");
        let rt = Runtime::new(base.join("cfg"), data, worker);
        rt.save_settings(&crate::runtime::Settings { python: Some(py), ..Default::default() }).unwrap();
        assert!(rt.status(true).ready, "环境应可用");

        let jobs = Arc::new(Jobs::default());
        let events = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
        let ev2 = events.clone();
        let cfg = TrainConfig {
            model: "frcnn_mobilenet".into(),
            epochs: 2,
            img_size: 224,
            batch: 4,
            lr: 0.01,
            device: "cpu".into(),
            only_reviewed: false,
            ratios: [0.75, 0.25, 0.0],
        };
        let id = start(p.clone(), &rt, jobs.clone(), cfg, Arc::new(move |_, e| ev2.lock().push(e))).unwrap();
        assert!(start(p.clone(), &rt, jobs.clone(), TrainConfig { epochs: 1, ..serde_json::from_value(get_run(&p, id).unwrap().config).unwrap() }, Arc::new(|_, _| {})).is_err(), "同时只能一个训练");
        for _ in 0..600 {
            if jobs.running().is_none() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        let d = detail(&p, id).unwrap();
        assert_eq!(d.run.status, "done", "日志：{:?} 错误：{:?}", d.log, d.error);
        assert_eq!(d.metrics.len(), 2);
        assert!(d.run.has_weights && d.eval.is_some());
        assert!(events.lock().iter().any(|e| e["type"] == "metric"));
        assert!(events.lock().iter().any(|e| e["type"] == "finished"));

        let pr = Predictor::default();
        let boxes = predict_one(&p, &rt, &pr, id, 1, 0.05).unwrap();
        eprintln!("预测：{boxes:?}");
        let r = prelabel_all(&p, &rt, &pr, id, 0.05, &|_, _| {}).unwrap();
        eprintln!("预标注：{r:?}");
        // 预标注的图仍是未标注
        let st: String = p.conn.lock().query_row("SELECT status FROM images WHERE id = 20", [], |r| r.get(0)).unwrap();
        assert_eq!(st, "todo");
        // 体检（模型检查）：故意把图 2 的框挪走，应该查出漏标和多余框
        annotations::save(&p, 2, &[bbox(red.id, 140.0, 100.0, 50.0, 40.0)]).unwrap();
        let predict = |ids: &[i64]| predict_many(&p, &rt, &pr, id, ids, 0.05, &|_, _| {});
        let report = crate::health::run(&p, Some((id, "r".into())), Some(&predict), &|_, _| {}).unwrap();
        let on2: Vec<&str> = report.issues.iter().filter(|i| i.image_id == 2).map(|i| i.kind.as_str()).collect();
        eprintln!("体检：{} 个问题，图 2：{on2:?}", report.issues.len());
        assert!(on2.contains(&"missing") && on2.contains(&"extra_box"), "{on2:?}");
        pr.shutdown();

        // 部署：服务包、多格式导出、实时验证
        let svc = base.join("服务包");
        crate::deploy::build_service(&p, &rt, id, &serde_json::json!({ "port": 8090, "streams": [] }), &svc, &|l| eprintln!("{l}")).unwrap();
        for f in ["model.onnx", "model.json", "service.py", "config.json", "run.bat", "README.md"] {
            assert!(svc.join(f).exists(), "服务包缺少 {f}");
        }
        let fmts = crate::deploy::export_formats(&p, &rt, id, &["onnx_static".into(), "openvino".into(), "rknn".into()], 224, &base.join("导出"), &|l| eprintln!("{l}")).unwrap();
        eprintln!("导出：{fmts:?}");
        assert!(fmts.iter().all(|f| f.ok), "{fmts:?}");
        if let Ok(video) = std::env::var("XLABEL_TEST_VIDEO") {
            let live = Arc::new(crate::deploy::Live::default());
            let evs = Arc::new(Mutex::new(Vec::<(String, serde_json::Value)>::new()));
            let e2 = evs.clone();
            live.start(p.clone(), &rt, id, &video, 0.3, Arc::new(move |k, v| e2.lock().push((k.to_string(), v)))).unwrap();
            for _ in 0..100 {
                if evs.lock().iter().filter(|e| e.0 == "frame").count() >= 3 {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
            let before = p.stats().unwrap().total;
            live.snapshot(&p).unwrap();
            for _ in 0..50 {
                if evs.lock().iter().any(|e| e.0 == "snapshot") {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
            live.stop();
            let frames = evs.lock().iter().filter(|e| e.0 == "frame").count();
            eprintln!("实时验证：{frames} 帧，截图事件 {:?}", evs.lock().iter().find(|e| e.0 == "snapshot"));
            assert!(frames >= 3);
            assert_eq!(p.stats().unwrap().total, before + 1, "截图要导入项目");
        }

        let onnx = export_onnx(&p, &rt, id, &|l| eprintln!("{l}")).unwrap();
        assert!(Path::new(&onnx).exists());
        assert!(get_run(&p, id).unwrap().has_onnx);
        delete_run(&p, &jobs, id).unwrap();
        assert!(list_runs(&p).unwrap().is_empty());
        let _ = fs::remove_dir_all(&base);
    }
}

/// 预测单张图，结果交给前端放进画布（不直接保存）
pub fn predict_one(p: &Project, rt: &Runtime, pr: &Predictor, run_id: i64, image_id: i64, conf: f64) -> AppResult<PredResult> {
    let w = weights_of(p, run_id)?;
    let (stored, _) = crate::images::locate(p, image_id)?;
    let mut raw = RawPred::default();
    let classes = pr.request(rt, &w, &[(image_id, p.resolve(&stored))], conf, &mut |_, r| raw = r)?;
    let map = class_map(p, &classes)?;
    Ok(PredResult { boxes: to_boxes(raw.boxes, &map), polys: to_polys(raw.polys, &map) })
}

/// 批量预测指定图片，返回每张图的预测框（体检用）
pub fn predict_many(
    p: &Project,
    rt: &Runtime,
    pr: &Predictor,
    run_id: i64,
    ids: &[i64],
    conf: f64,
    progress: &(dyn Fn(usize, usize) + Sync),
) -> AppResult<HashMap<i64, Vec<crate::health::Pred>>> {
    let w = weights_of(p, run_id)?;
    let items: Vec<(i64, PathBuf)> = {
        let conn = p.conn.lock();
        let mut stmt = conn.prepare("SELECT path FROM images WHERE id = ?1")?;
        ids.iter()
            .filter_map(|id| stmt.query_row([id], |r| r.get::<_, String>(0)).ok().map(|s| (*id, p.resolve(&s))))
            .collect()
    };
    let total = items.len();
    let mut out = HashMap::new();
    let mut map: Option<Vec<i64>> = None;
    progress(0, total);
    let mut done = 0;
    for chunk in items.chunks(16) {
        let mut got = Vec::new();
        let classes = pr.request(rt, &w, chunk, conf, &mut |id, r| got.push((id, r.boxes)))?;
        if map.is_none() {
            map = Some(class_map(p, &classes)?);
        }
        for (id, raw) in got {
            let boxes = to_boxes(raw, map.as_ref().unwrap())
                .into_iter()
                .map(|b| crate::health::Pred { label_id: b.label_id, bbox: [b.x, b.y, b.w, b.h], score: b.score })
                .collect();
            out.insert(id, boxes);
        }
        done += chunk.len();
        progress(done, total);
    }
    Ok(out)
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PrelabelResult {
    pub images: usize,
    pub boxes: usize,
}

/// 批量预标注所有「未标注」且没有任何框的图。结果存为模型来源，图片仍是「未标注」，等人工确认。
pub fn prelabel_all(
    p: &Project,
    rt: &Runtime,
    pr: &Predictor,
    run_id: i64,
    conf: f64,
    progress: &dyn Fn(usize, usize),
) -> AppResult<PrelabelResult> {
    let w = weights_of(p, run_id)?;
    let items: Vec<(i64, PathBuf)> = {
        let conn = p.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, path FROM images WHERE status = 'todo'
             AND NOT EXISTS (SELECT 1 FROM annotations a WHERE a.image_id = images.id) ORDER BY id",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
        rows.map(|r| r.map(|(id, path)| (id, p.resolve(&path)))).collect::<Result<_, _>>()?
    };
    let total = items.len();
    let mut result = PrelabelResult { images: 0, boxes: 0 };
    let mut map: Option<Vec<i64>> = None;
    let mut done = 0;
    progress(0, total);
    for chunk in items.chunks(16) {
        let mut got: Vec<(i64, RawPred)> = Vec::new();
        let classes = pr.request(rt, &w, chunk, conf, &mut |id, r| got.push((id, r)))?;
        if map.is_none() {
            map = Some(class_map(p, &classes)?);
        }
        let m = map.as_ref().unwrap();
        for (id, raw) in got {
            // 分割模型给多边形，检测模型给框
            let polys = to_polys(raw.polys, m);
            let anns: Vec<Annotation> = if !polys.is_empty() {
                polys
                    .iter()
                    .map(|pl| Annotation {
                        label_id: Some(pl.label_id),
                        kind: "polygon".into(),
                        data: serde_json::json!(pl.points),
                        score: Some(pl.score),
                        source: "model".into(),
                    })
                    .collect()
            } else {
                to_boxes(raw.boxes, m)
                    .iter()
                    .map(|b| Annotation {
                        label_id: Some(b.label_id),
                        kind: "bbox".into(),
                        data: serde_json::json!([b.x, b.y, b.w, b.h]),
                        score: Some(b.score),
                        source: "model".into(),
                    })
                    .collect()
            };
            if !anns.is_empty() {
                result.boxes += anns.len();
                result.images += 1;
                annotations::save_predictions(p, id, &anns)?;
            }
        }
        done += chunk.len();
        progress(done, total);
    }
    Ok(result)
}
