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
use crate::t;

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
    /// 用哪个数据集版本训练；为空用当前数据
    #[serde(default)]
    pub version: Option<i64>,
    /// 接着哪次训练的模型继续训；为空从预训练权重开始
    #[serde(default)]
    pub base_run: Option<i64>,
    /// 数据增强参数（原样交给 worker，见 worker 的 augment.py）；为空用默认
    #[serde(default)]
    pub aug: Option<serde_json::Value>,
    /// 训练引擎：torchvision（内置模型，默认）或 ultralytics（YOLOv8 / 11 等；model 是模型名或本地 .pt / .yaml 路径）
    #[serde(default = "default_engine")]
    pub engine: String,
    /// Ultralytics 的「更多参数」：一行一个 key=value，原样交给训练
    #[serde(default)]
    pub extra: Option<String>,
}

fn default_engine() -> String {
    TORCHVISION.into()
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
    /// 项目任务（detect / segment / classify / multilabel / pose），训练或导入时记下
    pub task: String,
    pub has_weights: bool,
    pub has_onnx: bool,
    /// 停止、中断或失败的训练，留有断点（resume.pt）：可以从最后一个完整的轮次接着训练
    pub can_resume: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunDetail {
    pub run: RunInfo,
    pub metrics: Vec<serde_json::Value>,
    pub eval: Option<serde_json::Value>,
    pub log: Vec<String>,
    /// 日志开头几行（设备、数据量、预训练权重），日志长了 log 里就没有了
    pub log_head: Vec<String>,
    /// worker 发来的醒目提示（warn 事件：code、msg），存在 warnings.jsonl
    pub warnings: Vec<serde_json::Value>,
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

/// 导入的外部模型（别处训好的 ONNX），训练记录里 engine 记为这个
pub const EXTERNAL: &str = "external";
/// 内置模型（torchvision、YOLOX）训练出的是 best.pt；其他引擎（Ultralytics、外部导入）统一用 model.onnx
pub const TORCHVISION: &str = "torchvision";
/// Ultralytics 训练引擎（YOLOv8 / 11 / 12 / 26、RT-DETR 等）：原生权重在 native/weights/，训完导出 model.onnx
pub const ULTRALYTICS: &str = "ultralytics";

/// Ultralytics 训练的原生权重（接着训练、断点续训用）
fn native_weights(dir: &Path, file: &str) -> PathBuf {
    dir.join("native").join("weights").join(file)
}

/// 这条训练记录的模型文件：内置模型是 best.pt，其余都是 model.onnx
pub fn model_path(dir: &Path, engine: &str) -> PathBuf {
    if engine == TORCHVISION {
        dir.join("best.pt")
    } else {
        dir.join("model.onnx")
    }
}

fn row_to_info(p: &Project, r: &rusqlite::Row) -> rusqlite::Result<RunInfo> {
    let dir: String = r.get(1)?;
    let abs = run_dir(p, &dir);
    let config: String = r.get(4)?;
    let engine: String = r.get(2)?;
    let status: String = r.get(3)?;
    let can_resume = matches!(status.as_str(), "stopped" | "interrupted" | "failed")
        && ((engine == TORCHVISION && abs.join("resume.pt").exists()) || (engine == ULTRALYTICS && native_weights(&abs, "last.pt").exists()));
    Ok(RunInfo {
        id: r.get(0)?,
        name: dir.clone(),
        dir: abs.to_string_lossy().into_owned(),
        status,
        config: serde_json::from_str(&config).unwrap_or(serde_json::Value::Null),
        best_map50: r.get(5)?,
        created_at: r.get(6)?,
        task: r.get::<_, Option<String>>(7)?.unwrap_or_default(),
        has_weights: model_path(&abs, &engine).exists(),
        has_onnx: abs.join("model.onnx").exists(),
        can_resume,
        engine,
    })
}

const RUN_COLS: &str = "id, dir, engine, status, config_json, best_metric, created_at, task";

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
        .ok_or_else(|| msg(t!("训练记录不存在")))
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
    let log_head = lines.iter().take(15).cloned().collect();
    let warnings = read_warnings(&dir);
    let error = fs::read_to_string(dir.join("error.txt")).ok();
    Ok(RunDetail { run, metrics, eval, log, log_head, warnings, error })
}

fn read_warnings(dir: &Path) -> Vec<serde_json::Value> {
    fs::read_to_string(dir.join("warnings.jsonl"))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// 记下一条醒目提示，同一种（code）只记一次
fn add_warning(dir: &Path, ev: &serde_json::Value) {
    if read_warnings(dir).iter().any(|w| w["code"] == ev["code"]) {
        return;
    }
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(dir.join("warnings.jsonl")) {
        let _ = writeln!(f, "{ev}");
    }
}

pub fn delete_run(p: &Project, jobs: &Jobs, id: i64) -> AppResult<()> {
    if jobs.running() == Some(id) {
        return Err(msg(t!("训练还在进行，先停止再删除")));
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

/// 准备数据集并启动训练进程，立即返回训练记录 id。prepare：准备数据（导出数据集副本）的进度
pub fn start(
    p: Arc<Project>,
    rt: &Runtime,
    jobs: Arc<Jobs>,
    cfg: TrainConfig,
    sink: EventSink,
    prepare: &dyn Fn(usize, usize),
) -> AppResult<i64> {
    if jobs.running().is_some() {
        return Err(msg(t!("已经有训练在进行中")));
    }
    let ultra = cfg.engine == ULTRALYTICS;
    if !ultra && cfg.engine != TORCHVISION {
        return Err(msg(t!("训练参数不对")));
    }
    // Ultralytics 的批大小可以填 0：按显存自动选
    if cfg.epochs == 0 || (cfg.batch == 0 && !ultra) || cfg.img_size < 128 {
        return Err(msg(t!("训练参数不对")));
    }
    let cmd = rt.worker(&["train"])?; // 先确认环境存在
    if ultra {
        if p.meta.task == "multilabel" {
            return Err(msg(t!("Ultralytics 的分类模型不支持多标签分类，请选内置模型")));
        }
        let m = cfg.model.trim();
        if m.is_empty() {
            return Err(msg(t!("这个模型和项目的任务类型不匹配，请重新选择模型")));
        }
        // 选的是本地文件（带路径）时，文件要在
        if (m.contains('/') || m.contains('\\')) && !Path::new(m).is_file() {
            return Err(msg(t!("模型文件不存在")));
        }
        // 第一次用：装上 Ultralytics 组件（右下角显示安装进度）
        rt.ensure_ultralytics(&|_| {})?;
    }
    let name = {
        let mut n = now_name();
        let mut i = 1;
        while run_dir(&p, &n).exists() {
            n = format!("{}_{i}", now_name());
            i += 1;
        }
        n
    };
    // 多标签分类用的也是分类模型（训练时按数据集换成多标签的训练方式）
    let task = if p.meta.task == "multilabel" { "classify" } else { p.meta.task.as_str() };
    // Ultralytics 的模型任务由 worker 加载模型后检查（本地 .pt 只有加载了才知道）
    if !ultra && model_task(&cfg.model) != task {
        return Err(msg(t!("这个模型和项目的任务类型不匹配，请重新选择模型")));
    }
    // 接着上次训练：上次要有模型文件，模型结构要一样
    let base = match cfg.base_run {
        Some(b) => {
            let r = get_run(&p, b)?;
            if r.engine == EXTERNAL {
                return Err(msg(t!("导入的外部模型不能接着训练")));
            }
            if r.engine != cfg.engine {
                return Err(msg(t!("接着训练要用和那次相同的模型")));
            }
            if ultra {
                // Ultralytics：从那次的 best.pt 接着训，类别变了会自动换输出层
                let w = native_weights(Path::new(&r.dir), "best.pt");
                if !w.exists() {
                    return Err(msg(t!("那次训练没有产出模型，不能接着训")));
                }
                Some((r.name, w))
            } else {
                let w = PathBuf::from(&r.dir).join("best.pt");
                if !w.exists() {
                    return Err(msg(t!("那次训练没有产出模型，不能接着训")));
                }
                if r.config["model"].as_str() != Some(cfg.model.as_str()) {
                    return Err(msg(t!("接着训练要用和那次相同的模型")));
                }
                Some((r.name, w))
            }
        }
        None => None,
    };
    let dir = run_dir(&p, &name);
    let ds = dir.join("dataset");
    // 分类项目导出分类文件夹，其他任务导出 COCO（框、多边形、关键点都在里面）
    let format = if crate::project::is_classify(&p.meta.task) { Format::Cls } else { Format::Coco };
    let opts = ExportOptions { format, out_dir: ds.clone(), only_reviewed: cfg.only_reviewed, ratios: cfg.ratios, copy_files: false };
    let res = match cfg.version {
        Some(v) => dataset::export_version(&p, v, &opts, prepare),
        None => dataset::export(&p, &opts, prepare),
    }
    .inspect_err(|_| {
        let _ = fs::remove_dir_all(&dir);
    })?;
    let usable = match p.meta.task.as_str() {
        "classify" | "multilabel" => res.tags,
        "pose" => res.kpts,
        _ => res.boxes + res.polys,
    };
    if usable == 0 {
        let _ = fs::remove_dir_all(&dir);
        return Err(msg(match p.meta.task.as_str() {
            "classify" | "multilabel" => t!("还没有给图片打类别标签，先标注一些图片再训练"),
            "pose" => t!("还没有关键点标注，先标注一些目标再训练"),
            _ => t!("没有任何标注，先标注一些图片再训练"),
        }));
    }
    // 手动把图都移出了训练集
    if res.train == 0 {
        let _ = fs::remove_dir_all(&dir);
        return Err(msg(t!("训练集是空的：在训练页「查看和调整」里把一些图移到训练集")));
    }
    // Ultralytics 训练检测、分割、关键点用 YOLO 格式（再导一份，划分和上面一样）；分类直接用分类文件夹。COCO 那份留给评估
    let ds_yolo = dir.join("dataset_yolo");
    if ultra && !crate::project::is_classify(&p.meta.task) {
        let yopts = ExportOptions { format: Format::Yolo, out_dir: ds_yolo.clone(), only_reviewed: cfg.only_reviewed, ratios: cfg.ratios, copy_files: false };
        match cfg.version {
            Some(v) => dataset::export_version(&p, v, &yopts, &|_, _| {}),
            None => dataset::export(&p, &yopts, &|_, _| {}),
        }
        .inspect_err(|_| {
            let _ = fs::remove_dir_all(&dir);
        })?;
    }
    let mut worker_cfg = serde_json::json!({
        "dataset": ds, "out": dir, "model": cfg.model.trim(), "classes": res.classes, "task": p.meta.task,
        "epochs": cfg.epochs, "img_size": cfg.img_size, "batch": cfg.batch, "lr": cfg.lr, "device": cfg.device,
    });
    if ultra {
        worker_cfg["engine"] = serde_json::json!(ULTRALYTICS);
        worker_cfg["extra"] = serde_json::json!(cfg.extra.clone().unwrap_or_default());
        if ds_yolo.exists() {
            worker_cfg["data_yolo"] = serde_json::json!(ds_yolo);
        }
    }
    if let Some(a) = &cfg.aug {
        worker_cfg["aug"] = a.clone();
    }
    if let Some((_, w)) = &base {
        worker_cfg["init_weights"] = serde_json::json!(w);
    }
    if p.meta.task == "pose" {
        let t = crate::keypoints::get(&p)?;
        worker_cfg["num_keypoints"] = serde_json::json!(t.points.len());
        worker_cfg["keypoints"] = serde_json::json!(t.points);
        worker_cfg["flip_index"] = serde_json::json!(crate::keypoints::flip_index(&t.points));
    }
    fs::write(dir.join("config.json"), serde_json::to_string_pretty(&worker_cfg)?)?;
    let mut saved = serde_json::to_value(&cfg)?;
    saved["classes"] = serde_json::json!(res.classes);
    saved["images"] = serde_json::json!({
        "train": res.train, "val": res.val, "test": res.test,
        "boxes": res.boxes, "polys": res.polys, "anns": res.total(),
    });
    if let Some(v) = cfg.version {
        saved["versionName"] = serde_json::json!(crate::versions::name_of(&p, v).unwrap_or_default());
    }
    if let Some((name, _)) = &base {
        saved["baseRunName"] = serde_json::json!(name);
    }
    let id = {
        let conn = p.conn.lock();
        conn.execute(
            "INSERT INTO runs (dir, engine, task, status, config_json, created_at) VALUES (?1, ?2, ?3, 'running', ?4, ?5)",
            params![name, cfg.engine, p.meta.task, saved.to_string(), db::now()],
        )?;
        conn.last_insert_rowid()
    };

    launch(p, jobs, id, dir, cmd, sink)?;
    Ok(id)
}

/// 从断点继续训练：同一条训练记录、同一份数据集副本，从最后一个完整的轮次接着跑（曲线、日志接在后面）
pub fn resume(p: Arc<Project>, rt: &Runtime, jobs: Arc<Jobs>, id: i64, sink: EventSink) -> AppResult<()> {
    if jobs.running().is_some() {
        return Err(msg(t!("已经有训练在进行中")));
    }
    let r = get_run(&p, id)?;
    if !r.can_resume {
        return Err(msg(t!("这次训练没有留下可以接着跑的进度")));
    }
    let dir = PathBuf::from(&r.dir);
    let cfg_path = dir.join("config.json");
    let mut cfg: serde_json::Value = serde_json::from_str(&fs::read_to_string(&cfg_path)?)?;
    let ds = PathBuf::from(cfg["dataset"].as_str().unwrap_or(""));
    if !ds.exists() {
        return Err(msg(t!("这次训练用的数据集副本不见了，没法接着训，请重新开始训练")));
    }
    cfg["resume"] = serde_json::json!(true);
    fs::write(&cfg_path, serde_json::to_string_pretty(&cfg)?)?;
    let cmd = rt.worker(&["train"])?;
    // 上次的出错信息、停止标记作废
    let _ = fs::remove_file(dir.join("error.txt"));
    let _ = fs::remove_file(dir.join("STOP"));
    set_status(&p, id, "running");
    launch(p, jobs, id, dir, cmd, sink)
}

/// 启动训练进程并接管它的输出（新开训练、从断点继续训练共用）
fn launch(p: Arc<Project>, jobs: Arc<Jobs>, id: i64, dir: PathBuf, mut cmd: std::process::Command, sink: EventSink) -> AppResult<()> {
    cmd.arg(dir.join("config.json")).current_dir(&dir).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            set_status(&p, id, "failed");
            let _ = fs::write(dir.join("error.txt"), t!("无法启动训练进程：{e}", e = e));
            return Err(msg(t!("无法启动训练进程：{e}", e = e)));
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
        // 还没下完的文件：训练进程被停掉、崩了时，告诉界面下载结束了（不然右下角的进度条一直挂着）
        let mut downloading: std::collections::HashSet<String> = std::collections::HashSet::new();
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let Ok(ev) = serde_json::from_str::<serde_json::Value>(&line) else {
                let _ = writeln!(log_file.lock(), "{line}");
                continue;
            };
            match ev["type"].as_str() {
                Some("log") => {
                    let _ = writeln!(log_file.lock(), "{}", ev["msg"].as_str().unwrap_or(""));
                }
                Some("warn") => add_warning(&dir, &ev),
                // 下载预训练权重的进度：右下角进度条
                Some("download") => {
                    let name = ev["name"].as_str().unwrap_or("").to_string();
                    if ev["finished"] == true || ev["failed"] == true {
                        downloading.remove(&name);
                    } else {
                        downloading.insert(name);
                    }
                    crate::dlprogress::forward(&ev);
                }
                Some("batch") => {
                    // 显存不够自动调小了批大小：训练记录里改成实际用的，原来设的另存一份
                    // 会从头重训，上一次的提示作废（预训练权重的提示重训时会再发）
                    let _ = fs::remove_file(dir.join("warnings.jsonl"));
                    if let Some(b) = ev["batch"].as_u64() {
                        let conn = p.conn.lock();
                        let saved: Option<String> =
                            conn.query_row("SELECT config_json FROM runs WHERE id = ?1", [id], |r| r.get(0)).ok();
                        if let Some(mut c) = saved.and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()) {
                            if c.get("batchRequested").is_none() {
                                c["batchRequested"] = c["batch"].clone();
                            }
                            c["batch"] = serde_json::json!(b);
                            let _ = conn.execute("UPDATE runs SET config_json = ?2 WHERE id = ?1", params![id, c.to_string()]);
                        }
                    }
                }
                Some("metric") => {
                    // 记录被选为最好模型的那一轮的 mAP50
                    if let (Some(m), true) = (ev["map50"].as_f64(), ev["best"] == true) {
                        let _ = p.conn.lock().execute("UPDATE runs SET best_metric = ?2 WHERE id = ?1", params![id, m]);
                    }
                    let _ = writeln!(
                        log_file.lock(),
                        "{}",
                        t!("第 {epoch} 轮：loss {loss}，mAP50 {map}",
                            epoch = ev["epoch"],
                            loss = format!("{:.4}", ev["loss"].as_f64().unwrap_or(f64::NAN)),
                            map = format!("{:.3}", ev["map50"].as_f64().unwrap_or(0.0))
                        )
                    );
                }
                Some("done") => {
                    final_status = Some(if ev["stopped"] == true { "stopped" } else { "done" });
                }
                Some("error") => {
                    let m = ev["msg"].as_str().map(String::from).unwrap_or_else(|| t!("未知错误"));
                    let _ = fs::write(dir.join("error.txt"), format!("{m}\n\n{}", ev["trace"].as_str().unwrap_or("")));
                    let _ = writeln!(log_file.lock(), "{}", t!("错误：{m}", m = m));
                    final_status = Some("failed");
                }
                _ => {}
            }
            sink(id, ev);
        }
        for name in downloading {
            crate::dlprogress::forward(&serde_json::json!({ "type": "download", "name": name, "done": 0, "total": 0, "speed": 0, "failed": true }));
        }
        let code = child.lock().wait().ok();
        let status = final_status.unwrap_or_else(|| {
            // 进程直接退出（崩溃、显存不足等）
            let tail = stderr_tail.lock().join("\n");
            let _ = fs::write(dir.join("error.txt"), format!("{}\n\n{tail}", t!("训练进程异常退出（{code}）", code = format!("{code:?}"))));
            if dir.join("STOP").exists() { "stopped" } else { "failed" }
        });
        let _ = fs::remove_file(dir.join("STOP"));
        set_status(&p, id, status);
        *jobs.current.lock() = None;
        sink(id, serde_json::json!({ "type": "finished", "status": status }));
    });
    Ok(())
}

/// 请求停止：写 STOP 文件让训练在当前迭代后停下，60 秒还没退出就强制结束
pub fn stop(jobs: &Jobs, id: i64) -> AppResult<()> {
    let cur = jobs.current.lock();
    let Some((rid, dir, child)) = cur.as_ref() else { return Err(msg(t!("没有正在进行的训练"))) };
    if *rid != id {
        return Err(msg(t!("这个训练没有在进行")));
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

/// 模型对应的任务（和 worker 的 models.MODELS 对应）
pub fn model_task(model: &str) -> &'static str {
    if model.starts_with("maskrcnn") {
        "segment"
    } else if model.starts_with("keypointrcnn") {
        "pose"
    } else if model.starts_with("cls_") {
        "classify"
    } else {
        "detect"
    }
}

/// 部署用途：导出模型文件 / 生成推理服务包 / 实时验证
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Deploy {
    Onnx,
    Service,
    Live,
}

/// 内置模型、Ultralytics 等引擎训练的模型、导入的外部模型都支持全部部署方式（后两种本身就是 ONNX），只要有模型文件
pub fn ensure_deployable(p: &Project, id: i64, _what: Deploy) -> AppResult<()> {
    let run = get_run(p, id)?;
    if !run.has_weights {
        return Err(msg(t!("这次训练没有产出模型")));
    }
    Ok(())
}

pub fn export_onnx(p: &Project, rt: &Runtime, id: i64, log: &(dyn Fn(String) + Sync)) -> AppResult<String> {
    ensure_deployable(p, id, Deploy::Onnx)?;
    let run = get_run(p, id)?;
    let dir = PathBuf::from(&run.dir);
    let out = dir.join("model.onnx");
    // Ultralytics 等引擎和外部导入的本身就是 ONNX
    if run.engine != TORCHVISION {
        return Ok(out.to_string_lossy().into_owned());
    }
    let w = dir.join("best.pt");
    if !w.exists() {
        return Err(msg(t!("这次训练没有产出模型")));
    }
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
        return Err(msg(t!("导出失败")));
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

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PredTag {
    pub label_id: i64,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PredPose {
    pub label_id: i64,
    /// 目标框 [x, y, w, h]
    #[serde(rename = "box")]
    pub bbox: [f64; 4],
    /// 关键点 [x, y, v, ...]，v：0 没找到、2 找到
    pub kpts: Vec<f64>,
    pub score: f64,
}

/// 预测结果：检测模型给框，分割模型给多边形，关键点模型给 poses，分类模型给 tags（按分数从高到低）
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PredResult {
    pub boxes: Vec<PredBox>,
    pub polys: Vec<PredPoly>,
    pub tags: Vec<PredTag>,
    pub poses: Vec<PredPose>,
}

/// 推理进程返回的一张图的原始结果
#[derive(Debug, Clone, Default)]
pub struct RawPred {
    /// [x, y, w, h, 类别序号, 分数]
    pub boxes: Vec<[f64; 6]>,
    /// (类别序号, 分数, [x1, y1, ...])
    pub polys: Vec<(usize, f64, Vec<f64>)>,
    /// (类别序号, 分数)
    pub tags: Vec<(usize, f64)>,
    /// (类别序号, 分数, 框 [x, y, w, h], 关键点 [x, y, v, ...])
    pub poses: Vec<(usize, f64, [f64; 4], Vec<f64>)>,
}

/// 解析推理进程一行 pred 事件
pub fn parse_pred(v: &serde_json::Value) -> RawPred {
    let boxes: Vec<[f64; 6]> = serde_json::from_value(v["boxes"].clone()).unwrap_or_default();
    let nums = |x: &serde_json::Value| -> Vec<f64> {
        x.as_array().map(|a| a.iter().filter_map(|n| n.as_f64()).collect()).unwrap_or_default()
    };
    let mut polys = Vec::new();
    for pl in v["polys"].as_array().into_iter().flatten() {
        // [类别序号, 分数, x1, y1, ...]，至少 3 个点
        let a = nums(pl);
        if a.len() >= 8 && a.len() % 2 == 0 {
            polys.push((a[0] as usize, a[1], a[2..].to_vec()));
        }
    }
    let mut tags = Vec::new();
    for t in v["tags"].as_array().into_iter().flatten() {
        let a = nums(t);
        if a.len() >= 2 {
            tags.push((a[0] as usize, a[1]));
        }
    }
    let mut poses = Vec::new();
    for ps in v["poses"].as_array().into_iter().flatten() {
        // [类别序号, 分数, x, y, w, h, kx1, ky1, kv1, ...]
        let a = nums(ps);
        if a.len() >= 6 && (a.len() - 6) % 3 == 0 {
            poses.push((a[0] as usize, a[1], [a[2], a[3], a[4], a[5]], a[6..].to_vec()));
        }
    }
    RawPred { boxes, polys, tags, poses }
}

impl Predictor {
    pub fn shutdown(&self) {
        if let Some(mut s) = self.proc.lock().take() {
            let _ = writeln!(s.stdin, "{{\"cmd\":\"quit\"}}");
            let _ = s.child.kill();
        }
    }

    /// 给常驻推理进程发一个请求（没启动就先启动），一直读到 done / error 为止。
    /// 中间的事件（pred、sam、log 等）交给 on_event；返回 done 事件。
    pub fn call(
        &self,
        rt: &Runtime,
        req: &serde_json::Value,
        on_event: &mut dyn FnMut(&serde_json::Value),
    ) -> AppResult<serde_json::Value> {
        let mut guard = self.proc.lock();
        let alive = guard.as_mut().map(|s| matches!(s.child.try_wait(), Ok(None))).unwrap_or(false);
        if !alive {
            let mut c = rt.worker(&["serve"])?;
            c.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
            let mut child = c.spawn().map_err(|e| msg(t!("无法启动推理进程：{e}", e = e)))?;
            let stdin = child.stdin.take().unwrap();
            let stdout = BufReader::new(child.stdout.take().unwrap());
            *guard = Some(ServeProc { child, stdin, stdout });
        }
        let s = guard.as_mut().unwrap();
        if writeln!(s.stdin, "{req}").and_then(|_| s.stdin.flush()).is_err() {
            *guard = None;
            return Err(msg(t!("推理进程已退出，请重试")));
        }
        let mut line = String::new();
        loop {
            line.clear();
            if s.stdout.read_line(&mut line)? == 0 {
                *guard = None;
                return Err(msg(t!("推理进程意外退出")));
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line.trim()) else { continue };
            match v["type"].as_str() {
                Some("done") => return Ok(v),
                Some("error") => return Err(msg(v["msg"].as_str().map(String::from).unwrap_or_else(|| t!("推理失败")))),
                // 第一次用 SAM、文本提示时下载模型的进度
                Some("download") => crate::dlprogress::forward(&v),
                _ => on_event(&v),
            }
        }
    }

    /// 发一次预测请求，每张图的结果回调一次；返回模型的类别名
    pub(crate) fn request(
        &self,
        rt: &Runtime,
        weights: &Path,
        items: &[(i64, PathBuf)],
        conf: f64,
        on_pred: &mut dyn FnMut(i64, RawPred),
    ) -> AppResult<Vec<String>> {
        let req = serde_json::json!({
            "cmd": "predict", "weights": weights, "conf": conf,
            "images": items.iter().map(|(id, p)| serde_json::json!({ "id": id, "path": p })).collect::<Vec<_>>(),
        });
        let done = self.call(rt, &req, &mut |v| {
            if v["type"] == "pred" {
                on_pred(v["id"].as_i64().unwrap_or(-1), parse_pred(v));
            }
        })?;
        Ok(serde_json::from_value(done["classes"].clone()).unwrap_or_default())
    }
}

/// 模型的类别名对应到项目的类别 id，项目里没有的自动新建
pub fn class_map(p: &Project, classes: &[String]) -> AppResult<Vec<i64>> {
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

/// 模型文件：训练出的 best.pt，或导入的外部模型 model.onnx
pub fn weights_of(p: &Project, run_id: i64) -> AppResult<PathBuf> {
    let run = get_run(p, run_id)?;
    let dir = PathBuf::from(&run.dir);
    let w = model_path(&dir, &run.engine);
    if !w.exists() {
        return Err(msg(t!("这次训练没有产出模型")));
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

fn to_tags(raw: Vec<(usize, f64)>, map: &[i64]) -> Vec<PredTag> {
    raw.into_iter().filter_map(|(cls, score)| Some(PredTag { label_id: *map.get(cls)?, score })).collect()
}

fn to_poses(raw: Vec<(usize, f64, [f64; 4], Vec<f64>)>, map: &[i64]) -> Vec<PredPose> {
    raw.into_iter()
        .filter_map(|(cls, score, bbox, kpts)| Some(PredPose { label_id: *map.get(cls)?, bbox, kpts, score }))
        .collect()
}

/// 推理进程的原始结果换成项目类别 id。map[i] 是类别序号 i 对应的类别 id
pub fn to_result(raw: RawPred, map: &[i64]) -> PredResult {
    PredResult {
        boxes: to_boxes(raw.boxes, map),
        polys: to_polys(raw.polys, map),
        tags: to_tags(raw.tags, map),
        poses: to_poses(raw.poses, map),
    }
}

/// 把一张图的预测结果转成待保存的标注（来源记为 source）：
/// 分类取最高分的类别（低于 conf 不要），关键点模型给关键点目标，分割模型给多边形，检测模型给框
pub fn result_to_annotations(r: &PredResult, conf: f64, source: &str) -> Vec<Annotation> {
    let ann = |label_id: i64, kind: &str, data: serde_json::Value, score: f64| Annotation {
        label_id: Some(label_id),
        kind: kind.into(),
        data,
        score: Some(score),
        source: source.into(),
    };
    if let Some(t) = r.tags.first() {
        return if t.score >= conf { vec![ann(t.label_id, "tag", serde_json::json!([]), t.score)] } else { Vec::new() };
    }
    if !r.poses.is_empty() {
        return r
            .poses
            .iter()
            .map(|ps| ann(ps.label_id, "keypoints", serde_json::json!({ "box": ps.bbox, "kpts": ps.kpts }), ps.score))
            .collect();
    }
    if !r.polys.is_empty() {
        return r.polys.iter().map(|pl| ann(pl.label_id, "polygon", serde_json::json!(pl.points), pl.score)).collect();
    }
    r.boxes.iter().map(|b| ann(b.label_id, "bbox", serde_json::json!([b.x, b.y, b.w, b.h]), b.score)).collect()
}

/// 同 result_to_annotations，但按项目任务转换只有框的结果（文本提示、巡瞳 AI 只给框）：
/// 分割项目把框变成矩形轮廓，关键点项目变成还没点关键点的目标
pub fn task_annotations(p: &Project, r: &PredResult, conf: f64, source: &str) -> AppResult<Vec<Annotation>> {
    let boxes_only = r.polys.is_empty() && r.poses.is_empty() && r.tags.is_empty() && !r.boxes.is_empty();
    let ann = |label_id: i64, kind: &str, data: serde_json::Value, score: f64| Annotation {
        label_id: Some(label_id),
        kind: kind.into(),
        data,
        score: Some(score),
        source: source.into(),
    };
    Ok(match p.meta.task.as_str() {
        "segment" if boxes_only => r
            .boxes
            .iter()
            .map(|b| {
                let pts = [b.x, b.y, b.x + b.w, b.y, b.x + b.w, b.y + b.h, b.x, b.y + b.h];
                ann(b.label_id, "polygon", serde_json::json!(pts), b.score)
            })
            .collect(),
        "pose" if boxes_only => {
            let nk = crate::keypoints::get(p)?.points.len();
            r.boxes
                .iter()
                .map(|b| {
                    let data = serde_json::json!({ "box": [b.x, b.y, b.w, b.h], "kpts": vec![0.0; nk * 3] });
                    ann(b.label_id, "keypoints", data, b.score)
                })
                .collect()
        }
        // 多标签分类：分数够的类别都打上
        "multilabel" => r
            .tags
            .iter()
            .filter(|t| t.score >= conf)
            .map(|t| ann(t.label_id, "tag", serde_json::json!([]), t.score))
            .collect(),
        _ => result_to_annotations(r, conf, source),
    })
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
            version: None,
            base_run: None,
            aug: None,
            engine: TORCHVISION.into(),
            extra: None,
        };
        let id = start(p.clone(), &rt, jobs.clone(), cfg, Arc::new(move |_, e| ev2.lock().push(e)), &|_, _| {}).unwrap();
        assert!(start(p.clone(), &rt, jobs.clone(), TrainConfig { epochs: 1, ..serde_json::from_value(get_run(&p, id).unwrap().config).unwrap() }, Arc::new(|_, _| {}), &|_, _| {}).is_err(), "同时只能一个训练");
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
        crate::deploy::build_service(&p, &rt, id, &serde_json::json!({ "port": 8090, "streams": [] }), &svc, &|l| eprintln!("{l}"), &|_, _, _| {}).unwrap();
        for f in ["model.onnx", "model.json", "service.py", "config.json", "run.bat", "README.md"] {
            assert!(svc.join(f).exists(), "服务包缺少 {f}");
        }
        let fmts = crate::deploy::export_formats(&p, &rt, id, &["onnx_static".into(), "openvino".into(), "rknn".into()], 224, &base.join("导出"), &|l| eprintln!("{l}"), &|_, _, _| {}).unwrap();
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
    Ok(to_result(raw, &map))
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
    let multi = p.meta.task == "multilabel";
    let mut dims: HashMap<i64, (f64, f64)> = HashMap::new();
    let items: Vec<(i64, PathBuf)> = {
        let conn = p.conn.lock();
        let mut stmt = conn.prepare("SELECT path, width, height FROM images WHERE id = ?1")?;
        ids.iter()
            .filter_map(|id| {
                let (s, iw, ih) = stmt
                    .query_row([id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?, r.get::<_, f64>(2)?)))
                    .ok()?;
                dims.insert(*id, (iw, ih));
                Some((*id, p.resolve(&s)))
            })
            .collect()
    };
    let total = items.len();
    let mut out = HashMap::new();
    let mut map: Option<Vec<i64>> = None;
    progress(0, total);
    let mut done = 0;
    for chunk in items.chunks(16) {
        let mut got = Vec::new();
        let classes = pr.request(rt, &w, chunk, conf, &mut |id, r| got.push((id, r)))?;
        if map.is_none() {
            map = Some(class_map(p, &classes)?);
        }
        for (id, raw) in got {
            let r = to_result(raw, map.as_ref().unwrap());
            let preds = match r.tags.first() {
                // 多标签分类：分数够的类别都当成整图的框
                Some(_) if multi => {
                    let (iw, ih) = dims.get(&id).copied().unwrap_or((1.0, 1.0));
                    r.tags
                        .iter()
                        .filter(|t| t.score >= conf)
                        .map(|t| crate::health::Pred { label_id: t.label_id, bbox: [0.0, 0.0, iw, ih], score: t.score })
                        .collect()
                }
                // 分类模型：最高分的类别当成整图一个框，和体检里分类标签的算法一致
                Some(t) => {
                    let (iw, ih) = dims.get(&id).copied().unwrap_or((1.0, 1.0));
                    if t.score >= conf {
                        vec![crate::health::Pred { label_id: t.label_id, bbox: [0.0, 0.0, iw, ih], score: t.score }]
                    } else {
                        Vec::new()
                    }
                }
                None => r
                    .boxes
                    .into_iter()
                    .map(|b| crate::health::Pred { label_id: b.label_id, bbox: [b.x, b.y, b.w, b.h], score: b.score })
                    .collect(),
            };
            out.insert(id, preds);
        }
        done += chunk.len();
        progress(done, total);
    }
    Ok(out)
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Uncertain {
    pub image_id: i64,
    /// 模型拿不准的程度，越大越值得先标
    pub score: f64,
    /// 模型在这张图上看到了什么（简短说明）
    pub detail: String,
}

/// 主动学习：用模型把「未标注」的图都看一遍，挑出模型最拿不准的，这些图先标对模型提升最大。
/// 检测类：分数在 0.1~0.9 之间的框越多越拿不准，什么都不确定也算；分类：第一名和第二名分数越接近越拿不准。
pub fn uncertain_images(
    p: &Project,
    rt: &Runtime,
    pr: &Predictor,
    run_id: i64,
    limit: usize,
    progress: &(dyn Fn(usize, usize) + Sync),
) -> AppResult<Vec<Uncertain>> {
    let w = weights_of(p, run_id)?;
    let multi = p.meta.task == "multilabel";
    let items: Vec<(i64, PathBuf)> = {
        let conn = p.conn.lock();
        let mut stmt = conn.prepare("SELECT id, path FROM images WHERE status = 'todo' ORDER BY id LIMIT 5000")?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
        rows.map(|r| r.map(|(id, path)| (id, p.resolve(&path)))).collect::<Result<_, _>>()?
    };
    if items.is_empty() {
        return Err(msg("没有未标注的图片"));
    }
    let total = items.len();
    let mut out = Vec::new();
    progress(0, total);
    let mut done = 0;
    for chunk in items.chunks(16) {
        let mut got: Vec<(i64, RawPred)> = Vec::new();
        pr.request(rt, &w, chunk, 0.05, &mut |id, r| got.push((id, r)))?;
        for (id, r) in got {
            let (score, detail) = if multi && !r.tags.is_empty() {
                // 多标签：每个类别各自判断，分数越接近 0.5 越拿不准
                let u: f64 = r.tags.iter().map(|t| 1.0 - (2.0 * t.1 - 1.0).abs()).sum();
                (u, t!("拿不准的类别 {n} 个", n = r.tags.iter().filter(|t| (0.2..0.8).contains(&t.1)).count()).to_string())
            } else if r.tags.len() >= 1 {
                let a = r.tags[0].1;
                let b = r.tags.get(1).map(|t| t.1).unwrap_or(0.0);
                (1.0 - (a - b), format!("前两名分数 {a:.2} / {b:.2}"))
            } else {
                let scores: Vec<f64> = r.boxes.iter().map(|b| b[5]).collect();
                let mid: Vec<f64> = scores.iter().copied().filter(|s| (0.1..0.9).contains(s)).collect();
                let mut u: f64 = mid.iter().map(|s| 1.0 - (2.0 * s - 1.0).abs()).sum();
                let sure = scores.iter().filter(|s| **s >= 0.5).count();
                // 一个有把握的目标都没有、又有些模糊的框：可能是新场景
                if sure == 0 && !mid.is_empty() {
                    u += 0.5;
                }
                (u, format!("{} 个拿不准的目标，{} 个有把握", mid.len(), sure))
            };
            out.push(Uncertain { image_id: id, score, detail });
        }
        done += chunk.len();
        progress(done, total);
    }
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(limit.max(1));
    Ok(out)
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PrelabelResult {
    pub images: usize,
    pub boxes: usize,
}

/// 批量预标注的停止开关（模型、文本提示、巡瞳 AI 共用）：每批之间检查，已经标好的保留
static PRELABEL_STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn prelabel_stop(on: bool) {
    PRELABEL_STOP.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn prelabel_stopped() -> bool {
    PRELABEL_STOP.load(std::sync::atomic::Ordering::Relaxed)
}

/// 批量预标注的对象：「未标注」且没有任何标注的图（id, 图片文件）
pub fn prelabel_targets(p: &Project) -> AppResult<Vec<(i64, PathBuf)>> {
    let conn = p.conn.lock();
    let mut stmt = conn.prepare(
        "SELECT id, path FROM images WHERE status = 'todo'
         AND NOT EXISTS (SELECT 1 FROM annotations a WHERE a.image_id = images.id) ORDER BY id",
    )?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
    let v = rows.map(|r| r.map(|(id, path)| (id, p.resolve(&path)))).collect::<Result<_, _>>()?;
    Ok(v)
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
    prelabel_stop(false);
    let w = weights_of(p, run_id)?;
    let items = prelabel_targets(p)?;
    let total = items.len();
    let mut result = PrelabelResult { images: 0, boxes: 0 };
    let mut map: Option<Vec<i64>> = None;
    let mut done = 0;
    progress(0, total);
    for chunk in items.chunks(16) {
        if prelabel_stopped() {
            break;
        }
        let mut got: Vec<(i64, RawPred)> = Vec::new();
        let classes = pr.request(rt, &w, chunk, conf, &mut |id, r| got.push((id, r)))?;
        if map.is_none() {
            map = Some(class_map(p, &classes)?);
        }
        let m = map.as_ref().unwrap();
        for (id, raw) in got {
            let anns = task_annotations(p, &to_result(raw, m), conf, "model")?;
            if !anns.is_empty() && annotations::save_predictions(p, id, &anns)? {
                result.boxes += anns.len();
                result.images += 1;
            }
        }
        done += chunk.len();
        progress(done, total);
    }
    Ok(result)
}
