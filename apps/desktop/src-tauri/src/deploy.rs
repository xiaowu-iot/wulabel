//! 部署：实时验证（RTSP / 视频文件）、多格式导出、推理服务包

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::error::{msg, AppResult};
use crate::images;
use crate::project::Project;
use crate::runtime::Runtime;
use crate::training;

fn weights(p: &Project, run_id: i64) -> AppResult<PathBuf> {
    let w = PathBuf::from(training::get_run(p, run_id)?.dir).join("best.pt");
    if !w.exists() {
        return Err(msg("这次训练没有产出模型"));
    }
    Ok(w)
}

fn now_stamp() -> String {
    chrono::Local::now().format("%Y%m%d_%H%M%S").to_string()
}

// ---------- 实时验证 ----------

struct LiveProc {
    child: Child,
    stdin: ChildStdin,
}

#[derive(Default)]
pub struct Live {
    proc: Mutex<Option<LiveProc>>,
}

pub type Sink = Arc<dyn Fn(&str, serde_json::Value) + Send + Sync>;

impl Live {
    fn send(&self, v: serde_json::Value) -> AppResult<()> {
        let mut g = self.proc.lock();
        let p = g.as_mut().ok_or_else(|| msg("实时验证没有在运行"))?;
        writeln!(p.stdin, "{v}").and_then(|_| p.stdin.flush()).map_err(|_| msg("实时验证进程已退出"))
    }

    pub fn stop(&self) {
        if let Some(mut p) = self.proc.lock().take() {
            let _ = writeln!(p.stdin, "{{\"cmd\":\"stop\"}}");
            let _ = p.stdin.flush();
            std::thread::sleep(std::time::Duration::from_millis(300));
            let _ = p.child.kill();
        }
    }

    pub fn set_conf(&self, conf: f64) -> AppResult<()> {
        self.send(serde_json::json!({ "cmd": "conf", "value": conf }))
    }

    /// 截取当前画面：worker 先存到临时文件，收到 snapshot 事件后导入项目
    pub fn snapshot(&self, p: &Project) -> AppResult<()> {
        let path = p.root.join(".cache").join("snapshots").join(format!("现场_{}.jpg", now_stamp()));
        self.send(serde_json::json!({ "cmd": "snapshot", "path": path }))
    }

    pub fn start(self: &Arc<Self>, p: Arc<Project>, rt: &Runtime, run_id: i64, source: &str, conf: f64, sink: Sink) -> AppResult<()> {
        self.stop();
        let source = source.trim();
        if source.is_empty() {
            return Err(msg("请填写 RTSP 地址或选择视频文件"));
        }
        if !source.contains("://") && !Path::new(source).is_file() {
            return Err(msg(format!("视频文件不存在：{source}")));
        }
        let s2 = sink.clone();
        rt.ensure_package("cv2", "opencv-python-headless", &move |l| s2("log", serde_json::json!(l)))?;
        let cfg_path = p.root.join(".cache").join("live.json");
        fs::create_dir_all(cfg_path.parent().unwrap())?;
        fs::write(
            &cfg_path,
            serde_json::json!({ "source": source, "weights": weights(&p, run_id)?, "conf": conf, "max_fps": 12, "preview_width": 960 }).to_string(),
        )?;
        let mut c = rt.worker(&["live"])?;
        c.arg(&cfg_path).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child = c.spawn().map_err(|e| msg(format!("无法启动实时验证：{e}")))?;
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        *self.proc.lock() = Some(LiveProc { child, stdin });

        let tail = Arc::new(Mutex::new(Vec::<String>::new()));
        {
            let tail = tail.clone();
            std::thread::spawn(move || {
                for l in BufReader::new(stderr).lines().map_while(Result::ok) {
                    let mut t = tail.lock();
                    t.push(l);
                    if t.len() > 12 {
                        t.remove(0);
                    }
                }
            });
        }
        let me = self.clone();
        std::thread::spawn(move || {
            let mut got_end = false;
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                let Ok(ev) = serde_json::from_str::<serde_json::Value>(&line) else { continue };
                match ev["type"].as_str() {
                    Some("snapshot") => {
                        let path = PathBuf::from(ev["path"].as_str().unwrap_or(""));
                        let r = if ev.get("error").is_some() {
                            Err(msg(ev["error"].as_str().unwrap_or("截图失败").to_string()))
                        } else {
                            images::import(&p, &[path.clone()], true, &|_, _| {})
                        };
                        let _ = fs::remove_file(&path);
                        sink(
                            "snapshot",
                            match r {
                                Ok(r) => serde_json::json!({ "added": r.added, "duplicated": r.duplicated }),
                                Err(e) => serde_json::json!({ "error": e.to_string() }),
                            },
                        );
                    }
                    Some("ended") | Some("error") => {
                        got_end = true;
                        let kind = ev["type"].as_str().unwrap_or("error").to_string();
                        sink(&kind, ev);
                    }
                    Some(t) => sink(t, ev.clone()),
                    None => {}
                }
            }
            if !got_end {
                let t = tail.lock().join("\n");
                if !t.trim().is_empty() && t.contains("Error") {
                    sink("error", serde_json::json!({ "msg": format!("实时验证意外退出：{}", t.lines().last().unwrap_or("")), "trace": t }));
                }
            }
            sink("stopped", serde_json::json!({}));
            let mut g = me.proc.lock();
            if let Some(p) = g.as_mut() {
                if matches!(p.child.try_wait(), Ok(Some(_))) {
                    *g = None;
                }
            }
        });
        Ok(())
    }
}

// ---------- 多格式导出 ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FormatResult {
    pub format: String,
    pub path: String,
    pub ok: bool,
    pub note: String,
}

fn check_out(out: &Path) -> AppResult<()> {
    if out.exists() && fs::read_dir(out)?.next().is_some() {
        return Err(msg(format!("输出目录不为空：{}", out.display())));
    }
    fs::create_dir_all(out)?;
    Ok(())
}

pub fn export_formats(
    p: &Project,
    rt: &Runtime,
    run_id: i64,
    formats: &[String],
    size: u32,
    out: &Path,
    log: &(dyn Fn(String) + Sync),
) -> AppResult<Vec<FormatResult>> {
    if formats.is_empty() {
        return Err(msg("至少选一种格式"));
    }
    if formats.iter().any(|f| f == "openvino") {
        rt.ensure_package("openvino", "openvino", log)?;
    }
    let w = weights(p, run_id)?;
    check_out(out)?;
    let cfg = out.join(".export.json");
    fs::write(&cfg, serde_json::json!({ "weights": w, "out": out, "formats": formats, "size": size }).to_string())?;
    let mut c = rt.worker(&["export-formats"])?;
    c.arg(&cfg);
    let results = Mutex::new(Vec::new());
    let r = crate::runtime::run_logged(c, &|line| {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
            match v["type"].as_str() {
                Some("log") => log(v["msg"].as_str().unwrap_or("").to_string()),
                Some("format") => {
                    if let Ok(f) = serde_json::from_value::<FormatResult>(v.clone()) {
                        log(format!("{}：{}", f.format, if f.ok { "完成" } else { &f.note }));
                        results.lock().push(f);
                    }
                }
                Some("error") => log(format!("错误：{}", v["msg"].as_str().unwrap_or(""))),
                _ => {}
            }
        }
    });
    let _ = fs::remove_file(&cfg);
    r?;
    Ok(results.into_inner())
}

// ---------- 推理服务包 ----------

const SERVICE_FILES: [&str; 5] = ["service.py", "requirements.txt", "run.bat", "run.sh", "README.md"];

/// 生成推理服务包：模型（ONNX）+ 服务程序 + 配置 + 说明
pub fn build_service(
    p: &Project,
    rt: &Runtime,
    run_id: i64,
    config: &serde_json::Value,
    out: &Path,
    log: &(dyn Fn(String) + Sync),
) -> AppResult<String> {
    let w = weights(p, run_id)?;
    let template = rt.worker_dir.join("service");
    for f in SERVICE_FILES {
        if !template.join(f).exists() {
            return Err(msg(format!("安装文件不完整，缺少服务模板 {f}")));
        }
    }
    check_out(out)?;
    log("导出模型（ONNX）…".into());
    let mut c = rt.worker(&["export"])?;
    c.arg(&w).arg(out.join("model.onnx"));
    crate::runtime::run_logged(c, &|line| {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(m) = v["msg"].as_str() {
                log(m.to_string());
            }
        }
    })?;
    for f in SERVICE_FILES {
        fs::copy(template.join(f), out.join(f))?;
    }
    let mut cfg = config.clone();
    cfg["model"] = serde_json::json!("model.onnx");
    fs::write(out.join("config.json"), serde_json::to_string_pretty(&cfg)?)?;
    log("服务包已生成".into());
    Ok(out.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_template_complete() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../worker/service");
        for f in SERVICE_FILES {
            assert!(dir.join(f).exists(), "缺少 {f}");
        }
        // run.bat 必须是 CRLF，否则部分 Windows 环境执行异常
        let bat = fs::read(dir.join("run.bat")).unwrap();
        assert!(bat.windows(2).any(|w| w == b"\r\n"));
    }
}
