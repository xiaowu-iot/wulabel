//! 部署：实时验证（视频流 / 视频文件）、多格式导出、推理服务包

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
use crate::t;

/// 模型文件：内置模型的 best.pt，或 Ultralytics 等引擎、外部导入的 model.onnx
fn weights(p: &Project, run_id: i64) -> AppResult<PathBuf> {
    training::weights_of(p, run_id)
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
        let p = g.as_mut().ok_or_else(|| msg(t!("实时验证没有在运行")))?;
        writeln!(p.stdin, "{v}").and_then(|_| p.stdin.flush()).map_err(|_| msg(t!("实时验证进程已退出")))
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
            return Err(msg(t!("请填写视频流地址或选择视频文件")));
        }
        if !source.contains("://") && !Path::new(source).is_file() {
            return Err(msg(t!("视频文件不存在：{source}", source = source)));
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
        let mut child = c.spawn().map_err(|e| msg(t!("无法启动实时验证：{e}", e = e)))?;
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
                            Err(msg(ev["error"].as_str().map(String::from).unwrap_or_else(|| t!("截图失败"))))
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
                    sink("error", serde_json::json!({ "msg": t!("实时验证意外退出：{e}", e = t.lines().last().unwrap_or("")), "trace": t }));
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
        return Err(msg(t!("输出目录不为空：{path}", path = out.display())));
    }
    fs::create_dir_all(out)?;
    Ok(())
}

/// step：步骤进度 (已完成几步, 共几步, 当前步骤)，共 0 步表示还不知道有几步。
/// worker 报的步骤名是格式（onnx、openvino 等），界面换成格式名显示
#[allow(clippy::too_many_arguments)]
pub fn export_formats(
    p: &Project,
    rt: &Runtime,
    run_id: i64,
    formats: &[String],
    size: u32,
    out: &Path,
    log: &(dyn Fn(String) + Sync),
    step: &(dyn Fn(usize, usize, String) + Sync),
) -> AppResult<Vec<FormatResult>> {
    if formats.is_empty() {
        return Err(msg(t!("至少选一种格式")));
    }
    // OpenVINO 要另外装组件；装不上（离线、网络差）时只跳过它，别的格式照常导出
    let mut skipped = Vec::new();
    let mut formats: Vec<String> = formats.to_vec();
    if formats.iter().any(|f| f == "openvino") {
        step(0, 0, t!("检查 OpenVINO 组件…"));
        if let Err(e) = rt.ensure_package("openvino", "openvino", log) {
            let note = t!("OpenVINO 组件没装上，这次没导出：{e}", e = e);
            log(note.clone());
            formats.retain(|f| f != "openvino");
            skipped.push(FormatResult { format: "openvino".into(), path: String::new(), ok: false, note });
            if formats.is_empty() {
                return Err(msg(skipped.remove(0).note));
            }
        }
    }
    let w = weights(p, run_id)?;
    check_out(out)?;
    let cfg = out.join(".export.json");
    fs::write(&cfg, serde_json::json!({ "weights": w, "out": out, "formats": formats, "size": size }).to_string())?;
    let mut c = rt.worker(&["export-formats"])?;
    c.arg(&cfg);
    step(0, 0, t!("正在加载模型…"));
    let results = Mutex::new(Vec::new());
    let r = crate::runtime::run_logged(c, &|line| {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
            match v["type"].as_str() {
                Some("log") => log(v["msg"].as_str().unwrap_or("").to_string()),
                Some("progress") => step(
                    v["step"].as_u64().unwrap_or(0) as usize,
                    v["total"].as_u64().unwrap_or(0) as usize,
                    v["name"].as_str().unwrap_or("").to_string(),
                ),
                Some("format") => {
                    if let Ok(f) = serde_json::from_value::<FormatResult>(v.clone()) {
                        log(t!("{name}：{result}", name = f.format, result = if f.ok { t!("完成") } else { f.note.clone() }));
                        results.lock().push(f);
                    }
                }
                Some("error") => log(t!("错误：{e}", e = v["msg"].as_str().unwrap_or(""))),
                _ => {}
            }
        }
    });
    let _ = fs::remove_file(&cfg);
    r?;
    let mut res = results.into_inner();
    res.extend(skipped);
    Ok(res)
}

// ---------- 推理服务包 ----------

const SERVICE_FILES: [&str; 5] = ["service.py", "requirements.txt", "run.bat", "run.sh", "README.md"];
const DECODER: &str = "onnx_decode.py";

/// 生成推理服务包：模型（ONNX）+ 服务程序 + 配置 + 说明
pub fn build_service(
    p: &Project,
    rt: &Runtime,
    run_id: i64,
    config: &serde_json::Value,
    out: &Path,
    log: &(dyn Fn(String) + Sync),
    step: &(dyn Fn(usize, usize, String) + Sync),
) -> AppResult<String> {
    // 三步：导出 ONNX → 复制服务程序和说明 → 写配置
    const STEPS: usize = 3;
    let w = weights(p, run_id)?;
    let template = rt.worker_dir.join("service");
    for f in SERVICE_FILES {
        if !template.join(f).exists() {
            return Err(msg(t!("安装文件不完整，缺少服务模板 {f}", f = f)));
        }
    }
    if !rt.worker_dir.join("xlabel_worker").join(DECODER).exists() {
        return Err(msg(t!("安装文件不完整，缺少服务模板 {f}", f = DECODER)));
    }
    check_out(out)?;
    log(t!("导出模型（ONNX）…"));
    step(0, STEPS, t!("导出模型（ONNX）…"));
    let mut c = rt.worker(&["export"])?;
    c.arg(&w).arg(out.join("model.onnx"));
    crate::runtime::run_logged(c, &|line| {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(m) = v["msg"].as_str() {
                log(m.to_string());
            }
        }
    })?;
    step(1, STEPS, t!("复制服务程序和说明…"));
    for f in SERVICE_FILES {
        fs::copy(template.join(f), out.join(f))?;
    }
    // 模型解码（检测、分割、关键点、分类，Wulabel 和 Ultralytics 等格式）和训练时用的是同一份代码
    fs::copy(rt.worker_dir.join("xlabel_worker").join(DECODER), out.join(DECODER))?;
    // 界面不是中文时，说明、启动脚本和服务提示换成对应语言（worker/service-i18n/，没有的语言用英文）
    let lang = crate::i18n::lang();
    if lang != "zh" {
        let i18n = rt.worker_dir.join("service-i18n");
        let pick = |name: &dyn Fn(&str) -> String| {
            [lang.as_str(), "en"].iter().map(|l| i18n.join(name(l))).find(|p| p.exists())
        };
        for (f, stem, ext) in [("README.md", "README", "md"), ("run.bat", "run", "bat"), ("run.sh", "run", "sh")] {
            if let Some(src) = pick(&|l| format!("{stem}.{l}.{ext}")) {
                fs::copy(src, out.join(f))?;
            }
        }
        if let Some(src) = pick(&|l| format!("{l}.json")) {
            fs::copy(src, out.join("locale.json"))?;
        }
    }
    step(2, STEPS, t!("写入配置…"));
    let mut cfg = config.clone();
    cfg["model"] = serde_json::json!("model.onnx");
    // 关键点模型：告警截图按项目的骨架连线画
    if training::get_run(p, run_id)?.task == "pose" {
        cfg["skeleton"] = serde_json::json!(crate::keypoints::get(p)?.skeleton);
    }
    fs::write(out.join("config.json"), serde_json::to_string_pretty(&cfg)?)?;
    log(t!("服务包已生成"));
    step(STEPS, STEPS, t!("服务包已生成"));
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
        assert!(dir.join("../xlabel_worker").join(DECODER).exists(), "缺少 {DECODER}");
        // run.bat 必须是 CRLF，否则部分 Windows 环境执行异常
        let bat = fs::read(dir.join("run.bat")).unwrap();
        assert!(bat.windows(2).any(|w| w == b"\r\n"));
    }
}
