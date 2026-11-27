//! 训练环境：设置、显卡检测、uv 下载、Python + PyTorch 安装、运行 worker

use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::error::{msg, AppResult};
use crate::t;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    /// 用户指定的 Python（已装好 torch 的环境），为空用软件自己管理的环境
    #[serde(default)]
    pub python: Option<String>,
    #[serde(default)]
    pub uv: Option<String>,
    /// "china" 用国内镜像，"official" 用官方源
    #[serde(default = "default_mirror")]
    pub mirror: String,
    /// 下载源是用户自己选的；没选过时跟着界面语言走（简体中文用国内镜像，其他语言用官方源）
    #[serde(default)]
    pub mirror_chosen: bool,
    /// 训练环境安装位置（Python、PyTorch、预训练权重），为空用系统默认的本机数据目录
    #[serde(default)]
    pub env_root: Option<String>,
    /// 用户指定的 ffmpeg（视频抽帧用），可以是 ffmpeg.exe 或其所在目录；为空则自动查找/下载
    #[serde(default)]
    pub ffmpeg: Option<String>,
    /// 巡瞳 AI 平台地址，为空用默认地址（私有化部署时填自己的）
    #[serde(default)]
    pub ai_url: Option<String>,
    /// 巡瞳 AI 的 API Key（只保存在本机设置文件里）
    #[serde(default)]
    pub ai_key: Option<String>,
}

fn default_mirror() -> String {
    "china".into()
}

/// 没选过下载源时的默认值：界面是简体中文用国内镜像，其他语言用官方源
fn auto_mirror() -> String {
    if crate::i18n::lang() == "zh" { "china".into() } else { "official".into() }
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            python: None,
            uv: None,
            mirror: default_mirror(),
            mirror_chosen: false,
            env_root: None,
            ffmpeg: None,
            ai_url: None,
            ai_key: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Gpu {
    pub name: String,
    pub vram_mb: u64,
    pub driver: String,
    pub cuda: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvStatus {
    pub python: Option<String>,
    pub managed: bool,
    pub ready: bool,
    pub info: Option<serde_json::Value>,
    pub gpu: Option<Gpu>,
    /// 推荐安装的 PyTorch 版本：cu128 / cu126 / cu118 / cpu
    pub variant: String,
    pub uv: Option<String>,
    pub installing: bool,
    pub error: Option<String>,
    pub settings: Settings,
    pub env_dir: String,
    /// 环境里缺的组件（pip 包名），可以在设置里「补装」
    pub missing: Vec<String>,
    /// 装的是显卡版 PyTorch，但显卡用不了，现在用 CPU 训练
    pub cpu_fallback: bool,
    /// 有 NVIDIA 显卡但驱动太旧、只能装 CPU 版时，要更新到的驱动版本
    pub min_driver: Option<String>,
}

/// 训练环境需要的 Python 组件：(模块名, pip 包名, 用户自己的环境也必须有)
const PACKAGES: [(&str, &str, bool); 5] = [
    ("numpy", "numpy", true),
    ("PIL", "pillow", true),
    ("onnx", "onnx", false),
    ("onnxruntime", "onnxruntime", false),
    ("cv2", "opencv-python-headless", false),
];

pub struct Runtime {
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub worker_dir: PathBuf,
    pub installing: Mutex<bool>,
    /// 点了「取消」：停掉正在下载的文件和 uv 子进程
    cancel: AtomicBool,
    check_cache: Mutex<Option<(String, serde_json::Value)>>,
}

/// 安装日志（<环境目录>/install.log）：每行带时间，追加写
type LogFile = Mutex<Option<fs::File>>;

fn write_log(f: &LogFile, line: &str) {
    if let Some(f) = f.lock().as_mut() {
        let _ = writeln!(f, "[{}] {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"), line);
    }
}

/// macOS / Linux 上常见的程序目录。从访达、Dock 启动的应用拿不到终端里的 PATH，Homebrew 装的 ffmpeg、uv 要单独找
fn unix_extra_bins() -> Vec<PathBuf> {
    if cfg!(windows) {
        return Vec::new();
    }
    ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/opt/local/bin"].iter().map(PathBuf::from).collect()
}

/// Windows 上启动子进程不弹黑窗口
pub fn command(program: impl AsRef<std::ffi::OsStr>) -> Command {
    #[allow(unused_mut)]
    let mut c = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000);
    }
    c
}

impl Runtime {
    pub fn new(config_dir: PathBuf, data_dir: PathBuf, worker_dir: PathBuf) -> Self {
        Runtime {
            config_dir,
            data_dir,
            worker_dir,
            installing: Mutex::new(false),
            cancel: AtomicBool::new(false),
            check_cache: Mutex::new(None),
        }
    }

    pub fn settings(&self) -> Settings {
        let mut s: Settings = fs::read_to_string(self.config_dir.join("settings.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        if !s.mirror_chosen {
            // 旧版本默认就是 china，存着 official 说明是用户自己改的
            if s.mirror == "official" {
                s.mirror_chosen = true;
            } else {
                s.mirror = auto_mirror();
            }
        }
        s
    }

    pub fn save_settings(&self, s: &Settings) -> AppResult<()> {
        let mut s = s.clone();
        // 下载源和现在的不一样，就是用户选过了，以后不再跟着界面语言变
        let cur = self.settings();
        s.mirror_chosen = cur.mirror_chosen || s.mirror != cur.mirror;
        fs::create_dir_all(&self.config_dir)?;
        fs::write(self.config_dir.join("settings.json"), serde_json::to_string_pretty(&s)?)?;
        *self.check_cache.lock() = None;
        Ok(())
    }

    pub fn install_log_path(&self) -> PathBuf {
        self.root().join("install.log")
    }

    fn open_install_log(&self) -> LogFile {
        let path = self.install_log_path();
        let _ = fs::create_dir_all(self.root());
        // 太大就重新开始记
        if fs::metadata(&path).map(|m| m.len() > 2 * 1024 * 1024).unwrap_or(false) {
            let _ = fs::remove_file(&path);
        }
        Mutex::new(fs::OpenOptions::new().create(true).append(true).open(&path).ok())
    }

    /// 取消正在进行的安装（下载停下、uv 进程结束）
    pub fn cancel_install(&self) {
        if *self.installing.lock() {
            self.cancel.store(true, Ordering::Relaxed);
        }
    }

    fn check_cancel(&self) -> AppResult<()> {
        if self.cancel.load(Ordering::Relaxed) {
            Err(msg(t!("已取消安装")))
        } else {
            Ok(())
        }
    }

    /// 运行安装命令：点了取消会结束这个进程
    fn run_step(&self, c: Command, log: &(dyn Fn(String) + Sync)) -> AppResult<()> {
        let r = run_cancellable(c, log, Some(&self.cancel));
        self.check_cancel()?;
        r
    }

    /// 安装类任务的公共部分：同一时间只能装一个、可取消、日志同时写到 install.log
    fn guarded(
        &self,
        title: &str,
        log: &(dyn Fn(String) + Sync),
        f: impl FnOnce(&(dyn Fn(String) + Sync)) -> AppResult<()>,
    ) -> AppResult<()> {
        {
            let mut g = self.installing.lock();
            if *g {
                return Err(msg(t!("已经在安装了")));
            }
            *g = true;
        }
        self.cancel.store(false, Ordering::Relaxed);
        let file = self.open_install_log();
        write_log(&file, &format!("===== {title} · Wulabel {} =====", env!("CARGO_PKG_VERSION")));
        let logf = |line: String| {
            write_log(&file, &line);
            log(line);
        };
        let r = f(&logf);
        if let Err(e) = &r {
            write_log(&file, &format!("ERROR: {e}"));
        }
        *self.installing.lock() = false;
        *self.check_cache.lock() = None;
        r
    }

    /// 训练环境根目录：下面有 env（虚拟环境）、python（解释器）、torch（预训练权重）、bin（uv）
    pub fn root(&self) -> PathBuf {
        match self.settings().env_root.filter(|s| !s.trim().is_empty()) {
            Some(r) => PathBuf::from(r),
            None => self.data_dir.clone(),
        }
    }

    pub fn env_dir(&self) -> PathBuf {
        self.root().join("env")
    }

    fn managed_python(&self) -> PathBuf {
        if cfg!(windows) { self.env_dir().join("Scripts").join("python.exe") } else { self.env_dir().join("bin").join("python3") }
    }

    /// 实际使用的 Python：用户指定的优先
    pub fn python(&self) -> Option<PathBuf> {
        if let Some(p) = self.settings().python.filter(|p| !p.trim().is_empty()) {
            return Some(PathBuf::from(p));
        }
        let m = self.managed_python();
        m.exists().then_some(m)
    }

    /// 运行 worker 的命令（已设置好 PYTHONPATH 等环境变量）
    pub fn worker(&self, args: &[&str]) -> AppResult<Command> {
        let py = self.python().ok_or_else(|| msg(t!("还没有训练环境，请先到「设置」里安装")))?;
        let mut c = command(py);
        c.arg("-m").arg("xlabel_worker").args(args);
        c.env("PYTHONPATH", &self.worker_dir)
            .env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONUNBUFFERED", "1")
            .env("TORCH_HOME", self.root().join("torch"))
            // SAM、文本提示等辅助模型从 Hugging Face 下载，缓存放训练环境目录
            .env("HF_HOME", self.root().join("hf"))
            .env("HF_HUB_DISABLE_TELEMETRY", "1")
            // 训练日志、报错用界面的语言
            .env("WULABEL_LANG", crate::i18n::lang());
        if self.settings().mirror == "china" {
            c.env("HF_ENDPOINT", "https://hf-mirror.com");
        }
        Ok(c)
    }

    pub fn uv(&self) -> Option<PathBuf> {
        let exe = if cfg!(windows) { "uv.exe" } else { "uv" };
        let mut cands: Vec<PathBuf> = Vec::new();
        if let Some(u) = self.settings().uv.filter(|s| !s.trim().is_empty()) {
            cands.push(u.into());
        }
        cands.push(self.root().join("bin").join(exe));
        if let Some(path) = std::env::var_os("PATH") {
            cands.extend(std::env::split_paths(&path).map(|d| d.join(exe)));
        }
        if let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
            let home = PathBuf::from(home);
            cands.push(home.join(".local").join("bin").join(exe));
            cands.push(home.join(".cargo").join("bin").join(exe));
        }
        cands.extend(unix_extra_bins().into_iter().map(|d| d.join(exe)));
        cands.into_iter().find(|p| p.is_file())
    }

    /// 找 ffmpeg：用户指定 → 训练环境 bin → 系统 PATH；都没有返回 None
    pub fn find_ffmpeg(&self) -> Option<PathBuf> {
        let exe = if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" };
        let mut cands: Vec<PathBuf> = Vec::new();
        if let Some(f) = self.settings().ffmpeg.filter(|s| !s.trim().is_empty()) {
            let p = PathBuf::from(f);
            cands.push(if p.is_dir() { p.join(exe) } else { p });
        }
        cands.push(self.root().join("bin").join(exe));
        if let Some(path) = std::env::var_os("PATH") {
            cands.extend(std::env::split_paths(&path).map(|d| d.join(exe)));
        }
        cands.extend(unix_extra_bins().into_iter().map(|d| d.join(exe)));
        cands.into_iter().find(|p| p.is_file())
    }

    /// 拿到 ffmpeg 路径：找不到就下载到训练环境 bin（带下载进度）
    pub fn ffmpeg(
        &self,
        log: &(dyn Fn(String) + Sync),
        progress: &(dyn Fn(&str, u64, u64, u64) + Sync),
    ) -> AppResult<PathBuf> {
        if let Some(found) = self.find_ffmpeg() {
            return Ok(found);
        }
        download_ffmpeg(&self.root().join("bin"), log, progress)
    }

    /// 运行 `python -m xlabel_worker check`，结果缓存
    pub fn check(&self, refresh: bool) -> AppResult<serde_json::Value> {
        let py = self.python().ok_or_else(|| msg(t!("还没有训练环境")))?;
        let key = py.to_string_lossy().into_owned();
        if !refresh {
            if let Some((k, v)) = self.check_cache.lock().clone() {
                if k == key {
                    return Ok(v);
                }
            }
        }
        let out = self.worker(&["check"])?.stdin(Stdio::null()).output()?;
        let text = String::from_utf8_lossy(&out.stdout);
        let v = text
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .find(|v| v["type"] == "env")
            .ok_or_else(|| {
                let err = String::from_utf8_lossy(&out.stderr);
                msg(t!("训练环境检查失败：{e}", e = last_lines(&err, 6)))
            })?;
        *self.check_cache.lock() = Some((key, v.clone()));
        Ok(v)
    }

    pub fn status(&self, refresh: bool) -> EnvStatus {
        let gpu = detect_gpu();
        let variant = torch_variant(gpu.as_ref()).to_string();
        let settings = self.settings();
        let python = self.python();
        let (info, error) = match &python {
            Some(_) => match self.check(refresh) {
                Ok(v) => (Some(v), None),
                Err(e) => (None, Some(e.to_string())),
            },
            None => (None, None),
        };
        let managed = settings.python.as_deref().map(str::trim).unwrap_or("").is_empty();
        let missing = info.as_ref().map(missing_packages).unwrap_or_default();
        // 缺必需的组件也不算就绪（软件自己装的环境，装的那几个都要齐）
        let required_missing = PACKAGES.iter().any(|(_, pkg, must)| (*must || managed) && missing.iter().any(|m| m == pkg));
        let ready = info.as_ref().map(|v| v.get("torch").is_some() && v.get("error").is_none()).unwrap_or(false) && !required_missing;
        // 显卡版 PyTorch（torch.version.cuda 有值）却用不了显卡
        let cpu_fallback = info
            .as_ref()
            .map(|v| v["cuda"] == false && v["cudaVersion"].as_str().is_some_and(|s| !s.is_empty()))
            .unwrap_or(false);
        let min_driver = (gpu.is_some() && variant == "cpu").then(|| oldest_cuda_driver().to_string());
        EnvStatus {
            managed,
            missing,
            cpu_fallback,
            min_driver,
            python: python.map(|p| p.to_string_lossy().into_owned()),
            ready,
            info,
            gpu,
            variant,
            uv: self.uv().map(|p| p.to_string_lossy().into_owned()),
            installing: *self.installing.lock(),
            error,
            settings,
            env_dir: self.env_dir().to_string_lossy().into_owned(),
        }
    }

    /// 确保环境里有某个 Python 包，没有就用 uv 装上（OpenCV、OpenVINO 等按需安装）。
    /// 安装时右下角显示「安装组件 xxx」，输出也写到 install.log
    pub fn ensure_package(&self, module: &str, package: &str, log: &(dyn Fn(String) + Sync)) -> AppResult<()> {
        let py = self.python().ok_or_else(|| msg(t!("还没有训练环境，请先到「设置」里安装")))?;
        let ok = command(&py).args(["-c", &format!("import {module}")]).output().map(|o| o.status.success()).unwrap_or(false);
        if ok {
            return Ok(());
        }
        let uv = self.uv().ok_or_else(|| msg(t!("缺少 {package}，并且没有找到 uv，无法自动安装", package = package)))?;
        log(t!("首次使用，正在安装 {package}…", package = package));
        let name = t!("安装组件 {package}", package = package);
        crate::dlprogress::forward(&serde_json::json!({ "type": "download", "name": &name, "done": 0, "total": 0, "speed": 0 }));
        let mut c = command(&uv);
        c.args(["pip", "install", "--python"]).arg(&py).arg(package);
        if self.settings().mirror == "china" {
            c.args(["--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple"]);
        }
        let file = self.open_install_log();
        write_log(&file, &format!("===== pip install {package} ====="));
        let r = run_logged(c, &|line: String| {
            write_log(&file, &line);
            log(line);
        });
        if let Err(e) = &r {
            write_log(&file, &format!("ERROR: {e}"));
        }
        let mut ev = serde_json::json!({ "type": "download", "name": &name, "done": 0, "total": 0, "speed": 0 });
        ev[if r.is_ok() { "finished" } else { "failed" }] = serde_json::Value::Bool(true);
        crate::dlprogress::forward(&ev);
        r
    }

    /// Ultralytics 训练组件（YOLOv8 / 11 等），第一次选这类模型训练时装上。
    /// 它依赖完整版 opencv-python，和环境里的 opencv-python-headless 都提供 cv2、会互相覆盖，装完删掉完整版、重装 headless 版
    pub fn ensure_ultralytics(&self, log: &(dyn Fn(String) + Sync)) -> AppResult<()> {
        let py = self.python().ok_or_else(|| msg(t!("还没有训练环境，请先到「设置」里安装")))?;
        let has = command(&py).args(["-c", "import ultralytics"]).output().map(|o| o.status.success()).unwrap_or(false);
        if has {
            return Ok(());
        }
        self.ensure_package("ultralytics", "ultralytics>=8.3,<9", log)?;
        if let Some(uv) = self.uv() {
            let mut c = command(&uv);
            c.args(["pip", "uninstall", "--python"]).arg(&py).arg("opencv-python");
            let _ = c.output();
            let mut c = command(&uv);
            c.args(["pip", "install", "--reinstall", "--python"]).arg(&py).arg("opencv-python-headless");
            if self.settings().mirror == "china" {
                c.args(["--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple"]);
            }
            let file = self.open_install_log();
            run_logged(c, &|line: String| {
                write_log(&file, &line);
                log(line);
            })?;
        }
        Ok(())
    }

    /// 安装训练环境：uv → Python → PyTorch → 其他依赖 → 自检。
    /// log 收到每一行输出；progress 收到大文件下载进度（文件名、已下载字节、总字节、瞬时速度 字节/秒）；
    /// step 收到当前第几步（第几步、共几步、这一步在做什么）。
    pub fn install(
        &self,
        variant: &str,
        log: &(dyn Fn(String) + Sync),
        progress: &(dyn Fn(&str, u64, u64, u64) + Sync),
        step: &(dyn Fn(u32, u32, String) + Sync),
    ) -> AppResult<()> {
        self.guarded(&format!("install {variant}"), log, |log| self.install_inner(variant, log, progress, step))
    }

    /// 只补装环境里缺的组件（numpy、pillow、onnx、opencv 等），不动 PyTorch
    pub fn install_missing(
        &self,
        log: &(dyn Fn(String) + Sync),
        progress: &(dyn Fn(&str, u64, u64, u64) + Sync),
        step: &(dyn Fn(u32, u32, String) + Sync),
    ) -> AppResult<()> {
        self.guarded("install missing", log, |log| {
            *self.check_cache.lock() = None;
            let info = self.check(true)?;
            let pkgs = missing_packages(&info);
            if pkgs.is_empty() {
                log(t!("组件都齐了，不用补装"));
                return Ok(());
            }
            let py = self.python().ok_or_else(|| msg(t!("还没有训练环境，请先到「设置」里安装")))?;
            step(1, 2, t!("安装 {pkgs}", pkgs = pkgs.join("、")));
            let uv = match self.uv() {
                Some(u) => u,
                None => download_uv(&self.root().join("bin"), self.settings().mirror == "china", log, progress, &self.cancel)?,
            };
            let mut c = command(&uv);
            c.args(["pip", "install", "--python"]).arg(&py).args(&pkgs);
            if self.settings().mirror == "china" {
                c.args(["--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple"]);
            }
            self.run_step(c, log)?;
            step(2, 2, t!("检查环境"));
            *self.check_cache.lock() = None;
            let v = self.check(true)?;
            log(t!("完成：{v}", v = v));
            Ok(())
        })
    }

    fn install_inner(
        &self,
        variant: &str,
        log: &(dyn Fn(String) + Sync),
        progress: &(dyn Fn(&str, u64, u64, u64) + Sync),
        step: &(dyn Fn(u32, u32, String) + Sync),
    ) -> AppResult<()> {
        let china = self.settings().mirror == "china";
        step(1, 5, t!("准备 uv（Python 环境管理工具）"));
        log(t!("【1/5】准备 uv（Python 环境管理工具）"));
        let uv = match self.uv() {
            Some(u) => u,
            None => download_uv(&self.root().join("bin"), china, log, progress, &self.cancel)?,
        };
        log(t!("使用 {path}", path = uv.display()));
        self.check_cancel()?;

        step(2, 5, t!("创建 Python 3.11 环境"));
        log(t!("【2/5】创建 Python 3.11 环境（首次需要下载约 30MB）"));
        let env = self.env_dir();
        let mut c = command(&uv);
        c.args(["venv", "--python", "3.11", "--allow-existing"]).arg(&env);
        c.env("UV_PYTHON_INSTALL_DIR", self.root().join("python"));
        if china {
            c.env("UV_PYTHON_INSTALL_MIRROR", "https://registry.npmmirror.com/-/binary/python-build-standalone");
        }
        self.run_step(c, log)?;
        let py = self.managed_python();

        let size = match variant {
            "cpu" => "200MB",
            "mps" => "300MB",
            _ => "3GB",
        };
        step(3, 5, t!("安装 PyTorch（{variant}，约 {size}，耗时最长）", variant = variant, size = size));
        log(t!("【3/5】安装 PyTorch（{variant}，约 {size}，耗时最长）", variant = variant, size = size));
        let pypi = if china { "https://mirrors.aliyun.com/pypi/simple" } else { "https://pypi.org/simple" };
        let official = format!("https://download.pytorch.org/whl/{variant}");
        if variant == "mps" {
            // macOS：PyPI 上的 torch 自带 Apple 芯片 MPS 加速，直接装
            let mut c = command(&uv);
            c.args(["pip", "install", "--python"]).arg(&py);
            c.args(["torch", "torchvision", "--index-url", pypi]);
            self.run_step(c, log)?;
        } else if variant == "cpu" {
            let mut c = command(&uv);
            c.args(["pip", "install", "--python"]).arg(&py);
            c.args(["torch", "torchvision", "--index-url"]).arg(&official).args(["--extra-index-url", pypi]);
            self.run_step(c, log)?;
        } else {
            // 精确指定带 +cuXXX 的版本，避免装成同版本号的 CPU 包
            let page = http_get_text(&format!("{official}/torch/"))
                .map_err(|e| msg(t!("无法访问 PyTorch 下载站：{e}", e = e)))?;
            let torch_v = best_version(&page, "torch", variant)
                .ok_or_else(|| msg(t!("PyTorch 官方没有适合这台电脑的 {variant} 版本", variant = variant)))?;
            let vision_v = torchvision_for(&torch_v);
            log(t!("选择 torch {torch_v}+{variant}、torchvision {vision_v}+{variant}", torch_v = torch_v, variant = variant, vision_v = vision_v));
            let on_aliyun = china
                && http_get_text(&format!("https://mirrors.aliyun.com/pytorch-wheels/{variant}/"))
                    .ok()
                    .and_then(|p| best_version_matching(&p, "torch", variant, &torch_v))
                    .is_some();
            let base = if on_aliyun {
                log(t!("阿里云镜像有这个版本，从国内镜像下载"));
                format!("https://mirrors.aliyun.com/pytorch-wheels/{variant}")
            } else {
                if china {
                    log(t!("国内镜像还没有这个版本，从 PyTorch 官方下载"));
                }
                official.clone()
            };
            // 先把两个大 wheel 自己下下来（带进度条、断线续传），再本地安装；下不下来再退回 uv 直接装
            match self.download_wheels(&base, &torch_v, &vision_v, variant, log, progress) {
                Ok((torch_whl, vision_whl)) => {
                    log(t!("下载完成，正在安装…"));
                    let mut c = command(&uv);
                    c.args(["pip", "install", "--python"]).arg(&py).arg(&torch_whl).arg(&vision_whl);
                    c.args(["--index-url", pypi]);
                    self.run_step(c, log)?;
                    let _ = fs::remove_dir_all(self.root().join("wheels"));
                }
                Err(e) => {
                    self.check_cancel()?;
                    log(t!("自带下载没成功（{e}），改用 uv 下载。uv 会从头重新下载、没有进度条，请耐心等", e = e));
                    let mut c = command(&uv);
                    c.args(["pip", "install", "--python"]).arg(&py);
                    c.arg(format!("torch=={torch_v}+{variant}")).arg(format!("torchvision=={vision_v}+{variant}"));
                    if on_aliyun {
                        c.args(["--index-url", pypi, "--find-links"]).arg(format!("https://mirrors.aliyun.com/pytorch-wheels/{variant}/"));
                    } else {
                        c.args(["--index-url"]).arg(&official).args(["--extra-index-url", pypi]);
                    }
                    self.run_step(c, log)?;
                    // 装好了，没下完的 .part 不用留着
                    let _ = fs::remove_dir_all(self.root().join("wheels"));
                }
            }
        }
        if !cuda_matches(&py, variant, &self.worker_dir) {
            return Err(msg(t!("装好的 PyTorch 不是显卡版本，请改用「官方源」后重新安装")));
        }

        step(4, 5, t!("安装 numpy、pillow、onnx、opencv 等组件"));
        log(t!("【4/5】安装其他依赖"));
        let mut c = command(&uv);
        c.args(["pip", "install", "--python"]).arg(&py).args(["numpy", "pillow", "onnx", "onnxruntime", "opencv-python-headless"]);
        if china {
            c.args(["--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple"]);
        }
        self.run_step(c, log)?;

        step(5, 5, t!("检查环境"));
        log(t!("【5/5】检查环境"));
        *self.check_cache.lock() = None;
        let v = self.check(true)?;
        log(t!("完成：{v}", v = v));
        // 新显卡配旧 CUDA 版本等问题，envcheck 会给出说明
        if let Some(e) = v.get("error").and_then(|e| e.as_str()) {
            return Err(msg(e.to_string()));
        }
        if variant.starts_with("cu") && v["cuda"] != true {
            return Err(msg(t!("PyTorch 已安装，但显卡不可用。请更新 NVIDIA 驱动后重新安装，或改用 CPU 版本")));
        }
        Ok(())
    }

    /// 自己下载 torch / torchvision 两个大 wheel（带进度回调），返回本地文件路径
    fn download_wheels(
        &self,
        base: &str,
        torch_v: &str,
        vision_v: &str,
        variant: &str,
        log: &(dyn Fn(String) + Sync),
        progress: &(dyn Fn(&str, u64, u64, u64) + Sync),
    ) -> AppResult<(PathBuf, PathBuf)> {
        let dir = self.root().join("wheels");
        fs::create_dir_all(&dir)?;
        let torch_whl = dir.join(format!("torch-{torch_v}+{variant}{}", wheel_suffix()));
        let vision_whl = dir.join(format!("torchvision-{vision_v}+{variant}{}", wheel_suffix()));
        log(t!("下载 torch（{torch_v}+{variant}）", torch_v = torch_v, variant = variant));
        download_file(&wheel_url(base, "torch", torch_v, variant), &torch_whl, log, &|d, t, s| progress("torch", d, t, s), &self.cancel)?;
        log(t!("下载 torchvision（{vision_v}+{variant}）", vision_v = vision_v, variant = variant));
        download_file(&wheel_url(base, "torchvision", vision_v, variant), &vision_whl, log, &|d, t, s| progress("torchvision", d, t, s), &self.cancel)?;
        Ok((torch_whl, vision_whl))
    }
}

/// torch / torchvision 的 wheel 直链（URL 里 + 要转义成 %2B）
fn wheel_url(base: &str, pkg: &str, version: &str, variant: &str) -> String {
    format!("{base}/{pkg}-{version}%2B{variant}{}", wheel_suffix())
}

/// 下载用的 HTTP 客户端：不限总时长（大文件、慢网络也能下完），连不上 20 秒、一分钟收不到数据才算断了
fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(20))
        .timeout_read(Duration::from_secs(60))
        .timeout_write(Duration::from_secs(60))
        .build()
}

/// Content-Range（bytes 100-199/1000 或 bytes */1000）里的总大小
fn range_total(h: Option<&str>) -> Option<u64> {
    h?.rsplit('/').next()?.trim().parse().ok()
}

/// 下载断了最多连续重试几次（这次有进展就重新计数）
const DOWNLOAD_RETRIES: u32 = 5;

/// 流式下载到文件，边下边回调（已下载字节（含续传的部分）、总字节、瞬时速度 字节/秒）。
/// 先写到 .part，断了用 Range 从断点接着下（上次没下完的也接着下），没进展的失败最多连续重试 5 次；
/// 服务器不支持断点续传时不重试，直接报错，由调用方换别的方式。cancel 置位时尽快停下。
fn download_file(
    url: &str,
    dest: &Path,
    log: &(dyn Fn(String) + Sync),
    progress: &(dyn Fn(u64, u64, u64) + Sync),
    cancel: &AtomicBool,
) -> AppResult<()> {
    if let Some(p) = dest.parent() {
        fs::create_dir_all(p)?;
    }
    let tmp = dest.with_extension("part");
    // 上次下完了、但后面的步骤失败：当成断点续传，服务器会告诉我们已经齐了
    if dest.is_file() && !tmp.exists() {
        let _ = fs::rename(dest, &tmp);
    }
    fn size(p: &Path) -> u64 {
        fs::metadata(p).map(|m| m.len()).unwrap_or(0)
    }
    let agent = agent();
    let start = std::time::Instant::now();
    let start_bytes = size(&tmp);
    let mut fails = 0u32;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(msg(t!("已取消安装")));
        }
        let before = size(&tmp);
        match download_once(&agent, url, &tmp, before, progress, cancel) {
            Ok(total) => {
                let got = size(&tmp);
                let avg = (got.saturating_sub(start_bytes) as f64 / start.elapsed().as_secs_f64().max(0.001)) as u64;
                progress(got, total.max(got), avg);
                fs::rename(&tmp, dest)?;
                return Ok(());
            }
            Err((e, retry)) => {
                if cancel.load(Ordering::Relaxed) {
                    return Err(msg(t!("已取消安装")));
                }
                if !retry {
                    return Err(msg(e));
                }
                // 这次下到了 1MB 以上就算有进展，重新计数
                if size(&tmp) > before + 1024 * 1024 {
                    fails = 0;
                } else {
                    fails += 1;
                }
                if fails >= DOWNLOAD_RETRIES {
                    return Err(msg(t!("重试 {n} 次还是没下下来：{e}", n = DOWNLOAD_RETRIES, e = e)));
                }
                let wait = 2 * (fails as u64 + 1);
                log(t!("下载中断（{e}），{s} 秒后从断点接着下…", e = e, s = wait));
                for _ in 0..wait * 10 {
                    if cancel.load(Ordering::Relaxed) {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }
    }
}

/// 一次下载尝试，从 have 字节处接着下。成功返回总大小（不知道时为 0）；失败返回（原因, 值不值得重试）
fn download_once(
    agent: &ureq::Agent,
    url: &str,
    tmp: &Path,
    have: u64,
    progress: &(dyn Fn(u64, u64, u64) + Sync),
    cancel: &AtomicBool,
) -> Result<u64, (String, bool)> {
    let mut req = agent.get(url);
    if have > 0 {
        req = req.set("Range", &format!("bytes={have}-"));
    }
    let resp = match req.call() {
        Ok(r) => r,
        Err(ureq::Error::Status(416, r)) if have > 0 => {
            // 续传的起点已经是文件末尾：其实已经下完了
            if range_total(r.header("Content-Range")) == Some(have) {
                return Ok(have);
            }
            let _ = fs::remove_file(tmp);
            return Err(("HTTP 416".to_string(), true));
        }
        // 服务器忙、限流可以重试；404 之类的直接失败
        Err(ureq::Error::Status(code, _)) => return Err((format!("HTTP {code}"), code == 408 || code == 429 || code >= 500)),
        Err(e) => return Err((e.to_string(), true)),
    };
    let partial = resp.status() == 206;
    // 能不能断点续传：这次就是续传的，或者服务器声明支持 Range
    let resumable = partial || resp.header("Accept-Ranges").is_some_and(|v| v.contains("bytes"));
    let total: u64 = if partial {
        range_total(resp.header("Content-Range")).unwrap_or(0)
    } else {
        resp.header("Content-Length").and_then(|s| s.parse().ok()).unwrap_or(0)
    };
    // 服务器不认 Range、返回了整个文件：从头写
    let opened = if partial { fs::OpenOptions::new().append(true).open(tmp) } else { fs::File::create(tmp) };
    let mut file = opened.map_err(|e| (e.to_string(), false))?;
    let mut done = if partial { have } else { 0 };
    progress(done, total, 0);
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 256 * 1024];
    let mut last = std::time::Instant::now();
    let mut last_bytes = done;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err((t!("已取消安装"), false));
        }
        let n = match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => return Err((e.to_string(), resumable)),
        };
        file.write_all(&buf[..n]).map_err(|e| (e.to_string(), false))?;
        done += n as u64;
        if last.elapsed() >= Duration::from_millis(300) {
            let dt = last.elapsed().as_secs_f64().max(0.001);
            progress(done, total, ((done - last_bytes) as f64 / dt) as u64);
            last = std::time::Instant::now();
            last_bytes = done;
        }
    }
    file.flush().map_err(|e| (e.to_string(), false))?;
    // 核对大小：连接提前断开时 read 也可能返回 0
    if total > 0 && done != total {
        if done > total {
            let _ = fs::remove_file(tmp);
        }
        return Err((t!("下载不完整（{done} / {total} 字节）", done = done, total = total), resumable));
    }
    Ok(total)
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeedResult {
    /// 国内镜像（阿里云）下载速度，MB/s；测不到为 null
    pub china: Option<f64>,
    /// 官方源（download.pytorch.org）下载速度，MB/s；测不到为 null
    pub official: Option<f64>,
    /// 推荐使用的下载源："china" 或 "official"
    pub recommend: String,
}

/// 实测当前网络下「国内镜像」vs「官方源」下载 torch 的速度，推荐更快的那个
pub fn speed_test(variant: &str) -> AppResult<SpeedResult> {
    if variant == "mps" || cfg!(target_os = "macos") {
        return Err(msg(t!("macOS 上 PyTorch 从 PyPI 安装，不需要测速；在国内选「国内镜像」即可")));
    }
    let official = format!("https://download.pytorch.org/whl/{variant}");
    let page = http_get_text(&format!("{official}/torch/")).map_err(|e| msg(t!("无法访问 PyTorch 下载站：{e}", e = e)))?;
    let torch_v = best_version(&page, "torch", variant).ok_or_else(|| msg(t!("没有找到适合的 {variant} 版本", variant = variant)))?;
    let file = format!("torch-{torch_v}%2B{variant}{}", wheel_suffix());
    let china = measure_speed(&format!("https://mirrors.aliyun.com/pytorch-wheels/{variant}/{file}"));
    let official_speed = measure_speed(&format!("{official}/{file}"));
    let recommend = match (china, official_speed) {
        (Some(c), Some(o)) => if c >= o { "china" } else { "official" },
        (None, Some(_)) => "official",
        _ => "china",
    }
    .to_string();
    Ok(SpeedResult { china, official: official_speed, recommend })
}

/// 从 url 拉最多 20MB（限时 6 秒），返回速度 MB/s；连不上或拉到的太少返回 None
fn measure_speed(url: &str) -> Option<f64> {
    let resp = ureq::get(url).set("Range", "bytes=0-20971519").timeout(std::time::Duration::from_secs(10)).call().ok()?;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 64 * 1024];
    let mut got: u64 = 0;
    let start = std::time::Instant::now();
    while start.elapsed() < std::time::Duration::from_secs(6) {
        match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => got += n as u64,
            Err(_) => break,
        }
        if got >= 20 * 1024 * 1024 {
            break;
        }
    }
    let secs = start.elapsed().as_secs_f64().max(0.001);
    if got < 64 * 1024 {
        return None;
    }
    Some(got as f64 / (1024.0 * 1024.0) / secs)
}

/// 环境检查结果里缺的组件（pip 包名）。旧版 worker 没报的模块不算缺
fn missing_packages(info: &serde_json::Value) -> Vec<String> {
    PACKAGES.iter().filter(|(m, _, _)| info[*m] == false).map(|(_, pkg, _)| pkg.to_string()).collect()
}

/// 装好的 torch 是不是想要的 CUDA 版本（镜像有时会装成 CPU 版）
fn cuda_matches(py: &Path, variant: &str, worker_dir: &Path) -> bool {
    if !variant.starts_with("cu") {
        return true;
    }
    let out = command(py)
        .args(["-c", "import torch;print(torch.version.cuda or 'cpu')"])
        .env("PYTHONPATH", worker_dir)
        .output();
    matches!(out, Ok(o) if !String::from_utf8_lossy(&o.stdout).trim().starts_with("cpu"))
}

fn parse_version(v: &str) -> Vec<u32> {
    v.split('.').map(|x| x.parse().unwrap_or(0)).collect()
}

/// 当前平台和 Python 3.11 对应的 wheel 文件名结尾
fn wheel_suffix() -> &'static str {
    if cfg!(windows) {
        "-cp311-cp311-win_amd64.whl"
    } else if cfg!(target_arch = "aarch64") {
        // ARM64 Linux（飞腾、鲲鹏等），PyTorch 官方有 CPU 和 CUDA 两种
        "-cp311-cp311-manylinux_2_28_aarch64.whl"
    } else {
        "-cp311-cp311-manylinux_2_28_x86_64.whl"
    }
}

/// 从下载页里找某个包的所有版本（文件名形如 torch-2.11.0+cu128-cp311-cp311-win_amd64.whl，「+」可能被转义）
fn versions(page: &str, pkg: &str, variant: &str) -> Vec<String> {
    let page = page.replace("&#43;", "+").replace("%2B", "+").replace("%2b", "+");
    let prefix = format!("{pkg}-");
    let tail = format!("+{variant}{}", wheel_suffix());
    let mut out: Vec<String> = page
        .match_indices(&prefix)
        .filter_map(|(i, _)| {
            let rest = &page[i + prefix.len()..];
            let end = rest.find(&tail)?;
            let v = &rest[..end];
            (!v.is_empty() && v.chars().all(|c| c.is_ascii_digit() || c == '.')).then(|| v.to_string())
        })
        .collect();
    out.sort_by_key(|v| parse_version(v));
    out.dedup();
    out
}

fn best_version(page: &str, pkg: &str, variant: &str) -> Option<String> {
    versions(page, pkg, variant).pop()
}

fn best_version_matching(page: &str, pkg: &str, variant: &str, want: &str) -> Option<String> {
    versions(page, pkg, variant).into_iter().find(|v| v == want)
}

/// torch 2.x 对应 torchvision 0.(x+15)，例如 2.11.0 → 0.26.0
fn torchvision_for(torch: &str) -> String {
    let v = parse_version(torch);
    format!("0.{}.{}", v.get(1).copied().unwrap_or(0) + 15, v.get(2).copied().unwrap_or(0))
}

fn http_get_text(url: &str) -> AppResult<String> {
    Ok(String::from_utf8_lossy(&http_get(url)?).into_owned())
}

fn last_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// 运行命令，stdout 和 stderr 逐行回调；失败时带上最后几行输出
pub fn run_logged(c: Command, log: &(dyn Fn(String) + Sync)) -> AppResult<()> {
    run_cancellable(c, log, None)
}

/// 同 run_logged；cancel 置位时结束这个进程
fn run_cancellable(mut c: Command, log: &(dyn Fn(String) + Sync), cancel: Option<&AtomicBool>) -> AppResult<()> {
    c.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = c.spawn().map_err(|e| msg(t!("无法启动 {prog}：{e}", prog = format!("{:?}", c.get_program()), e = e)))?;
    let out = child.stdout.take().unwrap();
    let err = child.stderr.take().unwrap();
    let tail = Mutex::new(Vec::<String>::new());
    let pump = |r: Box<dyn Read + Send>| {
        for line in BufReader::new(r).split(b'\n').map_while(Result::ok) {
            let line = String::from_utf8_lossy(&line).trim_end().to_string();
            // 进度条用 \r 刷新，只取最后一段
            let line = line.rsplit('\r').next().unwrap_or("").to_string();
            if line.is_empty() {
                continue;
            }
            {
                let mut t = tail.lock();
                t.push(line.clone());
                if t.len() > 8 {
                    t.remove(0);
                }
            }
            log(line);
        }
    };
    let status = std::thread::scope(|s| {
        s.spawn(|| pump(Box::new(out)));
        s.spawn(|| pump(Box::new(err)));
        // 等进程结束；点了取消就结束它
        loop {
            if let Some(st) = child.try_wait()? {
                return Ok::<_, std::io::Error>(st);
            }
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                let _ = child.kill();
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    })?;
    if !status.success() {
        return Err(msg(t!("命令失败（{status}）：{tail}", status = status, tail = tail.lock().join("\n"))));
    }
    Ok(())
}

pub fn detect_gpu() -> Option<Gpu> {
    let out = command("nvidia-smi")
        .args(["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let first = text.lines().next()?;
    let parts: Vec<&str> = first.split(',').map(str::trim).collect();
    if parts.len() < 3 {
        return None;
    }
    let full = command("nvidia-smi").output().ok()?;
    let full = String::from_utf8_lossy(&full.stdout);
    let cuda = full
        .split("CUDA Version:")
        .nth(1)
        .and_then(|s| s.split_whitespace().next())
        .unwrap_or("")
        .to_string();
    Some(Gpu { name: parts[0].into(), vram_mb: parts[1].parse().unwrap_or(0), driver: parts[2].into(), cuda })
}

/// 能装最老的 CUDA 版 PyTorch（cu118；ARM64 Linux 是 cu128）所需的 NVIDIA 驱动版本
fn oldest_cuda_driver() -> &'static str {
    if cfg!(target_arch = "aarch64") {
        "570"
    } else if cfg!(windows) {
        "522.06"
    } else {
        "520.61"
    }
}

/// 按驱动支持的 CUDA 版本选 PyTorch 安装包。macOS 用 PyPI 上自带 MPS（Apple 芯片加速）的版本
pub fn torch_variant(gpu: Option<&Gpu>) -> &'static str {
    if cfg!(target_os = "macos") {
        return "mps";
    }
    let Some(g) = gpu else { return "cpu" };
    let mut it = g.cuda.split('.').map(|x| x.parse::<u32>().unwrap_or(0));
    let v = (it.next().unwrap_or(0), it.next().unwrap_or(0));
    // ARM64 Linux 只有较新的 CUDA 版本有现成的 PyTorch，驱动太旧就用 CPU
    if cfg!(target_arch = "aarch64") {
        return if v >= (12, 8) { "cu128" } else { "cpu" };
    }
    if v >= (13, 0) {
        "cu130"
    } else if v >= (12, 8) {
        "cu128"
    } else if v >= (12, 6) {
        "cu126"
    } else if v >= (11, 8) {
        "cu118"
    } else {
        "cpu"
    }
}

/// 下载 uv（带进度）。国内镜像：先清华、阿里云的 PyPI 镜像，再 GitHub、PyPI；官方源：先 GitHub，再 PyPI。
/// PyPI 上的 uv wheel 也是 zip，里面带 uv.exe
fn download_uv(
    bin: &Path,
    china: bool,
    log: &(dyn Fn(String) + Sync),
    progress: &(dyn Fn(&str, u64, u64, u64) + Sync),
    cancel: &AtomicBool,
) -> AppResult<PathBuf> {
    fs::create_dir_all(bin)?;
    let exe = if cfg!(windows) { "uv.exe" } else { "uv" };
    let arm = cfg!(target_arch = "aarch64");
    let target = if cfg!(windows) {
        "x86_64-pc-windows-msvc"
    } else if cfg!(target_os = "macos") {
        if arm { "aarch64-apple-darwin" } else { "x86_64-apple-darwin" }
    } else if arm {
        "aarch64-unknown-linux-gnu"
    } else {
        "x86_64-unknown-linux-gnu"
    };
    let tag = if cfg!(windows) {
        "win_amd64"
    } else if cfg!(target_os = "macos") {
        if arm { "macosx_11_0_arm64" } else { "macosx_10_12_x86_64" }
    } else if arm {
        "manylinux_2_17_aarch64"
    } else {
        "manylinux_2_17_x86_64"
    };
    let sources: &[&str] = if china { &["tuna", "aliyun", "github", "pypi"] } else { &["github", "pypi"] };
    let tmp = bin.join("uv-download.zip");
    let mut errors = Vec::new();
    for src in sources {
        if cancel.load(Ordering::Relaxed) {
            return Err(msg(t!("已取消安装")));
        }
        let url = match *src {
            "tuna" => simple_index_wheel("https://pypi.tuna.tsinghua.edu.cn/simple/uv/", tag),
            "aliyun" => simple_index_wheel("https://mirrors.aliyun.com/pypi/simple/uv/", tag),
            "github" => Ok(format!("https://github.com/astral-sh/uv/releases/latest/download/uv-{target}.zip")),
            _ => pypi_uv_wheel(tag),
        };
        // 每个源的文件不一样，不能拿别的源下了一半的接着下
        let _ = fs::remove_file(&tmp);
        let _ = fs::remove_file(tmp.with_extension("part"));
        let r = url.and_then(|url| {
            log(t!("下载 uv：{github}", github = url));
            download_file(&url, &tmp, log, &|d, t, s| progress("uv", d, t, s), cancel)?;
            let bytes = fs::read(&tmp)?;
            extract_exe(&bytes, exe, &bin.join(exe))
        });
        let _ = fs::remove_file(&tmp);
        match r {
            Ok(p) => return Ok(p),
            Err(e) => {
                log(t!("下载失败：{e}", e = e));
                errors.push(e.to_string());
            }
        }
    }
    Err(msg(t!("下载 uv 失败：{errors}。可以手动安装 uv 后在设置里指定路径", errors = errors.join("；"))))
}

/// PyPI 官方：从 JSON 接口找最新的 uv wheel
fn pypi_uv_wheel(tag: &str) -> AppResult<String> {
    let meta: serde_json::Value = serde_json::from_slice(&http_get("https://pypi.org/pypi/uv/json")?)?;
    meta["urls"]
        .as_array()
        .and_then(|a| a.iter().find(|u| u["filename"].as_str().is_some_and(|f| f.ends_with(".whl") && f.contains(tag))))
        .and_then(|u| u["url"].as_str())
        .map(str::to_string)
        .ok_or_else(|| msg(t!("PyPI 上没有找到对应的 uv 安装包")))
}

/// PyPI 镜像：从简单索引页（/simple/uv/）里找最新版本的 uv wheel
fn simple_index_wheel(index: &str, tag: &str) -> AppResult<String> {
    let page = http_get_text(index)?;
    pick_wheel(&page, index, "uv", tag).ok_or_else(|| msg(t!("PyPI 上没有找到对应的 uv 安装包")))
}

/// 索引页里 pkg 的、文件名含 tag 的 wheel，取版本最高的正式版，返回完整地址
fn pick_wheel(page: &str, base: &str, pkg: &str, tag: &str) -> Option<String> {
    let prefix = format!("{pkg}-");
    let mut best: Option<(Vec<u32>, String)> = None;
    for part in page.split("href=\"").skip(1) {
        let href = part.split('"').next().unwrap_or("");
        let path = href.split('#').next().unwrap_or("");
        let name = path.rsplit('/').next().unwrap_or("");
        if !name.ends_with(".whl") || !name.contains(tag) {
            continue;
        }
        let Some(ver) = name.strip_prefix(&prefix).and_then(|r| r.split('-').next()) else { continue };
        if ver.is_empty() || !ver.chars().all(|c| c.is_ascii_digit() || c == '.') {
            continue;
        }
        let v = parse_version(ver);
        if best.as_ref().is_none_or(|(b, _)| v > *b) {
            best = Some((v, resolve_url(base, path)));
        }
    }
    best.map(|(_, u)| u)
}

/// 把页面里的相对链接（../../packages/…、/packages/…）拼成完整地址
fn resolve_url(base: &str, href: &str) -> String {
    if href.starts_with("http://") || href.starts_with("https://") {
        return href.to_string();
    }
    let (scheme, rest) = base.split_once("://").unwrap_or(("https", base));
    let (host, path) = rest.split_once('/').unwrap_or((rest, ""));
    if let Some(abs) = href.strip_prefix('/') {
        return format!("{scheme}://{host}/{abs}");
    }
    let mut segs: Vec<&str> = path.split('/').collect();
    // 去掉最后一段（文件名，或者末尾 / 后面的空串），剩下的是目录
    segs.pop();
    for s in href.split('/') {
        match s {
            ".." => {
                segs.pop();
            }
            "." | "" => {}
            s => segs.push(s),
        }
    }
    segs.retain(|s| !s.is_empty());
    format!("{scheme}://{host}/{}", segs.join("/"))
}

/// 下载 ffmpeg 静态版并解压出可执行文件。
/// Windows：先 gyan.dev，再退 BtbN 的 GitHub 构建；macOS：Apple 芯片用 osxexperts，Intel 用 evermeet（Apple 芯片也能经 Rosetta 跑）
fn download_ffmpeg(
    bin: &Path,
    log: &(dyn Fn(String) + Sync),
    progress: &(dyn Fn(&str, u64, u64, u64) + Sync),
) -> AppResult<PathBuf> {
    let (exe, sources): (&str, Vec<&str>) = if cfg!(windows) {
        (
            "ffmpeg.exe",
            vec![
                "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
                "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip",
            ],
        )
    } else if cfg!(target_os = "macos") {
        let intel = "https://evermeet.cx/ffmpeg/getrelease/zip";
        if cfg!(target_arch = "aarch64") {
            ("ffmpeg", vec!["https://www.osxexperts.net/ffmpeg71arm.zip", intel])
        } else {
            ("ffmpeg", vec![intel])
        }
    } else {
        return download_ffmpeg_linux(bin, log, progress);
    };
    fs::create_dir_all(bin)?;
    let out = bin.join(exe);
    let zip = bin.join("ffmpeg-download.zip");
    let mut errors = Vec::new();
    for url in sources {
        log(t!("下载 ffmpeg：{url}", url = url));
        // 换了下载源，上一个源没下完的不能接着用
        let _ = fs::remove_file(zip.with_extension("part"));
        let r = download_file(url, &zip, log, &|d, t, s| progress("ffmpeg", d, t, s), &AtomicBool::new(false))
            .and_then(|_| {
                let bytes = fs::read(&zip)?;
                extract_exe(&bytes, exe, &out)
            });
        let _ = fs::remove_file(&zip);
        match r {
            Ok(p) => return Ok(p),
            Err(e) => {
                log(t!("下载失败：{e}", e = e));
                errors.push(e.to_string());
            }
        }
    }
    Err(msg(t!("下载 ffmpeg 失败：{errors}。可以手动安装 ffmpeg 后在设置里指定路径", errors = errors.join("；"))))
}

/// Linux：下载静态编译的 ffmpeg（.tar.xz，用系统自带的 tar 解压），x86_64 和 ARM64 各有两个下载源
fn download_ffmpeg_linux(
    bin: &Path,
    log: &(dyn Fn(String) + Sync),
    progress: &(dyn Fn(&str, u64, u64, u64) + Sync),
) -> AppResult<PathBuf> {
    let sources: &[&str] = if cfg!(target_arch = "aarch64") {
        &[
            "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-linuxarm64-gpl.tar.xz",
            "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz",
        ]
    } else {
        &[
            "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-linux64-gpl.tar.xz",
            "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
        ]
    };
    fs::create_dir_all(bin)?;
    let out = bin.join("ffmpeg");
    let pkg = bin.join("ffmpeg-download.tar.xz");
    let tmp = bin.join("ffmpeg-extract");
    let mut errors = Vec::new();
    for url in sources {
        log(t!("下载 ffmpeg：{url}", url = url));
        let _ = fs::remove_dir_all(&tmp);
        let _ = fs::remove_file(pkg.with_extension("part"));
        let r = download_file(url, &pkg, log, &|d, t, s| progress("ffmpeg", d, t, s), &AtomicBool::new(false)).and_then(|_| {
            fs::create_dir_all(&tmp)?;
            let st = command("tar")
                .arg("-xJf")
                .arg(&pkg)
                .arg("-C")
                .arg(&tmp)
                .status()
                .map_err(|e| msg(t!("解压失败：{e}", e = e)))?;
            if !st.success() {
                return Err(msg(t!("解压失败：{e}", e = st)));
            }
            let found = find_file(&tmp, "ffmpeg").ok_or_else(|| msg(t!("压缩包里没有 {exe}", exe = "ffmpeg")))?;
            fs::copy(&found, &out)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&out, fs::Permissions::from_mode(0o755))?;
            }
            Ok(out.clone())
        });
        let _ = fs::remove_file(&pkg);
        let _ = fs::remove_dir_all(&tmp);
        match r {
            Ok(p) => return Ok(p),
            Err(e) => {
                log(t!("下载失败：{e}", e = e));
                errors.push(e.to_string());
            }
        }
    }
    let errors = errors.join("；");
    Err(msg(t!("下载 ffmpeg 失败：{errors}。可以用系统的包管理器安装（如 apt install ffmpeg），或在设置里指定路径", errors = errors)))
}

/// 在文件夹里（含子文件夹）找某个文件名
fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    for e in fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        if p.is_dir() {
            if let Some(f) = find_file(&p, name) {
                return Some(f);
            }
        } else if p.file_name().and_then(|n| n.to_str()) == Some(name) {
            return Some(p);
        }
    }
    None
}

fn http_get(url: &str) -> AppResult<Vec<u8>> {
    let resp = agent().get(url).call().map_err(|e| msg(e.to_string()))?;
    let mut buf = Vec::new();
    resp.into_reader().take(200 * 1024 * 1024).read_to_end(&mut buf)?;
    Ok(buf)
}

fn extract_exe(zip_bytes: &[u8], exe: &str, out: &Path) -> AppResult<PathBuf> {
    let mut z = zip::ZipArchive::new(std::io::Cursor::new(zip_bytes)).map_err(|e| msg(t!("解压失败：{e}", e = e)))?;
    for i in 0..z.len() {
        let mut f = z.by_index(i).map_err(|e| msg(t!("解压失败：{e}", e = e)))?;
        if f.name().rsplit('/').next() == Some(exe) {
            let mut w = fs::File::create(out)?;
            std::io::copy(&mut f, &mut w)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(out, fs::Permissions::from_mode(0o755))?;
            }
            return Ok(out.to_path_buf());
        }
    }
    Err(msg(t!("压缩包里没有 {exe}", exe = exe)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(cuda: &str) -> Gpu {
        Gpu { name: "x".into(), vram_mb: 8000, driver: "1".into(), cuda: cuda.into() }
    }

    /// 真实安装（下载几 GB），手动运行：
    /// XLABEL_DATA_DIR=... XLABEL_CONFIG_DIR=... cargo test install_real -- --ignored --nocapture
    #[test]
    #[ignore]
    fn install_real() {
        let data = PathBuf::from(std::env::var("XLABEL_DATA_DIR").unwrap());
        let cfg = PathBuf::from(std::env::var("XLABEL_CONFIG_DIR").unwrap());
        let worker = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../worker");
        let rt = Runtime::new(cfg, data, worker);
        if let Ok(root) = std::env::var("XLABEL_ENV_ROOT") {
            let mut s = rt.settings();
            s.env_root = Some(root);
            rt.save_settings(&s).unwrap();
        }
        let variant =std::env::var("XLABEL_VARIANT").unwrap_or_else(|_| torch_variant(detect_gpu().as_ref()).to_string());
        eprintln!("安装 {variant}");
        rt.install(&variant, &|l| eprintln!("{l}"), &|f, d, t, s| eprintln!("{f}: {d}/{t} {s}B/s"), &|n, all, s| eprintln!("[{n}/{all}] {s}"))
            .unwrap();
        eprintln!("{:?}", rt.check(true).unwrap());
    }

    #[test]
    fn variant_by_cuda() {
        assert_eq!(torch_variant(None), "cpu");
        assert_eq!(torch_variant(Some(&gpu("12.8"))), "cu128");
        assert_eq!(torch_variant(Some(&gpu("13.0"))), "cu130");
        assert_eq!(torch_variant(Some(&gpu("12.9"))), "cu128");
        assert_eq!(torch_variant(Some(&gpu("12.6"))), "cu126");
        assert_eq!(torch_variant(Some(&gpu("12.4"))), "cu118");
        assert_eq!(torch_variant(Some(&gpu("11.4"))), "cpu");
        assert_eq!(torch_variant(Some(&gpu(""))), "cpu");
    }

    #[test]
    fn pick_versions_from_index() {
        let s = wheel_suffix();
        let page = format!(
            r#"<a href="/whl/cu128/torch-2.9.1%2Bcu128{s}">x</a>
               <a href="torch-2.11.0%2Bcu128{s}">x</a>
               <a href="torch-2.10.0&#43;cu128{s}">x</a>
               <a href="torch-2.12.0+cu128-cp312-cp312-win_amd64.whl">其他 Python 版本</a>
               <a href="torch-2.13.0+cu130{s}">其他 CUDA</a>
               <a href="torchvision-0.26.0+cu128{s}">x</a>"#
        );
        assert_eq!(best_version(&page, "torch", "cu128").as_deref(), Some("2.11.0"));
        assert_eq!(versions(&page, "torch", "cu128"), vec!["2.9.1", "2.10.0", "2.11.0"]);
        assert_eq!(best_version(&page, "torchvision", "cu128").as_deref(), Some("0.26.0"));
        assert!(best_version_matching(&page, "torch", "cu128", "2.10.0").is_some());
        assert!(best_version_matching(&page, "torch", "cu128", "2.12.0").is_none());
        assert_eq!(torchvision_for("2.11.0"), "0.26.0");
        assert_eq!(torchvision_for("2.9.1"), "0.24.1");
    }

    #[test]
    fn uv_wheel_from_simple_index() {
        let base = "https://pypi.tuna.tsinghua.edu.cn/simple/uv/";
        let page = r#"<a href="../../packages/aa/bb/uv-0.9.5-py3-none-win_amd64.whl#sha256=1">uv-0.9.5-py3-none-win_amd64.whl</a>
            <a href="../../packages/cc/dd/uv-0.10.1-py3-none-win_amd64.whl#sha256=2">x</a>
            <a href="../../packages/ee/ff/uv-0.11.0rc1-py3-none-win_amd64.whl">预发布版</a>
            <a href="../../packages/gg/hh/uv-0.12.0-py3-none-win_arm64.whl">其他平台</a>
            <a href="../../packages/ii/jj/uv-0.12.0.tar.gz">源码包</a>"#;
        assert_eq!(
            pick_wheel(page, base, "uv", "win_amd64").as_deref(),
            Some("https://pypi.tuna.tsinghua.edu.cn/packages/cc/dd/uv-0.10.1-py3-none-win_amd64.whl")
        );
        assert!(pick_wheel(page, base, "uv", "macosx_11_0_arm64").is_none());
        assert_eq!(resolve_url(base, "/a/b.whl"), "https://pypi.tuna.tsinghua.edu.cn/a/b.whl");
        assert_eq!(resolve_url(base, "https://x.org/b.whl"), "https://x.org/b.whl");
        assert_eq!(range_total(Some("bytes 100-199/1000")), Some(1000));
        assert_eq!(range_total(Some("bytes */1000")), Some(1000));
        assert_eq!(range_total(Some("bytes 0-1/*")), None);
    }

    #[test]
    fn missing_from_check() {
        let v = serde_json::json!({ "torch": "2.11.0", "numpy": true, "PIL": false, "onnx": true, "onnxruntime": false });
        assert_eq!(missing_packages(&v), vec!["pillow", "onnxruntime"]);
    }

    #[test]
    fn extract_from_zip() {
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut w = zip::ZipWriter::new(&mut buf);
            let opt = zip::write::SimpleFileOptions::default();
            w.start_file("uv-0.1.data/scripts/uv.exe", opt).unwrap();
            std::io::Write::write_all(&mut w, b"binary").unwrap();
            w.finish().unwrap();
        }
        let dir = std::env::temp_dir().join(format!("xlabel-uv-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = extract_exe(buf.get_ref(), "uv.exe", &dir.join("uv.exe")).unwrap();
        assert_eq!(fs::read(p).unwrap(), b"binary");
        assert!(extract_exe(buf.get_ref(), "nope.exe", &dir.join("n")).is_err());
        let _ = fs::remove_dir_all(dir);
    }
}
