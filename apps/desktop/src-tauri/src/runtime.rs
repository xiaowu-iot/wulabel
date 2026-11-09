//! 训练环境：设置、显卡检测、uv 下载、Python + PyTorch 安装、运行 worker

use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::error::{msg, AppResult};

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

impl Default for Settings {
    fn default() -> Self {
        Settings { python: None, uv: None, mirror: default_mirror(), env_root: None, ffmpeg: None, ai_url: None, ai_key: None }
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
}

pub struct Runtime {
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub worker_dir: PathBuf,
    pub installing: Mutex<bool>,
    check_cache: Mutex<Option<(String, serde_json::Value)>>,
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
        Runtime { config_dir, data_dir, worker_dir, installing: Mutex::new(false), check_cache: Mutex::new(None) }
    }

    pub fn settings(&self) -> Settings {
        fs::read_to_string(self.config_dir.join("settings.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save_settings(&self, s: &Settings) -> AppResult<()> {
        fs::create_dir_all(&self.config_dir)?;
        fs::write(self.config_dir.join("settings.json"), serde_json::to_string_pretty(s)?)?;
        *self.check_cache.lock() = None;
        Ok(())
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
        let py = self.python().ok_or_else(|| msg("还没有训练环境，请先到「设置」里安装"))?;
        let mut c = command(py);
        c.arg("-m").arg("xlabel_worker").args(args);
        c.env("PYTHONPATH", &self.worker_dir)
            .env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONUNBUFFERED", "1")
            .env("TORCH_HOME", self.root().join("torch"))
            // SAM、文本提示等辅助模型从 Hugging Face 下载，缓存放训练环境目录
            .env("HF_HOME", self.root().join("hf"))
            .env("HF_HUB_DISABLE_TELEMETRY", "1");
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
        let py = self.python().ok_or_else(|| msg("还没有训练环境"))?;
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
                msg(format!("训练环境检查失败：{}", last_lines(&err, 6)))
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
        let ready = info.as_ref().map(|v| v.get("torch").is_some() && v.get("error").is_none()).unwrap_or(false);
        EnvStatus {
            managed: settings.python.as_deref().map(str::trim).unwrap_or("").is_empty(),
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

    /// 确保环境里有某个 Python 包，没有就用 uv 装上（OpenCV、OpenVINO 等按需安装）
    pub fn ensure_package(&self, module: &str, package: &str, log: &(dyn Fn(String) + Sync)) -> AppResult<()> {
        let py = self.python().ok_or_else(|| msg("还没有训练环境，请先到「设置」里安装"))?;
        let ok = command(&py).args(["-c", &format!("import {module}")]).output().map(|o| o.status.success()).unwrap_or(false);
        if ok {
            return Ok(());
        }
        let uv = self.uv().ok_or_else(|| msg(format!("缺少 {package}，并且没有找到 uv，无法自动安装")))?;
        log(format!("首次使用，正在安装 {package}…"));
        let mut c = command(&uv);
        c.args(["pip", "install", "--python"]).arg(&py).arg(package);
        if self.settings().mirror == "china" {
            c.args(["--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple"]);
        }
        run_logged(c, log)
    }

    /// 安装训练环境：uv → Python → PyTorch → 其他依赖 → 自检。
    /// log 收到每一行输出；progress 收到大文件下载进度（文件名、已下载字节、总字节、瞬时速度 字节/秒）。
    pub fn install(
        &self,
        variant: &str,
        log: &(dyn Fn(String) + Sync),
        progress: &(dyn Fn(&str, u64, u64, u64) + Sync),
    ) -> AppResult<()> {
        {
            let mut g = self.installing.lock();
            if *g {
                return Err(msg("已经在安装了"));
            }
            *g = true;
        }
        let r = self.install_inner(variant, log, progress);
        *self.installing.lock() = false;
        *self.check_cache.lock() = None;
        r
    }

    fn install_inner(
        &self,
        variant: &str,
        log: &(dyn Fn(String) + Sync),
        progress: &(dyn Fn(&str, u64, u64, u64) + Sync),
    ) -> AppResult<()> {
        let china = self.settings().mirror == "china";
        log(format!("【1/5】准备 uv（Python 环境管理工具）"));
        let uv = match self.uv() {
            Some(u) => u,
            None => download_uv(&self.root().join("bin"), log)?,
        };
        log(format!("使用 {}", uv.display()));

        log("【2/5】创建 Python 3.11 环境（首次需要下载约 30MB）".into());
        let env = self.env_dir();
        let mut c = command(&uv);
        c.args(["venv", "--python", "3.11", "--allow-existing"]).arg(&env);
        c.env("UV_PYTHON_INSTALL_DIR", self.root().join("python"));
        if china {
            c.env("UV_PYTHON_INSTALL_MIRROR", "https://registry.npmmirror.com/-/binary/python-build-standalone");
        }
        run_logged(c, log)?;
        let py = self.managed_python();

        let size = match variant {
            "cpu" => "200MB",
            "mps" => "300MB",
            _ => "3GB",
        };
        log(format!("【3/5】安装 PyTorch（{variant}，约 {size}，耗时最长）"));
        let pypi = if china { "https://mirrors.aliyun.com/pypi/simple" } else { "https://pypi.org/simple" };
        let official = format!("https://download.pytorch.org/whl/{variant}");
        if variant == "mps" {
            // macOS：PyPI 上的 torch 自带 Apple 芯片 MPS 加速，直接装
            let mut c = command(&uv);
            c.args(["pip", "install", "--python"]).arg(&py);
            c.args(["torch", "torchvision", "--index-url", pypi]);
            run_logged(c, log)?;
        } else if variant == "cpu" {
            let mut c = command(&uv);
            c.args(["pip", "install", "--python"]).arg(&py);
            c.args(["torch", "torchvision", "--index-url"]).arg(&official).args(["--extra-index-url", pypi]);
            run_logged(c, log)?;
        } else {
            // 精确指定带 +cuXXX 的版本，避免装成同版本号的 CPU 包
            let page = http_get_text(&format!("{official}/torch/"))
                .map_err(|e| msg(format!("无法访问 PyTorch 下载站：{e}")))?;
            let torch_v = best_version(&page, "torch", variant)
                .ok_or_else(|| msg(format!("PyTorch 官方没有适合这台电脑的 {variant} 版本")))?;
            let vision_v = torchvision_for(&torch_v);
            log(format!("选择 torch {torch_v}+{variant}、torchvision {vision_v}+{variant}"));
            let on_aliyun = china
                && http_get_text(&format!("https://mirrors.aliyun.com/pytorch-wheels/{variant}/"))
                    .ok()
                    .and_then(|p| best_version_matching(&p, "torch", variant, &torch_v))
                    .is_some();
            let base = if on_aliyun {
                log("阿里云镜像有这个版本，从国内镜像下载".into());
                format!("https://mirrors.aliyun.com/pytorch-wheels/{variant}")
            } else {
                if china {
                    log("国内镜像还没有这个版本，从 PyTorch 官方下载".into());
                }
                official.clone()
            };
            // 先把两个大 wheel 自己下下来（带进度条），再本地安装；任何环节出错则退回 uv 直接装
            match self.download_wheels(&base, &torch_v, &vision_v, variant, log, progress) {
                Ok((torch_whl, vision_whl)) => {
                    log("下载完成，正在安装…".into());
                    let mut c = command(&uv);
                    c.args(["pip", "install", "--python"]).arg(&py).arg(&torch_whl).arg(&vision_whl);
                    c.args(["--index-url", pypi]);
                    run_logged(c, log)?;
                    let _ = fs::remove_dir_all(self.root().join("wheels"));
                }
                Err(e) => {
                    log(format!("自带下载不可用（{e}），改用 uv 下载"));
                    let mut c = command(&uv);
                    c.args(["pip", "install", "--python"]).arg(&py);
                    c.arg(format!("torch=={torch_v}+{variant}")).arg(format!("torchvision=={vision_v}+{variant}"));
                    if on_aliyun {
                        c.args(["--index-url", pypi, "--find-links"]).arg(format!("https://mirrors.aliyun.com/pytorch-wheels/{variant}/"));
                    } else {
                        c.args(["--index-url"]).arg(&official).args(["--extra-index-url", pypi]);
                    }
                    run_logged(c, log)?;
                }
            }
        }
        if !cuda_matches(&py, variant, &self.worker_dir) {
            return Err(msg("装好的 PyTorch 不是显卡版本，请改用「官方源」后重新安装"));
        }

        log("【4/5】安装其他依赖".into());
        let mut c = command(&uv);
        c.args(["pip", "install", "--python"]).arg(&py).args(["numpy", "pillow", "onnx", "onnxruntime", "opencv-python-headless"]);
        if china {
            c.args(["--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple"]);
        }
        run_logged(c, log)?;

        log("【5/5】检查环境".into());
        *self.check_cache.lock() = None;
        let v = self.check(true)?;
        log(format!("完成：{}", v));
        if variant.starts_with("cu") && v["cuda"] != true {
            return Err(msg("PyTorch 已安装，但显卡不可用。请更新 NVIDIA 驱动后重新安装，或改用 CPU 版本"));
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
        log(format!("下载 torch（{torch_v}+{variant}）"));
        download_file(&wheel_url(base, "torch", torch_v, variant), &torch_whl, &|d, t, s| progress("torch", d, t, s))?;
        log(format!("下载 torchvision（{vision_v}+{variant}）"));
        download_file(&wheel_url(base, "torchvision", vision_v, variant), &vision_whl, &|d, t, s| progress("torchvision", d, t, s))?;
        Ok((torch_whl, vision_whl))
    }
}

/// torch / torchvision 的 wheel 直链（URL 里 + 要转义成 %2B）
fn wheel_url(base: &str, pkg: &str, version: &str, variant: &str) -> String {
    format!("{base}/{pkg}-{version}%2B{variant}{}", wheel_suffix())
}

/// 流式下载到文件，边下边回调（已下载字节、总字节、瞬时速度 字节/秒）
fn download_file(url: &str, dest: &Path, progress: &(dyn Fn(u64, u64, u64) + Sync)) -> AppResult<()> {
    let resp = ureq::get(url).timeout(std::time::Duration::from_secs(1800)).call().map_err(|e| msg(e.to_string()))?;
    let total: u64 = resp.header("Content-Length").and_then(|s| s.parse().ok()).unwrap_or(0);
    if let Some(p) = dest.parent() {
        fs::create_dir_all(p)?;
    }
    let tmp = dest.with_extension("part");
    let mut file = fs::File::create(&tmp)?;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 256 * 1024];
    let mut downloaded: u64 = 0;
    let start = std::time::Instant::now();
    let mut last = std::time::Instant::now();
    let mut last_bytes: u64 = 0;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        downloaded += n as u64;
        if last.elapsed() >= std::time::Duration::from_millis(300) {
            let dt = last.elapsed().as_secs_f64().max(0.001);
            let speed = ((downloaded - last_bytes) as f64 / dt) as u64;
            progress(downloaded, total, speed);
            last = std::time::Instant::now();
            last_bytes = downloaded;
        }
    }
    file.flush()?;
    drop(file);
    let avg = (downloaded as f64 / start.elapsed().as_secs_f64().max(0.001)) as u64;
    progress(downloaded, total.max(downloaded), avg);
    fs::rename(&tmp, dest)?;
    Ok(())
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
        return Err(msg("macOS 上 PyTorch 从 PyPI 安装，不需要测速；在国内选「国内镜像」即可"));
    }
    let official = format!("https://download.pytorch.org/whl/{variant}");
    let page = http_get_text(&format!("{official}/torch/")).map_err(|e| msg(format!("无法访问 PyTorch 下载站：{e}")))?;
    let torch_v = best_version(&page, "torch", variant).ok_or_else(|| msg(format!("没有找到适合的 {variant} 版本")))?;
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
    if cfg!(windows) { "-cp311-cp311-win_amd64.whl" } else { "-cp311-cp311-manylinux_2_28_x86_64.whl" }
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
pub fn run_logged(mut c: Command, log: &(dyn Fn(String) + Sync)) -> AppResult<()> {
    c.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = c.spawn().map_err(|e| msg(format!("无法启动 {:?}：{e}", c.get_program())))?;
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
    std::thread::scope(|s| {
        s.spawn(|| pump(Box::new(out)));
        s.spawn(|| pump(Box::new(err)));
    });
    let status = child.wait()?;
    if !status.success() {
        return Err(msg(format!("命令失败（{status}）：{}", tail.lock().join("\n"))));
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

/// 按驱动支持的 CUDA 版本选 PyTorch 安装包。macOS 用 PyPI 上自带 MPS（Apple 芯片加速）的版本
pub fn torch_variant(gpu: Option<&Gpu>) -> &'static str {
    if cfg!(target_os = "macos") {
        return "mps";
    }
    let Some(g) = gpu else { return "cpu" };
    let mut it = g.cuda.split('.').map(|x| x.parse::<u32>().unwrap_or(0));
    let v = (it.next().unwrap_or(0), it.next().unwrap_or(0));
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

/// 下载 uv：先 GitHub，失败再从 PyPI 下载 wheel（wheel 也是 zip，里面带 uv.exe）
fn download_uv(bin: &Path, log: &(dyn Fn(String) + Sync)) -> AppResult<PathBuf> {
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
    let mut errors = Vec::new();
    let github = format!("https://github.com/astral-sh/uv/releases/latest/download/uv-{target}.zip");
    log(format!("下载 uv：{github}"));
    match http_get(&github).and_then(|b| extract_exe(&b, exe, &bin.join(exe))) {
        Ok(p) => return Ok(p),
        Err(e) => {
            log(format!("GitHub 下载失败：{e}"));
            errors.push(e.to_string());
        }
    }
    log("改从 PyPI 下载 uv…".into());
    let r = (|| -> AppResult<PathBuf> {
        let meta: serde_json::Value = serde_json::from_slice(&http_get("https://pypi.org/pypi/uv/json")?)?;
        let tag = if cfg!(windows) {
            "win_amd64"
        } else if cfg!(target_os = "macos") {
            if arm { "macosx_11_0_arm64" } else { "macosx_10_12_x86_64" }
        } else if arm {
            "manylinux_2_17_aarch64"
        } else {
            "manylinux_2_17_x86_64"
        };
        let url = meta["urls"]
            .as_array()
            .and_then(|a| a.iter().find(|u| u["filename"].as_str().is_some_and(|f| f.ends_with(".whl") && f.contains(tag))))
            .and_then(|u| u["url"].as_str())
            .ok_or_else(|| msg("PyPI 上没有找到对应的 uv 安装包"))?
            .to_string();
        log(format!("下载 {url}"));
        extract_exe(&http_get(&url)?, exe, &bin.join(exe))
    })();
    r.map_err(|e| msg(format!("下载 uv 失败：{}；{e}。可以手动安装 uv 后在设置里指定路径", errors.join("；"))))
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
        return Err(msg("没有找到 ffmpeg，请先用系统的包管理器安装（如 apt install ffmpeg），或在设置里指定路径"));
    };
    fs::create_dir_all(bin)?;
    let out = bin.join(exe);
    let zip = bin.join("ffmpeg-download.zip");
    let mut errors = Vec::new();
    for url in sources {
        log(format!("下载 ffmpeg：{url}"));
        let r = download_file(url, &zip, &|d, t, s| progress("ffmpeg", d, t, s))
            .and_then(|_| {
                let bytes = fs::read(&zip)?;
                extract_exe(&bytes, exe, &out)
            });
        let _ = fs::remove_file(&zip);
        match r {
            Ok(p) => return Ok(p),
            Err(e) => {
                log(format!("下载失败：{e}"));
                errors.push(e.to_string());
            }
        }
    }
    Err(msg(format!("下载 ffmpeg 失败：{}。可以手动安装 ffmpeg 后在设置里指定路径", errors.join("；"))))
}

fn http_get(url: &str) -> AppResult<Vec<u8>> {
    let resp = ureq::get(url).timeout(std::time::Duration::from_secs(300)).call().map_err(|e| msg(e.to_string()))?;
    let mut buf = Vec::new();
    resp.into_reader().take(200 * 1024 * 1024).read_to_end(&mut buf)?;
    Ok(buf)
}

fn extract_exe(zip_bytes: &[u8], exe: &str, out: &Path) -> AppResult<PathBuf> {
    let mut z = zip::ZipArchive::new(std::io::Cursor::new(zip_bytes)).map_err(|e| msg(format!("解压失败：{e}")))?;
    for i in 0..z.len() {
        let mut f = z.by_index(i).map_err(|e| msg(format!("解压失败：{e}")))?;
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
    Err(msg(format!("压缩包里没有 {exe}")))
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
        rt.install(&variant, &|l| eprintln!("{l}"), &|f, d, t, s| eprintln!("{f}: {d}/{t} {s}B/s")).unwrap();
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
