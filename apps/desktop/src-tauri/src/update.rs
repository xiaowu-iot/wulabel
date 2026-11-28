//! 检查更新：问一下服务端有没有新版本。
//!
//! 请求里只带软件版本、系统类型和一个随机的匿名编号（和反馈用的是同一个），
//! 服务端据此统计大概有多少台电脑在用、各在用哪个版本。不带图片、项目和任何个人信息。
//! 用户可以在设置里关掉自动检查。

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::t;

// 和反馈服务是同一个后端（server/feedback/）。自建部署改成你自己的地址即可。
const UPDATE_ENDPOINT: &str = "https://feedback.xiaowuiot.com/update";
/// 服务器连不上时，从 GitHub 最新发行版里的清单看有没有新版本（打包时自动生成）
const GITHUB_LATEST: &str = "https://github.com/xiaowu-iot/wulabel/releases/latest/download/latest.json";
const GITHUB_RELEASES: &str = "https://github.com/xiaowu-iot/wulabel/releases/latest";
/// 自动更新的安装包一律从 GitHub 发行版下载（以前的版本从官网下）
const GITHUB_DL: &str = "https://github.com/xiaowu-iot/wulabel/releases/download/";
const SITE_DL: &str = "https://wulabel.xiaowuiot.com/download/";

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateInfo {
    pub current: String,
    pub latest: String,
    pub newer: bool,
    /// 下载页地址
    pub url: String,
    /// 新版本说明，可空
    pub notes: String,
    /// 本系统的安装包直链和 SHA-256，有的话界面上可以「立即更新」；没有就只能去下载页
    pub download_url: String,
    pub sha256: String,
}

#[derive(Deserialize, Default)]
struct DownloadResp {
    url: Option<String>,
    sha256: Option<String>,
}

#[derive(Deserialize)]
struct UpdateResp {
    latest: Option<String>,
    url: Option<String>,
    notes: Option<String>,
    download: Option<DownloadResp>,
}

fn parse_version(v: &str) -> [u64; 3] {
    let mut out = [0u64; 3];
    let core = v.trim().trim_start_matches('v').split('-').next().unwrap_or("");
    for (i, p) in core.split('.').take(3).enumerate() {
        out[i] = p.trim().parse().unwrap_or(0);
    }
    out
}

fn version_gt(a: &str, b: &str) -> bool {
    parse_version(a) > parse_version(b)
}

#[tauri::command]
pub async fn check_update(app: tauri::AppHandle) -> Result<UpdateInfo, String> {
    let current = env!("CARGO_PKG_VERSION").to_string();
    let device = crate::feedback::device_id(&app);
    let cur = current.clone();

    // ureq 是阻塞调用，放到阻塞线程池
    let result = tauri::async_runtime::spawn_blocking(move || {
        ureq::get(UPDATE_ENDPOINT)
            .timeout(Duration::from_secs(8))
            .query("v", &cur)
            .query("os", std::env::consts::OS)
            .query("arch", std::env::consts::ARCH)
            .query("id", &device)
            .call()
    })
    .await
    .map_err(|e| e.to_string())?;

    let text = match result {
        Ok(resp) => resp.into_string().map_err(|e| e.to_string())?,
        // 服务器连不上：看 GitHub 上的最新版本
        Err(e) => {
            let msg = match &e {
                ureq::Error::Status(code, _) => t!("检查更新失败（{code}）", code = code),
                _ => t!("网络连接失败：{e}", e = e),
            };
            return tauri::async_runtime::spawn_blocking(move || github_latest(current).ok_or(msg)).await.map_err(|e| e.to_string())?;
        }
    };
    let r: UpdateResp = serde_json::from_str(&text).map_err(|_| t!("检查更新失败：服务端返回的内容不对"))?;
    let latest = r.latest.filter(|s| !s.trim().is_empty()).unwrap_or_else(|| current.clone());
    let dl = r.download.unwrap_or_default();
    let download_url = dl.url.unwrap_or_default();
    // 安装包一律从 GitHub 发行版下载（服务器给的可能还是官网地址，文件一样，SHA-256 照样核对）
    let download_url = if download_url.starts_with(SITE_DL) { github_url(&download_url, &latest) } else { download_url };
    Ok(UpdateInfo {
        newer: version_gt(&latest, &current),
        current,
        latest,
        url: localize_site(r.url.unwrap_or_default()),
        notes: r.notes.unwrap_or_default(),
        download_url: if trusted_url(&download_url) { download_url } else { String::new() },
        sha256: dl.sha256.unwrap_or_default().to_lowercase(),
    })
}

#[derive(Deserialize)]
struct LatestJson {
    version: String,
    #[serde(default)]
    notes: String,
}

/// GitHub 最新发行版的版本号和说明（自动更新清单 latest.json）；连不上返回 None
fn github_latest(current: String) -> Option<UpdateInfo> {
    let text = ureq::get(GITHUB_LATEST).timeout(Duration::from_secs(10)).call().ok()?.into_string().ok()?;
    let j: LatestJson = serde_json::from_str(&text).ok()?;
    Some(UpdateInfo {
        newer: version_gt(&j.version, &current),
        current,
        latest: j.version.trim_start_matches('v').to_string(),
        url: GITHUB_RELEASES.to_string(),
        notes: j.notes,
        download_url: String::new(),
        sha256: String::new(),
    })
}

/// 安装包在 GitHub 发行版上的地址。官网上 Mac 更新包的名字带版本号（Wulabel_0.6.8_universal.app.tar.gz），发行版里不带
fn github_url(url: &str, version: &str) -> String {
    let name = url.rsplit('/').next().unwrap_or("");
    let mac_site = format!("Wulabel_{}_universal.app.tar.gz", version.trim_start_matches('v'));
    let name = if name == mac_site { "Wulabel_universal.app.tar.gz" } else { name };
    format!("{GITHUB_DL}v{}/{name}", version.trim_start_matches('v'))
}

/// 官网链接换成界面语言的版本（中文在根目录，其他语言在 /<语言>/ 下）
fn localize_site(url: String) -> String {
    const SITE: &str = "https://wulabel.xiaowuiot.com/";
    let lang = crate::i18n::lang();
    if let Some(rest) = url.strip_prefix(SITE) {
        if lang != "zh" && !rest.starts_with(&format!("{lang}/")) {
            return format!("{SITE}{lang}/{rest}");
        }
    }
    url
}

/// 只从官网和 GitHub 发行版下载安装包
fn trusted_url(url: &str) -> bool {
    url.starts_with(SITE_DL) || url.starts_with(GITHUB_DL)
}

#[derive(Serialize, Clone)]
struct Progress {
    downloaded: u64,
    total: u64,
}

/// 全自动更新（v0.4.1 起）：用 Tauri 官方更新组件，后台下载、校验签名、安装，然后自动重启。
/// Windows 用安装程序的「被动模式」（只显示进度条，不用点下一步）；Mac 直接替换应用；Linux 只支持 AppImage。
/// 返回 false 表示没有新版本；出错时界面退回旧的「下载安装包打开安装程序」方式。
#[tauri::command]
pub async fn auto_update(app: tauri::AppHandle) -> Result<bool, String> {
    use tauri::Emitter;
    use tauri_plugin_updater::UpdaterExt;

    let updater = app.updater().map_err(|e| t!("自动更新不可用：{e}", e = e))?;
    let Some(mut update) = updater.check().await.map_err(|e| t!("检查更新失败：{e}", e = e))? else {
        return Ok(false);
    };
    // 安装包一律从 GitHub 发行版下载（服务器的清单里可能还是官网地址）；签名照样校验
    if update.download_url.as_str().starts_with(SITE_DL) {
        if let Ok(u) = github_url(update.download_url.as_str(), &update.version).parse() {
            update.download_url = u;
        }
    }
    let a = app.clone();
    let mut downloaded = 0u64;
    update
        .download_and_install(
            move |chunk, total| {
                downloaded += chunk as u64;
                let _ = a.emit("update-progress", Progress { downloaded, total: total.unwrap_or(0) });
            },
            || {},
        )
        .await
        .map_err(|e| t!("自动更新失败：{e}", e = e))?;
    // Windows 上安装程序会自己结束软件；Mac / Linux 换好文件后重启
    app.restart();
}

/// 下载新版安装包、核对 SHA-256，然后打开安装程序并退出软件。
/// Windows：运行安装程序，按提示点「下一步」即可覆盖安装；macOS：打开 .dmg，把 Wulabel 拖进「应用程序」替换。
#[tauri::command]
pub async fn install_update(app: tauri::AppHandle, url: String, sha256: String) -> Result<(), String> {
    use std::io::{Read, Write};
    use tauri::Emitter;

    if !trusted_url(&url) {
        return Err(t!("安装包地址不对"));
    }
    let sha256 = sha256.trim().to_lowercase();
    if sha256.len() != 64 {
        return Err(t!("缺少安装包校验值，请到官网下载"));
    }
    let name = url.rsplit('/').next().unwrap_or("wulabel-setup").to_string();
    let dir = std::env::temp_dir().join("wulabel-update");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(&name);
    let app2 = app.clone();
    let path2 = path.clone();

    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        use sha2::{Digest, Sha256};

        // 不限总时长（慢网络也能下完），只在连不上、一分钟没收到数据时报错
        let resp = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(20))
            .timeout_read(Duration::from_secs(60))
            .build()
            .get(&url)
            .call()
            .map_err(|e| t!("下载失败：{e}", e = e))?;
        let total: u64 = resp.header("Content-Length").and_then(|v| v.parse().ok()).unwrap_or(0);
        let mut reader = resp.into_reader();
        let tmp = path2.with_extension("part");
        let mut file = std::fs::File::create(&tmp).map_err(|e| t!("保存安装包失败：{e}", e = e))?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 256 * 1024];
        let mut downloaded = 0u64;
        loop {
            let n = reader.read(&mut buf).map_err(|e| t!("下载中断：{e}", e = e))?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n]).map_err(|e| t!("保存安装包失败：{e}", e = e))?;
            hasher.update(&buf[..n]);
            downloaded += n as u64;
            let _ = app2.emit("update-progress", Progress { downloaded, total });
        }
        drop(file);
        let got: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
        if got != sha256 {
            let _ = std::fs::remove_file(&tmp);
            return Err(t!("安装包校验不通过（可能没下载完整），请重试或到官网下载"));
        }
        std::fs::rename(&tmp, &path2).map_err(|e| t!("保存安装包失败：{e}", e = e))?;
        Ok(())
    })
    .await
    .map_err(|e| e.to_string())??;

    #[cfg(target_os = "windows")]
    let r = std::process::Command::new(&path).spawn();
    #[cfg(target_os = "macos")]
    let r = std::process::Command::new("open").arg(&path).spawn();
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let r = std::process::Command::new("xdg-open").arg(&path).spawn();
    r.map_err(|e| t!("打开安装程序失败：{e}。安装包在 {path}", e = e, path = path.display()))?;

    // 留一点时间让安装程序起来，再退出软件，免得安装时文件被占用
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(Duration::from_millis(800)).await;
        app.exit(0);
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{github_url, trusted_url, version_gt};

    #[test]
    fn github_urls() {
        assert_eq!(
            github_url("https://wulabel.xiaowuiot.com/download/Wulabel_0.6.9_x64-setup.exe", "0.6.9"),
            "https://github.com/xiaowu-iot/wulabel/releases/download/v0.6.9/Wulabel_0.6.9_x64-setup.exe"
        );
        // Mac 更新包：官网带版本号，发行版不带
        assert_eq!(
            github_url("https://wulabel.xiaowuiot.com/download/Wulabel_0.6.9_universal.app.tar.gz", "0.6.9"),
            "https://github.com/xiaowu-iot/wulabel/releases/download/v0.6.9/Wulabel_universal.app.tar.gz"
        );
        assert!(trusted_url(&github_url("https://wulabel.xiaowuiot.com/download/Wulabel_0.6.9_amd64.deb", "0.6.9")));
        assert!(!trusted_url("https://example.com/Wulabel_0.6.9_x64-setup.exe"));
    }

    #[test]
    fn compare_versions() {
        assert!(version_gt("0.2.1", "0.2.0"));
        assert!(version_gt("0.10.0", "0.9.9"));
        assert!(version_gt("v1.0.0", "0.2.0"));
        assert!(!version_gt("0.2.0", "0.2.0"));
        assert!(!version_gt("0.1.9", "0.2.0"));
    }
}
