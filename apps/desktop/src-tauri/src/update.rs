//! 检查更新：问一下服务端有没有新版本。
//!
//! 请求里只带软件版本、系统类型和一个随机的匿名编号（和反馈用的是同一个），
//! 服务端据此统计大概有多少台电脑在用、各在用哪个版本。不带图片、项目和任何个人信息。
//! 用户可以在设置里关掉自动检查。

use std::time::Duration;

use serde::{Deserialize, Serialize};

// 和反馈服务是同一个后端（server/feedback/）。自建部署改成你自己的地址即可。
const UPDATE_ENDPOINT: &str = "https://feedback.xiaowuiot.com/update";

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
}

#[derive(Deserialize)]
struct UpdateResp {
    latest: Option<String>,
    url: Option<String>,
    notes: Option<String>,
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
        Err(ureq::Error::Status(code, _)) => return Err(format!("检查更新失败（{code}）")),
        Err(e) => return Err(format!("网络连接失败：{e}")),
    };
    let r: UpdateResp = serde_json::from_str(&text).map_err(|_| "检查更新失败：服务端返回的内容不对".to_string())?;
    let latest = r.latest.filter(|s| !s.trim().is_empty()).unwrap_or_else(|| current.clone());
    Ok(UpdateInfo {
        newer: version_gt(&latest, &current),
        current,
        latest,
        url: r.url.unwrap_or_default(),
        notes: r.notes.unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    use super::version_gt;

    #[test]
    fn compare_versions() {
        assert!(version_gt("0.2.1", "0.2.0"));
        assert!(version_gt("0.10.0", "0.9.9"));
        assert!(version_gt("v1.0.0", "0.2.0"));
        assert!(!version_gt("0.2.0", "0.2.0"));
        assert!(!version_gt("0.1.9", "0.2.0"));
    }
}
