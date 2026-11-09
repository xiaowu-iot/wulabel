//! 自定义协议 xl://，前端用它加载当前项目的图片和缩略图：
//!   /thumb/{id}  缩略图
//!   /image/{id}  原图
//! 只按图片 id 取文件，前端无法借此读取任意路径。
//! Windows 上实际地址是 http://xl.localhost/...

use std::fs;
use std::path::Path;

use tauri::http::{header, Request, Response, StatusCode};
use tauri::{Manager, Runtime, UriSchemeContext, UriSchemeResponder};

use crate::error::{msg, AppResult};
use crate::images;
use crate::state::AppState;

pub fn handle<R: Runtime>(ctx: UriSchemeContext<'_, R>, req: Request<Vec<u8>>, responder: UriSchemeResponder) {
    let app = ctx.app_handle().clone();
    let path = req.uri().path().to_string();
    // 读文件放到线程里，不阻塞界面
    std::thread::spawn(move || {
        let started = std::time::Instant::now();
        let result = serve(app.state::<AppState>().inner(), &path);
        let ms = started.elapsed().as_millis();
        if cfg!(debug_assertions) && ms > 200 {
            eprintln!("[xl] 慢请求 {path} 用时 {ms}ms");
        }
        let resp = match result {
            Ok(r) => r,
            Err(e) => Response::builder()
                .status(StatusCode::NOT_FOUND)
                .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
                .body(e.to_string().into_bytes())
                .unwrap(),
        };
        responder.respond(resp);
    });
}

fn serve(state: &AppState, path: &str) -> AppResult<Response<Vec<u8>>> {
    let project = state.current()?;
    let mut parts = path.trim_matches('/').splitn(2, '/');
    let kind = parts.next().unwrap_or("");
    let id: i64 = parts.next().and_then(|s| s.parse().ok()).ok_or_else(|| msg("地址格式不对"))?;
    let (stored, hash) = images::locate(&project, id)?;
    let file = match kind {
        "thumb" => images::ensure_thumb(&project, &stored, &hash)?,
        "image" => project.resolve(&stored),
        _ => return Err(msg("地址格式不对")),
    };
    let (bytes, content_type) = if is_tiff(&file) {
        // 浏览器内核不支持 TIFF，转成 JPEG
        (images::to_jpeg_bytes(&file)?, "image/jpeg")
    } else {
        (fs::read(&file)?, mime(&file))
    };
    Ok(Response::builder()
        .header(header::CONTENT_TYPE, content_type)
        // 同一个 id 的内容不会变（按哈希去重），可以放心缓存
        .header(header::CACHE_CONTROL, "max-age=86400")
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .body(bytes)
        .unwrap())
}

fn is_tiff(p: &Path) -> bool {
    matches!(mime(p), "image/tiff")
}

fn mime(p: &Path) -> &'static str {
    match p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref() {
        Some("png") => "image/png",
        Some("bmp") => "image/bmp",
        Some("webp") => "image/webp",
        Some("tif") | Some("tiff") => "image/tiff",
        _ => "image/jpeg",
    }
}
