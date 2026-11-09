use std::collections::HashMap;
use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, ImageDecoder, ImageReader};
use rayon::prelude::*;
use rusqlite::{params, OptionalExtension};
use serde::Serialize;

use crate::db;
use crate::error::{msg, AppResult};
use crate::project::{Project, IMAGES_DIR};

pub const IMAGE_EXTS: [&str; 7] = ["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"];
const THUMB_SIZE: u32 = 256;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportResult {
    pub added: usize,
    pub duplicated: usize,
    pub failed: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageItem {
    pub id: i64,
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub status: String,
    pub split: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImagePage {
    pub total: i64,
    pub items: Vec<ImageItem>,
}

fn is_image(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// 展开用户选的文件和文件夹（文件夹递归查找图片）
pub fn collect(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for p in paths {
        if p.is_dir() {
            for e in walkdir::WalkDir::new(p).follow_links(true).into_iter().flatten() {
                if e.file_type().is_file() && is_image(e.path()) {
                    out.push(e.into_path());
                }
            }
        } else if p.is_file() && is_image(p) {
            out.push(p.clone());
        }
    }
    out.sort();
    out.dedup();
    out
}

/// 解码图片并按 EXIF 方向摆正，宽高以摆正后的为准
fn decode(path: &Path) -> AppResult<DynamicImage> {
    let mut decoder = ImageReader::open(path)?.with_guessed_format()?.into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut img = DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    Ok(img)
}

pub fn write_thumb(img: &DynamicImage, out: &Path) -> AppResult<()> {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let thumb = img.thumbnail(THUMB_SIZE, THUMB_SIZE).to_rgb8();
    // 内容相同的图会并行写同一个缩略图，临时文件名必须各不相同
    let tmp = out.with_extension(format!("{}.tmp", SEQ.fetch_add(1, Ordering::Relaxed)));
    {
        let mut w = BufWriter::new(fs::File::create(&tmp)?);
        JpegEncoder::new_with_quality(&mut w, 82).encode_image(&thumb)?;
    }
    if let Err(e) = fs::rename(&tmp, out) {
        let _ = fs::remove_file(&tmp);
        // 别的线程已经写好了同一张缩略图，算成功
        if !out.exists() {
            return Err(e.into());
        }
    }
    Ok(())
}

pub fn to_jpeg_bytes(path: &Path) -> AppResult<Vec<u8>> {
    let img = decode(path)?.to_rgb8();
    let mut buf = Vec::new();
    JpegEncoder::new_with_quality(&mut buf, 92).encode_image(&img)?;
    Ok(buf)
}

/// 缩略图丢了（比如用户删了 .cache）就重新生成
pub fn ensure_thumb(project: &Project, stored_path: &str, hash: &str) -> AppResult<PathBuf> {
    let out = project.thumb_path(hash);
    if !out.exists() {
        let img = decode(&project.resolve(stored_path))?;
        write_thumb(&img, &out)?;
    }
    Ok(out)
}

struct Prepared {
    src: PathBuf,
    hash: String,
    width: u32,
    height: u32,
}

fn prepare(project: &Project, src: &Path) -> AppResult<Prepared> {
    let bytes = fs::read(src)?;
    let hash = blake3::hash(&bytes).to_hex()[..32].to_string();
    let img = decode(src)?;
    let thumb = project.thumb_path(&hash);
    if !thumb.exists() {
        write_thumb(&img, &thumb)?;
    }
    Ok(Prepared { src: src.to_path_buf(), hash, width: img.width(), height: img.height() })
}

/// 复制到项目 images/ 下，重名时加序号
fn copy_into(project: &Project, src: &Path) -> AppResult<String> {
    let dir = project.root.join(IMAGES_DIR);
    let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let ext = src.extension().and_then(|s| s.to_str()).unwrap_or("jpg").to_ascii_lowercase();
    let mut name = format!("{stem}.{ext}");
    let mut i = 1;
    while dir.join(&name).exists() {
        name = format!("{stem}_{i}.{ext}");
        i += 1;
    }
    fs::copy(src, dir.join(&name))?;
    Ok(format!("{IMAGES_DIR}/{name}"))
}

/// 存储路径：项目内用相对路径（整个文件夹拷走也能打开），项目外用绝对路径
fn stored_path(project: &Project, src: &Path) -> String {
    let abs = src.canonicalize().unwrap_or_else(|_| src.to_path_buf());
    let root = project.root.canonicalize().unwrap_or_else(|_| project.root.clone());
    match abs.strip_prefix(&root) {
        Ok(rel) => rel.to_string_lossy().replace('\\', "/"),
        Err(_) => src.to_string_lossy().into_owned(),
    }
}

/// 导入图片。copy=true 复制进项目，false 只记录原路径。按内容去重。
pub fn import(
    project: &Project,
    files: &[PathBuf],
    copy: bool,
    on_progress: &(dyn Fn(usize, usize) + Sync),
) -> AppResult<ImportResult> {
    import_mapped(project, files, copy, on_progress).map(|(r, _)| r)
}

/// 同 import，另外返回「源文件 → 图片 id」（重复的图对应已有的 id），导入带标注的数据集时用
pub fn import_mapped(
    project: &Project,
    files: &[PathBuf],
    copy: bool,
    on_progress: &(dyn Fn(usize, usize) + Sync),
) -> AppResult<(ImportResult, HashMap<PathBuf, i64>)> {
    let mut map = HashMap::new();
    let total = files.len();
    let done = AtomicUsize::new(0);
    on_progress(0, total);

    // 1. 并行：算哈希、读宽高、生成缩略图
    let prepared: Vec<Result<Prepared, String>> = files
        .par_iter()
        .map(|f| {
            let r = prepare(project, f).map_err(|e| format!("{}：{e}", f.display()));
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            on_progress(n, total);
            r
        })
        .collect();

    // 2. 串行：去重、复制、写库
    let mut result = ImportResult { added: 0, duplicated: 0, failed: Vec::new() };
    let mut seen: HashMap<String, i64> = HashMap::new();
    let mut conn = project.conn.lock();
    let tx = conn.transaction()?;
    {
        let mut exists = tx.prepare("SELECT id FROM images WHERE hash = ?1 LIMIT 1")?;
        let mut insert = tx.prepare(
            "INSERT INTO images (path, width, height, hash, created_at, batch) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )?;
        let now = db::now();
        let batch: i64 = tx.query_row("SELECT COALESCE(MAX(batch), 0) + 1 FROM images", [], |r| r.get(0))?;
        for item in prepared {
            let p = match item {
                Ok(p) => p,
                Err(e) => {
                    result.failed.push(e);
                    continue;
                }
            };
            let existing = match seen.get(&p.hash) {
                Some(id) => Some(*id),
                None => exists.query_row([&p.hash], |r| r.get::<_, i64>(0)).optional()?,
            };
            if let Some(id) = existing {
                result.duplicated += 1;
                map.insert(p.src, id);
                continue;
            }
            let path = if copy { copy_into(project, &p.src)? } else { stored_path(project, &p.src) };
            insert.execute(params![path, p.width, p.height, p.hash, now, batch])?;
            let id = tx.last_insert_rowid();
            seen.insert(p.hash, id);
            map.insert(p.src, id);
            result.added += 1;
        }
    }
    tx.commit()?;
    Ok((result, map))
}

pub fn list(project: &Project, status: Option<&str>, offset: u32, limit: u32) -> AppResult<ImagePage> {
    let conn = project.conn.lock();
    let status = status.filter(|s| !s.is_empty() && *s != "all");
    let total: i64 = conn.query_row(
        "SELECT COUNT(*) FROM images WHERE (?1 IS NULL OR status = ?1)",
        [status],
        |r| r.get(0),
    )?;
    let mut stmt = conn.prepare(
        "SELECT id, path, width, height, status, split FROM images
         WHERE (?1 IS NULL OR status = ?1) ORDER BY id LIMIT ?2 OFFSET ?3",
    )?;
    let items = stmt
        .query_map(params![status, limit.min(1000), offset], |r| {
            let path: String = r.get(1)?;
            let name = Path::new(&path)
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or(path.clone());
            Ok(ImageItem {
                id: r.get(0)?,
                name,
                width: r.get(2)?,
                height: r.get(3)?,
                status: r.get(4)?,
                split: r.get(5)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ImagePage { total, items })
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LastImport {
    pub count: i64,
    pub created_at: i64,
}

/// 最近一次导入的批次（批次 0 是旧数据，不算）
pub fn last_import(project: &Project) -> AppResult<Option<LastImport>> {
    let conn = project.conn.lock();
    Ok(conn
        .query_row(
            "SELECT COUNT(*), MAX(created_at) FROM images
             WHERE batch = (SELECT MAX(batch) FROM images) AND batch > 0",
            [],
            |r| Ok(LastImport { count: r.get(0)?, created_at: r.get::<_, Option<i64>>(1)?.unwrap_or(0) }),
        )
        .optional()?
        .filter(|l| l.count > 0))
}

pub enum DeleteScope {
    Ids(Vec<i64>),
    LastImport,
    All,
}

/// 删除图片及其标注。复制进项目的图片文件一起删；引用的外部原图不动。返回删除张数。
pub fn delete(project: &Project, scope: DeleteScope) -> AppResult<usize> {
    let mut conn = project.conn.lock();
    let tx = conn.transaction()?;
    let rows: Vec<(i64, String, String)> = {
        let (sql, ids): (&str, Vec<i64>) = match &scope {
            DeleteScope::All => ("SELECT id, path, hash FROM images", vec![]),
            DeleteScope::LastImport => (
                "SELECT id, path, hash FROM images WHERE batch = (SELECT MAX(batch) FROM images) AND batch > 0",
                vec![],
            ),
            DeleteScope::Ids(ids) => ("SELECT id, path, hash FROM images WHERE id = ?1", ids.clone()),
        };
        let mut stmt = tx.prepare(sql)?;
        let map = |r: &rusqlite::Row| Ok((r.get(0)?, r.get(1)?, r.get(2)?));
        if ids.is_empty() && !matches!(scope, DeleteScope::Ids(_)) {
            stmt.query_map([], map)?.collect::<Result<_, _>>()?
        } else {
            let mut out = Vec::new();
            for id in ids {
                if let Some(row) = stmt.query_row([id], map).optional()? {
                    out.push(row);
                }
            }
            out
        }
    };
    {
        let mut del = tx.prepare("DELETE FROM images WHERE id = ?1")?;
        for (id, _, _) in &rows {
            del.execute([id])?; // 标注随外键级联删除
        }
    }
    tx.commit()?;

    // 提交成功后再删文件：只删项目 images/ 下的副本
    let mut still_used = conn.prepare("SELECT 1 FROM images WHERE hash = ?1 LIMIT 1")?;
    for (_, path, hash) in &rows {
        if path.starts_with(&format!("{IMAGES_DIR}/")) {
            let _ = fs::remove_file(project.root.join(path));
        }
        if still_used.query_row([hash], |_| Ok(())).optional()?.is_none() {
            let _ = fs::remove_file(project.thumb_path(hash));
        }
    }
    Ok(rows.len())
}

/// 按 id 查图片的存储路径和哈希
pub fn locate(project: &Project, id: i64) -> AppResult<(String, String)> {
    let conn = project.conn.lock();
    conn.query_row("SELECT path, hash FROM images WHERE id = ?1", [id], |r| Ok((r.get(0)?, r.get(1)?)))
        .optional()?
        .ok_or_else(|| msg(format!("图片不存在：{id}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("xlabel-test-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn make_image(path: &Path, w: u32, h: u32, color: u8) {
        RgbImage::from_pixel(w, h, Rgb([color, 100, 50])).save(path).unwrap();
    }

    #[test]
    fn create_import_list() {
        let base = temp_dir("import");
        let src = base.join("src");
        fs::create_dir_all(src.join("sub")).unwrap();
        make_image(&src.join("a.png"), 640, 480, 10);
        make_image(&src.join("sub/b.jpg"), 800, 600, 200);
        fs::copy(src.join("a.png"), src.join("a_copy.png")).unwrap(); // 内容重复
        fs::write(src.join("broken.jpg"), b"not an image").unwrap();
        fs::write(src.join("readme.txt"), b"skip").unwrap();

        let project = Project::create(&base, "测试项目", "detect").unwrap();
        let files = collect(&[src.clone()]);
        assert_eq!(files.len(), 4, "txt 不算图片");

        let r = import(&project, &files, true, &|_, _| {}).unwrap();
        assert_eq!(r.added, 2);
        assert_eq!(r.duplicated, 1);
        assert_eq!(r.failed.len(), 1);

        let page = list(&project, None, 0, 100).unwrap();
        assert_eq!(page.total, 2);
        let b = page.items.iter().find(|i| i.name == "b.jpg").unwrap();
        assert_eq!((b.width, b.height), (800, 600));
        assert!(project.root.join("images/b.jpg").exists(), "复制模式要拷进项目");

        // 再导入一次全部算重复
        let r2 = import(&project, &files, true, &|_, _| {}).unwrap();
        assert_eq!((r2.added, r2.duplicated), (0, 3));

        // 缩略图删掉后能重建
        let (path, hash) = locate(&project, b.id).unwrap();
        fs::remove_file(project.thumb_path(&hash)).unwrap();
        let t = ensure_thumb(&project, &path, &hash).unwrap();
        let (tw, th) = image::image_dimensions(t).unwrap();
        assert!(tw <= THUMB_SIZE && th <= THUMB_SIZE);

        // 重新打开项目，数据还在
        drop(project);
        let reopened = Project::open(&base.join("测试项目")).unwrap();
        assert_eq!(reopened.stats().unwrap().total, 2);
        assert_eq!(reopened.stats().unwrap().todo, 2);
        drop(reopened);
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn reference_mode_keeps_original_path() {
        let base = temp_dir("ref");
        make_image(&base.join("x.png"), 32, 16, 1);
        let project = Project::create(&base, "p", "classify").unwrap();
        let r = import(&project, &[base.join("x.png")], false, &|_, _| {}).unwrap();
        assert_eq!(r.added, 1);
        let (path, _) = locate(&project, 1).unwrap();
        assert!(Path::new(&path).is_absolute());
        assert!(fs::read_dir(project.root.join(IMAGES_DIR)).unwrap().next().is_none());
        drop(project);
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn delete_scopes() {
        let base = temp_dir("del");
        for (n, c) in [("a", 1u8), ("b", 2), ("c", 3)] {
            make_image(&base.join(format!("{n}.png")), 40, 30, c);
        }
        make_image(&base.join("outside.png"), 40, 30, 9);
        let project = Project::create(&base, "p", "detect").unwrap();
        import(&project, &[base.join("a.png"), base.join("b.png")], true, &|_, _| {}).unwrap();
        import(&project, &[base.join("c.png")], true, &|_, _| {}).unwrap();

        let last = last_import(&project).unwrap().unwrap();
        assert_eq!(last.count, 1);
        let (_, hash_c) = locate(&project, 3).unwrap();
        assert_eq!(delete(&project, DeleteScope::LastImport).unwrap(), 1);
        assert!(!project.root.join("images/c.png").exists(), "项目里的副本要删掉");
        assert!(!project.thumb_path(&hash_c).exists(), "缩略图要删掉");
        assert!(base.join("c.png").exists(), "原图不能动");
        assert_eq!(last_import(&project).unwrap().unwrap().count, 2, "再上一批");

        // 标注随图片一起删掉
        project.conn.lock().execute(
            "INSERT INTO annotations (image_id, type, data_json, created_at, updated_at) VALUES (1, 'bbox', '[0,0,1,1]', 0, 0)",
            [],
        ).unwrap();
        assert_eq!(delete(&project, DeleteScope::Ids(vec![1, 999])).unwrap(), 1);
        let n: i64 = project.conn.lock().query_row("SELECT COUNT(*) FROM annotations", [], |r| r.get(0)).unwrap();
        assert_eq!(n, 0);

        // 删掉后可以重新导入
        assert_eq!(import(&project, &[base.join("a.png")], true, &|_, _| {}).unwrap().added, 1);
        // 引用模式的外部原图，清空时不能删
        import(&project, &[base.join("outside.png")], false, &|_, _| {}).unwrap();
        assert_eq!(delete(&project, DeleteScope::All).unwrap(), 3);
        assert!(base.join("outside.png").exists());
        assert_eq!(project.stats().unwrap().total, 0);
        assert!(last_import(&project).unwrap().is_none());
        drop(project);
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn create_rejects_bad_input() {
        let base = temp_dir("bad");
        assert!(Project::create(&base, "", "detect").is_err());
        assert!(Project::create(&base, "a/b", "detect").is_err());
        assert!(Project::create(&base, "ok", "ocr").is_err());
        assert!(Project::open(&base).is_err(), "普通文件夹不是项目");
        let _ = fs::remove_dir_all(&base);
    }
}
