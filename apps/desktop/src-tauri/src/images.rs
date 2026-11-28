use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, ImageDecoder, ImageReader};
use rayon::prelude::*;
use rusqlite::{params, OptionalExtension};
use serde::Serialize;

use crate::db;
use crate::error::{msg, AppResult};
use crate::project::{Project, IMAGES_DIR};
use crate::t;

pub const IMAGE_EXTS: [&str; 7] = ["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"];
const THUMB_SIZE: u32 = 256;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportResult {
    pub added: usize,
    pub duplicated: usize,
    pub failed: Vec<String>,
    /// 中途停止时没导入的张数
    pub stopped: usize,
    /// 批量识别的截图：用里面的原图入库、识别结果写成预标注的张数
    pub prelabeled: usize,
}

/// 数据页「停止导入」：置位后还没处理的图不再导入，已处理的照常入库
pub static STOP_IMPORT: AtomicBool = AtomicBool::new(false);

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
/// 读图并按 EXIF 摆正（和项目里记录的宽高、标注坐标一致）
pub fn decode(path: &Path) -> AppResult<DynamicImage> {
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

/// 在项目 images/ 下挑一个没占用的文件名，重名时加序号。
/// taken 是本批已经挑走的（还没复制完），按小写比较，Windows 上大小写不同也算重名。
fn unique_name(dir: &Path, src: &Path, taken: &mut HashSet<String>) -> String {
    let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let ext = src.extension().and_then(|s| s.to_str()).unwrap_or("jpg").to_ascii_lowercase();
    let mut name = format!("{stem}.{ext}");
    let mut i = 1;
    while taken.contains(&name.to_lowercase()) || dir.join(&name).exists() {
        name = format!("{stem}_{i}.{ext}");
        i += 1;
    }
    taken.insert(name.to_lowercase());
    name
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
/// 进度的 total 是 2 × 张数：前一半算指纹、生成缩略图，后一半复制并写库。
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
    import_batch(project, files, copy, None, on_progress)
}

/// 同 import_batch，可以用 STOP_IMPORT 中途停止（数据页导入、带批量识别截图时分两次导入共用一个批次）
pub(crate) fn import_batch_stoppable(
    project: &Project,
    files: &[PathBuf],
    copy: bool,
    batch: Option<i64>,
    on_progress: &(dyn Fn(usize, usize) + Sync),
) -> AppResult<(ImportResult, HashMap<PathBuf, i64>)> {
    import_inner(project, files, copy, batch, Some(&STOP_IMPORT), WRITE_CHUNK, on_progress)
}

/// 下一个导入批次号
pub fn next_batch(project: &Project) -> AppResult<i64> {
    Ok(project.conn.lock().query_row("SELECT COALESCE(MAX(batch), 0) + 1 FROM images", [], |r| r.get(0))?)
}

/// 同 import_mapped，可以指定批次号。视频流边采边导入时分多次调用、共用一个批次，
/// 这样「撤销上次导入」能把整次采集一起撤掉。
pub fn import_batch(
    project: &Project,
    files: &[PathBuf],
    copy: bool,
    batch: Option<i64>,
    on_progress: &(dyn Fn(usize, usize) + Sync),
) -> AppResult<(ImportResult, HashMap<PathBuf, i64>)> {
    import_inner(project, files, copy, batch, None, WRITE_CHUNK, on_progress)
}

/// 第二阶段每批多少张：每批一个小事务，批与批之间放开数据库锁，导入时缩略图、原图照常能看
const WRITE_CHUNK: usize = 500;

type PrepItem = Option<Result<Prepared, String>>;

/// 进度分两段报：前一半是算指纹、生成缩略图，后一半是复制并写库（total = 2 × 张数）
fn import_inner(
    project: &Project,
    files: &[PathBuf],
    copy: bool,
    batch: Option<i64>,
    stop: Option<&AtomicBool>,
    chunk: usize,
    on_progress: &(dyn Fn(usize, usize) + Sync),
) -> AppResult<(ImportResult, HashMap<PathBuf, i64>)> {
    let total = files.len();
    let total2 = total * 2;
    let stopped = || stop.is_some_and(|s| s.load(Ordering::Relaxed));
    let done = AtomicUsize::new(0);
    on_progress(0, total2);

    // 1. 并行：算哈希、读宽高、生成缩略图；停止后剩下的跳过（None）
    let prepared: Vec<PrepItem> = files
        .par_iter()
        .map(|f| {
            let r = if stopped() {
                None
            } else {
                Some(prepare(project, f).map_err(|e| format!("{}：{e}", f.display())))
            };
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            on_progress(n, total2);
            r
        })
        .collect();

    // 2. 分批：去重 → 复制（不持锁）→ 小事务写库
    let batch: i64 = match batch {
        Some(b) => b,
        None => next_batch(project)?,
    };
    let mut w = Writer {
        project,
        copy,
        batch,
        now: db::now(),
        seen: HashMap::new(),
        result: ImportResult { added: 0, duplicated: 0, failed: Vec::new(), stopped: 0, prelabeled: 0 },
        map: HashMap::new(),
    };
    let mut base = total;
    let mut rest = prepared.into_iter().peekable();
    while rest.peek().is_some() {
        let items: Vec<PrepItem> = rest.by_ref().take(chunk.max(1)).collect();
        let n = items.len();
        if stopped() {
            // 停止后还没写库的都算没导入（读取失败的照样报出来）
            for it in items {
                match it {
                    Some(Err(e)) => w.result.failed.push(e),
                    _ => w.result.stopped += 1,
                }
            }
        } else {
            let start = base;
            w.chunk(items, &|k| on_progress(start + k, total2))?;
        }
        base += n;
        on_progress(base, total2);
    }
    Ok((w.result, w.map))
}

/// 第二阶段的状态：已入库的内容哈希、结果统计、源文件 → id
struct Writer<'a> {
    project: &'a Project,
    copy: bool,
    batch: i64,
    now: i64,
    /// 内容哈希 → 图片 id（库里已有的和本次写入的）
    seen: HashMap<String, i64>,
    result: ImportResult,
    map: HashMap<PathBuf, i64>,
}

impl Writer<'_> {
    /// 处理一批。复制失败、写库失败的进 failed，不影响其他图；已经复制进来的文件删掉
    fn chunk(&mut self, items: Vec<PrepItem>, on_copied: &(dyn Fn(usize) + Sync)) -> AppResult<()> {
        let project = self.project;
        // 要新入库的（同一内容只留第一张）；和本批前面的图重复的，写完库再对应 id
        let mut fresh: Vec<Prepared> = Vec::new();
        let mut dups: Vec<(PathBuf, String)> = Vec::new();
        let mut in_chunk: HashSet<String> = HashSet::new();
        {
            let conn = project.conn.lock();
            let mut exists = conn.prepare("SELECT id FROM images WHERE hash = ?1 LIMIT 1")?;
            for it in items {
                let p = match it {
                    None => {
                        self.result.stopped += 1;
                        continue;
                    }
                    Some(Err(e)) => {
                        self.result.failed.push(e);
                        continue;
                    }
                    Some(Ok(p)) => p,
                };
                let existing = match self.seen.get(&p.hash) {
                    Some(id) => Some(*id),
                    None => exists.query_row([&p.hash], |r| r.get::<_, i64>(0)).optional()?,
                };
                if let Some(id) = existing {
                    self.seen.insert(p.hash, id);
                    self.result.duplicated += 1;
                    self.map.insert(p.src, id);
                } else if in_chunk.contains(&p.hash) {
                    dups.push((p.src, p.hash));
                } else {
                    in_chunk.insert(p.hash.clone());
                    fresh.push(p);
                }
            }
        }

        // 复制：先串行定好文件名（同名文件并行复制时不会互相覆盖），再并行复制，不持数据库锁
        let dir = project.root.join(IMAGES_DIR);
        let mut taken: HashSet<String> = HashSet::new();
        let copy = self.copy;
        let named: Vec<(Prepared, Option<String>)> = fresh
            .into_iter()
            .map(|p| {
                let name = if copy { Some(unique_name(&dir, &p.src, &mut taken)) } else { None };
                (p, name)
            })
            .collect();
        let copied = AtomicUsize::new(0);
        let outcomes: Vec<Result<(Prepared, String), (Prepared, String)>> = named
            .into_par_iter()
            .map(|(p, name)| {
                let r = match name {
                    Some(name) => {
                        let dest = dir.join(&name);
                        match fs::copy(&p.src, &dest) {
                            Ok(_) => Ok(format!("{IMAGES_DIR}/{name}")),
                            Err(e) => {
                                let _ = fs::remove_file(&dest);
                                Err(e.to_string())
                            }
                        }
                    }
                    None => Ok(stored_path(project, &p.src)),
                };
                on_copied(copied.fetch_add(1, Ordering::Relaxed) + 1);
                match r {
                    Ok(path) => Ok((p, path)),
                    Err(e) => Err((p, e)),
                }
            })
            .collect();

        // 失败原因按内容哈希记下，本批里和它重复的图一起算失败
        let mut errs: HashMap<String, String> = HashMap::new();
        let mut ok: Vec<(Prepared, String)> = Vec::new();
        for o in outcomes {
            match o {
                Ok(x) => ok.push(x),
                Err((p, e)) => self.fail(&p, e, &mut errs),
            }
        }

        // 写库：一个小事务
        match insert_rows(project, &ok, self.batch, self.now) {
            Ok(res) => {
                for ((p, path), r) in ok.into_iter().zip(res) {
                    match r {
                        Ok((id, added)) => {
                            if added {
                                self.result.added += 1;
                            } else {
                                // 复制期间别处已经导入了同样的图
                                self.remove_copy(&path);
                                self.result.duplicated += 1;
                            }
                            self.seen.insert(p.hash, id);
                            self.map.insert(p.src, id);
                        }
                        Err(e) => {
                            self.remove_copy(&path);
                            self.fail(&p, e, &mut errs);
                        }
                    }
                }
            }
            Err(e) => {
                // 整批没写进去：复制进来的文件删掉，都算失败
                let e = e.to_string();
                for (p, path) in ok {
                    self.remove_copy(&path);
                    self.fail(&p, e.clone(), &mut errs);
                }
            }
        }

        for (src, hash) in dups {
            match self.seen.get(&hash) {
                Some(id) => {
                    self.result.duplicated += 1;
                    self.map.insert(src, *id);
                }
                None => {
                    let e = errs.get(&hash).cloned().unwrap_or_default();
                    self.result.failed.push(format!("{}：{e}", src.display()));
                }
            }
        }
        Ok(())
    }

    fn fail(&mut self, p: &Prepared, e: String, errs: &mut HashMap<String, String>) {
        self.result.failed.push(format!("{}：{e}", p.src.display()));
        errs.insert(p.hash.clone(), e);
    }

    /// 删掉复制进项目的副本（引用模式下 path 是用户的原图，不能删）
    fn remove_copy(&self, path: &str) {
        if self.copy {
            let _ = fs::remove_file(self.project.root.join(path));
        }
    }
}

/// 一个事务写入一批图片。每张返回 Ok((id, 是否新增)) 或 Err(原因)
fn insert_rows(
    project: &Project,
    rows: &[(Prepared, String)],
    batch: i64,
    now: i64,
) -> AppResult<Vec<Result<(i64, bool), String>>> {
    let mut out = Vec::with_capacity(rows.len());
    if rows.is_empty() {
        return Ok(out);
    }
    let mut conn = project.conn.lock();
    let tx = conn.transaction()?;
    {
        let mut exists = tx.prepare("SELECT id FROM images WHERE hash = ?1 LIMIT 1")?;
        let mut insert = tx.prepare(
            "INSERT INTO images (path, width, height, hash, created_at, batch) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )?;
        for (p, path) in rows {
            // 复制时没持锁，再查一次有没有同样内容的图
            if let Some(id) = exists.query_row([&p.hash], |r| r.get::<_, i64>(0)).optional()? {
                out.push(Ok((id, false)));
                continue;
            }
            match insert.execute(params![path, p.width, p.height, p.hash, now, batch]) {
                Ok(_) => out.push(Ok((tx.last_insert_rowid(), true))),
                Err(e) => out.push(Err(e.to_string())),
            }
        }
    }
    tx.commit()?;
    Ok(out)
}

/// 按类别筛图的 SQL 条件（{n} 换成参数序号）
pub(crate) const LABEL_COND: &str = "(?{n} IS NULL
           OR (?{n} = -1 AND NOT EXISTS (SELECT 1 FROM annotations a WHERE a.image_id = images.id))
           OR (?{n} = -2 AND images.status = 'todo' AND EXISTS (SELECT 1 FROM annotations a WHERE a.image_id = images.id AND a.source IN ('model', 'text', 'ai')))
           OR (?{n} > 0 AND EXISTS (SELECT 1 FROM annotations a WHERE a.image_id = images.id AND a.label_id = ?{n})))";

/// 数据页的图片列表。label：只看有这个类别标注的图；-1 是一个标注都没有的，-2 是有预标注还没确认的
pub fn list(project: &Project, status: Option<&str>, label: Option<i64>, offset: u32, limit: u32) -> AppResult<ImagePage> {
    let conn = project.conn.lock();
    let status = status.filter(|s| !s.is_empty() && *s != "all");
    let total: i64 = conn.query_row(
        &format!(
            "SELECT COUNT(*) FROM images WHERE (?1 IS NULL OR status = ?1 OR (?1 = 'done' AND status IN ('labeled', 'reviewed'))) AND {}",
            LABEL_COND.replace("{n}", "2")
        ),
        params![status, label],
        |r| r.get(0),
    )?;
    let mut stmt = conn.prepare(&format!(
        "SELECT id, path, width, height, status, split FROM images
         WHERE (?1 IS NULL OR status = ?1 OR (?1 = 'done' AND status IN ('labeled', 'reviewed'))) AND {} ORDER BY id LIMIT ?2 OFFSET ?3",
        LABEL_COND.replace("{n}", "4")
    ))?;
    let items = stmt
        .query_map(params![status, limit.min(1000), offset, label], |r| {
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
#[cfg_attr(not(test), allow(dead_code))] // 界面用 delete_with，这个给测试用
pub fn delete(project: &Project, scope: DeleteScope) -> AppResult<usize> {
    delete_with(project, scope, &|_, _| {})
}

/// 同 delete，删文件时报进度（删几万张时文件要删一会儿）
pub fn delete_with(project: &Project, scope: DeleteScope, progress: &(dyn Fn(usize, usize) + Sync)) -> AppResult<usize> {
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

    // 内容相同的图可能还有别的在用缩略图：先查好，然后放开数据库锁再删文件
    let unused: HashSet<&str> = {
        let mut still_used = conn.prepare("SELECT 1 FROM images WHERE hash = ?1 LIMIT 1")?;
        let mut set = HashSet::new();
        for (_, _, hash) in &rows {
            if !set.contains(hash.as_str()) && still_used.query_row([hash], |_| Ok(())).optional()?.is_none() {
                set.insert(hash.as_str());
            }
        }
        set
    };
    drop(conn);

    // 提交成功后再删文件：只删项目 images/ 下的副本，并行删
    let total = rows.len();
    let done = AtomicUsize::new(0);
    let prefix = format!("{IMAGES_DIR}/");
    progress(0, total);
    rows.par_iter().for_each(|(_, path, hash)| {
        if path.starts_with(&prefix) {
            let _ = fs::remove_file(project.root.join(path));
        }
        if unused.contains(hash.as_str()) {
            let _ = fs::remove_file(project.thumb_path(hash));
        }
        progress(done.fetch_add(1, Ordering::Relaxed) + 1, total);
    });
    Ok(total)
}

/// 按 id 查图片的存储路径和哈希
pub fn locate(project: &Project, id: i64) -> AppResult<(String, String)> {
    let conn = project.conn.lock();
    conn.query_row("SELECT path, hash FROM images WHERE id = ?1", [id], |r| Ok((r.get(0)?, r.get(1)?)))
        .optional()?
        .ok_or_else(|| msg(t!("图片不存在：{id}", id = id)))
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
    fn chunked_import_dedup_and_stop() {
        let base = temp_dir("chunk");
        fs::create_dir_all(base.join("x")).unwrap();
        fs::create_dir_all(base.join("y")).unwrap();
        for (n, c) in [("a", 1u8), ("b", 2), ("c", 3), ("d", 4)] {
            make_image(&base.join(format!("{n}.png")), 20, 10, c);
        }
        fs::copy(base.join("a.png"), base.join("a_dup.png")).unwrap();
        fs::copy(base.join("b.png"), base.join("b_dup.png")).unwrap();
        make_image(&base.join("x/img.png"), 20, 10, 5);
        make_image(&base.join("y/img.png"), 20, 10, 6);
        let project = Project::create(&base, "p", "detect").unwrap();

        // 每批 2 张：同一批里重复、和前一批重复、同名不同内容都要对
        let files: Vec<PathBuf> = ["a", "a_dup", "b", "c", "b_dup", "d"]
            .iter()
            .map(|n| base.join(format!("{n}.png")))
            .chain([base.join("x/img.png"), base.join("y/img.png")])
            .collect();
        let last = AtomicUsize::new(0);
        let (r, map) = import_inner(&project, &files, true, None, None, 2, &|d, t| {
            assert_eq!(t, files.len() * 2);
            last.fetch_max(d, Ordering::Relaxed);
        })
        .unwrap();
        assert_eq!((r.added, r.duplicated, r.failed.len(), r.stopped), (6, 2, 0, 0));
        assert_eq!(last.load(Ordering::Relaxed), files.len() * 2, "进度走到头");
        assert_eq!(map[&files[0]], map[&files[1]], "a 和 a_dup 是同一张");
        assert_eq!(map[&files[2]], map[&files[4]], "b 和 b_dup 是同一张");
        assert!(project.root.join("images/img.png").exists());
        assert!(project.root.join("images/img_1.png").exists(), "同名文件不能互相覆盖");
        assert_eq!(project.stats().unwrap().total, 6);

        // 第一批写完就停：后面的算没导入，也没留下复制的文件
        for (n, c) in [("e", 7u8), ("f", 8), ("g", 9), ("h", 10)] {
            make_image(&base.join(format!("{n}.png")), 20, 10, c);
        }
        let more: Vec<PathBuf> = ["e", "f", "g", "h"].iter().map(|n| base.join(format!("{n}.png"))).collect();
        let stop = AtomicBool::new(false);
        let (r, _) = import_inner(&project, &more, true, None, Some(&stop), 2, &|d, t| {
            if d >= t / 2 + 2 {
                stop.store(true, Ordering::Relaxed);
            }
        })
        .unwrap();
        assert_eq!((r.added, r.stopped), (2, 2));
        assert!(!project.root.join("images/g.png").exists());
        assert!(!project.root.join("images/h.png").exists());

        // 一开始就停：一张都不导入
        stop.store(true, Ordering::Relaxed);
        let (r, _) = import_inner(&project, &more[2..], true, None, Some(&stop), 2, &|_, _| {}).unwrap();
        assert_eq!((r.added, r.stopped), (0, 2));
        assert_eq!(project.stats().unwrap().total, 8);
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
