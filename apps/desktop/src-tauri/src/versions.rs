//! 数据集版本：把当前可训练的图片（已标注、已审核）和它们的标注存成快照。
//! 版本可以导出、用来训练（训练记录会记下用的哪个版本）、和当前数据对比、恢复。
//! 标注按类别名保存，之后改类别名、删类别都不影响已有版本。

use std::collections::{HashMap, HashSet};

use rusqlite::{params, OptionalExtension};
use serde::Serialize;

use crate::annotations;
use crate::dataset::{self, ExportImage};
use crate::db;
use crate::error::{msg, AppResult};
use crate::project::Project;

const PALETTE: [&str; 10] = ["#3d8bfd", "#e5534b", "#2fb67c", "#e7a33e", "#9b7bf2", "#14b8a6", "#f472b6", "#a3e635", "#f97316", "#60a5fa"];

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassCount {
    pub name: String,
    pub count: i64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VersionInfo {
    pub id: i64,
    pub name: String,
    pub note: String,
    pub created_at: i64,
    pub images: i64,
    pub annotations: i64,
    pub train: i64,
    pub val: i64,
    pub test: i64,
    /// 各类别的标注数（按快照时的类别顺序）
    pub classes: Vec<ClassCount>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassDelta {
    pub name: String,
    /// 版本里的标注数
    pub before: i64,
    /// 当前的标注数
    pub after: i64,
}

/// 版本和当前数据的差别（当前 = 现在的已标注、已审核图片）
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VersionDiff {
    /// 当前有、版本里没有的图
    pub added: Vec<i64>,
    /// 版本里有、当前不算可训练（或已删除）的图
    pub removed: Vec<i64>,
    /// 两边都有、标注不一样的图
    pub changed: Vec<i64>,
    pub classes: Vec<ClassDelta>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RestoreResult {
    /// 恢复了标注的图片数
    pub images: usize,
    /// 版本里有、但图片已经从项目删掉的数量（没法恢复）
    pub missing: usize,
    /// 恢复前自动保存的备份版本
    pub backup: Option<i64>,
}

/// 把当前可训练的图片和标注存成一个新版本。还没划分训练/验证/测试的图先按 ratios 划分，版本里的划分就固定下来了
pub fn create(p: &Project, name: &str, note: &str, ratios: [f64; 3]) -> AppResult<VersionInfo> {
    dataset::assign_splits(p, ratios, false)?;
    let vid = {
        let mut conn = p.conn.lock();
        let tx = conn.transaction()?;
        let n: i64 = tx.query_row("SELECT COUNT(*) FROM dataset_versions", [], |r| r.get(0))?;
        let name = match name.trim() {
            "" => format!("v{}", n + 1),
            s => s.chars().take(60).collect(),
        };
        let classes: Vec<String> = {
            let mut stmt = tx.prepare("SELECT name FROM labels ORDER BY sort, id")?;
            let v = stmt.query_map([], |r| r.get(0))?.collect::<Result<_, _>>()?;
            v
        };
        tx.execute(
            "INSERT INTO dataset_versions (name, note, created_at, classes_json) VALUES (?1, ?2, ?3, ?4)",
            params![name, note.trim(), db::now(), serde_json::to_string(&classes)?],
        )?;
        let vid = tx.last_insert_rowid();
        let n_img = tx.execute(
            "INSERT INTO version_images (version_id, image_id, path, width, height, hash, split, status)
             SELECT ?1, id, path, width, height, hash, split, status FROM images WHERE status IN ('labeled', 'reviewed')",
            [vid],
        )?;
        if n_img == 0 {
            return Err(msg("还没有已标注或已审核的图片，没法建版本"));
        }
        tx.execute(
            "INSERT INTO version_annotations (version_id, image_id, label_name, type, data_json)
             SELECT ?1, a.image_id, l.name, a.type, a.data_json
             FROM annotations a JOIN images i ON i.id = a.image_id LEFT JOIN labels l ON l.id = a.label_id
             WHERE i.status IN ('labeled', 'reviewed') ORDER BY a.id",
            [vid],
        )?;
        tx.commit()?;
        vid
    };
    info(p, vid)
}

pub fn info(p: &Project, vid: i64) -> AppResult<VersionInfo> {
    let conn = p.conn.lock();
    let (name, note, created_at, classes_json): (String, String, i64, String) = conn
        .query_row("SELECT name, note, created_at, classes_json FROM dataset_versions WHERE id = ?1", [vid], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
        })
        .optional()?
        .ok_or_else(|| msg("数据集版本不存在"))?;
    let mut v = VersionInfo {
        id: vid,
        name,
        note,
        created_at,
        images: 0,
        annotations: 0,
        train: 0,
        val: 0,
        test: 0,
        classes: Vec::new(),
    };
    {
        let mut stmt = conn.prepare("SELECT split, COUNT(*) FROM version_images WHERE version_id = ?1 GROUP BY split")?;
        for row in stmt.query_map([vid], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))? {
            let (s, n) = row?;
            v.images += n;
            match s.as_str() {
                "val" => v.val += n,
                "test" => v.test += n,
                _ => v.train += n,
            }
        }
    }
    let mut counts: HashMap<String, i64> = HashMap::new();
    {
        let mut stmt = conn.prepare(
            "SELECT COALESCE(label_name, '未分类'), COUNT(*) FROM version_annotations WHERE version_id = ?1 GROUP BY label_name",
        )?;
        for row in stmt.query_map([vid], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))? {
            let (n, c) = row?;
            v.annotations += c;
            counts.insert(n, c);
        }
    }
    let order: Vec<String> = serde_json::from_str(&classes_json).unwrap_or_default();
    for name in &order {
        v.classes.push(ClassCount { name: name.clone(), count: counts.remove(name).unwrap_or(0) });
    }
    let mut rest: Vec<(String, i64)> = counts.into_iter().collect();
    rest.sort();
    v.classes.extend(rest.into_iter().map(|(name, count)| ClassCount { name, count }));
    Ok(v)
}

pub fn list(p: &Project) -> AppResult<Vec<VersionInfo>> {
    let ids: Vec<i64> = {
        let conn = p.conn.lock();
        let mut stmt = conn.prepare("SELECT id FROM dataset_versions ORDER BY id DESC")?;
        let v = stmt.query_map([], |r| r.get(0))?.collect::<Result<_, _>>()?;
        v
    };
    ids.into_iter().map(|id| info(p, id)).collect()
}

pub fn delete(p: &Project, vid: i64) -> AppResult<()> {
    let n = p.conn.lock().execute("DELETE FROM dataset_versions WHERE id = ?1", [vid])?;
    if n == 0 {
        return Err(msg("数据集版本不存在"));
    }
    Ok(())
}

pub fn rename(p: &Project, vid: i64, name: &str, note: &str) -> AppResult<()> {
    let name = name.trim();
    if name.is_empty() {
        return Err(msg("版本名不能为空"));
    }
    p.conn.lock().execute(
        "UPDATE dataset_versions SET name = ?2, note = ?3 WHERE id = ?1",
        params![vid, name.chars().take(60).collect::<String>(), note.trim()],
    )?;
    Ok(())
}

/// 版本名（训练记录里显示用）
pub fn name_of(p: &Project, vid: i64) -> AppResult<String> {
    p.conn
        .lock()
        .query_row("SELECT name FROM dataset_versions WHERE id = ?1", [vid], |r| r.get(0))
        .optional()?
        .ok_or_else(|| msg("数据集版本不存在"))
}

/// 读出版本里的图片和标注，给导出 / 训练用。图片文件已经不在了的跳过
pub fn collect(p: &Project, vid: i64, only_reviewed: bool) -> AppResult<(Vec<String>, Vec<ExportImage>)> {
    let conn = p.conn.lock();
    let classes_json: String = conn
        .query_row("SELECT classes_json FROM dataset_versions WHERE id = ?1", [vid], |r| r.get(0))
        .optional()?
        .ok_or_else(|| msg("数据集版本不存在"))?;
    let mut classes: Vec<String> = serde_json::from_str(&classes_json).unwrap_or_default();
    // 快照里出现、但不在类别表里的类别名（理论上没有）补到后面
    {
        let mut stmt =
            conn.prepare("SELECT DISTINCT label_name FROM version_annotations WHERE version_id = ?1 AND label_name IS NOT NULL")?;
        for n in stmt.query_map([vid], |r| r.get::<_, String>(0))? {
            let n = n?;
            if !classes.contains(&n) {
                classes.push(n);
            }
        }
    }
    let index: HashMap<String, usize> = classes.iter().enumerate().map(|(i, n)| (n.clone(), i)).collect();
    let mut images = Vec::new();
    {
        let cond = if only_reviewed { "AND status = 'reviewed'" } else { "" };
        let mut stmt = conn.prepare(&format!(
            "SELECT image_id, path, width, height, split FROM version_images WHERE version_id = ?1 {cond} ORDER BY image_id"
        ))?;
        let rows = stmt.query_map([vid], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, u32>(2)?, r.get::<_, u32>(3)?, r.get::<_, String>(4)?))
        })?;
        for row in rows {
            let (id, path, width, height, split) = row?;
            let src = p.resolve(&path);
            if !src.exists() {
                continue;
            }
            let file_name = src.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| format!("{id}.jpg"));
            images.push(ExportImage {
                id,
                src,
                file_name,
                width,
                height,
                split,
                boxes: Vec::new(),
                polys: Vec::new(),
                tag: None,
                kpts: Vec::new(),
            });
        }
    }
    if images.is_empty() {
        return Err(msg("这个版本里的图片都找不到了（可能已经从项目里删除）"));
    }
    let pos: HashMap<i64, usize> = images.iter().enumerate().map(|(i, im)| (im.id, i)).collect();
    let mut stmt = conn.prepare("SELECT image_id, label_name, type, data_json FROM version_annotations WHERE version_id = ?1")?;
    let rows = stmt.query_map([vid], |r| {
        Ok((r.get::<_, i64>(0)?, r.get::<_, Option<String>>(1)?, r.get::<_, String>(2)?, r.get::<_, String>(3)?))
    })?;
    for row in rows {
        let (image_id, label, kind, data) = row?;
        let (Some(&i), Some(label)) = (pos.get(&image_id), label) else { continue };
        let Some(&cls) = index.get(&label) else { continue };
        dataset::push_annotation(&mut images[i], cls, &kind, &data);
    }
    dataset::dedupe_names(&mut images);
    Ok((classes, images))
}

/// 每张图的标注指纹（类别名 + 类型 + 数据，排好序），用来比较两边是否一样
fn fingerprints(conn: &rusqlite::Connection, sql: &str, arg: Option<i64>) -> AppResult<HashMap<i64, Vec<String>>> {
    let mut out: HashMap<i64, Vec<String>> = HashMap::new();
    let mut stmt = conn.prepare(sql)?;
    let map = |r: &rusqlite::Row| {
        Ok((r.get::<_, i64>(0)?, r.get::<_, Option<String>>(1)?, r.get::<_, String>(2)?, r.get::<_, String>(3)?))
    };
    let rows: Vec<_> = match arg {
        Some(a) => stmt.query_map([a], map)?.collect::<Result<_, _>>()?,
        None => stmt.query_map([], map)?.collect::<Result<_, _>>()?,
    };
    for (id, label, kind, data) in rows {
        out.entry(id).or_default().push(format!("{}|{kind}|{data}", label.unwrap_or_default()));
    }
    for v in out.values_mut() {
        v.sort();
    }
    Ok(out)
}

/// 版本和当前数据对比
pub fn diff(p: &Project, vid: i64) -> AppResult<VersionDiff> {
    let conn = p.conn.lock();
    let ver_imgs: HashSet<i64> = {
        let mut stmt = conn.prepare("SELECT image_id FROM version_images WHERE version_id = ?1")?;
        let v = stmt.query_map([vid], |r| r.get(0))?.collect::<Result<_, _>>()?;
        v
    };
    let cur_imgs: HashSet<i64> = {
        let mut stmt = conn.prepare("SELECT id FROM images WHERE status IN ('labeled', 'reviewed')")?;
        let v = stmt.query_map([], |r| r.get(0))?.collect::<Result<_, _>>()?;
        v
    };
    let ver = fingerprints(
        &conn,
        "SELECT image_id, label_name, type, data_json FROM version_annotations WHERE version_id = ?1",
        Some(vid),
    )?;
    let cur = fingerprints(
        &conn,
        "SELECT a.image_id, l.name, a.type, a.data_json FROM annotations a JOIN images i ON i.id = a.image_id
         LEFT JOIN labels l ON l.id = a.label_id WHERE i.status IN ('labeled', 'reviewed')",
        None,
    )?;
    let mut added: Vec<i64> = cur_imgs.difference(&ver_imgs).copied().collect();
    let mut removed: Vec<i64> = ver_imgs.difference(&cur_imgs).copied().collect();
    let empty = Vec::new();
    let mut changed: Vec<i64> = ver_imgs
        .intersection(&cur_imgs)
        .filter(|id| ver.get(*id).unwrap_or(&empty) != cur.get(*id).unwrap_or(&empty))
        .copied()
        .collect();
    added.sort();
    removed.sort();
    changed.sort();
    // 各类别标注数的变化
    let count = |m: &HashMap<i64, Vec<String>>| {
        let mut c: HashMap<String, i64> = HashMap::new();
        for v in m.values() {
            for s in v {
                *c.entry(s.split('|').next().unwrap_or("").to_string()).or_default() += 1;
            }
        }
        c
    };
    let (cv, cc) = (count(&ver), count(&cur));
    let mut names: Vec<String> = cv.keys().chain(cc.keys()).cloned().collect::<HashSet<_>>().into_iter().collect();
    names.sort();
    let classes = names
        .into_iter()
        .map(|n| ClassDelta {
            before: cv.get(&n).copied().unwrap_or(0),
            after: cc.get(&n).copied().unwrap_or(0),
            name: if n.is_empty() { "未分类".into() } else { n },
        })
        .filter(|d| d.before != d.after)
        .collect();
    added.truncate(1000);
    removed.truncate(1000);
    changed.truncate(1000);
    Ok(VersionDiff { added, removed, changed, classes })
}

/// 把版本里图片的标注、划分、状态恢复回去（先自动把当前数据存成一个备份版本）。
/// 版本里没有的图不动；图片已经删掉的没法恢复。
pub fn restore(p: &Project, vid: i64) -> AppResult<RestoreResult> {
    let target = info(p, vid)?;
    let backup = create(p, "恢复前自动备份", &format!("恢复到「{}」之前自动保存的当前数据", target.name), [0.8, 0.2, 0.0])
        .ok()
        .map(|v| v.id);

    // 版本里用到的类别名，项目里没有的先建出来
    let names: Vec<String> = {
        let conn = p.conn.lock();
        let mut stmt = conn
            .prepare("SELECT DISTINCT label_name FROM version_annotations WHERE version_id = ?1 AND label_name IS NOT NULL")?;
        let v = stmt.query_map([vid], |r| r.get(0))?.collect::<Result<_, _>>()?;
        v
    };
    let mut labels: HashMap<String, i64> = annotations::list_labels(p)?.into_iter().map(|l| (l.name, l.id)).collect();
    for n in &names {
        if !labels.contains_key(n) {
            let l = annotations::add_label(p, n, PALETTE[labels.len() % PALETTE.len()])?;
            labels.insert(n.clone(), l.id);
        }
    }

    let mut conn = p.conn.lock();
    let tx = conn.transaction()?;
    let rows: Vec<(i64, String, String, String)> = {
        let mut stmt = tx.prepare(
            "SELECT v.image_id, v.split, v.status, v.hash FROM version_images v WHERE v.version_id = ?1",
        )?;
        let v = stmt.query_map([vid], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?.collect::<Result<_, _>>()?;
        v
    };
    let mut images = 0;
    let mut missing = 0;
    {
        let mut exists = tx.prepare("SELECT hash FROM images WHERE id = ?1")?;
        let mut del = tx.prepare("DELETE FROM annotations WHERE image_id = ?1")?;
        let mut anns = tx.prepare("SELECT label_name, type, data_json FROM version_annotations WHERE version_id = ?1 AND image_id = ?2")?;
        let mut ins = tx.prepare(
            "INSERT INTO annotations (image_id, label_id, type, data_json, score, source, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, NULL, 'manual', ?5, ?5)",
        )?;
        let mut up = tx.prepare("UPDATE images SET split = ?2, status = ?3 WHERE id = ?1")?;
        let now = db::now();
        for (id, split, status, hash) in &rows {
            // 同一个 id 的图换成了别的内容（删了又导入）也不恢复
            let cur: Option<String> = exists.query_row([id], |r| r.get(0)).optional()?;
            if cur.as_deref() != Some(hash.as_str()) {
                missing += 1;
                continue;
            }
            del.execute([id])?;
            let list: Vec<(Option<String>, String, String)> =
                anns.query_map(params![vid, id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.collect::<Result<_, _>>()?;
            for (label, kind, data) in list {
                let label_id = label.and_then(|n| labels.get(&n).copied());
                ins.execute(params![id, label_id, kind, data, now])?;
            }
            up.execute(params![id, split, status])?;
            images += 1;
        }
    }
    tx.commit()?;
    Ok(RestoreResult { images, missing, backup })
}
