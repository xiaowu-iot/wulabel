//! 多人协作（离线）：把图片分成几个标注包，每个包就是一个独立的 Wulabel 项目，发给不同的人标；
//! 标完把包合并回来。包里记着「包内图片 id → 原项目图片 id」，所以包里的图打过码（内容变了）也能对回去。

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::annotations::{self, Annotation};
use crate::db;
use crate::error::{msg, AppResult};
use crate::evaltools::{self, Anonymize};
use crate::images;
use crate::keypoints;
use crate::project::Project;
use crate::runtime::Runtime;

const PACKAGE_FILE: &str = "package.json";

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SplitOptions {
    /// 标注包放在哪个文件夹下
    pub out_dir: PathBuf,
    /// 每人一个包，名字写在包名里；人数就是包数
    pub assignees: Vec<String>,
    /// todo 未标注 / unreviewed 未审核（未标注 + 已标注）/ all 全部（不含跳过）
    pub scope: String,
    /// 带上已有的标注（比如模型预标注、需要复核的）
    #[serde(default = "yes")]
    pub with_annotations: bool,
    /// 包里的图片先打码再发出去（人脸、车牌等）
    #[serde(default)]
    pub anonymize: Option<Anonymize>,
}

fn yes() -> bool {
    true
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PackageInfo {
    pub id: i64,
    pub name: String,
    pub assignee: String,
    pub dir: String,
    pub images: i64,
    pub created_at: i64,
    pub merged_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PackageMeta {
    wulabel_package: u32,
    source: String,
    assignee: String,
    created_at: i64,
    /// [包内图片 id, 原项目图片 id]
    map: Vec<[i64; 2]>,
}

fn safe(name: &str) -> String {
    let s: String = name.chars().map(|c| if r#"\/:*?"<>|"#.contains(c) || c.is_control() { '_' } else { c }).collect();
    s.trim().trim_end_matches('.').to_string()
}

/// 把一张图的标注换成另一个项目的类别 id（按类别名对应）
fn remap(anns: Vec<Annotation>, from: &HashMap<i64, String>, to: &HashMap<String, i64>) -> Vec<Annotation> {
    anns.into_iter()
        .map(|mut a| {
            a.label_id = a.label_id.and_then(|id| from.get(&id)).and_then(|n| to.get(n)).copied();
            a
        })
        .collect()
}

fn label_names(p: &Project) -> AppResult<HashMap<i64, String>> {
    Ok(annotations::list_labels(p)?.into_iter().map(|l| (l.id, l.name)).collect())
}

/// 确保 to 项目里有这些类别（按名字），返回 名字 → id
fn ensure_labels(to: &Project, from: &Project, added: &mut Vec<String>) -> AppResult<HashMap<String, i64>> {
    let mut have: HashMap<String, i64> = annotations::list_labels(to)?.into_iter().map(|l| (l.name, l.id)).collect();
    for l in annotations::list_labels(from)? {
        if !have.contains_key(&l.name) {
            let n = annotations::add_label(to, &l.name, &l.color)?;
            if !l.prompt.is_empty() {
                annotations::update_label(to, n.id, &l.name, &l.color, Some(&l.prompt))?;
            }
            have.insert(l.name.clone(), n.id);
            added.push(l.name);
        }
    }
    Ok(have)
}

/// 分包：按图片顺序（视频连续帧会分在一起）平均分给每个人
pub fn split(
    p: &Project,
    opts: &SplitOptions,
    rt: &Runtime,
    log: &(dyn Fn(String) + Sync),
    progress: &(dyn Fn(usize, usize) + Sync),
) -> AppResult<Vec<PackageInfo>> {
    let people: Vec<String> = opts.assignees.iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    if people.is_empty() {
        return Err(msg("至少填一个标注人"));
    }
    let cond = match opts.scope.as_str() {
        "todo" => "status = 'todo'",
        "unreviewed" => "status IN ('todo', 'labeled')",
        _ => "status != 'skipped'",
    };
    let ids: Vec<(i64, String, String)> = {
        let conn = p.conn.lock();
        let mut stmt = conn.prepare(&format!("SELECT id, path, status FROM images WHERE {cond} ORDER BY id"))?;
        let v = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.collect::<Result<_, _>>()?;
        v
    };
    if ids.is_empty() {
        return Err(msg("没有符合条件的图片"));
    }
    if ids.len() < people.len() {
        return Err(msg(format!("只有 {} 张图，分不成 {} 个包", ids.len(), people.len())));
    }
    fs::create_dir_all(&opts.out_dir)?;
    let src_names = label_names(p)?;
    let total = ids.len();
    let k = people.len();
    let mut done = 0;
    let mut out = Vec::new();
    progress(0, total);
    let stamp = chrono::Local::now().format("%m%d_%H%M").to_string();
    for (i, who) in people.iter().enumerate() {
        // 尽量平均：每人 total/k 张，多出来的分给前面几个人
        let chunk = &ids[i * total / k..(i + 1) * total / k];
        let name = safe(&format!("{}_分包{}_{}_{}", p.meta.name, i + 1, who, stamp));
        let np = Project::create(&opts.out_dir, &name, &p.meta.task)?;
        let mut _added = Vec::new();
        let to_labels = ensure_labels(&np, p, &mut _added)?;
        if p.meta.task == "pose" {
            keypoints::save(&np, &keypoints::get(p)?)?;
        }
        let files: Vec<PathBuf> = chunk.iter().map(|(_, path, _)| p.resolve(path)).collect();
        let (_, newids) = images::import_mapped(&np, &files, true, &|_, _| {})?;
        let mut map = Vec::new();
        for ((sid, _, status), file) in chunk.iter().zip(files.iter()) {
            let Some(&nid) = newids.get(file) else { continue };
            map.push([nid, *sid]);
            if opts.with_annotations {
                let anns = remap(annotations::load(p, *sid)?.annotations, &src_names, &to_labels);
                if !anns.is_empty() {
                    annotations::save(&np, nid, &anns)?;
                }
                if status != "todo" {
                    annotations::set_status(&np, nid, status)?;
                }
            }
            done += 1;
            if done % 10 == 0 {
                progress(done, total);
            }
        }
        if let Some(a) = &opts.anonymize {
            // 包里的图是复制进去的独立文件，直接改写；标注区域从原项目读
            let mut seen = std::collections::HashSet::new();
            let mut files = Vec::new();
            let mut hashes = Vec::new();
            {
                let conn = np.conn.lock();
                let mut stmt = conn.prepare("SELECT path, hash FROM images WHERE id = ?1")?;
                for [nid, sid] in &map {
                    if !seen.insert(*nid) {
                        continue;
                    }
                    let (path, hash): (String, String) = stmt.query_row([nid], |r| Ok((r.get(0)?, r.get(1)?)))?;
                    files.push((np.resolve(&path), *sid));
                    hashes.push(hash);
                }
            }
            log(format!("给「{who}」的包打码…"));
            evaltools::anonymize(p, rt, &files, a, log, &|_, _| {})?;
            // 缩略图是打码前生成的，删掉让它重新生成
            for h in hashes {
                let _ = fs::remove_file(np.thumb_path(&h));
            }
        }
        let meta = PackageMeta {
            wulabel_package: 1,
            source: p.meta.name.clone(),
            assignee: who.clone(),
            created_at: db::now(),
            map: map.clone(),
        };
        fs::write(np.root.join(PACKAGE_FILE), serde_json::to_string_pretty(&meta)?)?;
        let dir = np.root.to_string_lossy().into_owned();
        let conn = p.conn.lock();
        conn.execute(
            "INSERT INTO packages (name, assignee, dir, images, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![name, who, dir, map.len() as i64, meta.created_at],
        )?;
        out.push(PackageInfo {
            id: conn.last_insert_rowid(),
            name,
            assignee: who.clone(),
            dir,
            images: map.len() as i64,
            created_at: meta.created_at,
            merged_at: None,
        });
    }
    progress(total, total);
    Ok(out)
}

pub fn list(p: &Project) -> AppResult<Vec<PackageInfo>> {
    let conn = p.conn.lock();
    let mut stmt = conn.prepare("SELECT id, name, assignee, dir, images, created_at, merged_at FROM packages ORDER BY id DESC")?;
    let v = stmt
        .query_map([], |r| {
            Ok(PackageInfo {
                id: r.get(0)?,
                name: r.get(1)?,
                assignee: r.get(2)?,
                dir: r.get(3)?,
                images: r.get(4)?,
                created_at: r.get(5)?,
                merged_at: r.get(6)?,
            })
        })?
        .collect::<Result<_, _>>()?;
    Ok(v)
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MergeOptions {
    pub dir: PathBuf,
    /// 原项目里这些图在分包后也改过时，用包里的覆盖；否则跳过这些图
    #[serde(default)]
    pub overwrite: bool,
    /// 标注人在包里新加的图也导入
    #[serde(default = "yes")]
    pub import_new: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MergeResult {
    pub assignee: String,
    /// 合并了标注的图
    pub updated: usize,
    /// 包里没动过、或和原项目一样的图
    pub unchanged: usize,
    /// 原项目在分包后也改过、没有覆盖的图
    pub conflicts: Vec<i64>,
    /// 包里新加的图
    pub added: usize,
    /// 原项目里已经删掉的图
    pub missing: usize,
    pub labels_added: Vec<String>,
}

fn fingerprint(anns: &[Annotation], names: &HashMap<i64, String>) -> Vec<String> {
    let mut v: Vec<String> = anns
        .iter()
        .map(|a| format!("{}|{}|{}", a.label_id.and_then(|id| names.get(&id)).cloned().unwrap_or_default(), a.kind, a.data))
        .collect();
    v.sort();
    v
}

/// 把一个标注包合并回来。没有 package.json 的普通 Wulabel 项目也能合并（按图片内容对应）
pub fn merge(p: &Project, opts: &MergeOptions) -> AppResult<MergeResult> {
    if opts.dir.canonicalize().ok() == p.root.canonicalize().ok() {
        return Err(msg("不能把项目合并到它自己"));
    }
    let np = Project::open(&opts.dir)?;
    if np.meta.task != p.meta.task {
        return Err(msg("这个包的任务类型和当前项目不一样，不能合并"));
    }
    let meta: Option<PackageMeta> = fs::read_to_string(np.root.join(PACKAGE_FILE)).ok().and_then(|t| serde_json::from_str(&t).ok());
    let map: HashMap<i64, i64> = meta.as_ref().map(|m| m.map.iter().map(|[a, b]| (*a, *b)).collect()).unwrap_or_default();
    let since = meta.as_ref().map(|m| m.created_at).unwrap_or(0);
    let mut res = MergeResult {
        assignee: meta.as_ref().map(|m| m.assignee.clone()).unwrap_or_default(),
        updated: 0,
        unchanged: 0,
        conflicts: Vec::new(),
        added: 0,
        missing: 0,
        labels_added: Vec::new(),
    };
    let to_labels = ensure_labels(p, &np, &mut res.labels_added)?;
    if p.meta.task == "pose" {
        let (a, b) = (keypoints::get(p)?, keypoints::get(&np)?);
        if a.points.len() != b.points.len() {
            return Err(msg("包里的关键点模板点数和当前项目不一样，不能合并"));
        }
    }
    let pkg_names = label_names(&np)?;
    let src_names = label_names(p)?;
    let pkg_imgs: Vec<(i64, String, String, String)> = {
        let conn = np.conn.lock();
        let mut stmt = conn.prepare("SELECT id, path, status, hash FROM images ORDER BY id")?;
        let v = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?.collect::<Result<_, _>>()?;
        v
    };
    for (pid, path, status, hash) in pkg_imgs {
        let pkg = annotations::load(&np, pid)?.annotations;
        // 对应原项目的图：先看包里的对照表，没有再按图片内容找
        let sid: Option<i64> = match map.get(&pid) {
            Some(s) => Some(*s),
            None => p.conn.lock().query_row("SELECT id FROM images WHERE hash = ?1", [&hash], |r| r.get(0)).optional()?,
        };
        let exists = match sid {
            Some(s) => p.conn.lock().query_row("SELECT 1 FROM images WHERE id = ?1", [s], |_| Ok(())).optional()?.is_some(),
            None => false,
        };
        if sid.is_some() && !exists {
            res.missing += 1;
            continue;
        }
        let Some(sid) = sid else {
            // 包里新加的图
            if !opts.import_new || (pkg.is_empty() && status == "todo") {
                continue;
            }
            let file = np.resolve(&path);
            let (_, ids) = images::import_mapped(p, &[file.clone()], true, &|_, _| {})?;
            if let Some(&nid) = ids.get(&file) {
                let anns = remap(pkg, &pkg_names, &to_labels);
                if !anns.is_empty() {
                    annotations::save(p, nid, &anns)?;
                }
                if status != "todo" {
                    annotations::set_status(p, nid, &status)?;
                }
                res.added += 1;
            }
            continue;
        };
        if pkg.is_empty() && status == "todo" {
            res.unchanged += 1;
            continue;
        }
        let cur = annotations::load(p, sid)?;
        let same = fingerprint(&pkg, &pkg_names) == fingerprint(&cur.annotations, &src_names) && cur.status == status;
        if same {
            res.unchanged += 1;
            continue;
        }
        // 原项目在分包之后也改过这张图
        let touched: Option<i64> =
            p.conn.lock().query_row("SELECT MAX(updated_at) FROM annotations WHERE image_id = ?1", [sid], |r| r.get(0))?;
        if touched.unwrap_or(0) > since && !opts.overwrite {
            res.conflicts.push(sid);
            continue;
        }
        annotations::save(p, sid, &remap(pkg, &pkg_names, &to_labels))?;
        if status != "todo" {
            annotations::set_status(p, sid, &status)?;
        }
        res.updated += 1;
    }
    let dir = np.root.to_string_lossy().into_owned();
    p.conn.lock().execute("UPDATE packages SET merged_at = ?2 WHERE dir = ?1", params![dir, db::now()])?;
    Ok(res)
}
