use std::fs;
use std::path::{Path, PathBuf};

use parking_lot::Mutex;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::db;
use crate::error::{msg, AppResult};

pub const PROJECT_FILE: &str = "project.json";
pub const DB_FILE: &str = "annotations.db";
pub const IMAGES_DIR: &str = "images";
pub const THUMBS_DIR: &str = ".cache/thumbs";
const FORMAT_VERSION: u32 = 1;
const TASKS: [&str; 4] = ["detect", "segment", "classify", "pose"];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectMeta {
    pub format_version: u32,
    pub name: String,
    pub task: String,
    pub created_at: i64,
}

pub struct Project {
    pub root: PathBuf,
    pub meta: ProjectMeta,
    pub conn: Mutex<Connection>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Stats {
    pub total: i64,
    pub todo: i64,
    pub labeled: i64,
    pub reviewed: i64,
    pub skipped: i64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectInfo {
    pub path: String,
    pub name: String,
    pub task: String,
    pub stats: Stats,
}

impl Project {
    /// 在 parent 下新建名为 name 的项目文件夹
    pub fn create(parent: &Path, name: &str, task: &str) -> AppResult<Project> {
        let name = name.trim();
        if name.is_empty() {
            return Err(msg("项目名不能为空"));
        }
        if name.chars().any(|c| r#"\/:*?"<>|"#.contains(c)) {
            return Err(msg(r#"项目名不能包含 \ / : * ? " < > | 这些字符"#));
        }
        if !TASKS.contains(&task) {
            return Err(msg(format!("不支持的任务类型：{task}")));
        }
        let root = parent.join(name);
        if root.exists() && fs::read_dir(&root)?.next().is_some() {
            return Err(msg(format!("文件夹已存在且不为空：{}", root.display())));
        }
        fs::create_dir_all(root.join(IMAGES_DIR))?;
        fs::create_dir_all(root.join(THUMBS_DIR))?;
        let meta = ProjectMeta {
            format_version: FORMAT_VERSION,
            name: name.to_string(),
            task: task.to_string(),
            created_at: db::now(),
        };
        fs::write(root.join(PROJECT_FILE), serde_json::to_string_pretty(&meta)?)?;
        let conn = db::open(&root.join(DB_FILE))?;
        Ok(Project { root, meta, conn: Mutex::new(conn) })
    }

    pub fn open(root: &Path) -> AppResult<Project> {
        let file = root.join(PROJECT_FILE);
        if !file.exists() {
            return Err(msg(format!("这个文件夹不是项目（缺少 {PROJECT_FILE}）：{}", root.display())));
        }
        let meta: ProjectMeta = serde_json::from_str(&fs::read_to_string(&file)?)?;
        if meta.format_version > FORMAT_VERSION {
            return Err(msg("项目由更新版本的软件创建，请先升级软件"));
        }
        fs::create_dir_all(root.join(THUMBS_DIR))?;
        let conn = db::open(&root.join(DB_FILE))?;
        Ok(Project { root: root.to_path_buf(), meta, conn: Mutex::new(conn) })
    }

    pub fn stats(&self) -> AppResult<Stats> {
        let conn = self.conn.lock();
        let mut s = Stats { total: 0, todo: 0, labeled: 0, reviewed: 0, skipped: 0 };
        let mut stmt = conn.prepare("SELECT status, COUNT(*) FROM images GROUP BY status")?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?;
        for row in rows {
            let (status, n) = row?;
            s.total += n;
            match status.as_str() {
                "todo" => s.todo = n,
                "labeled" => s.labeled = n,
                "reviewed" => s.reviewed = n,
                "skipped" => s.skipped = n,
                _ => {}
            }
        }
        Ok(s)
    }

    pub fn info(&self) -> AppResult<ProjectInfo> {
        Ok(ProjectInfo {
            path: self.root.to_string_lossy().into_owned(),
            name: self.meta.name.clone(),
            task: self.meta.task.clone(),
            stats: self.stats()?,
        })
    }

    /// 数据库里存的路径换成磁盘上的绝对路径
    pub fn resolve(&self, stored: &str) -> PathBuf {
        let p = Path::new(stored);
        if p.is_absolute() { p.to_path_buf() } else { self.root.join(p) }
    }

    pub fn thumb_path(&self, hash: &str) -> PathBuf {
        self.root.join(THUMBS_DIR).join(format!("{hash}.jpg"))
    }
}

// ---------- 最近项目 ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecentProject {
    pub path: String,
    pub name: String,
    pub task: String,
    pub opened_at: i64,
    #[serde(default)]
    pub exists: bool,
}

const RECENT_MAX: usize = 20;

pub fn load_recent(config_dir: &Path) -> Vec<RecentProject> {
    let list: Vec<RecentProject> = fs::read_to_string(config_dir.join("recent.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    list.into_iter()
        .map(|mut r| {
            r.exists = Path::new(&r.path).join(PROJECT_FILE).exists();
            r
        })
        .collect()
}

fn save_recent(config_dir: &Path, list: &[RecentProject]) -> AppResult<()> {
    fs::create_dir_all(config_dir)?;
    fs::write(config_dir.join("recent.json"), serde_json::to_string_pretty(list)?)?;
    Ok(())
}

pub fn touch_recent(config_dir: &Path, p: &Project) -> AppResult<()> {
    let path = p.root.to_string_lossy().into_owned();
    let mut list = load_recent(config_dir);
    list.retain(|r| !same_path(&r.path, &path));
    list.insert(0, RecentProject {
        path,
        name: p.meta.name.clone(),
        task: p.meta.task.clone(),
        opened_at: db::now(),
        exists: true,
    });
    list.truncate(RECENT_MAX);
    save_recent(config_dir, &list)
}

pub fn remove_recent(config_dir: &Path, path: &str) -> AppResult<()> {
    let mut list = load_recent(config_dir);
    list.retain(|r| !same_path(&r.path, path));
    save_recent(config_dir, &list)
}

fn same_path(a: &str, b: &str) -> bool {
    // Windows 路径不区分大小写
    if cfg!(windows) { a.eq_ignore_ascii_case(b) } else { a == b }
}
