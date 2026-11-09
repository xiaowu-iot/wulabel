use std::path::Path;

use rusqlite::Connection;

use crate::error::AppResult;

/// 坐标统一存原图像素坐标，导出时再按格式换算。
const SCHEMA_V1: &str = r#"
CREATE TABLE images (
    id         INTEGER PRIMARY KEY,
    path       TEXT    NOT NULL UNIQUE,   -- 项目内为相对路径，引用外部文件为绝对路径
    width      INTEGER NOT NULL,
    height     INTEGER NOT NULL,
    hash       TEXT    NOT NULL,
    split      TEXT    NOT NULL DEFAULT 'unset',  -- train / val / test / unset
    status     TEXT    NOT NULL DEFAULT 'todo',   -- todo / labeled / reviewed / skipped
    created_at INTEGER NOT NULL
);
CREATE INDEX idx_images_hash   ON images(hash);
CREATE INDEX idx_images_status ON images(status);

CREATE TABLE labels (
    id    INTEGER PRIMARY KEY,
    name  TEXT    NOT NULL UNIQUE,
    color TEXT    NOT NULL,
    kind  TEXT    NOT NULL DEFAULT 'object',
    sort  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE annotations (
    id         INTEGER PRIMARY KEY,
    image_id   INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    label_id   INTEGER REFERENCES labels(id) ON DELETE SET NULL,
    type       TEXT    NOT NULL,          -- bbox / polygon / mask / keypoints / tag
    data_json  TEXT    NOT NULL,
    score      REAL,
    source     TEXT    NOT NULL DEFAULT 'manual',  -- manual / model / sam / ai
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX idx_ann_image ON annotations(image_id);

CREATE TABLE keypoint_templates (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    points_json   TEXT NOT NULL,
    skeleton_json TEXT NOT NULL
);

CREATE TABLE runs (
    id          INTEGER PRIMARY KEY,
    dir         TEXT    NOT NULL,
    engine      TEXT    NOT NULL,
    task        TEXT    NOT NULL,
    status      TEXT    NOT NULL,
    config_json TEXT    NOT NULL,
    best_metric REAL,
    created_at  INTEGER NOT NULL
);
"#;

pub fn open(path: &Path) -> AppResult<Connection> {
    let conn = Connection::open(path)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    migrate(&conn)?;
    Ok(conn)
}

fn migrate(conn: &Connection) -> AppResult<()> {
    let version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if version < 1 {
        conn.execute_batch(&format!("BEGIN;{SCHEMA_V1}PRAGMA user_version = 1;COMMIT;"))?;
    }
    if version < 2 {
        // 导入批次：用于「撤销上次导入」。老数据记为批次 0。
        conn.execute_batch(
            "BEGIN;
             ALTER TABLE images ADD COLUMN batch INTEGER NOT NULL DEFAULT 0;
             CREATE INDEX idx_images_batch ON images(batch);
             PRAGMA user_version = 2;
             COMMIT;",
        )?;
    }
    if version < 3 {
        // 类别的英文提示词：文本提示自动标注（开放词汇检测模型只懂英文）用
        conn.execute_batch(
            "BEGIN;
             ALTER TABLE labels ADD COLUMN prompt TEXT NOT NULL DEFAULT '';
             PRAGMA user_version = 3;
             COMMIT;",
        )?;
    }
    Ok(())
}

pub fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
