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
    if version < 4 {
        // 数据集版本：可训练图片和标注的快照。标注按类别名保存，之后改名、删类别都不影响已有版本
        conn.execute_batch(
            "BEGIN;
             CREATE TABLE dataset_versions (
                 id         INTEGER PRIMARY KEY,
                 name       TEXT    NOT NULL,
                 note       TEXT    NOT NULL DEFAULT '',
                 created_at INTEGER NOT NULL,
                 classes_json TEXT  NOT NULL      -- 快照时的类别名（按顺序）
             );
             CREATE TABLE version_images (
                 version_id INTEGER NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
                 image_id   INTEGER NOT NULL,
                 path       TEXT    NOT NULL,
                 width      INTEGER NOT NULL,
                 height     INTEGER NOT NULL,
                 hash       TEXT    NOT NULL,
                 split      TEXT    NOT NULL,
                 status     TEXT    NOT NULL,
                 PRIMARY KEY (version_id, image_id)
             );
             CREATE TABLE version_annotations (
                 version_id INTEGER NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
                 image_id   INTEGER NOT NULL,
                 label_name TEXT,
                 type       TEXT    NOT NULL,
                 data_json  TEXT    NOT NULL
             );
             CREATE INDEX idx_va_version ON version_annotations(version_id, image_id);
             PRAGMA user_version = 4;
             COMMIT;",
        )?;
    }
    if version < 5 {
        // 多人协作：分出去的标注包
        conn.execute_batch(
            "BEGIN;
             CREATE TABLE packages (
                 id         INTEGER PRIMARY KEY,
                 name       TEXT    NOT NULL,
                 assignee   TEXT    NOT NULL DEFAULT '',
                 dir        TEXT    NOT NULL,
                 images     INTEGER NOT NULL,
                 created_at INTEGER NOT NULL,
                 merged_at  INTEGER
             );
             PRAGMA user_version = 5;
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
