use std::path::Path;

use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::db;
use crate::error::{msg, AppResult};
use crate::project::Project;

const STATUSES: [&str; 4] = ["todo", "labeled", "reviewed", "skipped"];
const TYPES: [&str; 5] = ["bbox", "polygon", "mask", "keypoints", "tag"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Annotation {
    pub label_id: Option<i64>,
    #[serde(rename = "type")]
    pub kind: String,
    /// bbox: [x, y, w, h]，原图像素坐标
    pub data: serde_json::Value,
    #[serde(default)]
    pub score: Option<f64>,
    #[serde(default = "manual")]
    pub source: String,
}

fn manual() -> String {
    "manual".into()
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageAnnotations {
    pub id: i64,
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub status: String,
    pub annotations: Vec<Annotation>,
}

fn validate(a: &Annotation, w: u32, h: u32) -> AppResult<()> {
    if !TYPES.contains(&a.kind.as_str()) {
        return Err(msg(format!("不支持的标注类型：{}", a.kind)));
    }
    if a.kind == "bbox" {
        let v: Vec<f64> = serde_json::from_value(a.data.clone()).map_err(|_| msg("矩形框数据格式不对"))?;
        if v.len() != 4 || v.iter().any(|x| !x.is_finite()) {
            return Err(msg("矩形框数据格式不对"));
        }
        let (x, y, bw, bh) = (v[0], v[1], v[2], v[3]);
        // 允许 1 像素的误差
        if bw <= 0.0 || bh <= 0.0 || x < -1.0 || y < -1.0 || x + bw > w as f64 + 1.0 || y + bh > h as f64 + 1.0 {
            return Err(msg("矩形框超出了图片范围"));
        }
    }
    if a.kind == "polygon" {
        // 分割多边形：[x1, y1, x2, y2, ...]，至少 3 个点，原图像素坐标
        let v: Vec<f64> = serde_json::from_value(a.data.clone()).map_err(|_| msg("多边形数据格式不对"))?;
        if v.len() < 6 || v.len() % 2 != 0 || v.iter().any(|x| !x.is_finite()) {
            return Err(msg("多边形至少要有 3 个点"));
        }
        for pt in v.chunks(2) {
            if pt[0] < -1.0 || pt[1] < -1.0 || pt[0] > w as f64 + 1.0 || pt[1] > h as f64 + 1.0 {
                return Err(msg("多边形超出了图片范围"));
            }
        }
    }
    if a.kind == "tag" {
        // 图像分类：整张图一个类别标签，类别就是 label_id，data 不用
        if a.label_id.is_none() {
            return Err(msg("分类标签缺少类别"));
        }
    }
    if a.kind == "keypoints" {
        // {"box": [x, y, w, h], "kpts": [x, y, v, ...]}
        let (b, k) = crate::keypoints::parse(&a.data).ok_or_else(|| msg("关键点数据格式不对"))?;
        if b[2] <= 0.0 || b[3] <= 0.0 || b[0] < -1.0 || b[1] < -1.0 || b[0] + b[2] > w as f64 + 1.0 || b[1] + b[3] > h as f64 + 1.0 {
            return Err(msg("关键点目标的框超出了图片范围"));
        }
        for pt in k.chunks(3) {
            let v = pt[2];
            if v != 0.0 && v != 1.0 && v != 2.0 {
                return Err(msg("关键点可见性只能是 0、1、2"));
            }
            if v > 0.0 && (pt[0] < -1.0 || pt[1] < -1.0 || pt[0] > w as f64 + 1.0 || pt[1] > h as f64 + 1.0) {
                return Err(msg("关键点超出了图片范围"));
            }
        }
    }
    Ok(())
}

pub fn load(p: &Project, image_id: i64) -> AppResult<ImageAnnotations> {
    let conn = p.conn.lock();
    let (path, width, height, status): (String, u32, u32, String) = conn
        .query_row("SELECT path, width, height, status FROM images WHERE id = ?1", [image_id], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
        })
        .optional()?
        .ok_or_else(|| msg(format!("图片不存在：{image_id}")))?;
    let mut stmt =
        conn.prepare("SELECT label_id, type, data_json, score, source FROM annotations WHERE image_id = ?1 ORDER BY id")?;
    let annotations = stmt
        .query_map([image_id], |r| {
            let data: String = r.get(2)?;
            Ok(Annotation {
                label_id: r.get(0)?,
                kind: r.get(1)?,
                data: serde_json::from_str(&data).unwrap_or(serde_json::Value::Null),
                score: r.get(3)?,
                source: r.get(4)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    let name = Path::new(&path).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or(path);
    Ok(ImageAnnotations { id: image_id, name, width, height, status, annotations })
}

/// 整图保存：替换这张图的全部标注，并按有无标注更新状态。返回新状态。
pub fn save(p: &Project, image_id: i64, anns: &[Annotation]) -> AppResult<String> {
    let mut conn = p.conn.lock();
    let (w, h): (u32, u32) = conn
        .query_row("SELECT width, height FROM images WHERE id = ?1", [image_id], |r| Ok((r.get(0)?, r.get(1)?)))
        .optional()?
        .ok_or_else(|| msg(format!("图片不存在：{image_id}")))?;
    for a in anns {
        validate(a, w, h)?;
    }
    let tx = conn.transaction()?;
    tx.execute("DELETE FROM annotations WHERE image_id = ?1", [image_id])?;
    {
        let now = db::now();
        let mut ins = tx.prepare(
            "INSERT INTO annotations (image_id, label_id, type, data_json, score, source, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)",
        )?;
        for a in anns {
            ins.execute(params![image_id, a.label_id, a.kind, a.data.to_string(), a.score, a.source, now])?;
        }
    }
    // 有标注：未标注 → 已标注；标注删光：已标注 → 未标注。已审核、已跳过不自动改。
    tx.execute(
        "UPDATE images SET status = CASE
            WHEN ?2 > 0 AND status = 'todo' THEN 'labeled'
            WHEN ?2 = 0 AND status = 'labeled' THEN 'todo'
            ELSE status END
         WHERE id = ?1",
        params![image_id, anns.len() as i64],
    )?;
    let status: String = tx.query_row("SELECT status FROM images WHERE id = ?1", [image_id], |r| r.get(0))?;
    tx.commit()?;
    Ok(status)
}

/// 给一张图追加一个框（体检里「加为标注」用），返回新状态
pub fn append(p: &Project, image_id: i64, ann: Annotation) -> AppResult<String> {
    let mut all = load(p, image_id)?.annotations;
    all.push(ann);
    save(p, image_id, &all)
}

/// 保存模型预标注：替换这张图上模型来源的框，不改图片状态（仍等人工确认）
pub fn save_predictions(p: &Project, image_id: i64, anns: &[Annotation]) -> AppResult<()> {
    let mut conn = p.conn.lock();
    let tx = conn.transaction()?;
    tx.execute("DELETE FROM annotations WHERE image_id = ?1 AND source = 'model'", [image_id])?;
    {
        let now = db::now();
        let mut ins = tx.prepare(
            "INSERT INTO annotations (image_id, label_id, type, data_json, score, source, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)",
        )?;
        for a in anns {
            ins.execute(params![image_id, a.label_id, a.kind, a.data.to_string(), a.score, a.source, now])?;
        }
    }
    tx.commit()?;
    Ok(())
}

pub fn set_status(p: &Project, image_id: i64, status: &str) -> AppResult<()> {
    if !STATUSES.contains(&status) {
        return Err(msg(format!("不支持的状态：{status}")));
    }
    let n = p.conn.lock().execute("UPDATE images SET status = ?2 WHERE id = ?1", params![image_id, status])?;
    if n == 0 {
        return Err(msg(format!("图片不存在：{image_id}")));
    }
    Ok(())
}

/// 标注页翻图用：按筛选条件返回全部图片 id
pub fn list_ids(p: &Project, status: Option<&str>) -> AppResult<Vec<i64>> {
    let status = status.filter(|s| !s.is_empty() && *s != "all");
    let conn = p.conn.lock();
    let mut stmt = conn.prepare("SELECT id FROM images WHERE (?1 IS NULL OR status = ?1) ORDER BY id")?;
    let ids = stmt.query_map([status], |r| r.get(0))?.collect::<Result<Vec<i64>, _>>()?;
    Ok(ids)
}

// ---------- 类别 ----------

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Label {
    pub id: i64,
    pub name: String,
    pub color: String,
    /// 使用这个类别的标注数量
    pub count: i64,
    /// 英文提示词（文本提示自动标注用），可为空
    pub prompt: String,
    /// 识别说明（巡瞳 AI 按类别识别时用，告诉 AI 这个类别具体指什么），可为空
    pub description: String,
}

fn check_name(name: &str) -> AppResult<String> {
    let name = name.trim();
    if name.is_empty() {
        return Err(msg("类别名不能为空"));
    }
    if name.chars().count() > 40 {
        return Err(msg("类别名不能超过 40 个字"));
    }
    Ok(name.to_string())
}

fn map_unique(e: rusqlite::Error, name: &str) -> crate::error::AppError {
    match e {
        rusqlite::Error::SqliteFailure(f, _) if f.code == rusqlite::ErrorCode::ConstraintViolation => {
            msg(format!("类别「{name}」已存在"))
        }
        e => e.into(),
    }
}

pub fn list_labels(p: &Project) -> AppResult<Vec<Label>> {
    let conn = p.conn.lock();
    let mut stmt = conn.prepare(
        "SELECT l.id, l.name, l.color, (SELECT COUNT(*) FROM annotations a WHERE a.label_id = l.id), l.prompt, l.description
         FROM labels l ORDER BY l.sort, l.id",
    )?;
    let rows = stmt
        .query_map([], |r| {
            Ok(Label { id: r.get(0)?, name: r.get(1)?, color: r.get(2)?, count: r.get(3)?, prompt: r.get(4)?, description: r.get(5)? })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

pub fn add_label(p: &Project, name: &str, color: &str) -> AppResult<Label> {
    let name = check_name(name)?;
    let conn = p.conn.lock();
    let sort: i64 = conn.query_row("SELECT COALESCE(MAX(sort), 0) + 1 FROM labels", [], |r| r.get(0))?;
    conn.execute("INSERT INTO labels (name, color, sort) VALUES (?1, ?2, ?3)", params![name, color, sort])
        .map_err(|e| map_unique(e, &name))?;
    Ok(Label { id: conn.last_insert_rowid(), name, color: color.to_string(), count: 0, prompt: String::new(), description: String::new() })
}

/// 改类别名、颜色；prompt 为 None 时不改英文提示词
pub fn update_label(p: &Project, id: i64, name: &str, color: &str, prompt: Option<&str>) -> AppResult<()> {
    let name = check_name(name)?;
    let conn = p.conn.lock();
    let n = conn
        .execute("UPDATE labels SET name = ?2, color = ?3 WHERE id = ?1", params![id, name, color])
        .map_err(|e| map_unique(e, &name))?;
    if n == 0 {
        return Err(msg("类别不存在"));
    }
    if let Some(pr) = prompt {
        let pr: String = pr.trim().chars().take(120).collect();
        conn.execute("UPDATE labels SET prompt = ?2 WHERE id = ?1", params![id, pr])?;
    }
    Ok(())
}

/// 改识别说明（最多 500 字）
pub fn set_label_description(p: &Project, id: i64, description: &str) -> AppResult<()> {
    let d: String = description.trim().chars().take(500).collect();
    let n = p.conn.lock().execute("UPDATE labels SET description = ?2 WHERE id = ?1", params![id, d])?;
    if n == 0 {
        return Err(msg("类别不存在"));
    }
    Ok(())
}

/// 删除类别和它的全部标注，返回删掉的标注数。删完没有标注的图退回「未标注」。
pub fn delete_label(p: &Project, id: i64) -> AppResult<usize> {
    let mut conn = p.conn.lock();
    let tx = conn.transaction()?;
    let removed = tx.execute("DELETE FROM annotations WHERE label_id = ?1", [id])?;
    tx.execute("DELETE FROM labels WHERE id = ?1", [id])?;
    tx.execute(
        "UPDATE images SET status = 'todo'
         WHERE status = 'labeled' AND NOT EXISTS (SELECT 1 FROM annotations a WHERE a.image_id = images.id)",
        [],
    )?;
    tx.commit()?;
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::images;
    use image::{Rgb, RgbImage};
    use std::fs;

    fn setup(tag: &str) -> (std::path::PathBuf, Project) {
        let base = std::env::temp_dir().join(format!("xlabel-ann-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        for (i, c) in [10u8, 200].iter().enumerate() {
            RgbImage::from_pixel(100, 50, Rgb([*c, 0, 0])).save(base.join(format!("{i}.png"))).unwrap();
        }
        let p = Project::create(&base, "p", "detect").unwrap();
        let files = images::collect(&[base.join("0.png"), base.join("1.png")]);
        images::import(&p, &files, true, &|_, _| {}).unwrap();
        (base, p)
    }

    fn bbox(label: i64, x: f64, y: f64, w: f64, h: f64) -> Annotation {
        Annotation {
            label_id: Some(label),
            kind: "bbox".into(),
            data: serde_json::json!([x, y, w, h]),
            score: None,
            source: "manual".into(),
        }
    }

    #[test]
    fn save_load_and_status() {
        let (base, p) = setup("save");
        let car = add_label(&p, "车", "#3d8bfd").unwrap();
        let anns = vec![bbox(car.id, 1.0, 2.0, 30.0, 20.0), bbox(car.id, 50.0, 10.0, 50.0, 40.0)];
        assert_eq!(save(&p, 1, &anns).unwrap(), "labeled");
        let loaded = load(&p, 1).unwrap();
        assert_eq!(loaded.annotations, anns);
        assert_eq!((loaded.width, loaded.height), (100, 50));

        // 删光标注退回未标注
        assert_eq!(save(&p, 1, &[]).unwrap(), "todo");
        // 已审核的图删光标注仍保持已审核（负样本）
        set_status(&p, 1, "reviewed").unwrap();
        assert_eq!(save(&p, 1, &[]).unwrap(), "reviewed");

        assert_eq!(list_ids(&p, Some("reviewed")).unwrap(), vec![1]);
        assert_eq!(list_ids(&p, None).unwrap(), vec![1, 2]);
        assert!(set_status(&p, 1, "done").is_err());
        drop(p);
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn rejects_bad_boxes() {
        let (base, p) = setup("bad");
        assert!(save(&p, 1, &[bbox(1, 90.0, 0.0, 20.0, 10.0)]).is_err(), "超出右边界");
        assert!(save(&p, 1, &[bbox(1, 0.0, 0.0, 0.0, 10.0)]).is_err(), "宽为 0");
        let mut bad = bbox(1, 0.0, 0.0, 5.0, 5.0);
        bad.data = serde_json::json!([1, 2, 3]);
        assert!(save(&p, 1, &[bad]).is_err());
        assert!(save(&p, 99, &[]).is_err(), "图片不存在");
        drop(p);
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn label_crud() {
        let (base, p) = setup("label");
        let a = add_label(&p, " 安全帽 ", "#3d8bfd").unwrap();
        assert_eq!(a.name, "安全帽");
        assert!(add_label(&p, "安全帽", "#fff").is_err(), "重名");
        let b = add_label(&p, "人", "#2fb67c").unwrap();
        update_label(&p, b.id, "行人", "#000000", Some("pedestrian")).unwrap();
        assert!(update_label(&p, b.id, "安全帽", "#000", None).is_err(), "改成重名");

        save(&p, 1, &[bbox(a.id, 0.0, 0.0, 10.0, 10.0)]).unwrap();
        save(&p, 2, &[bbox(a.id, 0.0, 0.0, 10.0, 10.0), bbox(b.id, 5.0, 5.0, 10.0, 10.0)]).unwrap();
        let labels = list_labels(&p).unwrap();
        assert_eq!(labels.iter().map(|l| l.count).collect::<Vec<_>>(), vec![2, 1]);

        // 删「安全帽」：图 1 没标注了退回未标注，图 2 还剩「行人」
        assert_eq!(delete_label(&p, a.id).unwrap(), 2);
        assert_eq!(load(&p, 1).unwrap().status, "todo");
        assert_eq!(load(&p, 2).unwrap().annotations.len(), 1);
        assert_eq!(list_labels(&p).unwrap().len(), 1);
        drop(p);
        let _ = fs::remove_dir_all(base);
    }
}
