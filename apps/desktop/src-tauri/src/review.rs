//! 审核页（只在团队协作时有）：按成员、状态、文件名等筛选图片，批量通过 / 打回，撤销刚才那一批。
//! 「成员」指这张图的标注人：标过的按 labeled_by，还没标的按分给谁（assignee）。

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use std::sync::Mutex;
use rusqlite::types::Value as Sql;
use rusqlite::{params, params_from_iter, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::annotations::Annotation;
use crate::db;
use crate::error::{msg, AppResult};
use crate::project::Project;
use crate::t;
use crate::team;

/// 撤销的有效期：界面上只给 10 秒，后端多留一点余量
const UNDO_TTL: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ReviewQuery {
    /// 标注人（成员 id，发起人是 0）；不填是全部
    pub member: Option<i64>,
    /// todo 未标注 / labeled 待审核 / reviewed 已通过 / rejected 被打回 / skipped 已跳过；空表示全部
    pub statuses: Vec<String>,
    /// 文件名包含
    pub name: String,
    /// 含这个类别的标注
    pub label_id: Option<i64>,
    /// 只看没有目标的（一个标注都没有）
    pub empty: bool,
    /// 只看曾经被打回过的
    pub rejected_before: bool,
    /// time 按提交时间（新的在前）/ name 按文件名
    pub sort: String,
    /// 只要这几张（逐张审核时取单张）；空表示不限
    pub ids: Vec<i64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReviewItem {
    pub id: i64,
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub status: String,
    pub reject_reason: String,
    pub rejected_before: bool,
    pub member_id: Option<i64>,
    pub member: Option<String>,
    /// 最近一次提交的时间（秒），没提交过是 0
    pub submitted_at: i64,
    pub annotations: Vec<Annotation>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReviewPage {
    pub total: i64,
    pub items: Vec<ReviewItem>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemberSummary {
    pub member_id: i64,
    pub name: String,
    pub role: String,
    /// 当前筛选条件下（不算成员这一条）属于他的图
    pub count: i64,
    /// 他的待审核图
    pub pending: i64,
    /// 他标的图被通过、被打回的次数（全部历史）
    pub passed: i64,
    pub rejected: i64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchResult {
    pub done: usize,
    /// 有人正在改，跳过
    pub skipped_locked: usize,
    /// 不是待审核（已经通过、被打回、未标注等），跳过
    pub skipped_not_pending: usize,
    /// 撤销用；一张都没处理时为空
    pub token: String,
    /// 处理了的图
    pub ids: Vec<i64>,
}

const OWNER: &str = "COALESCE(labeled_by, assignee)";

fn status_cond(s: &str) -> AppResult<&'static str> {
    Ok(match s {
        "todo" => "(status = 'todo' AND reject_reason = '')",
        "labeled" => "status = 'labeled'",
        "reviewed" => "status = 'reviewed'",
        "rejected" => "reject_reason != ''",
        "skipped" => "status = 'skipped'",
        _ => return Err(msg(t!("筛选条件不对"))),
    })
}

/// 拼 WHERE 条件（with_member 为 false 时不算成员，给成员列表统计用）
fn where_of(q: &ReviewQuery, with_member: bool) -> AppResult<(String, Vec<Sql>)> {
    let mut conds: Vec<String> = Vec::new();
    let mut args: Vec<Sql> = Vec::new();
    if with_member {
        if let Some(m) = q.member {
            conds.push(format!("{OWNER} = ?"));
            args.push(Sql::Integer(m));
        }
    }
    if !q.ids.is_empty() {
        let list: Vec<String> = q.ids.iter().map(|i| i.to_string()).collect();
        conds.push(format!("id IN ({})", list.join(",")));
    }
    if !q.statuses.is_empty() {
        let parts = q.statuses.iter().map(|s| status_cond(s)).collect::<AppResult<Vec<_>>>()?;
        conds.push(format!("({})", parts.join(" OR ")));
    }
    let name = q.name.trim();
    if !name.is_empty() {
        let esc = name.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_");
        conds.push("path LIKE ? ESCAPE '\\'".into());
        args.push(Sql::Text(format!("%{esc}%")));
    }
    if let Some(l) = q.label_id {
        conds.push("EXISTS (SELECT 1 FROM annotations a WHERE a.image_id = images.id AND a.label_id = ?)".into());
        args.push(Sql::Integer(l));
    }
    if q.empty {
        conds.push("NOT EXISTS (SELECT 1 FROM annotations a WHERE a.image_id = images.id)".into());
    }
    if q.rejected_before {
        conds.push("EXISTS (SELECT 1 FROM activity x WHERE x.image_id = images.id AND x.action = 'reject')".into());
    }
    let w = if conds.is_empty() { "1".to_string() } else { conds.join(" AND ") };
    Ok((w, args))
}

fn order_of(q: &ReviewQuery) -> &'static str {
    if q.sort == "name" {
        "path, id"
    } else {
        "submitted_at DESC, id"
    }
}

/// 只有审核员和发起人能用审核页
fn ensure_reviewer(m: &team::Member) -> AppResult<()> {
    if m.can_review() {
        Ok(())
    } else {
        Err(msg(t!("只有审核员能审核")))
    }
}

pub fn query(p: &Project, m: &team::Member, q: &ReviewQuery, offset: u32, limit: u32, host_name: &str) -> AppResult<ReviewPage> {
    ensure_reviewer(m)?;
    let (w, args) = where_of(q, true)?;
    let names = team::names(p, host_name)?;
    let conn = p.conn.lock();
    let total: i64 = conn.query_row(&format!("SELECT COUNT(*) FROM images WHERE {w}"), params_from_iter(args.iter()), |r| r.get(0))?;
    let sql = format!(
        "SELECT id, path, width, height, status, reject_reason, {OWNER}, submitted_at,
                EXISTS (SELECT 1 FROM activity x WHERE x.image_id = images.id AND x.action = 'reject')
         FROM images WHERE {w} ORDER BY {} LIMIT {} OFFSET {}",
        order_of(q),
        limit.clamp(1, 200),
        offset
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut items = stmt
        .query_map(params_from_iter(args.iter()), |r| {
            let path: String = r.get(1)?;
            let name = Path::new(&path).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or(path);
            let owner: Option<i64> = r.get(6)?;
            Ok(ReviewItem {
                id: r.get(0)?,
                name,
                width: r.get(2)?,
                height: r.get(3)?,
                status: r.get(4)?,
                reject_reason: r.get(5)?,
                member_id: owner,
                member: owner.and_then(|o| names.get(&o).cloned()),
                submitted_at: r.get(7)?,
                rejected_before: r.get::<_, i64>(8)? != 0,
                annotations: Vec::new(),
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    // 这一页的标注一次取回来，宫格里直接画在缩略图上
    if !items.is_empty() {
        let ids: Vec<String> = items.iter().map(|i| i.id.to_string()).collect();
        let mut stmt = conn.prepare(&format!(
            "SELECT image_id, label_id, type, data_json, score, source FROM annotations WHERE image_id IN ({}) ORDER BY id",
            ids.join(",")
        ))?;
        let mut by: HashMap<i64, Vec<Annotation>> = HashMap::new();
        let rows = stmt.query_map([], |r| {
            let data: String = r.get(3)?;
            Ok((
                r.get::<_, i64>(0)?,
                Annotation {
                    label_id: r.get(1)?,
                    kind: r.get(2)?,
                    data: serde_json::from_str(&data).unwrap_or(serde_json::Value::Null),
                    score: r.get(4)?,
                    source: r.get(5)?,
                },
            ))
        })?;
        for row in rows {
            let (id, a) = row?;
            by.entry(id).or_default().push(a);
        }
        for it in &mut items {
            it.annotations = by.remove(&it.id).unwrap_or_default();
        }
    }
    Ok(ReviewPage { total, items })
}

/// 符合条件的全部图片 id（「全选筛选结果」、逐张审核的队列、抽检用）
pub fn ids(p: &Project, m: &team::Member, q: &ReviewQuery) -> AppResult<Vec<i64>> {
    ensure_reviewer(m)?;
    let (w, args) = where_of(q, true)?;
    let conn = p.conn.lock();
    let mut stmt = conn.prepare(&format!("SELECT id FROM images WHERE {w} ORDER BY {}", order_of(q)))?;
    let v = stmt.query_map(params_from_iter(args.iter()), |r| r.get(0))?.collect::<Result<Vec<i64>, _>>()?;
    Ok(v)
}

/// 成员列表：每人在当前筛选下有几张、待审核几张、历史通过率
pub fn members(p: &Project, m: &team::Member, q: &ReviewQuery, host_name: &str) -> AppResult<Vec<MemberSummary>> {
    ensure_reviewer(m)?;
    let mut people: Vec<(i64, String, String)> = vec![(team::HOST, host_name.to_string(), "admin".to_string())];
    people.extend(team::list_members(p)?.into_iter().map(|x| (x.id, x.name, x.role)));
    let (w, args) = where_of(q, false)?;
    let conn = p.conn.lock();
    let group = |sql: &str, args: &[Sql]| -> rusqlite::Result<HashMap<i64, i64>> {
        let mut stmt = conn.prepare(sql)?;
        let v = stmt
            .query_map(params_from_iter(args.iter()), |r| Ok((r.get::<_, Option<i64>>(0)?, r.get::<_, i64>(1)?)))?
            .filter_map(|x| x.ok())
            .filter_map(|(k, n)| k.map(|k| (k, n)))
            .collect();
        Ok(v)
    };
    let count = group(&format!("SELECT {OWNER} AS o, COUNT(*) FROM images WHERE {w} GROUP BY o"), &args)?;
    let pending = group(&format!("SELECT {OWNER} AS o, COUNT(*) FROM images WHERE status = 'labeled' GROUP BY o"), &[])?;
    let passed = group("SELECT target, COUNT(*) FROM activity WHERE action = 'pass' GROUP BY target", &[])?;
    let rejected = group("SELECT target, COUNT(*) FROM activity WHERE action = 'reject' GROUP BY target", &[])?;
    let g = |m: &HashMap<i64, i64>, id: i64| m.get(&id).copied().unwrap_or(0);
    Ok(people
        .into_iter()
        // 发起人自己没标过图就不列出来；审核员也一样
        .filter(|(id, _, role)| role == "annotator" || g(&count, *id) > 0 || g(&pending, *id) > 0)
        .map(|(id, name, role)| MemberSummary {
            member_id: id,
            name,
            role,
            count: g(&count, id),
            pending: g(&pending, id),
            passed: g(&passed, id),
            rejected: g(&rejected, id),
        })
        .collect())
}

// ---------- 批量审核、撤销 ----------

struct UndoEntry {
    image_id: i64,
    status: String,
    reviewed_by: Option<i64>,
    reject_reason: String,
    /// 这次写进去的状态和打回原因：撤销时还是这样才恢复（之后又被人动过的不碰）
    new_status: String,
    new_reason: String,
    activity_id: i64,
}

struct UndoBatch {
    project: std::path::PathBuf,
    reviewer: i64,
    at: Instant,
    entries: Vec<UndoEntry>,
}

static UNDO: Mutex<Vec<(String, UndoBatch)>> = Mutex::new(Vec::new());

/// 批量通过 / 打回。只处理「待审核」的图；locked 返回 true 的（有人正在改）跳过。
/// 打回的图回到原标注员的任务里，原因所有图一样。
pub fn review_many(
    p: &Project,
    reviewer: &team::Member,
    ids: &[i64],
    pass: bool,
    reason: &str,
    locked: &dyn Fn(i64) -> bool,
) -> AppResult<BatchResult> {
    ensure_reviewer(reviewer)?;
    let reason = reason.trim();
    let reason = if pass {
        String::new()
    } else if reason.is_empty() {
        t!("需要修改")
    } else {
        reason.to_string()
    };
    let now = db::now();
    let mut entries = Vec::new();
    let (mut skipped_locked, mut skipped_not_pending) = (0, 0);
    {
        let mut conn = p.conn.lock();
        let tx = conn.transaction()?;
        for &id in ids {
            let row: Option<(String, Option<i64>, Option<i64>, String, i64)> = tx
                .query_row(
                    "SELECT status, labeled_by, reviewed_by, reject_reason, (SELECT COUNT(*) FROM annotations WHERE image_id = images.id)
                     FROM images WHERE id = ?1",
                    [id],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
                )
                .optional()?;
            let Some((status, labeled_by, reviewed_by, old_reason, objects)) = row else { continue };
            if status != "labeled" {
                skipped_not_pending += 1;
                continue;
            }
            if locked(id) {
                skipped_locked += 1;
                continue;
            }
            let new_status = if pass { "reviewed" } else { "todo" };
            tx.execute(
                "UPDATE images SET status = ?2, reviewed_by = ?3, reject_reason = ?4 WHERE id = ?1",
                params![id, new_status, reviewer.id, reason],
            )?;
            tx.execute(
                "INSERT INTO activity (member_id, image_id, action, target, objects, detail, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![reviewer.id, id, if pass { "pass" } else { "reject" }, labeled_by, objects, reason, now],
            )?;
            entries.push(UndoEntry {
                image_id: id,
                status,
                reviewed_by,
                reject_reason: old_reason,
                new_status: new_status.to_string(),
                new_reason: reason.clone(),
                activity_id: tx.last_insert_rowid(),
            });
        }
        tx.commit()?;
    }
    let done_ids: Vec<i64> = entries.iter().map(|e| e.image_id).collect();
    let token = if entries.is_empty() {
        String::new()
    } else {
        let token = team::token(12);
        let mut u = UNDO.lock().unwrap_or_else(|e| e.into_inner());
        u.retain(|(_, b)| b.at.elapsed() < UNDO_TTL);
        u.push((token.clone(), UndoBatch { project: p.root.clone(), reviewer: reviewer.id, at: Instant::now(), entries }));
        token
    };
    Ok(BatchResult { done: done_ids.len(), skipped_locked, skipped_not_pending, token, ids: done_ids })
}

/// 撤销刚才那一批：之后没再被人动过的图恢复原来的状态，操作记录也删掉。返回恢复了的图
pub fn undo(p: &Project, reviewer: &team::Member, token: &str) -> AppResult<Vec<i64>> {
    ensure_reviewer(reviewer)?;
    let batch = {
        let mut u = UNDO.lock().unwrap_or_else(|e| e.into_inner());
        u.retain(|(_, b)| b.at.elapsed() < UNDO_TTL);
        let i = u
            .iter()
            .position(|(k, b)| k == token && b.reviewer == reviewer.id && b.project == p.root)
            .ok_or_else(|| msg(t!("已经过了撤销时间")))?;
        u.remove(i).1
    };
    let mut restored = Vec::new();
    let mut conn = p.conn.lock();
    let tx = conn.transaction()?;
    for e in &batch.entries {
        let n = tx.execute(
            "UPDATE images SET status = ?2, reviewed_by = ?3, reject_reason = ?4
             WHERE id = ?1 AND status = ?5 AND reject_reason = ?6",
            params![e.image_id, e.status, e.reviewed_by, e.reject_reason, e.new_status, e.new_reason],
        )?;
        if n > 0 {
            tx.execute("DELETE FROM activity WHERE id = ?1", [e.activity_id])?;
            restored.push(e.image_id);
        }
    }
    tx.commit()?;
    Ok(restored)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_review_and_undo() {
        let base = std::env::temp_dir().join(format!("xlabel-test-review-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let p = Project::create(&base, "审核", "detect").unwrap();
        {
            let conn = p.conn.lock();
            for i in 0..6 {
                conn.execute(
                    "INSERT INTO images (path, width, height, hash, created_at, batch) VALUES (?1, 100, 100, ?2, 0, 1)",
                    params![format!("images/cam_{i}.jpg"), format!("h{i}")],
                )
                .unwrap();
            }
        }
        let a = team::add_member(&p, "张三", "annotator", "1234").unwrap();
        let r = team::add_member(&p, "王审", "reviewer", "pass").unwrap();
        let host = team::Member::host("管理员");
        team::assign(&p, &team::AssignOptions { member_ids: vec![a.id], scope: "unassigned".into(), per_person: None }).unwrap();
        let ids: Vec<i64> = team::list_ids(&p, &a, None).unwrap();
        // 前 4 张标了一个框再提交；第 5 张什么都没标也提交（负样本，也要进待审核）
        for &id in &ids[..4] {
            p.conn
                .lock()
                .execute("INSERT INTO annotations (image_id, label_id, type, data_json, source, created_at, updated_at) VALUES (?1, NULL, 'bbox', '[1,1,5,5]', 'manual', 0, 0)", [id])
                .unwrap();
            team::submit(&p, a.id, id).unwrap();
        }
        assert_eq!(team::submit(&p, a.id, ids[4]).unwrap(), "labeled", "没有目标的图提交后也是待审核");

        let pending = ReviewQuery { statuses: vec!["labeled".into()], member: Some(a.id), ..Default::default() };
        assert_eq!(ids_of(&p, &r, &pending).len(), 5);
        let page = query(&p, &r, &pending, 0, 2, "管理员").unwrap();
        assert_eq!((page.total, page.items.len()), (5, 2));
        assert_eq!(page.items[0].member.as_deref(), Some("张三"));
        let empty = ReviewQuery { empty: true, ..pending.clone() };
        assert_eq!(ids_of(&p, &r, &empty), vec![ids[4]]);
        let named = ReviewQuery { name: "cam_2".into(), ..pending.clone() };
        assert_eq!(ids_of(&p, &r, &named), vec![ids[2]]);
        assert!(query(&p, &a, &pending, 0, 10, "管理员").is_err(), "标注员不能用审核页");

        // 批量通过：第 2 张有人正在改跳过，第 6 张没提交过（不是待审核）跳过
        let res = review_many(&p, &host, &[ids[0], ids[1], ids[5]], true, "", &|id| id == ids[1]).unwrap();
        assert_eq!((res.done, res.skipped_locked, res.skipped_not_pending), (1, 1, 1));
        // 批量打回
        let rej = review_many(&p, &r, &[ids[1], ids[2]], false, "漏标", &|_| false).unwrap();
        assert_eq!(rej.done, 2);
        assert_eq!(team::list_ids(&p, &a, Some("rejected")).unwrap(), vec![ids[1], ids[2]]);
        let before = ReviewQuery { rejected_before: true, ..Default::default() };
        assert_eq!(ids_of(&p, &r, &before), vec![ids[1], ids[2]]);
        let ms = members(&p, &r, &ReviewQuery::default(), "管理员").unwrap();
        let za = ms.iter().find(|m| m.member_id == a.id).unwrap();
        assert_eq!((za.pending, za.passed, za.rejected), (2, 1, 2));

        // 别人不能撤销；撤销后恢复待审核、操作记录删掉；撤销过的不能再撤
        assert!(undo(&p, &host, &rej.token).is_err());
        let back = undo(&p, &r, &rej.token).unwrap();
        assert_eq!(back.len(), 2);
        assert!(team::list_ids(&p, &a, Some("rejected")).unwrap().is_empty());
        assert_eq!(ids_of(&p, &r, &pending).len(), 4);
        assert!(ids_of(&p, &r, &before).is_empty());
        assert!(undo(&p, &r, &rej.token).is_err());
        let _ = std::fs::remove_dir_all(&base);
    }

    fn ids_of(p: &Project, m: &team::Member, q: &ReviewQuery) -> Vec<i64> {
        let mut v = ids(p, m, q).unwrap();
        v.sort();
        v
    }

    #[test]
    fn status_filter_rejects_unknown() {
        assert!(status_cond("labeled").is_ok());
        assert!(status_cond("bogus").is_err());
    }

    #[test]
    fn where_clause_escapes_like() {
        let q = ReviewQuery { name: "a_1%".into(), statuses: vec!["labeled".into(), "rejected".into()], member: Some(3), ..Default::default() };
        let (w, args) = where_of(&q, true).unwrap();
        assert!(w.contains("COALESCE(labeled_by, assignee) = ?"));
        assert!(w.contains("status = 'labeled' OR reject_reason != ''"));
        assert_eq!(args.len(), 2);
        assert_eq!(args[1], Sql::Text("%a\\_1\\%%".into()));
        let (w2, a2) = where_of(&q, false).unwrap();
        assert!(!w2.contains("COALESCE"));
        assert_eq!(a2.len(), 1);
    }
}
