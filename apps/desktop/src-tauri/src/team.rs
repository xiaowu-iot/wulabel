//! 实时协作的数据部分：成员和密码、任务分配、提交和审核打回、操作记录、工作量统计。
//! 发起人（开主机的人）是管理员，成员 id 记为 0，不存在 members 表里。

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::db;
use crate::error::{msg, AppResult};
use crate::project::Project;

/// 发起人（管理员）的成员 id
pub const HOST: i64 = 0;
pub const ROLES: [&str; 2] = ["annotator", "reviewer"];

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Member {
    pub id: i64,
    pub name: String,
    /// annotator 标注员 / reviewer 审核员 / admin 管理员（发起人）
    pub role: String,
    pub disabled: bool,
    pub created_at: i64,
    /// 分给他的图
    pub assigned: i64,
    /// 其中已提交（已标注 / 已审核，且没被打回）的
    pub done: i64,
    /// 被打回、还没改的
    pub rejected: i64,
}

impl Member {
    pub fn host(name: &str) -> Member {
        Member { id: HOST, name: name.into(), role: "admin".into(), disabled: false, created_at: 0, assigned: 0, done: 0, rejected: 0 }
    }
    pub fn is_annotator(&self) -> bool {
        self.role == "annotator"
    }
    /// 审核员和管理员能看全部图、能审核
    pub fn can_review(&self) -> bool {
        self.role == "reviewer" || self.role == "admin"
    }
}

fn random_hex(bytes: usize) -> String {
    let mut buf = vec![0u8; bytes];
    if getrandom::getrandom(&mut buf).is_err() {
        // 取不到系统随机数时退而用时间和地址混合（几乎不会发生）
        let seed = format!("{:?}{}", std::time::SystemTime::now(), &buf as *const _ as usize);
        let h = blake3::hash(seed.as_bytes());
        buf.iter_mut().zip(h.as_bytes().iter().cycle()).for_each(|(b, s)| *b = *s);
    }
    buf.iter().map(|b| format!("{b:02x}")).collect()
}

/// 随机令牌（登录会话、共享编号等）
pub fn token(bytes: usize) -> String {
    random_hex(bytes)
}

/// 加盐、多轮的密码哈希（blake3），不存明文
fn hash(salt: &str, password: &str) -> String {
    let mut h = blake3::hash(format!("{salt}:{password}").as_bytes());
    for _ in 0..20_000 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(salt.as_bytes());
        hasher.update(h.as_bytes());
        h = hasher.finalize();
    }
    h.to_hex().to_string()
}

fn check_password(pw: &str) -> AppResult<()> {
    if pw.chars().count() < 4 {
        return Err(msg("密码至少 4 位"));
    }
    Ok(())
}

fn check_role(role: &str) -> AppResult<()> {
    if !ROLES.contains(&role) {
        return Err(msg(format!("不支持的角色：{role}")));
    }
    Ok(())
}

const MEMBER_SQL: &str = "SELECT m.id, m.name, m.role, m.disabled, m.created_at,
        (SELECT COUNT(*) FROM images i WHERE i.assignee = m.id),
        (SELECT COUNT(*) FROM images i WHERE i.assignee = m.id AND i.status IN ('labeled', 'reviewed') AND i.reject_reason = ''),
        (SELECT COUNT(*) FROM images i WHERE i.assignee = m.id AND i.reject_reason != '')
     FROM members m";

fn row_member(r: &rusqlite::Row) -> rusqlite::Result<Member> {
    Ok(Member {
        id: r.get(0)?,
        name: r.get(1)?,
        role: r.get(2)?,
        disabled: r.get::<_, i64>(3)? != 0,
        created_at: r.get(4)?,
        assigned: r.get(5)?,
        done: r.get(6)?,
        rejected: r.get(7)?,
    })
}

pub fn list_members(p: &Project) -> AppResult<Vec<Member>> {
    let conn = p.conn.lock();
    let mut stmt = conn.prepare(&format!("{MEMBER_SQL} ORDER BY m.id"))?;
    let v = stmt.query_map([], row_member)?.collect::<Result<Vec<_>, _>>()?;
    Ok(v)
}

pub fn get_member(p: &Project, id: i64) -> AppResult<Member> {
    let conn = p.conn.lock();
    conn.query_row(&format!("{MEMBER_SQL} WHERE m.id = ?1"), [id], row_member)
        .optional()?
        .ok_or_else(|| msg("成员不存在"))
}

pub fn add_member(p: &Project, name: &str, role: &str, password: &str) -> AppResult<Member> {
    let name = name.trim();
    if name.is_empty() || name.chars().count() > 20 {
        return Err(msg("名字不能为空，最多 20 个字"));
    }
    check_role(role)?;
    check_password(password)?;
    let salt = random_hex(16);
    let id = {
        let conn = p.conn.lock();
        let exists: Option<i64> = conn.query_row("SELECT id FROM members WHERE name = ?1", [name], |r| r.get(0)).optional()?;
        if exists.is_some() {
            return Err(msg(format!("已经有叫「{name}」的成员了")));
        }
        conn.execute(
            "INSERT INTO members (name, role, pass_hash, salt, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![name, role, hash(&salt, password), salt, db::now()],
        )?;
        conn.last_insert_rowid()
    };
    get_member(p, id)
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemberPatch {
    pub name: Option<String>,
    pub role: Option<String>,
    pub password: Option<String>,
    pub disabled: Option<bool>,
}

pub fn update_member(p: &Project, id: i64, patch: &MemberPatch) -> AppResult<Member> {
    get_member(p, id)?;
    {
        let conn = p.conn.lock();
        if let Some(name) = patch.name.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            let dup: Option<i64> =
                conn.query_row("SELECT id FROM members WHERE name = ?1 AND id != ?2", params![name, id], |r| r.get(0)).optional()?;
            if dup.is_some() {
                return Err(msg(format!("已经有叫「{name}」的成员了")));
            }
            conn.execute("UPDATE members SET name = ?2 WHERE id = ?1", params![id, name])?;
        }
        if let Some(role) = &patch.role {
            check_role(role)?;
            conn.execute("UPDATE members SET role = ?2 WHERE id = ?1", params![id, role])?;
        }
        if let Some(pw) = patch.password.as_deref().filter(|s| !s.is_empty()) {
            check_password(pw)?;
            let salt = random_hex(16);
            conn.execute("UPDATE members SET pass_hash = ?2, salt = ?3 WHERE id = ?1", params![id, hash(&salt, pw), salt])?;
        }
        if let Some(d) = patch.disabled {
            conn.execute("UPDATE members SET disabled = ?2 WHERE id = ?1", params![id, d as i64])?;
        }
    }
    get_member(p, id)
}

/// 删除成员：他手上没做完的图收回（变成未分配），已做的保留记录
pub fn delete_member(p: &Project, id: i64) -> AppResult<()> {
    let conn = p.conn.lock();
    conn.execute("UPDATE images SET assignee = NULL WHERE assignee = ?1", [id])?;
    conn.execute("DELETE FROM members WHERE id = ?1", [id])?;
    Ok(())
}

/// 登录：按名字和密码查成员
pub fn verify(p: &Project, name: &str, password: &str) -> AppResult<Member> {
    let row: Option<(i64, String, String, i64)> = p
        .conn
        .lock()
        .query_row(
            "SELECT id, pass_hash, salt, disabled FROM members WHERE name = ?1",
            [name.trim()],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .optional()?;
    let Some((id, pass_hash, salt, disabled)) = row else {
        return Err(msg("名字或密码不对"));
    };
    if hash(&salt, password) != pass_hash {
        return Err(msg("名字或密码不对"));
    }
    if disabled != 0 {
        return Err(msg("这个账号已被停用，请联系管理员"));
    }
    get_member(p, id)
}

// ---------- 任务分配 ----------

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssignOptions {
    pub member_ids: Vec<i64>,
    /// unassigned 还没分给人的未标注图 / todo 全部未标注（已分配的也重新分）/ unreviewed 未审核的（未标注 + 已标注）
    pub scope: String,
    /// 每人分几张；不填就把范围内的图平均分完
    pub per_person: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AssignResult {
    pub member_id: i64,
    pub name: String,
    pub count: usize,
}

pub fn assign(p: &Project, opts: &AssignOptions) -> AppResult<Vec<AssignResult>> {
    let mut people = Vec::new();
    for id in &opts.member_ids {
        let m = get_member(p, *id)?;
        if !m.disabled {
            people.push(m);
        }
    }
    if people.is_empty() {
        return Err(msg("请选择要分配的成员"));
    }
    let cond = match opts.scope.as_str() {
        "unassigned" => "status = 'todo' AND assignee IS NULL",
        "todo" => "status = 'todo'",
        "unreviewed" => "status IN ('todo', 'labeled')",
        _ => return Err(msg("分配范围不对")),
    };
    let mut conn = p.conn.lock();
    // 按 id 顺序分，视频的连续帧会落在同一个人手里
    let ids: Vec<i64> = {
        let mut stmt = conn.prepare(&format!("SELECT id FROM images WHERE {cond} ORDER BY id"))?;
        let v = stmt.query_map([], |r| r.get(0))?.collect::<Result<Vec<i64>, _>>()?;
        v
    };
    if ids.is_empty() {
        return Err(msg("这个范围里没有可以分配的图"));
    }
    let k = people.len();
    let total = match opts.per_person {
        Some(n) if n > 0 => (n * k).min(ids.len()),
        _ => ids.len(),
    };
    let tx = conn.transaction()?;
    let mut out = Vec::new();
    {
        let mut up = tx.prepare("UPDATE images SET assignee = ?2 WHERE id = ?1")?;
        for (i, m) in people.iter().enumerate() {
            let chunk = &ids[i * total / k..(i + 1) * total / k];
            for id in chunk {
                up.execute(params![id, m.id])?;
            }
            out.push(AssignResult { member_id: m.id, name: m.name.clone(), count: chunk.len() });
        }
    }
    tx.commit()?;
    Ok(out)
}

/// 收回分配：only_unfinished 时只收回还没提交的（和被打回的）
pub fn unassign(p: &Project, member_id: i64, only_unfinished: bool) -> AppResult<usize> {
    let sql = if only_unfinished {
        "UPDATE images SET assignee = NULL WHERE assignee = ?1 AND (status = 'todo' OR reject_reason != '')"
    } else {
        "UPDATE images SET assignee = NULL WHERE assignee = ?1"
    };
    Ok(p.conn.lock().execute(sql, [member_id])?)
}

/// 标注员只能碰分给自己的图；审核员、管理员都可以
pub fn can_access(p: &Project, m: &Member, image_id: i64) -> AppResult<bool> {
    if !m.is_annotator() {
        return Ok(true);
    }
    let a: Option<Option<i64>> =
        p.conn.lock().query_row("SELECT assignee FROM images WHERE id = ?1", [image_id], |r| r.get(0)).optional()?;
    Ok(a.flatten() == Some(m.id))
}

pub fn ensure_access(p: &Project, m: &Member, image_id: i64) -> AppResult<()> {
    if can_access(p, m, image_id)? {
        Ok(())
    } else {
        Err(msg("这张图没有分给你"))
    }
}

/// 标注页翻图用的图片列表。标注员只看分给自己的图。
/// filter：all / todo / labeled / reviewed / skipped / rejected（被打回）/ mine（审核员：自己审过的）
pub fn list_ids(p: &Project, m: &Member, filter: Option<&str>) -> AppResult<Vec<i64>> {
    let filter = filter.filter(|s| !s.is_empty()).unwrap_or("all");
    let mut conds: Vec<String> = Vec::new();
    if m.is_annotator() {
        conds.push(format!("assignee = {}", m.id));
    }
    match filter {
        "all" => {}
        "rejected" => conds.push("reject_reason != ''".into()),
        "todo" | "labeled" | "reviewed" | "skipped" => conds.push(format!("status = '{filter}'")),
        _ => return Err(msg("筛选条件不对")),
    }
    let sql = if conds.is_empty() {
        "SELECT id FROM images ORDER BY id".to_string()
    } else {
        format!("SELECT id FROM images WHERE {} ORDER BY id", conds.join(" AND "))
    };
    let conn = p.conn.lock();
    let mut stmt = conn.prepare(&sql)?;
    let v = stmt.query_map([], |r| r.get(0))?.collect::<Result<Vec<i64>, _>>()?;
    Ok(v)
}

// ---------- 提交、审核、记录 ----------

pub fn log(p: &Project, member_id: i64, image_id: Option<i64>, action: &str, target: Option<i64>, objects: i64, detail: &str) -> AppResult<()> {
    p.conn.lock().execute(
        "INSERT INTO activity (member_id, image_id, action, target, objects, detail, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![member_id, image_id, action, target, objects, detail, db::now()],
    )?;
    Ok(())
}

fn object_count(p: &Project, image_id: i64) -> AppResult<i64> {
    Ok(p.conn.lock().query_row("SELECT COUNT(*) FROM annotations WHERE image_id = ?1", [image_id], |r| r.get(0))?)
}

/// 保存标注之后调用：记下是谁标的
pub fn after_save(p: &Project, member_id: i64, image_id: i64) -> AppResult<()> {
    let n = object_count(p, image_id)?;
    p.conn.lock().execute("UPDATE images SET labeled_by = ?2 WHERE id = ?1", params![image_id, member_id])?;
    log(p, member_id, Some(image_id), "save", None, n, "")
}

/// 标注员提交（标注页的「提交并下一张」）：变成已标注，打回原因清掉
pub fn submit(p: &Project, member_id: i64, image_id: i64) -> AppResult<String> {
    let n = object_count(p, image_id)?;
    let status = if n > 0 { "labeled" } else { "todo" };
    p.conn.lock().execute(
        "UPDATE images SET status = CASE WHEN status = 'reviewed' THEN status ELSE ?2 END, reject_reason = '', labeled_by = ?3 WHERE id = ?1",
        params![image_id, status, member_id],
    )?;
    log(p, member_id, Some(image_id), "submit", None, n, "")?;
    Ok(p.conn.lock().query_row("SELECT status FROM images WHERE id = ?1", [image_id], |r| r.get(0))?)
}

/// 审核：通过变成已审核；打回退回未标注，记下原因，回到原标注员的任务里
pub fn review(p: &Project, reviewer: i64, image_id: i64, pass: bool, reason: &str) -> AppResult<String> {
    let labeled_by: Option<i64> = p
        .conn
        .lock()
        .query_row("SELECT labeled_by FROM images WHERE id = ?1", [image_id], |r| r.get(0))
        .optional()?
        .ok_or_else(|| msg("图片不存在"))?;
    let n = object_count(p, image_id)?;
    if pass {
        p.conn.lock().execute(
            "UPDATE images SET status = 'reviewed', reviewed_by = ?2, reject_reason = '' WHERE id = ?1",
            params![image_id, reviewer],
        )?;
        log(p, reviewer, Some(image_id), "pass", labeled_by, n, "")?;
        Ok("reviewed".into())
    } else {
        let reason = reason.trim();
        let reason = if reason.is_empty() { "需要修改" } else { reason };
        p.conn.lock().execute(
            "UPDATE images SET status = 'todo', reviewed_by = ?2, reject_reason = ?3 WHERE id = ?1",
            params![image_id, reviewer, reason],
        )?;
        log(p, reviewer, Some(image_id), "reject", labeled_by, n, reason)?;
        Ok("todo".into())
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageTeamInfo {
    pub assignee: Option<String>,
    pub labeled_by: Option<String>,
    pub reviewed_by: Option<String>,
    pub reject_reason: String,
}

fn names(p: &Project, host_name: &str) -> AppResult<HashMap<i64, String>> {
    let conn = p.conn.lock();
    let mut stmt = conn.prepare("SELECT id, name FROM members")?;
    let mut map: HashMap<i64, String> = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_, _>>()?;
    map.insert(HOST, host_name.to_string());
    Ok(map)
}

pub fn image_info(p: &Project, image_id: i64, host_name: &str) -> AppResult<ImageTeamInfo> {
    let (a, l, r, reason): (Option<i64>, Option<i64>, Option<i64>, String) = p
        .conn
        .lock()
        .query_row("SELECT assignee, labeled_by, reviewed_by, reject_reason FROM images WHERE id = ?1", [image_id], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
        })
        .optional()?
        .ok_or_else(|| msg("图片不存在"))?;
    let n = names(p, host_name)?;
    let name = |id: Option<i64>| id.and_then(|i| n.get(&i).cloned());
    Ok(ImageTeamInfo { assignee: name(a), labeled_by: name(l), reviewed_by: name(r), reject_reason: reason })
}

// ---------- 统计 ----------

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemberStats {
    pub member_id: i64,
    pub name: String,
    pub role: String,
    /// 当前分给他的图
    pub assigned: i64,
    /// 时间段内提交的图（去重）
    pub submitted: i64,
    /// 这些图现在的标注对象数
    pub objects: i64,
    /// 他标的图被审核通过 / 打回的次数
    pub passed: i64,
    pub rejected: i64,
    /// 他审核过的次数（审核员）
    pub reviews: i64,
    /// 估算的工作时长（秒）：相邻两次操作间隔不超过 5 分钟的累加
    pub active_secs: i64,
}

pub fn stats(p: &Project, from: i64, to: i64, host_name: &str) -> AppResult<Vec<MemberStats>> {
    let mut people: Vec<(i64, String, String)> = vec![(HOST, host_name.to_string(), "admin".to_string())];
    people.extend(list_members(p)?.into_iter().map(|m| (m.id, m.name, m.role)));
    let conn = p.conn.lock();
    let mut out = Vec::new();
    for (id, name, role) in people {
        let one = |sql: &str| -> rusqlite::Result<i64> { conn.query_row(sql, params![id, from, to], |r| r.get(0)) };
        let assigned = conn.query_row("SELECT COUNT(*) FROM images WHERE assignee = ?1", [id], |r| r.get(0))?;
        let submitted = one(
            "SELECT COUNT(DISTINCT image_id) FROM activity WHERE member_id = ?1 AND action = 'submit' AND created_at BETWEEN ?2 AND ?3",
        )?;
        let objects = one(
            "SELECT COUNT(*) FROM annotations WHERE image_id IN (
                 SELECT DISTINCT image_id FROM activity WHERE member_id = ?1 AND action = 'submit' AND created_at BETWEEN ?2 AND ?3)",
        )?;
        let passed = one("SELECT COUNT(*) FROM activity WHERE target = ?1 AND action = 'pass' AND created_at BETWEEN ?2 AND ?3")?;
        let rejected = one("SELECT COUNT(*) FROM activity WHERE target = ?1 AND action = 'reject' AND created_at BETWEEN ?2 AND ?3")?;
        let reviews =
            one("SELECT COUNT(*) FROM activity WHERE member_id = ?1 AND action IN ('pass', 'reject') AND created_at BETWEEN ?2 AND ?3")?;
        let times: Vec<i64> = {
            let mut stmt =
                conn.prepare("SELECT created_at FROM activity WHERE member_id = ?1 AND created_at BETWEEN ?2 AND ?3 ORDER BY created_at")?;
            let v = stmt.query_map(params![id, from, to], |r| r.get(0))?.collect::<Result<Vec<i64>, _>>()?;
            v
        };
        let active_secs: i64 = times.windows(2).map(|w| w[1] - w[0]).filter(|d| *d <= 300).sum();
        // 没参与的管理员不显示
        if id == HOST && submitted == 0 && reviews == 0 && assigned == 0 {
            continue;
        }
        out.push(MemberStats { member_id: id, name, role, assigned, submitted, objects, passed, rejected, reviews, active_secs });
    }
    Ok(out)
}

/// 导出统计为 CSV（Excel 能直接打开，带 BOM）
pub fn export_csv(p: &Project, from: i64, to: i64, host_name: &str, path: &Path) -> AppResult<usize> {
    let rows = stats(p, from, to, host_name)?;
    let role_name = |r: &str| match r {
        "annotator" => "标注员",
        "reviewer" => "审核员",
        _ => "管理员",
    };
    let mut s = String::from("\u{feff}成员,角色,当前分配,提交张数,标注对象数,被通过,被打回,通过率,审核次数,工作时长(小时)\n");
    for r in &rows {
        let rate = if r.passed + r.rejected > 0 { format!("{:.1}%", r.passed as f64 * 100.0 / (r.passed + r.rejected) as f64) } else { "-".into() };
        s.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{:.2}\n",
            r.name.replace(',', " "),
            role_name(&r.role),
            r.assigned,
            r.submitted,
            r.objects,
            r.passed,
            r.rejected,
            rate,
            r.reviews,
            r.active_secs as f64 / 3600.0
        ));
    }
    fs::write(path, s)?;
    Ok(rows.len())
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ActivityRow {
    pub id: i64,
    pub time: i64,
    pub member: String,
    pub image_id: Option<i64>,
    pub action: String,
    pub objects: i64,
    pub detail: String,
}

/// 最近的操作记录（自动保存太多，默认不列出）
pub fn activity(p: &Project, limit: usize, member_id: Option<i64>, with_saves: bool, host_name: &str) -> AppResult<Vec<ActivityRow>> {
    let n = names(p, host_name)?;
    let conn = p.conn.lock();
    let mut stmt = conn.prepare(
        "SELECT id, created_at, member_id, image_id, action, objects, detail FROM activity
         WHERE (?1 IS NULL OR member_id = ?1) AND (?2 = 1 OR action != 'save')
         ORDER BY id DESC LIMIT ?3",
    )?;
    let v = stmt
        .query_map(params![member_id, with_saves as i64, limit as i64], |r| {
            let mid: i64 = r.get(2)?;
            Ok(ActivityRow {
                id: r.get(0)?,
                time: r.get(1)?,
                member: n.get(&mid).cloned().unwrap_or_else(|| format!("已删除成员 #{mid}")),
                image_id: r.get(3)?,
                action: r.get(4)?,
                objects: r.get(5)?,
                detail: r.get(6)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(v)
}

// ---------- 项目设置（键值） ----------

pub fn get_setting(p: &Project, key: &str) -> AppResult<Option<String>> {
    Ok(p.conn.lock().query_row("SELECT value FROM settings WHERE key = ?1", [key], |r| r.get(0)).optional()?)
}

pub fn set_setting(p: &Project, key: &str, value: &str) -> AppResult<()> {
    p.conn.lock().execute(
        "INSERT INTO settings (key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![key, value],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn members_assign_review() {
        let base = std::env::temp_dir().join(format!("xlabel-test-team-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let p = Project::create(&base, "协作", "detect").unwrap();
        {
            let conn = p.conn.lock();
            for i in 0..10 {
                conn.execute(
                    "INSERT INTO images (path, width, height, hash, created_at, batch) VALUES (?1, 100, 100, ?2, 0, 1)",
                    params![format!("images/{i}.jpg"), format!("h{i}")],
                )
                .unwrap();
            }
        }
        let a = add_member(&p, "张三", "annotator", "1234").unwrap();
        let b = add_member(&p, "李四", "annotator", "abcd").unwrap();
        let r = add_member(&p, "王审", "reviewer", "pass").unwrap();
        assert!(add_member(&p, "张三", "annotator", "1234").is_err(), "重名");
        assert!(verify(&p, "张三", "wrong").is_err());
        assert_eq!(verify(&p, "张三", "1234").unwrap().id, a.id);

        let res = assign(&p, &AssignOptions { member_ids: vec![a.id, b.id], scope: "unassigned".into(), per_person: None }).unwrap();
        assert_eq!(res.iter().map(|x| x.count).sum::<usize>(), 10);
        let a = get_member(&p, a.id).unwrap();
        let mine = list_ids(&p, &a, None).unwrap();
        assert_eq!(mine.len(), 5);
        assert!(can_access(&p, &a, mine[0]).unwrap());
        let others = list_ids(&p, &get_member(&p, b.id).unwrap(), None).unwrap();
        assert!(!can_access(&p, &a, others[0]).unwrap());
        assert_eq!(list_ids(&p, &r, None).unwrap().len(), 10, "审核员看全部");

        // 标注、提交、打回、再提交、通过
        p.conn.lock().execute("INSERT INTO annotations (image_id, label_id, type, data_json, source, created_at, updated_at) VALUES (?1, NULL, 'bbox', '[1,1,5,5]', 'manual', 0, 0)", [mine[0]]).unwrap();
        after_save(&p, a.id, mine[0]).unwrap();
        assert_eq!(submit(&p, a.id, mine[0]).unwrap(), "labeled");
        assert_eq!(review(&p, r.id, mine[0], false, "漏标").unwrap(), "todo");
        assert_eq!(list_ids(&p, &a, Some("rejected")).unwrap(), vec![mine[0]]);
        assert_eq!(image_info(&p, mine[0], "管理员").unwrap().reject_reason, "漏标");
        submit(&p, a.id, mine[0]).unwrap();
        assert_eq!(review(&p, r.id, mine[0], true, "").unwrap(), "reviewed");

        let st = stats(&p, 0, i64::MAX, "管理员").unwrap();
        let sa = st.iter().find(|s| s.member_id == a.id).unwrap();
        assert_eq!((sa.submitted, sa.passed, sa.rejected, sa.objects), (1, 1, 1, 1));
        assert_eq!(st.iter().find(|s| s.member_id == r.id).unwrap().reviews, 2);

        assert_eq!(unassign(&p, b.id, true).unwrap(), 5);
        delete_member(&p, a.id).unwrap();
        assert!(get_member(&p, a.id).is_err());
        let _ = fs::remove_dir_all(&base);
    }
}
