//! 实时协作主机：发起人的 Wulabel 开一个 HTTP + WebSocket 服务，其他人用浏览器（或 Wulabel）连进来一起标。
//!
//! - 界面：同一套前端页面，浏览器打开主机地址就是登录页，登录后进标注页。
//! - 接口：POST /api/login、POST /api/call/{命令}（带 Authorization: Bearer 令牌）、
//!   GET /api/image/{id}、/api/thumb/{id}（?t=令牌）、GET /api/ws（?t=令牌，实时推送）。
//! - 权限：标注员只能看、改分给自己的图；审核员能看全部、能审核；删类别、训练、导出等只在主机上做。
//! - 按张加锁：一张图同一时间只有一个人能改，别人打开是只读，并实时看到改动。
//! - 在线：局域网直接连；外网通过中转服务器（见 tunnel.rs）。

use std::collections::HashMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path as AxPath, Query, State as AxState};
use axum::http::{header, HeaderMap, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::{SinkExt, Stream, StreamExt};
use parking_lot::Mutex;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager};
use tokio::sync::{broadcast, mpsc, oneshot, watch};

use crate::annotations::{self, Annotation};
use crate::assist;
use crate::error::{msg, AppResult};
use crate::keypoints;
use crate::project::Project;
use crate::protocol;
use crate::runtime::Runtime;
use crate::team::{self, Member, HOST};
use crate::training::{self, Predictor};

/// 锁多久没续就自动放开（人走了、断网了）
const LOCK_TTL: Duration = Duration::from_secs(90);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ShareOptions {
    /// lan 只在局域网 / relay 自建中转 / xuntong 巡瞳官方中转
    pub mode: String,
    pub port: u16,
    /// 自建中转的地址（wss:// 或 https://）和密钥
    pub relay_url: String,
    pub relay_secret: String,
    /// 巡瞳官方中转地址（默认 https://xuntongai.com/wulabel-relay/）
    pub xuntong_relay_url: String,
    /// 发起人在协作里显示的名字
    pub host_name: String,
    /// 成员能用 AI 辅助（SAM、模型预标注、文本提示，在主机上算）
    pub allow_ai: bool,
    /// 成员的画面叠加带名字的水印
    pub watermark: bool,
    /// 成员可以新增、修改类别和识别说明（删除类别只有发起人能做）
    pub members_edit_labels: bool,
    /// 固定的共享编号（在线模式的地址不变），第一次开启时生成
    pub share_id: String,
}

impl Default for ShareOptions {
    fn default() -> Self {
        ShareOptions {
            mode: "lan".into(),
            port: 8765,
            relay_url: String::new(),
            relay_secret: String::new(),
            xuntong_relay_url: "https://xuntongai.com/wulabel-relay/".into(),
            host_name: "管理员".into(),
            allow_ai: true,
            watermark: false,
            members_edit_labels: true,
            share_id: String::new(),
        }
    }
}

const OPTIONS_KEY: &str = "share";

pub fn load_options(p: &Project) -> ShareOptions {
    team::get_setting(p, OPTIONS_KEY)
        .ok()
        .flatten()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save_options(p: &Project, o: &ShareOptions) -> AppResult<()> {
    team::set_setting(p, OPTIONS_KEY, &serde_json::to_string(o)?)
}

#[derive(Debug, Clone)]
struct Session {
    member_id: i64,
}

#[derive(Debug, Clone)]
struct Lock {
    member_id: i64,
    name: String,
    at: Instant,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Presence {
    pub member_id: i64,
    pub name: String,
    pub role: String,
    pub image_id: Option<i64>,
    /// 打开的窗口数（同一个人可能开了两个浏览器标签）
    #[serde(skip)]
    conns: usize,
    pub since: i64,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RelayStatus {
    /// off / connecting / online / error
    pub state: String,
    pub url: String,
    pub error: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ShareStatus {
    pub running: bool,
    pub options: ShareOptions,
    /// 局域网访问地址
    pub lan_urls: Vec<String>,
    pub relay: RelayStatus,
    pub online: Vec<Presence>,
}

pub struct Hub {
    pub project: Arc<Project>,
    pub app: AppHandle,
    pub opts: Mutex<ShareOptions>,
    pub port: u16,
    sessions: Mutex<HashMap<String, Session>>,
    locks: Mutex<HashMap<i64, Lock>>,
    presence: Mutex<HashMap<i64, Presence>>,
    tx: broadcast::Sender<String>,
    closed: watch::Sender<bool>,
    shutdown: Mutex<Option<oneshot::Sender<()>>>,
    pub relay: Mutex<RelayStatus>,
    pub router: Mutex<Option<Router>>,
}

/// 由 Tauri 管理的共享状态：没开共享时是 None
#[derive(Default)]
pub struct Share(Mutex<Option<Arc<Hub>>>);

impl Share {
    pub fn hub(&self) -> Option<Arc<Hub>> {
        self.0.lock().clone()
    }

    /// 当前项目正在共享时返回 hub（主机这边的命令用来加锁、广播）
    pub fn hub_for(&self, p: &Project) -> Option<Arc<Hub>> {
        self.hub().filter(|h| h.project.root == p.root)
    }

    pub fn status(&self, p: Option<&Project>) -> ShareStatus {
        match self.hub() {
            Some(h) => h.status(),
            None => ShareStatus {
                running: false,
                options: p.map(load_options).unwrap_or_default(),
                lan_urls: Vec::new(),
                relay: RelayStatus { state: "off".into(), ..Default::default() },
                online: Vec::new(),
            },
        }
    }

    pub async fn start(&self, app: AppHandle, project: Arc<Project>, mut opts: ShareOptions) -> AppResult<ShareStatus> {
        self.stop();
        if opts.port < 1024 {
            return Err(msg("端口请用 1024 以上的数字，比如 8765"));
        }
        if opts.host_name.trim().is_empty() {
            opts.host_name = "管理员".into();
        }
        if opts.share_id.is_empty() {
            opts.share_id = team::token(6);
        }
        match opts.mode.as_str() {
            "lan" => {}
            "relay" => {
                if opts.relay_url.trim().is_empty() {
                    return Err(msg("请填写中转服务器地址"));
                }
            }
            "xuntong" => {
                let key = app.state::<Runtime>().settings().ai_key.unwrap_or_default();
                if key.trim().is_empty() {
                    return Err(msg("用巡瞳官方中转需要巡瞳 AI 的 API Key，请先到「设置 → 巡瞳 AI」填写"));
                }
            }
            _ => return Err(msg("共享方式不对")),
        }
        save_options(&project, &opts)?;
        let listener = tokio::net::TcpListener::bind(("0.0.0.0", opts.port))
            .await
            .map_err(|e| msg(format!("端口 {} 打不开：{e}。可能被别的程序占用了，换一个端口试试", opts.port)))?;
        let (sd_tx, sd_rx) = oneshot::channel::<()>();
        let (tx, _) = broadcast::channel(256);
        let (closed, _) = watch::channel(false);
        let hub = Arc::new(Hub {
            project,
            app,
            port: opts.port,
            opts: Mutex::new(opts.clone()),
            sessions: Mutex::new(HashMap::new()),
            locks: Mutex::new(HashMap::new()),
            presence: Mutex::new(HashMap::new()),
            tx,
            closed,
            shutdown: Mutex::new(Some(sd_tx)),
            relay: Mutex::new(RelayStatus { state: "off".into(), ..Default::default() }),
            router: Mutex::new(None),
        });
        let app_router = router(hub.clone());
        *hub.router.lock() = Some(app_router.clone());
        tauri::async_runtime::spawn(async move {
            let _ = axum::serve(listener, app_router)
                .with_graceful_shutdown(async move {
                    let _ = sd_rx.await;
                })
                .await;
        });
        if opts.mode != "lan" {
            crate::tunnel::start(hub.clone());
        }
        *self.0.lock() = Some(hub.clone());
        hub.emit_status();
        Ok(hub.status())
    }

    pub fn stop(&self) {
        if let Some(h) = self.0.lock().take() {
            h.close();
        }
    }
}

/// 本机的局域网 IP（不发数据包，只是让系统选一下出口网卡）
pub fn lan_ips() -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(s) = std::net::UdpSocket::bind("0.0.0.0:0") {
        if s.connect("8.8.8.8:80").is_ok() {
            if let Ok(a) = s.local_addr() {
                out.push(a.ip().to_string());
            }
        }
    }
    if out.is_empty() {
        out.push("127.0.0.1".into());
    }
    out
}

impl Hub {
    pub fn host_member(&self) -> Member {
        Member::host(&self.opts.lock().host_name)
    }

    pub fn status(&self) -> ShareStatus {
        let opts = self.opts.lock().clone();
        let mut online: Vec<Presence> = self.presence.lock().values().cloned().collect();
        online.sort_by_key(|p| p.since);
        ShareStatus {
            running: true,
            lan_urls: lan_ips().into_iter().map(|ip| format!("http://{ip}:{}/", self.port)).collect(),
            relay: self.relay.lock().clone(),
            options: opts,
            online,
        }
    }

    fn close(&self) {
        let _ = self.closed.send(true);
        if let Some(tx) = self.shutdown.lock().take() {
            let _ = tx.send(());
        }
        self.sessions.lock().clear();
        self.locks.lock().clear();
        self.presence.lock().clear();
        *self.relay.lock() = RelayStatus { state: "off".into(), ..Default::default() };
        let _ = self.app.emit("share-status", json!({ "running": false }));
    }

    pub fn closed_rx(&self) -> watch::Receiver<bool> {
        self.closed.subscribe()
    }

    /// 推送一条事件：所有连着的成员 + 主机自己的界面都收到（事件名 share-event）
    pub fn broadcast(&self, kind: &str, mut payload: Value) {
        payload["kind"] = json!(kind);
        let _ = self.tx.send(json!({ "name": "share-event", "payload": payload }).to_string());
        let _ = self.app.emit("share-event", payload);
    }

    pub fn emit_status(&self) {
        let _ = self.app.emit("share-status", self.status());
    }

    fn broadcast_presence(&self) {
        let mut list: Vec<Presence> = self.presence.lock().values().cloned().collect();
        list.sort_by_key(|p| p.since);
        self.broadcast("presence", json!({ "online": list }));
    }

    // ---------- 登录、会话 ----------

    fn login(&self, name: &str, password: &str) -> AppResult<(String, Member)> {
        let m = team::verify(&self.project, name, password)?;
        let t = team::token(24);
        self.sessions.lock().insert(t.clone(), Session { member_id: m.id });
        team::log(&self.project, m.id, None, "login", None, 0, "")?;
        Ok((t, m))
    }

    /// 令牌对应的成员（每次都从数据库取，停用、改角色立即生效）
    pub fn member_of(&self, token: &str) -> Option<Member> {
        let id = self.sessions.lock().get(token)?.member_id;
        match team::get_member(&self.project, id) {
            Ok(m) if !m.disabled => Some(m),
            _ => {
                self.sessions.lock().remove(token);
                None
            }
        }
    }

    /// 停用、删除成员后把他踢下线
    pub fn kick(&self, member_id: i64) {
        self.sessions.lock().retain(|_, s| s.member_id != member_id);
        self.locks.lock().retain(|_, l| l.member_id != member_id);
        self.presence.lock().remove(&member_id);
        self.broadcast("kick", json!({ "memberId": member_id }));
        self.broadcast_presence();
    }

    // ---------- 在线状态 ----------

    pub fn join(&self, m: &Member) {
        {
            let mut pr = self.presence.lock();
            let e = pr.entry(m.id).or_insert_with(|| Presence {
                member_id: m.id,
                name: m.name.clone(),
                role: m.role.clone(),
                image_id: None,
                conns: 0,
                since: crate::db::now(),
            });
            e.conns += 1;
        }
        self.broadcast_presence();
    }

    pub fn leave(&self, m: &Member) {
        let gone = {
            let mut pr = self.presence.lock();
            match pr.get_mut(&m.id) {
                Some(e) if e.conns > 1 => {
                    e.conns -= 1;
                    false
                }
                Some(_) => {
                    pr.remove(&m.id);
                    true
                }
                None => false,
            }
        };
        if gone {
            self.release(m.id);
        }
        self.broadcast_presence();
    }

    fn view(&self, member_id: i64, image_id: Option<i64>) {
        let changed = {
            let mut pr = self.presence.lock();
            match pr.get_mut(&member_id) {
                Some(e) if e.image_id != image_id => {
                    e.image_id = image_id;
                    true
                }
                _ => false,
            }
        };
        self.touch(member_id);
        if changed {
            self.broadcast_presence();
        }
    }

    // ---------- 锁 ----------

    /// 打开一张图时加锁。别人正在改就返回他的名字（只读打开）
    pub fn lock(&self, m: &Member, image_id: i64) -> Option<String> {
        let mut released = Vec::new();
        let holder = {
            let mut locks = self.locks.lock();
            // 一个人同时只锁一张：换图就放开上一张
            locks.retain(|img, l| {
                let mine_other = l.member_id == m.id && *img != image_id;
                if mine_other {
                    released.push(*img);
                }
                !mine_other
            });
            match locks.get(&image_id) {
                Some(l) if l.member_id != m.id && l.at.elapsed() < LOCK_TTL => Some(l.name.clone()),
                _ => {
                    locks.insert(image_id, Lock { member_id: m.id, name: m.name.clone(), at: Instant::now() });
                    None
                }
            }
        };
        for img in released {
            self.broadcast("lock", json!({ "imageId": img, "holder": Value::Null }));
        }
        if holder.is_none() {
            self.broadcast("lock", json!({ "imageId": image_id, "holder": m.name, "memberId": m.id }));
        }
        // 主机自己也算在线（显示在标哪张）
        if m.id == HOST {
            let mut pr = self.presence.lock();
            let e = pr.entry(HOST).or_insert_with(|| Presence {
                member_id: HOST,
                name: m.name.clone(),
                role: "admin".into(),
                image_id: None,
                conns: 1,
                since: crate::db::now(),
            });
            e.image_id = Some(image_id);
        }
        self.view(m.id, Some(image_id));
        holder
    }

    /// 放开某人的全部锁
    pub fn release(&self, member_id: i64) {
        let mut freed = Vec::new();
        self.locks.lock().retain(|img, l| {
            if l.member_id == member_id {
                freed.push(*img);
                false
            } else {
                true
            }
        });
        for img in freed {
            self.broadcast("lock", json!({ "imageId": img, "holder": Value::Null }));
        }
    }

    fn touch(&self, member_id: i64) {
        for l in self.locks.lock().values_mut() {
            if l.member_id == member_id {
                l.at = Instant::now();
            }
        }
    }

    /// 写之前检查：这张图被别人锁着就不让改
    pub fn check_write(&self, m: &Member, image_id: i64) -> AppResult<()> {
        if let Some(l) = self.locks.lock().get(&image_id) {
            if l.member_id != m.id && l.at.elapsed() < LOCK_TTL {
                return Err(msg(format!("「{}」正在标这张图，等他标完再改", l.name)));
            }
        }
        Ok(())
    }

    // ---------- 写操作的统一出口（主机、成员都走这里，好记录和广播） ----------

    pub fn saved(&self, m: &Member, image_id: i64, status: &str) {
        let _ = team::after_save(&self.project, m.id, image_id);
        self.broadcast("ann", json!({ "imageId": image_id, "by": m.name, "memberId": m.id, "status": status }));
    }

    pub fn status_changed(&self, m: &Member, image_id: i64, status: &str, action: &str) {
        self.broadcast("status", json!({ "imageId": image_id, "by": m.name, "memberId": m.id, "status": status, "action": action }));
    }

    pub fn labels_changed(&self, m: &Member, what: &str) {
        let _ = team::log(&self.project, m.id, None, "label", None, 0, what);
        self.broadcast("labels", json!({ "by": m.name, "what": what }));
    }

    /// 提交、审核、改状态（主机的「通过并下一张」也走这里）
    pub fn set_status(&self, m: &Member, image_id: i64, status: &str) -> AppResult<String> {
        let p = &self.project;
        let (new_status, action) = match (status, m.is_annotator()) {
            // 标注员的「通过」就是提交，等审核
            ("reviewed" | "labeled", true) => (team::submit(p, m.id, image_id)?, "submit"),
            ("reviewed", false) => (team::review(p, m.id, image_id, true, "")?, "pass"),
            ("labeled", false) => (team::submit(p, m.id, image_id)?, "submit"),
            (s, _) => {
                annotations::set_status(p, image_id, s)?;
                team::log(p, m.id, Some(image_id), s, None, 0, "")?;
                (s.to_string(), s)
            }
        };
        self.status_changed(m, image_id, &new_status, action);
        Ok(new_status)
    }

    pub fn review(&self, m: &Member, image_id: i64, pass: bool, reason: &str) -> AppResult<String> {
        if !m.can_review() {
            return Err(msg("只有审核员能审核"));
        }
        let s = team::review(&self.project, m.id, image_id, pass, reason)?;
        self.status_changed(m, image_id, &s, if pass { "pass" } else { "reject" });
        Ok(s)
    }

    fn on_client_msg(&self, m: &Member, text: &str) {
        let Ok(v) = serde_json::from_str::<Value>(text) else { return };
        match v["type"].as_str() {
            Some("view") => self.view(m.id, v["imageId"].as_i64()),
            Some("ping") => self.touch(m.id),
            _ => {}
        }
    }
}

// ---------- 成员调用的命令 ----------

fn arg<T: DeserializeOwned>(args: &Value, key: &str) -> AppResult<T> {
    serde_json::from_value(args.get(key).cloned().unwrap_or(Value::Null)).map_err(|e| msg(format!("参数 {key} 不对：{e}")))
}

fn need_ai(hub: &Hub) -> AppResult<()> {
    if hub.opts.lock().allow_ai {
        Ok(())
    } else {
        Err(msg("发起人没有开放 AI 辅助"))
    }
}

fn need_labels(hub: &Hub, m: &Member) -> AppResult<()> {
    if m.role == "admin" || hub.opts.lock().members_edit_labels {
        Ok(())
    } else {
        Err(msg("发起人没有开放修改类别，请联系发起人"))
    }
}

/// 成员通过网络调用的命令（在阻塞线程里跑）。只开放标注需要的，训练、导出、删除等只在主机上做
fn call(hub: &Hub, m: &Member, cmd: &str, args: &Value) -> AppResult<Value> {
    let p = &*hub.project;
    let app = &hub.app;
    let host_name = hub.opts.lock().host_name.clone();
    let v = match cmd {
        "whoami" => {
            let o = hub.opts.lock().clone();
            json!({
                "member": m,
                "project": { "name": p.meta.name, "task": p.meta.task },
                "hostName": o.host_name,
                "options": { "allowAi": o.allow_ai, "watermark": o.watermark, "membersEditLabels": o.members_edit_labels },
            })
        }
        "current_project" => {
            let mut info = p.info()?;
            // 不把主机上的文件夹路径给出去
            info.path = format!("share:{}", p.meta.name);
            serde_json::to_value(info)?
        }
        "project_stats" => serde_json::to_value(p.stats()?)?,
        "my_stats" => {
            let today = chrono::Local::now().date_naive().and_hms_opt(0, 0, 0).map(|d| d.and_local_timezone(chrono::Local).single()).flatten();
            let from = today.map(|d| d.timestamp()).unwrap_or(0);
            let all = team::stats(p, 0, i64::MAX, &host_name)?;
            let day = team::stats(p, from, i64::MAX, &host_name)?;
            let pick = |v: &Vec<team::MemberStats>| v.iter().find(|s| s.member_id == m.id).cloned();
            json!({ "member": team::get_member(p, m.id).ok(), "all": pick(&all), "today": pick(&day) })
        }
        "list_labels" => serde_json::to_value(annotations::list_labels(p)?)?,
        "add_label" => {
            need_labels(hub, m)?;
            let name: String = arg(args, "name")?;
            let l = annotations::add_label(p, &name, &arg::<String>(args, "color")?)?;
            hub.labels_changed(m, &format!("新增类别「{name}」"));
            serde_json::to_value(l)?
        }
        "update_label" => {
            need_labels(hub, m)?;
            let id: i64 = arg(args, "id")?;
            let name: String = arg(args, "name")?;
            let prompt: Option<String> = arg(args, "prompt")?;
            annotations::update_label(p, id, &name, &arg::<String>(args, "color")?, prompt.as_deref())?;
            if let Some(d) = arg::<Option<String>>(args, "description")? {
                annotations::set_label_description(p, id, &d)?;
            }
            hub.labels_changed(m, &format!("修改类别「{name}」"));
            Value::Null
        }
        "delete_label" => return Err(msg("删除类别会删掉它的全部标注，只有发起人能删")),
        "keypoint_template" => serde_json::to_value(keypoints::get(p)?)?,
        "list_image_ids" => {
            let status: Option<String> = arg(args, "status")?;
            serde_json::to_value(team::list_ids(p, m, status.as_deref())?)?
        }
        "load_annotations" => {
            let id: i64 = arg(args, "imageId")?;
            team::ensure_access(p, m, id)?;
            serde_json::to_value(annotations::load(p, id)?)?
        }
        "save_annotations" => {
            let id: i64 = arg(args, "imageId")?;
            team::ensure_access(p, m, id)?;
            hub.check_write(m, id)?;
            let anns: Vec<Annotation> = arg(args, "annotations")?;
            let status = annotations::save(p, id, &anns)?;
            hub.saved(m, id, &status);
            json!(status)
        }
        "set_image_status" => {
            let id: i64 = arg(args, "imageId")?;
            team::ensure_access(p, m, id)?;
            hub.check_write(m, id)?;
            json!(hub.set_status(m, id, &arg::<String>(args, "status")?)?)
        }
        "review_image" => {
            let id: i64 = arg(args, "imageId")?;
            json!(hub.review(m, id, arg(args, "pass")?, &arg::<Option<String>>(args, "reason")?.unwrap_or_default())?)
        }
        "lock_image" => {
            let id: i64 = arg(args, "imageId")?;
            team::ensure_access(p, m, id)?;
            let holder = hub.lock(m, id);
            json!({ "holder": holder, "info": team::image_info(p, id, &host_name)? })
        }
        "unlock_image" => {
            hub.release(m.id);
            hub.view(m.id, None);
            Value::Null
        }
        "image_info" => {
            let id: i64 = arg(args, "imageId")?;
            team::ensure_access(p, m, id)?;
            serde_json::to_value(team::image_info(p, id, &host_name)?)?
        }
        "list_runs" => {
            need_ai(hub)?;
            serde_json::to_value(training::list_runs(p)?)?
        }
        "predict_image" => {
            need_ai(hub)?;
            let id: i64 = arg(args, "imageId")?;
            team::ensure_access(p, m, id)?;
            let r = training::predict_one(p, &app.state::<Runtime>(), &app.state::<Predictor>(), arg(args, "runId")?, id, arg(args, "conf")?)?;
            serde_json::to_value(r)?
        }
        "sam_segment" => {
            need_ai(hub)?;
            let id: i64 = arg(args, "imageId")?;
            team::ensure_access(p, m, id)?;
            let points: Vec<[f64; 3]> = arg(args, "points")?;
            let bbox: Option<[f64; 4]> = arg(args, "bbox")?;
            let r = assist::sam_segment(p, &app.state::<Runtime>(), &app.state::<Predictor>(), id, &points, bbox, &|_| {})?;
            serde_json::to_value(r)?
        }
        "text_detect" => {
            need_ai(hub)?;
            let id: i64 = arg(args, "imageId")?;
            team::ensure_access(p, m, id)?;
            let r = assist::text_detect(p, &app.state::<Runtime>(), &app.state::<Predictor>(), id, arg(args, "conf")?, &|_| {})?;
            serde_json::to_value(r)?
        }
        "ai_rules" => serde_json::to_value(crate::ai::rules(p)?)?,
        "ai_save_rules" => {
            need_labels(hub, m)?;
            let instructions: Option<String> = arg(args, "instructions")?;
            let classes: Vec<crate::ai::ClassRule> = arg(args, "classes")?;
            crate::ai::save_rules(p, instructions.as_deref(), &classes)?;
            hub.labels_changed(m, "更新了识别规则");
            Value::Null
        }
        "ai_chat_rules" => {
            // 用的是发起人的巡瞳额度，跟着「允许 AI 辅助」开关走
            need_ai(hub)?;
            let messages: Vec<crate::ai::ChatMsg> = arg(args, "messages")?;
            for id in messages.iter().filter_map(|x| x.image_id) {
                team::ensure_access(p, m, id)?;
            }
            let r = crate::ai::chat_rules(p, &app.state::<Runtime>(), &app.state::<Predictor>(), &messages, &|_| {})?;
            serde_json::to_value(r)?
        }
        _ => return Err(msg("这个功能只能在发起人的电脑上用")),
    };
    Ok(v)
}

// ---------- HTTP ----------

type HubState = AxState<Arc<Hub>>;

fn router(hub: Arc<Hub>) -> Router {
    Router::new()
        .route("/api/info", get(info))
        .route("/api/login", post(login))
        .route("/api/logout", post(logout))
        .route("/api/call/{cmd}", post(rpc))
        .route("/api/image/{id}", get(image))
        .route("/api/thumb/{id}", get(thumb))
        .route("/api/ws", get(ws))
        .fallback(static_file)
        .with_state(hub)
}

fn err(status: StatusCode, text: impl Into<String>) -> Response {
    (status, Json(json!({ "error": text.into() }))).into_response()
}

fn bearer(headers: &HeaderMap) -> Option<String> {
    headers.get(header::AUTHORIZATION)?.to_str().ok()?.strip_prefix("Bearer ").map(|s| s.trim().to_string())
}

async fn info(AxState(hub): HubState) -> Response {
    let o = hub.opts.lock().clone();
    Json(json!({ "app": "wulabel", "project": hub.project.meta.name, "task": hub.project.meta.task, "hostName": o.host_name })).into_response()
}

#[derive(Deserialize)]
struct LoginReq {
    name: String,
    password: String,
}

async fn login(AxState(hub): HubState, Json(req): Json<LoginReq>) -> Response {
    let h = hub.clone();
    let r = tokio::task::spawn_blocking(move || h.login(&req.name, &req.password)).await;
    match r {
        Ok(Ok((token, m))) => {
            let me = m.clone();
            let h = hub.clone();
            let who = tokio::task::spawn_blocking(move || call(&h, &me, "whoami", &Value::Null)).await;
            let mut v = match who {
                Ok(Ok(v)) => v,
                _ => json!({ "member": m }),
            };
            v["token"] = json!(token);
            Json(v).into_response()
        }
        Ok(Err(e)) => err(StatusCode::UNAUTHORIZED, e.to_string()),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

async fn logout(AxState(hub): HubState, headers: HeaderMap) -> Response {
    if let Some(t) = bearer(&headers) {
        if let Some(m) = hub.member_of(&t) {
            hub.release(m.id);
        }
        hub.sessions.lock().remove(&t);
    }
    Json(json!({})).into_response()
}

async fn rpc(AxState(hub): HubState, AxPath(cmd): AxPath<String>, headers: HeaderMap, body: axum::body::Bytes) -> Response {
    let Some(token) = bearer(&headers) else { return err(StatusCode::UNAUTHORIZED, "请先登录") };
    let Some(m) = hub.member_of(&token) else { return err(StatusCode::UNAUTHORIZED, "登录已失效，请重新登录") };
    let args: Value = if body.is_empty() { json!({}) } else { serde_json::from_slice(&body).unwrap_or(json!({})) };
    let h = hub.clone();
    match tokio::task::spawn_blocking(move || call(&h, &m, &cmd, &args)).await {
        Ok(Ok(v)) => Json(v).into_response(),
        Ok(Err(e)) => err(StatusCode::BAD_REQUEST, e.to_string()),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[derive(Deserialize)]
struct TokenQ {
    t: Option<String>,
}

async fn image(state: HubState, id: AxPath<i64>, q: Query<TokenQ>) -> Response {
    send_image(state, id, q, "image").await
}

async fn thumb(state: HubState, id: AxPath<i64>, q: Query<TokenQ>) -> Response {
    send_image(state, id, q, "thumb").await
}

async fn send_image(AxState(hub): HubState, AxPath(id): AxPath<i64>, Query(q): Query<TokenQ>, kind: &'static str) -> Response {
    let Some(m) = q.t.as_deref().and_then(|t| hub.member_of(t)) else { return err(StatusCode::UNAUTHORIZED, "请先登录") };
    let h = hub.clone();
    let r = tokio::task::spawn_blocking(move || -> AppResult<(Vec<u8>, &'static str)> {
        team::ensure_access(&h.project, &m, id)?;
        protocol::image_bytes(&h.project, kind, id)
    })
    .await;
    match r {
        Ok(Ok((bytes, ct))) => ([(header::CONTENT_TYPE, ct), (header::CACHE_CONTROL, "private, max-age=86400")], bytes).into_response(),
        Ok(Err(e)) => err(StatusCode::NOT_FOUND, e.to_string()),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

async fn ws(AxState(hub): HubState, Query(q): Query<TokenQ>, upgrade: WebSocketUpgrade) -> Response {
    let Some(m) = q.t.as_deref().and_then(|t| hub.member_of(t)) else { return err(StatusCode::UNAUTHORIZED, "请先登录") };
    upgrade.on_upgrade(move |socket: WebSocket| async move {
        let (mut sink, stream) = socket.split();
        let (out_tx, mut out_rx) = mpsc::channel::<String>(128);
        let writer = tokio::spawn(async move {
            while let Some(s) = out_rx.recv().await {
                if sink.send(Message::Text(s.into())).await.is_err() {
                    break;
                }
            }
        });
        let incoming = stream
            .take_while(|r| futures_util::future::ready(matches!(r, Ok(m) if !matches!(m, Message::Close(_)))))
            .filter_map(|r| {
                futures_util::future::ready(match r {
                    Ok(Message::Text(t)) => Some(t.as_str().to_string()),
                    _ => None,
                })
            });
        session(hub, m, Box::pin(incoming), out_tx).await;
        writer.abort();
    })
}

/// 一个成员连接的实时会话：收他的「在看哪张」、心跳；把事件推给他。真连接和中转来的连接共用
pub async fn session(hub: Arc<Hub>, m: Member, mut incoming: Pin<Box<dyn Stream<Item = String> + Send>>, out: mpsc::Sender<String>) {
    let mut rx = hub.tx.subscribe();
    let mut closed = hub.closed_rx();
    hub.join(&m);
    let online: Vec<Presence> = hub.presence.lock().values().cloned().collect();
    let _ = out.send(json!({ "name": "share-event", "payload": { "kind": "presence", "online": online } }).to_string()).await;
    loop {
        tokio::select! {
            t = incoming.next() => match t {
                Some(text) => hub.on_client_msg(&m, &text),
                None => break,
            },
            ev = rx.recv() => match ev {
                Ok(s) => {
                    if out.send(s).await.is_err() {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => break,
            },
            _ = closed.changed() => {
                let _ = out.send(json!({ "name": "share-event", "payload": { "kind": "closed" } }).to_string()).await;
                break;
            }
        }
    }
    hub.leave(&m);
}

// ---------- 界面文件 ----------

fn mime_of(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" | "tsx" | "ts" | "jsx" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "ico" => "image/x-icon",
        "json" => "application/json",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "ttf" => "font/ttf",
        _ => "application/octet-stream",
    }
}

async fn static_file(AxState(hub): HubState, uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/').to_string();
    if path.starts_with("api/") {
        return err(StatusCode::NOT_FOUND, "没有这个接口");
    }
    let app = hub.app.clone();
    let q = uri.query().map(|s| s.to_string());
    let r = tokio::task::spawn_blocking(move || find_asset(&app, &path, q.as_deref())).await;
    match r {
        Ok(Some((bytes, ct))) => ([(header::CONTENT_TYPE, ct)], Body::from(bytes)).into_response(),
        _ => err(StatusCode::NOT_FOUND, "找不到页面"),
    }
}

/// 找界面文件：安装包里内嵌的 → 开发时的 dist 文件夹 → 开发时代理到 vite（npm run tauri dev）
fn find_asset(app: &AppHandle, path: &str, query: Option<&str>) -> Option<(Vec<u8>, String)> {
    let path = if path.is_empty() { "index.html" } else { path };
    let resolver = app.asset_resolver();
    if let Some(a) = resolver.get(path.to_string()) {
        return Some((a.bytes().to_vec(), a.mime_type().to_string()));
    }
    if !path.contains('.') {
        // 单页应用：没有扩展名的路径都给首页
        if let Some(a) = resolver.get("index.html".to_string()) {
            return Some((a.bytes().to_vec(), a.mime_type().to_string()));
        }
    }
    let dist = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("dist");
    if dist.join("index.html").exists() && !path.contains("..") {
        let f = dist.join(path);
        let f = if f.is_file() { f } else { dist.join("index.html") };
        if let Ok(b) = std::fs::read(&f) {
            return Some((b, mime_of(&f.to_string_lossy()).to_string()));
        }
    }
    if cfg!(debug_assertions) {
        let url = match query {
            Some(q) => format!("http://localhost:1420/{path}?{q}"),
            None => format!("http://localhost:1420/{path}"),
        };
        if let Ok(resp) = ureq::get(&url).timeout(Duration::from_secs(20)).call() {
            let ct = resp.content_type().to_string();
            let mut buf = Vec::new();
            if std::io::Read::read_to_end(&mut resp.into_reader(), &mut buf).is_ok() {
                let ct = if ct.is_empty() { mime_of(path).to_string() } else { ct };
                return Some((buf, ct));
            }
        }
    }
    None
}
