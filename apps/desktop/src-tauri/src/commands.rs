use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

use crate::ai;
use crate::collab;
use crate::annotations::{self, Annotation, ImageAnnotations, Label};
use crate::assist;
use crate::dataset::{self, ExportOptions, ExportResult, SplitSummary};
use crate::dataset_import::{self, DatasetImportResult};
use crate::deploy;
use crate::evaltools;
use crate::error::{msg, AppResult};
use crate::health::{self, HealthReport};
use crate::images::{self, DeleteScope, ImagePage, ImportResult, LastImport};
use crate::keypoints::{self, Template};
use crate::project::{self, Project, ProjectInfo, RecentProject, Stats};
use crate::runtime::{self, EnvStatus, Runtime, Settings, SpeedResult};
use crate::share::{self, Share, ShareOptions, ShareStatus};
use crate::state::AppState;
use crate::team::{self, AssignOptions, AssignResult, Member, MemberPatch};
use crate::quality;
use crate::report;
use crate::track;
use crate::versions;
use crate::video;
use crate::training::{self, Jobs, Predictor, PredResult, PrelabelResult, RunDetail, RunInfo, TrainConfig};
use crate::t;

fn config_dir(app: &AppHandle) -> AppResult<PathBuf> {
    Ok(app.path().app_config_dir()?)
}

fn opened(app: &AppHandle, state: &AppState, p: Project) -> AppResult<ProjectInfo> {
    project::touch_recent(&config_dir(app)?, &p)?;
    training::recover(&p, &app.state::<Arc<Jobs>>());
    // 换项目后推理进程里缓存的模型没用了
    app.state::<Predictor>().shutdown();
    // 换项目就停止共享（成员连的是原来的项目）
    app.state::<Share>().stop();
    let info = p.info()?;
    state.set(Some(p));
    Ok(info)
}

#[tauri::command]
pub async fn create_project(
    app: AppHandle,
    state: State<'_, AppState>,
    parent: String,
    name: String,
    task: String,
) -> AppResult<ProjectInfo> {
    let p = Project::create(&PathBuf::from(parent), &name, &task)?;
    opened(&app, &state, p)
}

#[tauri::command]
pub async fn open_project(app: AppHandle, state: State<'_, AppState>, path: String) -> AppResult<ProjectInfo> {
    let p = Project::open(&PathBuf::from(path))?;
    opened(&app, &state, p)
}

#[tauri::command]
pub async fn close_project(state: State<'_, AppState>, share: State<'_, Share>) -> AppResult<()> {
    share.stop();
    state.set(None);
    Ok(())
}

#[tauri::command]
pub async fn current_project(state: State<'_, AppState>) -> AppResult<Option<ProjectInfo>> {
    state.try_current().map(|p| p.info()).transpose()
}

#[tauri::command]
pub async fn recent_projects(app: AppHandle) -> AppResult<Vec<RecentProject>> {
    Ok(project::load_recent(&config_dir(&app)?))
}

/// 首页项目卡片的概况（图片数、已标注、训练次数、封面），项目不存在时返回 null
#[tauri::command]
pub async fn project_summary(path: String) -> AppResult<Option<project::ProjectSummary>> {
    Ok(tauri::async_runtime::spawn_blocking(move || project::summary(std::path::Path::new(&path)))
        .await
        .unwrap_or(None))
}

#[tauri::command]
pub async fn remove_recent(app: AppHandle, path: String) -> AppResult<()> {
    project::remove_recent(&config_dir(&app)?, &path)
}

#[derive(Clone, Serialize)]
struct ImportProgress {
    done: usize,
    total: usize,
}

#[tauri::command]
pub async fn import_images(
    app: AppHandle,
    state: State<'_, AppState>,
    paths: Vec<String>,
    copy: bool,
) -> AppResult<ImportResult> {
    let project = state.current()?;
    let paths: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
    images::STOP_IMPORT.store(false, std::sync::atomic::Ordering::Relaxed);
    tauri::async_runtime::spawn_blocking(move || {
        let files = images::collect(&paths);
        if files.is_empty() {
            return Err(msg(t!("没有找到图片（支持 jpg、png、bmp、webp、tif）")));
        }
        let last = std::sync::atomic::AtomicUsize::new(0);
        crate::shotfile::import(&project, &files, copy, &|done, total| {
            // 每 1% 或最后一张推一次进度，避免事件太多
            let step = (total / 100).max(1);
            if done == total || done >= last.load(std::sync::atomic::Ordering::Relaxed) + step {
                last.store(done, std::sync::atomic::Ordering::Relaxed);
                let _ = app.emit("import-progress", ImportProgress { done, total });
            }
        })
    })
    .await
    .map_err(|e| msg(t!("导入任务异常：{e}", e = e)))?
}

/// 停止正在进行的图片导入，已处理的图照常入库
#[tauri::command]
pub async fn stop_import() -> AppResult<()> {
    images::STOP_IMPORT.store(true, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct VideoProgress {
    stage: String,
    done: u64,
    total: u64,
    speed: u64,
    detail: String,
}

#[derive(Clone, Serialize)]
struct StreamFrames {
    ids: Vec<i64>,
    dir: String,
}

/// 视频抽帧：按 every 秒抽 1 帧（每个视频最多 max_frames 帧，0 不限），抽出的帧导入项目
#[tauri::command]
pub async fn extract_video(
    app: AppHandle,
    state: State<'_, AppState>,
    paths: Vec<String>,
    every: f64,
    max_frames: u64,
) -> AppResult<video::ExtractResult> {
    let project = state.current()?;
    let paths: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
    tauri::async_runtime::spawn_blocking(move || {
        let _busy = BusyGuard::new(&BUSY_VIDEO);
        let rt = app.state::<Runtime>();
        let ext = app.state::<video::Extractor>();
        let a = app.clone();
        let progress = move |stage: &str, done: u64, total: u64, speed: u64, detail: &str| {
            let _ = a.emit(
                "video-progress",
                VideoProgress { stage: stage.to_string(), done, total, speed, detail: detail.to_string() },
            );
        };
        video::extract(&project, &rt, &ext, &paths, every, max_frames, &progress, &|_line| {})
    })
    .await
    .map_err(|e| msg(t!("抽帧任务异常：{e}", e = e)))?
}

/// 测试连接：探测视频流信息（可达性、分辨率、直播还是回放）
#[tauri::command]
pub async fn probe_stream(app: AppHandle, url: String) -> AppResult<video::StreamInfo> {
    tauri::async_runtime::spawn_blocking(move || {
        let rt = app.state::<Runtime>();
        let a = app.clone();
        let progress = move |stage: &str, done: u64, total: u64, speed: u64, detail: &str| {
            let _ = a.emit(
                "video-progress",
                VideoProgress { stage: stage.to_string(), done, total, speed, detail: detail.to_string() },
            );
        };
        video::probe_stream(&rt, &url, &|_line| {}, &progress)
    })
    .await
    .map_err(|e| msg(t!("探测任务异常：{e}", e = e)))?
}

/// 从视频流抽帧：直播用最多帧数 / 最长时长做上限，回放读到结尾；dedup 跳过相似帧
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn extract_stream(
    app: AppHandle,
    state: State<'_, AppState>,
    url: String,
    every: f64,
    max_frames: u64,
    max_minutes: f64,
    dedup: bool,
) -> AppResult<video::ExtractResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let _busy = BusyGuard::new(&BUSY_STREAM);
        let rt = app.state::<Runtime>();
        let ext = app.state::<video::Extractor>();
        let a = app.clone();
        let progress = move |stage: &str, done: u64, total: u64, speed: u64, detail: &str| {
            let _ = a.emit(
                "video-progress",
                VideoProgress { stage: stage.to_string(), done, total, speed, detail: detail.to_string() },
            );
        };
        // 边采边入库：新入库的图片 id 推给前端实时显示
        let b = app.clone();
        let on_frames = move |ids: &[i64], dir: &str| {
            let _ = b.emit("stream-frames", StreamFrames { ids: ids.to_vec(), dir: dir.to_string() });
        };
        video::extract_stream(
            &project, &rt, &ext, &url, every, max_frames, max_minutes, dedup, &progress, &on_frames, &|_line| {},
        )
    })
    .await
    .map_err(|e| msg(t!("采集任务异常：{e}", e = e)))?
}

// ---------- 外部模型、重新评估、误报测试 ----------

#[tauri::command]
pub async fn inspect_onnx(app: AppHandle, path: String) -> AppResult<serde_json::Value> {
    tauri::async_runtime::spawn_blocking(move || evaltools::inspect_onnx(&app.state::<Runtime>(), &path))
        .await
        .map_err(|e| msg(t!("读取模型异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn import_onnx(state: State<'_, AppState>, path: String, model: evaltools::ExternalModel) -> AppResult<i64> {
    let project = state.current()?;
    evaltools::import_onnx(&project, &path, &model)
}

#[tauri::command]
pub async fn update_external(state: State<'_, AppState>, run_id: i64, model: evaltools::ExternalModel) -> AppResult<()> {
    let project = state.current()?;
    evaltools::update_external(&project, run_id, &model)
}

#[tauri::command]
pub async fn evaluate_run(app: AppHandle, state: State<'_, AppState>, run_id: i64) -> AppResult<f64> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let a = app.clone();
        let b = app.clone();
        // 导出数据时每张图回调一次，按 1% 推；每个阶段的开头（done 为 0 或变小）一定推
        let last = std::sync::atomic::AtomicUsize::new(usize::MAX);
        let progress = move |stage: &str, done: usize, total: usize| {
            use std::sync::atomic::Ordering::Relaxed;
            let prev = last.load(Relaxed);
            if prev == usize::MAX || done == 0 || done == total || done < prev || done >= prev + (total / 100).max(1) {
                last.store(done, Relaxed);
                let _ = b.emit("eval-progress", serde_json::json!({ "stage": stage, "done": done, "total": total }));
            }
        };
        evaltools::evaluate_run(&project, &app.state::<Runtime>(), run_id, &move |l| {
            let _ = a.emit("eval-log", l);
        }, &progress)
    })
    .await
    .map_err(|e| msg(t!("评估异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn fp_test(app: AppHandle, state: State<'_, AppState>, config: evaltools::FpConfig) -> AppResult<serde_json::Value> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let a = app.clone();
        let sink = move |v: &serde_json::Value| {
            let _ = a.emit("fptest", v);
        };
        evaltools::fp_test(&project, &app.state::<Runtime>(), &app.state::<evaltools::FpTest>(), &config, &sink)
    })
    .await
    .map_err(|e| msg(t!("误报测试异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn stop_fp_test(job: State<'_, evaltools::FpTest>) -> AppResult<()> {
    job.stop();
    Ok(())
}

#[tauri::command]
pub async fn split_packages(app: AppHandle, state: State<'_, AppState>, options: collab::SplitOptions) -> AppResult<Vec<collab::PackageInfo>> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let progress = progress_emitter(app.clone(), "export-progress");
        let rt = app.state::<Runtime>();
        let log_app = app.clone();
        let log = move |m: String| {
            let _ = log_app.emit("export-log", m);
        };
        collab::split(&project, &options, &rt, &log, &progress)
    })
    .await
    .map_err(|e| msg(t!("分包异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn list_packages(state: State<'_, AppState>) -> AppResult<Vec<collab::PackageInfo>> {
    let project = state.current()?;
    collab::list(&project)
}

#[tauri::command]
pub async fn merge_package(app: AppHandle, state: State<'_, AppState>, options: collab::MergeOptions) -> AppResult<collab::MergeResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let progress = progress_emitter(app, "merge-progress");
        collab::merge(&project, &options, &progress)
    })
    .await
    .map_err(|e| msg(t!("合并异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn export_report(state: State<'_, AppState>, run_id: i64, path: String) -> AppResult<()> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || report::export(&project, run_id, &PathBuf::from(path)))
        .await
        .map_err(|e| msg(t!("导出报告异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn quality_scan(app: AppHandle, state: State<'_, AppState>, only_todo: bool) -> AppResult<quality::QualityReport> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let progress = progress_emitter(app.clone(), "quality-progress");
        quality::scan(&project, only_todo, &progress)
    })
    .await
    .map_err(|e| msg(t!("质量筛查异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn uncertain_images(app: AppHandle, state: State<'_, AppState>, run_id: i64, limit: usize) -> AppResult<Vec<training::Uncertain>> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let progress = progress_emitter(app.clone(), "prelabel-progress");
        training::uncertain_images(&project, &app.state::<Runtime>(), &app.state::<Predictor>(), run_id, limit, &progress)
    })
    .await
    .map_err(|e| msg(t!("挑图异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn aug_preview(app: AppHandle, state: State<'_, AppState>, aug: serde_json::Value, img_size: u32) -> AppResult<Vec<String>> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || evaltools::aug_preview(&project, &app.state::<Runtime>(), &aug, img_size, 4))
        .await
        .map_err(|e| msg(t!("预览异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn import_negatives(state: State<'_, AppState>, paths: Vec<String>) -> AppResult<usize> {
    let project = state.current()?;
    evaltools::import_negatives(&project, &paths)
}

// ---------- 数据集版本 ----------

#[tauri::command]
pub async fn list_versions(state: State<'_, AppState>) -> AppResult<Vec<versions::VersionInfo>> {
    let project = state.current()?;
    versions::list(&project)
}

#[tauri::command]
pub async fn create_version(
    state: State<'_, AppState>,
    name: String,
    note: String,
    ratios: [f64; 3],
) -> AppResult<versions::VersionInfo> {
    let project = state.current()?;
    versions::create(&project, &name, &note, ratios)
}

#[tauri::command]
pub async fn rename_version(state: State<'_, AppState>, id: i64, name: String, note: String) -> AppResult<()> {
    let project = state.current()?;
    versions::rename(&project, id, &name, &note)
}

#[tauri::command]
pub async fn delete_version(state: State<'_, AppState>, id: i64) -> AppResult<()> {
    let project = state.current()?;
    versions::delete(&project, id)
}

#[tauri::command]
pub async fn diff_version(state: State<'_, AppState>, id: i64) -> AppResult<versions::VersionDiff> {
    let project = state.current()?;
    versions::diff(&project, id)
}

#[tauri::command]
pub async fn restore_version(state: State<'_, AppState>, id: i64) -> AppResult<versions::RestoreResult> {
    let project = state.current()?;
    versions::restore(&project, id)
}

#[tauri::command]
pub async fn export_version(
    app: AppHandle,
    state: State<'_, AppState>,
    id: i64,
    mut options: ExportOptions,
    anonymize: Option<evaltools::Anonymize>,
) -> AppResult<ExportResult> {
    let project = state.current()?;
    if anonymize.is_some() {
        options.copy_files = true;
    }
    tauri::async_runtime::spawn_blocking(move || {
        let progress = progress_emitter(app.clone(), "export-progress");
        let res = dataset::export_version(&project, id, &options, &progress)?;
        if let Some(a) = anonymize {
            anonymize_export(&app, &project, &res, &a)?;
        }
        Ok(res)
    })
    .await
    .map_err(|e| msg(t!("导出异常：{e}", e = e)))?
}

/// 关键点项目的模板（点名 + 骨架），没有时返回默认的 COCO 人体 17 点
#[tauri::command]
pub async fn keypoint_template(state: State<'_, AppState>) -> AppResult<Template> {
    let project = state.current()?;
    keypoints::get(&project)
}

#[tauri::command]
pub async fn save_keypoint_template(state: State<'_, AppState>, template: Template) -> AppResult<()> {
    let project = state.current()?;
    keypoints::save(&project, &template)
}

/// 停止正在进行的视频流采集，已抽的帧会照常导入
#[tauri::command]
pub async fn stop_extract(ext: State<'_, video::Extractor>) -> AppResult<()> {
    ext.stop();
    Ok(())
}

#[tauri::command]
pub async fn list_images(
    state: State<'_, AppState>,
    status: Option<String>,
    label: Option<i64>,
    offset: u32,
    limit: u32,
) -> AppResult<ImagePage> {
    let project = state.current()?;
    images::list(&project, status.as_deref(), label, offset, limit)
}

#[tauri::command]
pub async fn project_stats(state: State<'_, AppState>) -> AppResult<Stats> {
    state.current()?.stats()
}

#[tauri::command]
pub async fn last_import(state: State<'_, AppState>) -> AppResult<Option<LastImport>> {
    images::last_import(&*state.current()?)
}

/// scope: "ids"（配合 ids）/ "last"（上次导入）/ "all"（全部）
#[tauri::command]
pub async fn delete_images(app: AppHandle, state: State<'_, AppState>, scope: String, ids: Option<Vec<i64>>) -> AppResult<usize> {
    let project = state.current()?;
    let scope = match scope.as_str() {
        "ids" => DeleteScope::Ids(ids.unwrap_or_default()),
        "last" => DeleteScope::LastImport,
        "all" => DeleteScope::All,
        s => return Err(msg(t!("不支持的删除范围：{s}", s = s))),
    };
    tauri::async_runtime::spawn_blocking(move || {
        let progress = progress_emitter(app, "delete-progress");
        images::delete_with(&project, scope, &progress)
    })
    .await
    .map_err(|e| msg(t!("删除任务异常：{e}", e = e)))?
}

// ---------- 训练环境 ----------

/// 检查要启动 Python 导入 torch，放到后台线程
#[tauri::command]
pub async fn env_status(app: AppHandle, refresh: bool) -> AppResult<EnvStatus> {
    tauri::async_runtime::spawn_blocking(move || app.state::<Runtime>().status(refresh))
        .await
        .map_err(|e| msg(t!("环境检查异常：{e}", e = e)))
}

#[tauri::command]
pub async fn save_settings(rt: State<'_, Runtime>, settings: Settings) -> AppResult<()> {
    rt.save_settings(&settings)
}

#[tauri::command]
pub async fn env_install(app: AppHandle, variant: String) -> AppResult<()> {
    env_task(app, move |rt, log, progress, step| rt.install(&variant, log, progress, step)).await
}

/// 只补装环境里缺的组件
#[tauri::command]
pub async fn env_repair(app: AppHandle) -> AppResult<()> {
    env_task(app, |rt, log, progress, step| rt.install_missing(log, progress, step)).await
}

/// 取消正在进行的安装
#[tauri::command]
pub async fn env_cancel(rt: State<'_, Runtime>) -> AppResult<()> {
    rt.cancel_install();
    Ok(())
}

type EnvLog<'a> = &'a (dyn Fn(String) + Sync);
type EnvProgress<'a> = &'a (dyn Fn(&str, u64, u64, u64) + Sync);
type EnvStep<'a> = &'a (dyn Fn(u32, u32, String) + Sync);

/// 安装类任务：日志（env-log）、下载进度（env-progress）、第几步（env-step）推给设置页
async fn env_task(
    app: AppHandle,
    f: impl FnOnce(&Runtime, EnvLog<'_>, EnvProgress<'_>, EnvStep<'_>) -> AppResult<()> + Send + 'static,
) -> AppResult<()> {
    tauri::async_runtime::spawn_blocking(move || {
        let rt = app.state::<Runtime>();
        let a = app.clone();
        let ap = app.clone();
        let st = app.clone();
        f(
            rt.inner(),
            &move |line: String| {
                let _ = a.emit("env-log", line);
            },
            &move |file: &str, downloaded: u64, total: u64, speed: u64| {
                let _ = ap.emit(
                    "env-progress",
                    serde_json::json!({ "file": file, "downloaded": downloaded, "total": total, "speed": speed }),
                );
            },
            &move |step: u32, total: u32, text: String| {
                let _ = st.emit("env-step", serde_json::json!({ "step": step, "total": total, "text": text }));
            },
        )
    })
    .await
    .map_err(|e| msg(t!("安装任务异常：{e}", e = e)))?
}

/// 实测当前网络下国内镜像 / 官方源的下载速度，推荐更快的
#[tauri::command]
pub async fn speed_test(variant: String) -> AppResult<SpeedResult> {
    tauri::async_runtime::spawn_blocking(move || runtime::speed_test(&variant))
        .await
        .map_err(|e| msg(t!("测速任务异常：{e}", e = e)))?
}

// ---------- 训练 ----------

#[tauri::command]
pub async fn list_runs(state: State<'_, AppState>) -> AppResult<Vec<RunInfo>> {
    training::list_runs(&*state.current()?)
}

#[tauri::command]
pub async fn run_detail(state: State<'_, AppState>, id: i64) -> AppResult<RunDetail> {
    training::detail(&*state.current()?, id)
}

#[tauri::command]
pub async fn running_run(jobs: State<'_, Arc<Jobs>>) -> AppResult<Option<i64>> {
    Ok(jobs.running())
}

#[tauri::command]
pub async fn start_training(app: AppHandle, state: State<'_, AppState>, config: TrainConfig) -> AppResult<i64> {
    let project = state.current()?;
    let jobs = app.state::<Arc<Jobs>>().inner().clone();
    let sink = train_sink(&app);
    tauri::async_runtime::spawn_blocking(move || {
        let rt = app.state::<Runtime>();
        // 开始前要导出一份数据集副本，图多时要一会儿，给界面推进度
        let prepare = progress_emitter(app.clone(), "train-prepare");
        training::start(project, &rt, jobs, config, sink, &prepare)
    })
    .await
    .map_err(|e| msg(t!("启动训练异常：{e}", e = e)))?
}

/// 训练进程的事件转给前端（train-event），结束时发系统通知
fn train_sink(app: &AppHandle) -> training::EventSink {
    let a = app.clone();
    Arc::new(move |id, ev| {
        if ev["type"] == "finished" {
            notify_training(&a, ev["status"].as_str().unwrap_or(""));
        }
        let _ = a.emit("train-event", serde_json::json!({ "runId": id, "event": ev }));
    })
}

/// 从断点继续训练（停止、中断、失败的训练，留有每轮的断点）
#[tauri::command]
pub async fn resume_training(app: AppHandle, state: State<'_, AppState>, run_id: i64) -> AppResult<()> {
    let project = state.current()?;
    let jobs = app.state::<Arc<Jobs>>().inner().clone();
    let sink = train_sink(&app);
    tauri::async_runtime::spawn_blocking(move || {
        let rt = app.state::<Runtime>();
        training::resume(project, &rt, jobs, run_id, sink)
    })
    .await
    .map_err(|e| msg(t!("启动训练异常：{e}", e = e)))?
}

/// 训练结束：发系统通知、任务栏闪一下（训练常常要几十分钟，人多半不在电脑前）
fn notify_training(app: &AppHandle, status: &str) {
    use tauri_plugin_notification::NotificationExt;

    let (title, body) = match status {
        "done" => (t!("Wulabel 训练完成"), t!("可以去评估页看效果，或者用训好的模型预标注剩下的图")),
        "failed" => (t!("Wulabel 训练失败"), t!("打开 Wulabel 的训练页看日志")),
        _ => return,
    };
    let _ = app.notification().builder().title(title).body(body).show();
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.request_user_attention(Some(tauri::UserAttentionType::Informational));
    }
}

#[tauri::command]
pub async fn stop_training(jobs: State<'_, Arc<Jobs>>, id: i64) -> AppResult<()> {
    training::stop(&jobs, id)
}

#[tauri::command]
pub async fn delete_run(state: State<'_, AppState>, jobs: State<'_, Arc<Jobs>>, id: i64) -> AppResult<()> {
    training::delete_run(&*state.current()?, &jobs, id)
}

#[tauri::command]
pub async fn export_onnx(app: AppHandle, state: State<'_, AppState>, id: i64) -> AppResult<String> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let rt = app.state::<Runtime>();
        let a = app.clone();
        training::export_onnx(&project, &rt, id, &move |line| {
            let _ = a.emit("onnx-log", line);
        })
    })
    .await
    .map_err(|e| msg(t!("导出异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn predict_image(app: AppHandle, state: State<'_, AppState>, run_id: i64, image_id: i64, conf: f64) -> AppResult<PredResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        training::predict_one(&project, &app.state::<Runtime>(), &app.state::<Predictor>(), run_id, image_id, conf)
    })
    .await
    .map_err(|e| msg(t!("预测异常：{e}", e = e)))?
}

// ---------- 标注辅助：SAM、文本提示、巡瞳 AI ----------

/// 辅助模型准备过程（安装组件、下载模型）的日志推给前端显示
fn assist_log(app: &AppHandle) -> impl Fn(String) + Sync {
    let a = app.clone();
    move |line: String| {
        let _ = a.emit("assist-log", line);
    }
}

#[tauri::command]
pub async fn sam_segment(
    app: AppHandle,
    state: State<'_, AppState>,
    image_id: i64,
    points: Vec<[f64; 3]>,
    bbox: Option<[f64; 4]>,
) -> AppResult<assist::SamResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let log = assist_log(&app);
        assist::sam_segment(&project, &app.state::<Runtime>(), &app.state::<Predictor>(), image_id, &points, bbox, &log)
    })
    .await
    .map_err(|e| msg(t!("SAM 异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn text_detect(app: AppHandle, state: State<'_, AppState>, image_id: i64, conf: f64) -> AppResult<PredResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let log = assist_log(&app);
        assist::text_detect(&project, &app.state::<Runtime>(), &app.state::<Predictor>(), image_id, conf, &log)
    })
    .await
    .map_err(|e| msg(t!("文本提示异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn text_prelabel_all(app: AppHandle, state: State<'_, AppState>, conf: f64) -> AppResult<PrelabelResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let _busy = BusyGuard::new(&BUSY_PRELABEL);
        let log = assist_log(&app);
        let progress = progress_emitter(app.clone(), "prelabel-progress");
        assist::text_prelabel_all(&project, &app.state::<Runtime>(), &app.state::<Predictor>(), conf, &progress, &log)
    })
    .await
    .map_err(|e| msg(t!("预标注异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn ai_models(app: AppHandle) -> AppResult<Vec<ai::AiModel>> {
    tauri::async_runtime::spawn_blocking(move || ai::models(&app.state::<Runtime>()))
        .await
        .map_err(|e| msg(t!("读取模型异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn ai_detect(app: AppHandle, state: State<'_, AppState>, image_id: i64, model: String, conf: f64) -> AppResult<PredResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let log = assist_log(&app);
        ai::detect(&project, &app.state::<Runtime>(), &app.state::<Predictor>(), image_id, &model, conf, &log)
    })
    .await
    .map_err(|e| msg(t!("巡瞳 AI 异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn ai_prelabel_all(app: AppHandle, state: State<'_, AppState>, model: String, conf: f64) -> AppResult<PrelabelResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let _busy = BusyGuard::new(&BUSY_PRELABEL);
        let log = assist_log(&app);
        let progress = progress_emitter(app.clone(), "prelabel-progress");
        ai::prelabel_all(&project, &app.state::<Runtime>(), &app.state::<Predictor>(), &model, conf, &progress, &log)
    })
    .await
    .map_err(|e| msg(t!("预标注异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn prelabel_all(app: AppHandle, state: State<'_, AppState>, run_id: i64, conf: f64) -> AppResult<PrelabelResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let _busy = BusyGuard::new(&BUSY_PRELABEL);
        let progress = progress_emitter(app.clone(), "prelabel-progress");
        training::prelabel_all(&project, &app.state::<Runtime>(), &app.state::<Predictor>(), run_id, conf, &progress)
    })
    .await
    .map_err(|e| msg(t!("预标注异常：{e}", e = e)))?
}

/// 在资源管理器里打开文件夹（或选中文件）
#[tauri::command]
pub async fn reveal_path(path: String) -> AppResult<()> {
    let p = PathBuf::from(&path);
    if !p.exists() {
        return Err(msg(t!("路径不存在")));
    }
    #[cfg(windows)]
    {
        let mut c = std::process::Command::new("explorer");
        if p.is_file() {
            c.arg(format!("/select,{}", p.display()));
        } else {
            c.arg(&p);
        }
        c.spawn()?;
    }
    #[cfg(target_os = "macos")]
    std::process::Command::new("open").arg("-R").arg(&p).spawn()?;
    #[cfg(all(unix, not(target_os = "macos")))]
    std::process::Command::new("xdg-open").arg(p.parent().unwrap_or(&p)).spawn()?;
    Ok(())
}

// ---------- 部署 ----------

#[tauri::command]
pub async fn live_start(app: AppHandle, state: State<'_, AppState>, run_id: i64, source: String, conf: f64) -> AppResult<()> {
    let project = state.current()?;
    training::ensure_deployable(&project, run_id, training::Deploy::Live)?;
    tauri::async_runtime::spawn_blocking(move || {
        // 这次实时验证的编号：进程结束（stopped）时只清掉自己的，不误清后来启动的
        let live_no = LIVE_NEXT.fetch_add(1, Ordering::Relaxed) + 1;
        LIVE_ON.store(live_no, Ordering::Relaxed);
        let a = app.clone();
        let sink: deploy::Sink = Arc::new(move |kind, data| {
            if kind == "stopped" {
                let _ = LIVE_ON.compare_exchange(live_no, 0, Ordering::Relaxed, Ordering::Relaxed);
            }
            let _ = a.emit("live", serde_json::json!({ "kind": kind, "data": data }));
        });
        let live = app.state::<Arc<deploy::Live>>().inner().clone();
        let r = live.start(project, &app.state::<Runtime>(), run_id, &source, conf, sink);
        if r.is_err() {
            let _ = LIVE_ON.compare_exchange(live_no, 0, Ordering::Relaxed, Ordering::Relaxed);
        }
        r
    })
    .await
    .map_err(|e| msg(t!("启动实时验证异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn live_stop(live: State<'_, Arc<deploy::Live>>) -> AppResult<()> {
    live.stop();
    LIVE_ON.store(0, Ordering::Relaxed);
    Ok(())
}

// ---------- 关闭软件、更新前：看看有没有正在进行的任务 ----------

static BUSY_STREAM: AtomicUsize = AtomicUsize::new(0);
static BUSY_VIDEO: AtomicUsize = AtomicUsize::new(0);
static BUSY_PRELABEL: AtomicUsize = AtomicUsize::new(0);
/// 正在运行的实时验证的编号，0 表示没有
static LIVE_ON: AtomicU64 = AtomicU64::new(0);
static LIVE_NEXT: AtomicU64 = AtomicU64::new(0);

/// 长任务进行中计数，结束（出错、panic 也算）时自动减一
struct BusyGuard(&'static AtomicUsize);

impl BusyGuard {
    fn new(c: &'static AtomicUsize) -> Self {
        c.fetch_add(1, Ordering::Relaxed);
        BusyGuard(c)
    }
}

impl Drop for BusyGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BusyTasks {
    /// 正在训练的记录 id
    pub training: Option<i64>,
    /// 正在安装训练环境（或补装组件）
    pub installing: bool,
    /// 实时验证
    pub live: bool,
    /// 实时协作共享
    pub share: bool,
    /// 视频流采集
    pub stream: bool,
    /// 本地视频抽帧
    pub video: bool,
    /// 批量预标注
    pub prelabel: bool,
}

impl BusyTasks {
    pub fn any(&self) -> bool {
        self.training.is_some() || self.installing || self.live || self.share || self.stream || self.video || self.prelabel
    }
}

pub fn busy_of(app: &AppHandle) -> BusyTasks {
    BusyTasks {
        training: app.state::<Arc<Jobs>>().running(),
        installing: *app.state::<Runtime>().installing.lock(),
        live: LIVE_ON.load(Ordering::Relaxed) != 0,
        share: app.state::<Share>().hub().is_some(),
        stream: BUSY_STREAM.load(Ordering::Relaxed) > 0,
        video: BUSY_VIDEO.load(Ordering::Relaxed) > 0,
        prelabel: BUSY_PRELABEL.load(Ordering::Relaxed) > 0,
    }
}

#[tauri::command]
pub async fn busy_tasks(app: AppHandle) -> AppResult<BusyTasks> {
    Ok(busy_of(&app))
}

/// 用户在确认框里点了「仍然关闭」
static QUIT_CONFIRMED: AtomicBool = AtomicBool::new(false);
/// 拦下关闭时发给前端的编号，前端收到后回 close_ack
static CLOSE_SEQ: AtomicU64 = AtomicU64::new(0);
static CLOSE_ACK: AtomicU64 = AtomicU64::new(0);

/// 关主窗口时还有任务在跑：发 app-close-blocked 让前端弹确认框，返回 true 表示要拦下。
/// 前端几秒内没回应（界面卡住、还没加载完）就直接退出，免得软件关不掉。
pub fn intercept_close(app: &AppHandle) -> bool {
    if QUIT_CONFIRMED.load(Ordering::Relaxed) {
        return false;
    }
    let busy = busy_of(app);
    if !busy.any() {
        return false;
    }
    let seq = CLOSE_SEQ.fetch_add(1, Ordering::Relaxed) + 1;
    let _ = app.emit("app-close-blocked", serde_json::json!({ "seq": seq, "busy": busy }));
    let a = app.clone();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(4));
        if CLOSE_ACK.load(Ordering::Relaxed) < seq {
            a.exit(0);
        }
    });
    true
}

/// 前端收到 app-close-blocked、已经弹出确认框
#[tauri::command]
pub async fn close_ack(seq: u64) -> AppResult<()> {
    CLOSE_ACK.fetch_max(seq, Ordering::Relaxed);
    Ok(())
}

/// 确认框里点了「仍然关闭」：退出软件（退出时会结束训练等后台进程）
#[tauri::command]
pub async fn quit_app(app: AppHandle) -> AppResult<()> {
    QUIT_CONFIRMED.store(true, Ordering::Relaxed);
    app.exit(0);
    Ok(())
}

#[tauri::command]
pub async fn live_conf(live: State<'_, Arc<deploy::Live>>, conf: f64) -> AppResult<()> {
    live.set_conf(conf)
}

#[tauri::command]
pub async fn live_snapshot(state: State<'_, AppState>, live: State<'_, Arc<deploy::Live>>) -> AppResult<()> {
    live.snapshot(&*state.current()?)
}

#[tauri::command]
pub async fn export_formats(
    app: AppHandle,
    state: State<'_, AppState>,
    run_id: i64,
    formats: Vec<String>,
    size: u32,
    out_dir: String,
) -> AppResult<Vec<deploy::FormatResult>> {
    let project = state.current()?;
    training::ensure_deployable(&project, run_id, training::Deploy::Onnx)?;
    tauri::async_runtime::spawn_blocking(move || {
        let a = app.clone();
        let step = deploy_step(app.clone());
        deploy::export_formats(&project, &app.state::<Runtime>(), run_id, &formats, size, &PathBuf::from(out_dir), &move |l| {
            let _ = a.emit("deploy-log", l);
        }, &step)
    })
    .await
    .map_err(|e| msg(t!("导出异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn build_service(
    app: AppHandle,
    state: State<'_, AppState>,
    run_id: i64,
    config: serde_json::Value,
    out_dir: String,
) -> AppResult<String> {
    let project = state.current()?;
    training::ensure_deployable(&project, run_id, training::Deploy::Service)?;
    tauri::async_runtime::spawn_blocking(move || {
        let a = app.clone();
        let step = deploy_step(app.clone());
        deploy::build_service(&project, &app.state::<Runtime>(), run_id, &config, &PathBuf::from(out_dir), &move |l| {
            let _ = a.emit("deploy-log", l);
        }, &step)
    })
    .await
    .map_err(|e| msg(t!("生成服务包异常：{e}", e = e)))?
}

/// 部署页导出、生成服务包的步骤进度（deploy-progress：done / total / name）
fn deploy_step(app: AppHandle) -> impl Fn(usize, usize, String) + Sync {
    move |done, total, name| {
        let _ = app.emit("deploy-progress", serde_json::json!({ "done": done, "total": total, "name": name }));
    }
}

/// 最近一次导入的那批图的 id（现场回流导入后直接去标注这批）
#[tauri::command]
pub async fn last_import_ids(state: State<'_, AppState>) -> AppResult<Vec<i64>> {
    let project = state.current()?;
    let conn = project.conn.lock();
    let mut stmt = conn.prepare("SELECT id FROM images WHERE batch = (SELECT MAX(batch) FROM images) AND batch > 0 ORDER BY id")?;
    let ids = stmt.query_map([], |r| r.get::<_, i64>(0))?.collect::<Result<Vec<_>, _>>()?;
    Ok(ids)
}

// ---------- 体检 ----------

#[tauri::command]
pub async fn health_load(state: State<'_, AppState>) -> AppResult<Option<HealthReport>> {
    health::load(&*state.current()?)
}

/// run_id 为空只做规则检查。同一时间只跑一个；进度推 health-progress，结束推 health-done（是否成功）
#[tauri::command]
pub async fn health_run(app: AppHandle, state: State<'_, AppState>, run_id: Option<i64>) -> AppResult<HealthReport> {
    let project = state.current()?;
    let guard = health::begin(&project)?;
    let done_app = app.clone();
    let res = tauri::async_runtime::spawn_blocking(move || {
        let _guard = guard;
        let prog_app = app.clone();
        let progress = move |stage: &str, done: usize, total: usize| {
            if let Some(pg) = health::note(stage, done, total) {
                let _ = prog_app.emit("health-progress", pg);
            }
        };
        let run = match run_id {
            Some(id) => Some((id, training::get_run(&project, id)?.name)),
            None => None,
        };
        let rt = app.state::<Runtime>();
        let pr = app.state::<Predictor>();
        let predict = |ids: &[i64], prog: &(dyn Fn(usize, usize) + Sync)| {
            training::predict_many(&project, &rt, &pr, run_id.unwrap(), ids, 0.05, prog)
        };
        let predict_ref: Option<&health::Predict<'_>> = if run_id.is_some() { Some(&predict) } else { None };
        health::run_with(&project, run, predict_ref, &health::STOP_HEALTH, &progress)
    })
    .await
    .map_err(|e| msg(t!("体检任务异常：{e}", e = e)))
    .and_then(|r| r);
    let _ = done_app.emit("health-done", res.is_ok());
    res
}

/// 正在跑的体检进度（离开体检页再回来时用），没在跑返回 null
#[tauri::command]
pub async fn health_status(state: State<'_, AppState>) -> AppResult<Option<health::HealthProgress>> {
    Ok(health::status(&*state.current()?))
}

/// 停止正在跑的体检，不保存报告
#[tauri::command]
pub async fn health_stop() -> AppResult<()> {
    health::stop();
    Ok(())
}

#[tauri::command]
pub async fn health_ignore(state: State<'_, AppState>, keys: Vec<String>) -> AppResult<()> {
    health::ignore(&*state.current()?, &keys)
}

/// 恢复这份体检报告里忽略掉的问题
#[tauri::command]
pub async fn health_unignore(state: State<'_, AppState>) -> AppResult<()> {
    health::unignore(&*state.current()?)
}

#[tauri::command]
pub async fn append_annotation(state: State<'_, AppState>, image_id: i64, annotation: Annotation) -> AppResult<String> {
    annotations::append(&*state.current()?, image_id, annotation)
}

// ---------- 数据集 ----------

#[tauri::command]
pub async fn unedited_prelabels(state: State<'_, AppState>) -> AppResult<Vec<i64>> {
    annotations::unedited_prelabels(&*state.current()?)
}

#[tauri::command]
pub async fn confirm_prelabels(state: State<'_, AppState>, ids: Vec<i64>) -> AppResult<usize> {
    annotations::confirm_prelabels(&*state.current()?, &ids)
}

#[tauri::command]
pub async fn split_summary(state: State<'_, AppState>) -> AppResult<SplitSummary> {
    dataset::split_summary(&*state.current()?)
}

#[tauri::command]
pub async fn assign_splits(state: State<'_, AppState>, ratios: [f64; 3], all: bool) -> AppResult<SplitSummary> {
    dataset::assign_splits(&*state.current()?, ratios, all)
}

#[tauri::command]
pub async fn list_split_images(state: State<'_, AppState>, split: String, offset: u32, limit: u32) -> AppResult<ImagePage> {
    dataset::list_split(&*state.current()?, &split, offset, limit)
}

#[tauri::command]
pub async fn set_split(state: State<'_, AppState>, ids: Vec<i64>, split: String) -> AppResult<SplitSummary> {
    dataset::set_split(&*state.current()?, &ids, &split)
}

fn progress_emitter(app: AppHandle, event: &'static str) -> impl Fn(usize, usize) + Sync {
    let last = std::sync::atomic::AtomicUsize::new(0);
    move |done, total| {
        let step = (total / 100).max(1);
        if done == total || done >= last.load(std::sync::atomic::Ordering::Relaxed) + step {
            last.store(done, std::sync::atomic::Ordering::Relaxed);
            let _ = app.emit(event, ImportProgress { done, total });
        }
    }
}

/// 视频标注：从上一个关键帧插值到这一帧
#[tauri::command]
pub async fn interpolate_frames(state: State<'_, AppState>, image_id: i64) -> AppResult<track::InterpResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || track::interpolate(&project, image_id))
        .await
        .map_err(|e| msg(t!("插值异常：{e}", e = e)))?
}

/// 视频标注：从这一帧往后跟踪 count 帧
#[tauri::command]
pub async fn track_frames(app: AppHandle, state: State<'_, AppState>, image_id: i64, count: usize) -> AppResult<track::TrackResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let rt = app.state::<Runtime>();
        let progress = progress_emitter(app.clone(), "track-progress");
        let log_app = app.clone();
        let log = move |m: String| {
            let _ = log_app.emit("track-log", m);
        };
        track::track(&project, &rt, image_id, count, &log, &progress)
    })
    .await
    .map_err(|e| msg(t!("跟踪异常：{e}", e = e)))?
}

/// 导出完再给导出的图片打码（导出时已经强制复制，不会动原图）
fn anonymize_export(app: &AppHandle, project: &Project, res: &ExportResult, a: &evaltools::Anonymize) -> AppResult<()> {
    let rt = app.state::<Runtime>();
    let log_app = app.clone();
    let log = move |m: String| {
        let _ = log_app.emit("export-log", m);
    };
    // 进度条从 0 重新走一遍
    let progress = progress_emitter(app.clone(), "export-progress");
    evaltools::anonymize(project, &rt, &res.files, a, &log, &progress)?;
    Ok(())
}

#[tauri::command]
pub async fn export_dataset(
    app: AppHandle,
    state: State<'_, AppState>,
    mut options: ExportOptions,
    anonymize: Option<evaltools::Anonymize>,
) -> AppResult<ExportResult> {
    let project = state.current()?;
    // 要打码就复制图片（不能用硬链接，不然会改到原图）
    if anonymize.is_some() {
        options.copy_files = true;
    }
    tauri::async_runtime::spawn_blocking(move || {
        let progress = progress_emitter(app.clone(), "export-progress");
        let res = dataset::export(&project, &options, &progress)?;
        if let Some(a) = anonymize {
            anonymize_export(&app, &project, &res, &a)?;
        }
        Ok(res)
    })
    .await
    .map_err(|e| msg(t!("导出任务异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn import_dataset(app: AppHandle, state: State<'_, AppState>, dir: String) -> AppResult<DatasetImportResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let progress = progress_emitter(app, "import-progress");
        dataset_import::import(&project, &PathBuf::from(dir), &progress)
    })
    .await
    .map_err(|e| msg(t!("导入任务异常：{e}", e = e)))?
}

/// 用训练好的模型批量识别一个文件夹：导出表格，可以按类别把图片复制到各自的文件夹
#[tauri::command]
pub async fn batch_predict(app: AppHandle, state: State<'_, AppState>, opts: crate::batch::BatchOptions) -> AppResult<crate::batch::BatchResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let progress = progress_emitter(app.clone(), "batch-progress");
        crate::batch::run(&project, &app.state::<Runtime>(), &app.state::<Predictor>(), &opts, &progress)
    })
    .await
    .map_err(|e| msg(t!("批量识别异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn batch_predict_stop() -> AppResult<()> {
    crate::batch::stop();
    Ok(())
}

/// 当前项目（检测、分割、关键点）转成多标签分类项目：新建项目，图片复制过去
#[tauri::command]
pub async fn convert_to_multilabel(
    app: AppHandle,
    state: State<'_, AppState>,
    parent: String,
    name: String,
    empty_label: Option<String>,
) -> AppResult<crate::convert::ConvertResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let progress = progress_emitter(app, "import-progress");
        crate::convert::to_multilabel(&project, &PathBuf::from(parent), &name, empty_label.as_deref(), &progress)
    })
    .await
    .map_err(|e| msg(t!("转换任务异常：{e}", e = e)))?
}

// ---------- 标注 ----------

#[tauri::command]
pub async fn list_image_ids(state: State<'_, AppState>, status: Option<String>, label: Option<i64>) -> AppResult<Vec<i64>> {
    annotations::list_ids(&*state.current()?, status.as_deref(), label)
}

#[tauri::command]
pub async fn load_annotations(state: State<'_, AppState>, image_id: i64) -> AppResult<ImageAnnotations> {
    annotations::load(&*state.current()?, image_id)
}

#[tauri::command]
pub async fn save_annotations(
    state: State<'_, AppState>,
    share: State<'_, Share>,
    image_id: i64,
    annotations: Vec<Annotation>,
) -> AppResult<String> {
    let project = state.current()?;
    // 正在共享时：别人锁着这张图就不让改；改完通知大家
    let hub = share.hub_for(&project);
    if let Some(h) = &hub {
        h.check_write(&h.host_member(), image_id)?;
    }
    let status = annotations::save(&project, image_id, &annotations)?;
    if let Some(h) = hub {
        h.saved(&h.host_member(), image_id, &status);
    }
    Ok(status)
}

#[tauri::command]
pub async fn set_image_status(state: State<'_, AppState>, share: State<'_, Share>, image_id: i64, status: String) -> AppResult<()> {
    let project = state.current()?;
    match share.hub_for(&project) {
        Some(h) => {
            let me = h.host_member();
            h.check_write(&me, image_id)?;
            h.set_status(&me, image_id, &status)?;
            Ok(())
        }
        None => annotations::set_status(&project, image_id, &status),
    }
}

/// 数据页批量改状态（设为跳过、恢复未标注），返回改了几张。共享中别人正在标的图不动
#[tauri::command]
pub async fn set_status_many(state: State<'_, AppState>, share: State<'_, Share>, ids: Vec<i64>, status: String) -> AppResult<usize> {
    let project = state.current()?;
    let hub = share.hub_for(&project);
    let ids: Vec<i64> = match &hub {
        Some(h) => {
            let me = h.host_member();
            ids.into_iter().filter(|id| h.check_write(&me, *id).is_ok()).collect()
        }
        None => ids,
    };
    let changed = annotations::set_status_many(&project, &ids, &status)?;
    if let Some(h) = hub {
        let me = h.host_member();
        for (id, s) in &changed {
            h.status_changed(&me, *id, s, &status);
        }
    }
    Ok(changed.len())
}

// ---------- 类别 ----------

#[tauri::command]
pub async fn list_labels(state: State<'_, AppState>) -> AppResult<Vec<Label>> {
    annotations::list_labels(&*state.current()?)
}

#[tauri::command]
pub async fn add_label(state: State<'_, AppState>, share: State<'_, Share>, name: String, color: String) -> AppResult<Label> {
    let project = state.current()?;
    let l = annotations::add_label(&project, &name, &color)?;
    if let Some(h) = share.hub_for(&project) {
        h.labels_changed(&h.host_member(), &t!("新增类别「{name}」", name = name));
    }
    Ok(l)
}

#[tauri::command]
pub async fn update_label(
    state: State<'_, AppState>,
    id: i64,
    name: String,
    color: String,
    prompt: Option<String>,
    description: Option<String>,
    share: State<'_, Share>,
) -> AppResult<()> {
    let project = state.current()?;
    annotations::update_label(&project, id, &name, &color, prompt.as_deref())?;
    if let Some(d) = description {
        annotations::set_label_description(&project, id, &d)?;
    }
    if let Some(h) = share.hub_for(&project) {
        h.labels_changed(&h.host_member(), &t!("修改类别「{name}」", name = name));
    }
    Ok(())
}

/// 中文类别没填英文提示词的，按对照表自动补上（文本提示自动标注用）
#[tauri::command]
pub async fn fill_label_prompts(state: State<'_, AppState>, share: State<'_, Share>) -> AppResult<usize> {
    let project = state.current()?;
    let n = annotations::fill_prompts(&project)?;
    if n > 0 {
        if let Some(h) = share.hub_for(&project) {
            h.labels_changed(&h.host_member(), &t!("自动补了英文提示词"));
        }
    }
    Ok(n)
}

#[tauri::command]
pub async fn delete_label(state: State<'_, AppState>, share: State<'_, Share>, id: i64) -> AppResult<usize> {
    let project = state.current()?;
    let n = annotations::delete_label(&project, id)?;
    if let Some(h) = share.hub_for(&project) {
        h.labels_changed(&h.host_member(), &t!("删除了一个类别"));
    }
    Ok(n)
}

// ---------- 标注页：缩略图状态、合并类别、停止批量预标注 ----------

/// 全部图片的 (id, 状态)，缩略图条上的状态点用
#[tauri::command]
pub async fn image_statuses(state: State<'_, AppState>) -> AppResult<Vec<(i64, String)>> {
    annotations::statuses(&*state.current()?)
}

/// 把类别 from 的标注合并到类别 to，再删掉 from
#[tauri::command]
pub async fn merge_label(state: State<'_, AppState>, share: State<'_, Share>, from: i64, to: i64) -> AppResult<usize> {
    let project = state.current()?;
    let n = annotations::merge_label(&project, from, to)?;
    if let Some(h) = share.hub_for(&project) {
        h.labels_changed(&h.host_member(), &t!("合并了两个类别"));
    }
    Ok(n)
}

/// 停止批量预标注，已经标好的保留
#[tauri::command]
pub async fn stop_prelabel() -> AppResult<()> {
    training::prelabel_stop(true);
    Ok(())
}

// ---------- 实时协作（主机这边） ----------

fn host_name(share: &Share, p: &Project) -> String {
    share.hub_for(p).map(|h| h.opts.lock().host_name.clone()).unwrap_or_else(|| share::load_options(p).host_name)
}

#[tauri::command]
pub async fn share_status(state: State<'_, AppState>, share: State<'_, Share>) -> AppResult<ShareStatus> {
    let p = state.try_current();
    Ok(share.status(p.as_deref()))
}

#[tauri::command]
pub async fn share_start(app: AppHandle, state: State<'_, AppState>, share: State<'_, Share>, options: ShareOptions) -> AppResult<ShareStatus> {
    let project = state.current()?;
    share.start(app, project, options).await
}

#[tauri::command]
pub async fn share_stop(share: State<'_, Share>) -> AppResult<()> {
    share.stop();
    Ok(())
}

/// 改协作选项。正在共享时立即生效（端口、共享方式要重新开启才生效）
#[tauri::command]
pub async fn share_save_options(state: State<'_, AppState>, share: State<'_, Share>, options: ShareOptions) -> AppResult<()> {
    let project = state.current()?;
    share::save_options(&project, &options)?;
    if let Some(h) = share.hub_for(&project) {
        {
            let mut o = h.opts.lock();
            o.host_name = options.host_name.clone();
            o.allow_ai = options.allow_ai;
            o.watermark = options.watermark;
            o.members_edit_labels = options.members_edit_labels;
        }
        h.broadcast(
            "options",
            serde_json::json!({ "allowAi": options.allow_ai, "watermark": options.watermark, "membersEditLabels": options.members_edit_labels }),
        );
        h.emit_status();
    }
    Ok(())
}

/// 主机打开一张图：正在共享时加锁，返回锁的人（别人在改就只读）和这张图的分配、审核信息
#[tauri::command]
pub async fn share_lock(state: State<'_, AppState>, share: State<'_, Share>, image_id: i64) -> AppResult<serde_json::Value> {
    let project = state.current()?;
    let holder = share.hub_for(&project).and_then(|h| h.lock(&h.host_member(), image_id));
    let info = team::image_info(&project, image_id, &host_name(&share, &project))?;
    Ok(serde_json::json!({ "holder": holder, "info": info }))
}

#[tauri::command]
pub async fn share_unlock(state: State<'_, AppState>, share: State<'_, Share>) -> AppResult<()> {
    let project = state.current()?;
    if let Some(h) = share.hub_for(&project) {
        h.release(team::HOST);
    }
    Ok(())
}

#[tauri::command]
pub async fn share_review(state: State<'_, AppState>, share: State<'_, Share>, image_id: i64, pass: bool, reason: String) -> AppResult<String> {
    let project = state.current()?;
    match share.hub_for(&project) {
        Some(h) => h.review(&h.host_member(), image_id, pass, &reason),
        None => team::review(&project, team::HOST, image_id, pass, &reason),
    }
}

// ---------- 审核页（发起人这边；成员走 share.rs 的同名接口） ----------

#[tauri::command]
pub async fn review_query(
    state: State<'_, AppState>,
    share: State<'_, Share>,
    query: crate::review::ReviewQuery,
    offset: u32,
    limit: u32,
) -> AppResult<crate::review::ReviewPage> {
    let project = state.current()?;
    let name = host_name(&share, &project);
    crate::review::query(&project, &Member::host(&name), &query, offset, limit, &name)
}

#[tauri::command]
pub async fn review_ids(state: State<'_, AppState>, share: State<'_, Share>, query: crate::review::ReviewQuery) -> AppResult<Vec<i64>> {
    let project = state.current()?;
    crate::review::ids(&project, &Member::host(&host_name(&share, &project)), &query)
}

#[tauri::command]
pub async fn review_members(
    state: State<'_, AppState>,
    share: State<'_, Share>,
    query: crate::review::ReviewQuery,
) -> AppResult<Vec<crate::review::MemberSummary>> {
    let project = state.current()?;
    let name = host_name(&share, &project);
    crate::review::members(&project, &Member::host(&name), &query, &name)
}

#[tauri::command]
pub async fn review_many(
    state: State<'_, AppState>,
    share: State<'_, Share>,
    ids: Vec<i64>,
    pass: bool,
    reason: Option<String>,
) -> AppResult<crate::review::BatchResult> {
    let project = state.current()?;
    let reason = reason.unwrap_or_default();
    match share.hub_for(&project) {
        Some(h) => h.review_many(&h.host_member(), &ids, pass, &reason),
        None => crate::review::review_many(&project, &Member::host(&host_name(&share, &project)), &ids, pass, &reason, &|_| false),
    }
}

#[tauri::command]
pub async fn review_undo(state: State<'_, AppState>, share: State<'_, Share>, token: String) -> AppResult<Vec<i64>> {
    let project = state.current()?;
    match share.hub_for(&project) {
        Some(h) => h.review_undo(&h.host_member(), &token),
        None => crate::review::undo(&project, &Member::host(&host_name(&share, &project)), &token),
    }
}

#[tauri::command]
pub async fn team_members(state: State<'_, AppState>) -> AppResult<Vec<Member>> {
    team::list_members(&*state.current()?)
}

#[tauri::command]
pub async fn team_add_member(state: State<'_, AppState>, name: String, role: String, password: String) -> AppResult<Member> {
    team::add_member(&*state.current()?, &name, &role, &password)
}

#[tauri::command]
pub async fn team_update_member(state: State<'_, AppState>, share: State<'_, Share>, id: i64, patch: MemberPatch) -> AppResult<Member> {
    let project = state.current()?;
    let m = team::update_member(&project, id, &patch)?;
    if let Some(h) = share.hub_for(&project) {
        // 停用、改了密码就让他重新登录
        if patch.disabled == Some(true) || patch.password.as_deref().is_some_and(|s| !s.is_empty()) {
            h.kick(id);
        }
        h.emit_status();
    }
    Ok(m)
}

#[tauri::command]
pub async fn team_delete_member(state: State<'_, AppState>, share: State<'_, Share>, id: i64) -> AppResult<()> {
    let project = state.current()?;
    team::delete_member(&project, id)?;
    if let Some(h) = share.hub_for(&project) {
        h.kick(id);
        h.broadcast("assign", serde_json::json!({}));
    }
    Ok(())
}

#[tauri::command]
pub async fn team_assign(state: State<'_, AppState>, share: State<'_, Share>, options: AssignOptions) -> AppResult<Vec<AssignResult>> {
    let project = state.current()?;
    let r = team::assign(&project, &options)?;
    let total: usize = r.iter().map(|x| x.count).sum();
    team::log(&project, team::HOST, None, "assign", None, total as i64, &t!("分配 {total} 张", total = total))?;
    if let Some(h) = share.hub_for(&project) {
        h.broadcast("assign", serde_json::json!({}));
    }
    Ok(r)
}

#[tauri::command]
pub async fn team_unassign(state: State<'_, AppState>, share: State<'_, Share>, member_id: i64, only_unfinished: bool) -> AppResult<usize> {
    let project = state.current()?;
    let n = team::unassign(&project, member_id, only_unfinished)?;
    if let Some(h) = share.hub_for(&project) {
        h.broadcast("assign", serde_json::json!({}));
    }
    Ok(n)
}

#[tauri::command]
pub async fn team_stats(state: State<'_, AppState>, share: State<'_, Share>, from: i64, to: i64) -> AppResult<Vec<team::MemberStats>> {
    let project = state.current()?;
    let name = host_name(&share, &project);
    tauri::async_runtime::spawn_blocking(move || team::stats(&project, from, to, &name))
        .await
        .map_err(|e| msg(t!("统计异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn team_export_csv(state: State<'_, AppState>, share: State<'_, Share>, from: i64, to: i64, path: String) -> AppResult<usize> {
    let project = state.current()?;
    let name = host_name(&share, &project);
    tauri::async_runtime::spawn_blocking(move || team::export_csv(&project, from, to, &name, &PathBuf::from(path)))
        .await
        .map_err(|e| msg(t!("导出异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn team_activity(
    state: State<'_, AppState>,
    share: State<'_, Share>,
    limit: usize,
    member_id: Option<i64>,
    with_saves: bool,
) -> AppResult<Vec<team::ActivityRow>> {
    let project = state.current()?;
    let name = host_name(&share, &project);
    team::activity(&project, limit.min(2000), member_id, with_saves, &name)
}

// ---------- 巡瞳 AI：识别规则、聊天定规则 ----------

#[tauri::command]
pub async fn ai_rules(state: State<'_, AppState>) -> AppResult<ai::Rules> {
    ai::rules(&*state.current()?)
}

#[tauri::command]
pub async fn ai_save_rules(
    state: State<'_, AppState>,
    share: State<'_, Share>,
    instructions: Option<String>,
    classes: Vec<ai::ClassRule>,
    app: AppHandle,
) -> AppResult<Option<String>> {
    let project = state.current()?;
    ai::save_rules(&project, instructions.as_deref(), &classes)?;
    if let Some(h) = share.hub_for(&project) {
        h.labels_changed(&h.host_member(), &t!("更新了识别规则"));
    }
    sync_if_bound(app, project).await
}

/// 绑定了巡瞳平台规则的项目，改完规则顺手同步；同步失败只提醒，不影响本地保存
async fn sync_if_bound(app: AppHandle, project: Arc<Project>) -> AppResult<Option<String>> {
    if ai::rules(&project)?.rule_set.is_empty() {
        return Ok(None);
    }
    let r = tauri::async_runtime::spawn_blocking(move || ai::sync_rules(&project, &app.state::<Runtime>()))
        .await
        .map_err(|e| msg(t!("同步异常：{e}", e = e)))?;
    Ok(r.err().map(|e| t!("规则已保存，但同步到巡瞳平台失败：{e}", e = e)))
}

/// 把识别规则同步到巡瞳平台（没绑定就新建并绑定），返回平台规则编号
#[tauri::command]
pub async fn ai_sync_rules(app: AppHandle, state: State<'_, AppState>) -> AppResult<String> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || ai::sync_rules(&project, &app.state::<Runtime>()))
        .await
        .map_err(|e| msg(t!("同步异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn ai_unbind_rules(state: State<'_, AppState>) -> AppResult<()> {
    ai::unbind_rules(&*state.current()?)
}

#[tauri::command]
pub async fn ai_chat_rules(app: AppHandle, state: State<'_, AppState>, messages: Vec<ai::ChatMsg>) -> AppResult<ai::ChatReply> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let log = assist_log(&app);
        ai::chat_rules(&project, &app.state::<Runtime>(), &app.state::<Predictor>(), &messages, &log)
    })
    .await
    .map_err(|e| msg(t!("聊天异常：{e}", e = e)))?
}

// ---------- 巡瞳 AI 平台：上传模型、拉告警图片 ----------

#[tauri::command]
pub async fn xt_upload_model(app: AppHandle, state: State<'_, AppState>, run_id: i64, name: String) -> AppResult<crate::xuntong::UploadResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let a = app.clone();
        let log = move |line: String| {
            let _ = a.emit("xuntong-log", line);
        };
        crate::xuntong::upload_model(&project, &app.state::<Runtime>(), run_id, &name, &log)
    })
    .await
    .map_err(|e| msg(t!("上传异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn xt_models(app: AppHandle) -> AppResult<Vec<crate::xuntong::PlatformModel>> {
    tauri::async_runtime::spawn_blocking(move || crate::xuntong::list_models(&app.state::<Runtime>()))
        .await
        .map_err(|e| msg(t!("读取异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn xt_pull_alarms(app: AppHandle, state: State<'_, AppState>, options: crate::xuntong::PullOptions) -> AppResult<crate::xuntong::PullResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let progress = progress_emitter(app.clone(), "xuntong-progress");
        let a = app.clone();
        let log = move |line: String| {
            let _ = a.emit("xuntong-log", line);
        };
        crate::xuntong::pull_alarms(&project, &app.state::<Runtime>(), &app.state::<Predictor>(), &options, &progress, &log)
    })
    .await
    .map_err(|e| msg(t!("拉取异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn xt_delete_model(app: AppHandle, id: String) -> AppResult<()> {
    tauri::async_runtime::spawn_blocking(move || crate::xuntong::delete_model(&app.state::<Runtime>(), &id))
        .await
        .map_err(|e| msg(t!("删除异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn ai_rule_set_info(app: AppHandle, state: State<'_, AppState>) -> AppResult<serde_json::Value> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || ai::rule_set_info(&project, &app.state::<Runtime>()))
        .await
        .map_err(|e| msg(t!("读取异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn ai_rules_platform_enabled(app: AppHandle, state: State<'_, AppState>, enabled: bool) -> AppResult<()> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || ai::set_platform_enabled(&project, &app.state::<Runtime>(), enabled))
        .await
        .map_err(|e| msg(t!("设置异常：{e}", e = e)))?
}

// ---------- 登录巡瞳账号 ----------

#[tauri::command]
pub async fn xt_login_start(app: AppHandle) -> AppResult<crate::xuntong::DeviceCode> {
    tauri::async_runtime::spawn_blocking(move || crate::xuntong::login_start(&app.state::<Runtime>()))
        .await
        .map_err(|e| msg(t!("登录异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn xt_login_poll(app: AppHandle, device_code: String) -> AppResult<serde_json::Value> {
    tauri::async_runtime::spawn_blocking(move || crate::xuntong::login_poll(&app.state::<Runtime>(), &device_code))
        .await
        .map_err(|e| msg(t!("登录异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn xt_account(app: AppHandle) -> AppResult<serde_json::Value> {
    tauri::async_runtime::spawn_blocking(move || crate::xuntong::account(&app.state::<Runtime>()))
        .await
        .map_err(|e| msg(t!("读取异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn xt_logout(app: AppHandle) -> AppResult<()> {
    tauri::async_runtime::spawn_blocking(move || crate::xuntong::logout(&app.state::<Runtime>()))
        .await
        .map_err(|e| msg(t!("退出异常：{e}", e = e)))?
}

#[tauri::command]
pub async fn open_url(url: String) -> AppResult<()> {
    crate::xuntong::open_url(&url)
}
