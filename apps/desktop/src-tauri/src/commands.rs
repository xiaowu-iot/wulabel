use std::path::PathBuf;
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
    tauri::async_runtime::spawn_blocking(move || {
        let files = images::collect(&paths);
        if files.is_empty() {
            return Err(msg("没有找到图片（支持 jpg、png、bmp、webp、tif）"));
        }
        let last = std::sync::atomic::AtomicUsize::new(0);
        images::import(&project, &files, copy, &|done, total| {
            // 每 1% 或最后一张推一次进度，避免事件太多
            let step = (total / 100).max(1);
            if done == total || done >= last.load(std::sync::atomic::Ordering::Relaxed) + step {
                last.store(done, std::sync::atomic::Ordering::Relaxed);
                let _ = app.emit("import-progress", ImportProgress { done, total });
            }
        })
    })
    .await
    .map_err(|e| msg(format!("导入任务异常：{e}")))?
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
        let rt = app.state::<Runtime>();
        let a = app.clone();
        let progress = move |stage: &str, done: u64, total: u64, speed: u64, detail: &str| {
            let _ = a.emit(
                "video-progress",
                VideoProgress { stage: stage.to_string(), done, total, speed, detail: detail.to_string() },
            );
        };
        video::extract(&project, &rt, &paths, every, max_frames, &progress, &|_line| {})
    })
    .await
    .map_err(|e| msg(format!("抽帧任务异常：{e}")))?
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
    .map_err(|e| msg(format!("探测任务异常：{e}")))?
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
    .map_err(|e| msg(format!("采集任务异常：{e}")))?
}

// ---------- 外部模型、重新评估、误报测试 ----------

#[tauri::command]
pub async fn inspect_onnx(app: AppHandle, path: String) -> AppResult<serde_json::Value> {
    tauri::async_runtime::spawn_blocking(move || evaltools::inspect_onnx(&app.state::<Runtime>(), &path))
        .await
        .map_err(|e| msg(format!("读取模型异常：{e}")))?
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
        evaltools::evaluate_run(&project, &app.state::<Runtime>(), run_id, &move |l| {
            let _ = a.emit("eval-log", l);
        })
    })
    .await
    .map_err(|e| msg(format!("评估异常：{e}")))?
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
    .map_err(|e| msg(format!("误报测试异常：{e}")))?
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
    .map_err(|e| msg(format!("分包异常：{e}")))?
}

#[tauri::command]
pub async fn list_packages(state: State<'_, AppState>) -> AppResult<Vec<collab::PackageInfo>> {
    let project = state.current()?;
    collab::list(&project)
}

#[tauri::command]
pub async fn merge_package(state: State<'_, AppState>, options: collab::MergeOptions) -> AppResult<collab::MergeResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || collab::merge(&project, &options))
        .await
        .map_err(|e| msg(format!("合并异常：{e}")))?
}

#[tauri::command]
pub async fn export_report(state: State<'_, AppState>, run_id: i64, path: String) -> AppResult<()> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || report::export(&project, run_id, &PathBuf::from(path)))
        .await
        .map_err(|e| msg(format!("导出报告异常：{e}")))?
}

#[tauri::command]
pub async fn quality_scan(app: AppHandle, state: State<'_, AppState>, only_todo: bool) -> AppResult<quality::QualityReport> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let progress = progress_emitter(app.clone(), "quality-progress");
        quality::scan(&project, only_todo, &progress)
    })
    .await
    .map_err(|e| msg(format!("质量筛查异常：{e}")))?
}

#[tauri::command]
pub async fn uncertain_images(app: AppHandle, state: State<'_, AppState>, run_id: i64, limit: usize) -> AppResult<Vec<training::Uncertain>> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let progress = progress_emitter(app.clone(), "prelabel-progress");
        training::uncertain_images(&project, &app.state::<Runtime>(), &app.state::<Predictor>(), run_id, limit, &progress)
    })
    .await
    .map_err(|e| msg(format!("挑图异常：{e}")))?
}

#[tauri::command]
pub async fn aug_preview(app: AppHandle, state: State<'_, AppState>, aug: serde_json::Value, img_size: u32) -> AppResult<Vec<String>> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || evaltools::aug_preview(&project, &app.state::<Runtime>(), &aug, img_size, 4))
        .await
        .map_err(|e| msg(format!("预览异常：{e}")))?
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
    .map_err(|e| msg(format!("导出异常：{e}")))?
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
    offset: u32,
    limit: u32,
) -> AppResult<ImagePage> {
    let project = state.current()?;
    images::list(&project, status.as_deref(), offset, limit)
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
pub async fn delete_images(state: State<'_, AppState>, scope: String, ids: Option<Vec<i64>>) -> AppResult<usize> {
    let project = state.current()?;
    let scope = match scope.as_str() {
        "ids" => DeleteScope::Ids(ids.unwrap_or_default()),
        "last" => DeleteScope::LastImport,
        "all" => DeleteScope::All,
        s => return Err(msg(format!("不支持的删除范围：{s}"))),
    };
    tauri::async_runtime::spawn_blocking(move || images::delete(&project, scope))
        .await
        .map_err(|e| msg(format!("删除任务异常：{e}")))?
}

// ---------- 训练环境 ----------

/// 检查要启动 Python 导入 torch，放到后台线程
#[tauri::command]
pub async fn env_status(app: AppHandle, refresh: bool) -> AppResult<EnvStatus> {
    tauri::async_runtime::spawn_blocking(move || app.state::<Runtime>().status(refresh))
        .await
        .map_err(|e| msg(format!("环境检查异常：{e}")))
}

#[tauri::command]
pub async fn save_settings(rt: State<'_, Runtime>, settings: Settings) -> AppResult<()> {
    rt.save_settings(&settings)
}

#[tauri::command]
pub async fn env_install(app: AppHandle, variant: String) -> AppResult<()> {
    tauri::async_runtime::spawn_blocking(move || {
        let rt = app.state::<Runtime>();
        let a = app.clone();
        let ap = app.clone();
        rt.install(
            &variant,
            &move |line| {
                let _ = a.emit("env-log", line);
            },
            &move |file, downloaded, total, speed| {
                let _ = ap.emit(
                    "env-progress",
                    serde_json::json!({ "file": file, "downloaded": downloaded, "total": total, "speed": speed }),
                );
            },
        )
    })
    .await
    .map_err(|e| msg(format!("安装任务异常：{e}")))?
}

/// 实测当前网络下国内镜像 / 官方源的下载速度，推荐更快的
#[tauri::command]
pub async fn speed_test(variant: String) -> AppResult<SpeedResult> {
    tauri::async_runtime::spawn_blocking(move || runtime::speed_test(&variant))
        .await
        .map_err(|e| msg(format!("测速任务异常：{e}")))?
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
    let a = app.clone();
    let sink: training::EventSink = Arc::new(move |id, ev| {
        let _ = a.emit("train-event", serde_json::json!({ "runId": id, "event": ev }));
    });
    tauri::async_runtime::spawn_blocking(move || {
        let rt = app.state::<Runtime>();
        training::start(project, &rt, jobs, config, sink)
    })
    .await
    .map_err(|e| msg(format!("启动训练异常：{e}")))?
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
    .map_err(|e| msg(format!("导出异常：{e}")))?
}

#[tauri::command]
pub async fn predict_image(app: AppHandle, state: State<'_, AppState>, run_id: i64, image_id: i64, conf: f64) -> AppResult<PredResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        training::predict_one(&project, &app.state::<Runtime>(), &app.state::<Predictor>(), run_id, image_id, conf)
    })
    .await
    .map_err(|e| msg(format!("预测异常：{e}")))?
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
    .map_err(|e| msg(format!("SAM 异常：{e}")))?
}

#[tauri::command]
pub async fn text_detect(app: AppHandle, state: State<'_, AppState>, image_id: i64, conf: f64) -> AppResult<PredResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let log = assist_log(&app);
        assist::text_detect(&project, &app.state::<Runtime>(), &app.state::<Predictor>(), image_id, conf, &log)
    })
    .await
    .map_err(|e| msg(format!("文本提示异常：{e}")))?
}

#[tauri::command]
pub async fn text_prelabel_all(app: AppHandle, state: State<'_, AppState>, conf: f64) -> AppResult<PrelabelResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let log = assist_log(&app);
        let progress = progress_emitter(app.clone(), "prelabel-progress");
        assist::text_prelabel_all(&project, &app.state::<Runtime>(), &app.state::<Predictor>(), conf, &progress, &log)
    })
    .await
    .map_err(|e| msg(format!("预标注异常：{e}")))?
}

#[tauri::command]
pub async fn ai_models(app: AppHandle) -> AppResult<Vec<ai::AiModel>> {
    tauri::async_runtime::spawn_blocking(move || ai::models(&app.state::<Runtime>()))
        .await
        .map_err(|e| msg(format!("读取模型异常：{e}")))?
}

#[tauri::command]
pub async fn ai_detect(app: AppHandle, state: State<'_, AppState>, image_id: i64, model: String, conf: f64) -> AppResult<PredResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let log = assist_log(&app);
        ai::detect(&project, &app.state::<Runtime>(), &app.state::<Predictor>(), image_id, &model, conf, &log)
    })
    .await
    .map_err(|e| msg(format!("巡瞳 AI 异常：{e}")))?
}

#[tauri::command]
pub async fn ai_prelabel_all(app: AppHandle, state: State<'_, AppState>, model: String, conf: f64) -> AppResult<PrelabelResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let log = assist_log(&app);
        let progress = progress_emitter(app.clone(), "prelabel-progress");
        ai::prelabel_all(&project, &app.state::<Runtime>(), &app.state::<Predictor>(), &model, conf, &progress, &log)
    })
    .await
    .map_err(|e| msg(format!("预标注异常：{e}")))?
}

#[tauri::command]
pub async fn prelabel_all(app: AppHandle, state: State<'_, AppState>, run_id: i64, conf: f64) -> AppResult<PrelabelResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let progress = progress_emitter(app.clone(), "prelabel-progress");
        training::prelabel_all(&project, &app.state::<Runtime>(), &app.state::<Predictor>(), run_id, conf, &progress)
    })
    .await
    .map_err(|e| msg(format!("预标注异常：{e}")))?
}

/// 在资源管理器里打开文件夹（或选中文件）
#[tauri::command]
pub async fn reveal_path(path: String) -> AppResult<()> {
    let p = PathBuf::from(&path);
    if !p.exists() {
        return Err(msg("路径不存在"));
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
        let a = app.clone();
        let sink: deploy::Sink = Arc::new(move |kind, data| {
            let _ = a.emit("live", serde_json::json!({ "kind": kind, "data": data }));
        });
        let live = app.state::<Arc<deploy::Live>>().inner().clone();
        live.start(project, &app.state::<Runtime>(), run_id, &source, conf, sink)
    })
    .await
    .map_err(|e| msg(format!("启动实时验证异常：{e}")))?
}

#[tauri::command]
pub async fn live_stop(live: State<'_, Arc<deploy::Live>>) -> AppResult<()> {
    live.stop();
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
        deploy::export_formats(&project, &app.state::<Runtime>(), run_id, &formats, size, &PathBuf::from(out_dir), &move |l| {
            let _ = a.emit("deploy-log", l);
        })
    })
    .await
    .map_err(|e| msg(format!("导出异常：{e}")))?
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
        deploy::build_service(&project, &app.state::<Runtime>(), run_id, &config, &PathBuf::from(out_dir), &move |l| {
            let _ = a.emit("deploy-log", l);
        })
    })
    .await
    .map_err(|e| msg(format!("生成服务包异常：{e}")))?
}

// ---------- 体检 ----------

#[tauri::command]
pub async fn health_load(state: State<'_, AppState>) -> AppResult<Option<HealthReport>> {
    health::load(&*state.current()?)
}

/// run_id 为空只做规则检查
#[tauri::command]
pub async fn health_run(app: AppHandle, state: State<'_, AppState>, run_id: Option<i64>) -> AppResult<HealthReport> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let progress = progress_emitter(app.clone(), "health-progress");
        let run = match run_id {
            Some(id) => Some((id, training::get_run(&project, id)?.name)),
            None => None,
        };
        let rt = app.state::<Runtime>();
        let pr = app.state::<Predictor>();
        let predict = |ids: &[i64]| training::predict_many(&project, &rt, &pr, run_id.unwrap(), ids, 0.05, &progress);
        let predict_ref: Option<&dyn Fn(&[i64]) -> AppResult<_>> = if run_id.is_some() { Some(&predict) } else { None };
        health::run(&project, run, predict_ref, &progress)
    })
    .await
    .map_err(|e| msg(format!("体检任务异常：{e}")))?
}

#[tauri::command]
pub async fn health_ignore(state: State<'_, AppState>, keys: Vec<String>) -> AppResult<()> {
    health::ignore(&*state.current()?, &keys)
}

#[tauri::command]
pub async fn append_annotation(state: State<'_, AppState>, image_id: i64, annotation: Annotation) -> AppResult<String> {
    annotations::append(&*state.current()?, image_id, annotation)
}

// ---------- 数据集 ----------

#[tauri::command]
pub async fn split_summary(state: State<'_, AppState>) -> AppResult<SplitSummary> {
    dataset::split_summary(&*state.current()?)
}

#[tauri::command]
pub async fn assign_splits(state: State<'_, AppState>, ratios: [f64; 3], all: bool) -> AppResult<SplitSummary> {
    dataset::assign_splits(&*state.current()?, ratios, all)
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
        .map_err(|e| msg(format!("插值异常：{e}")))?
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
    .map_err(|e| msg(format!("跟踪异常：{e}")))?
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
    .map_err(|e| msg(format!("导出任务异常：{e}")))?
}

#[tauri::command]
pub async fn import_dataset(app: AppHandle, state: State<'_, AppState>, dir: String) -> AppResult<DatasetImportResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let progress = progress_emitter(app, "import-progress");
        dataset_import::import(&project, &PathBuf::from(dir), &progress)
    })
    .await
    .map_err(|e| msg(format!("导入任务异常：{e}")))?
}

// ---------- 标注 ----------

#[tauri::command]
pub async fn list_image_ids(state: State<'_, AppState>, status: Option<String>) -> AppResult<Vec<i64>> {
    annotations::list_ids(&*state.current()?, status.as_deref())
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
        h.labels_changed(&h.host_member(), &format!("新增类别「{name}」"));
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
        h.labels_changed(&h.host_member(), &format!("修改类别「{name}」"));
    }
    Ok(())
}

#[tauri::command]
pub async fn delete_label(state: State<'_, AppState>, share: State<'_, Share>, id: i64) -> AppResult<usize> {
    let project = state.current()?;
    let n = annotations::delete_label(&project, id)?;
    if let Some(h) = share.hub_for(&project) {
        h.labels_changed(&h.host_member(), "删除了一个类别");
    }
    Ok(n)
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
    team::log(&project, team::HOST, None, "assign", None, total as i64, &format!("分配 {total} 张"))?;
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
        .map_err(|e| msg(format!("统计异常：{e}")))?
}

#[tauri::command]
pub async fn team_export_csv(state: State<'_, AppState>, share: State<'_, Share>, from: i64, to: i64, path: String) -> AppResult<usize> {
    let project = state.current()?;
    let name = host_name(&share, &project);
    tauri::async_runtime::spawn_blocking(move || team::export_csv(&project, from, to, &name, &PathBuf::from(path)))
        .await
        .map_err(|e| msg(format!("导出异常：{e}")))?
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
        h.labels_changed(&h.host_member(), "更新了识别规则");
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
        .map_err(|e| msg(format!("同步异常：{e}")))?;
    Ok(r.err().map(|e| format!("规则已保存，但同步到巡瞳平台失败：{e}")))
}

/// 把识别规则同步到巡瞳平台（没绑定就新建并绑定），返回平台规则编号
#[tauri::command]
pub async fn ai_sync_rules(app: AppHandle, state: State<'_, AppState>) -> AppResult<String> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || ai::sync_rules(&project, &app.state::<Runtime>()))
        .await
        .map_err(|e| msg(format!("同步异常：{e}")))?
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
    .map_err(|e| msg(format!("聊天异常：{e}")))?
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
    .map_err(|e| msg(format!("上传异常：{e}")))?
}

#[tauri::command]
pub async fn xt_models(app: AppHandle) -> AppResult<Vec<crate::xuntong::PlatformModel>> {
    tauri::async_runtime::spawn_blocking(move || crate::xuntong::list_models(&app.state::<Runtime>()))
        .await
        .map_err(|e| msg(format!("读取异常：{e}")))?
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
    .map_err(|e| msg(format!("拉取异常：{e}")))?
}

#[tauri::command]
pub async fn xt_delete_model(app: AppHandle, id: String) -> AppResult<()> {
    tauri::async_runtime::spawn_blocking(move || crate::xuntong::delete_model(&app.state::<Runtime>(), &id))
        .await
        .map_err(|e| msg(format!("删除异常：{e}")))?
}

#[tauri::command]
pub async fn ai_rule_set_info(app: AppHandle, state: State<'_, AppState>) -> AppResult<serde_json::Value> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || ai::rule_set_info(&project, &app.state::<Runtime>()))
        .await
        .map_err(|e| msg(format!("读取异常：{e}")))?
}

#[tauri::command]
pub async fn ai_rules_platform_enabled(app: AppHandle, state: State<'_, AppState>, enabled: bool) -> AppResult<()> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || ai::set_platform_enabled(&project, &app.state::<Runtime>(), enabled))
        .await
        .map_err(|e| msg(format!("设置异常：{e}")))?
}

// ---------- 登录巡瞳账号 ----------

#[tauri::command]
pub async fn xt_login_start(app: AppHandle) -> AppResult<crate::xuntong::DeviceCode> {
    tauri::async_runtime::spawn_blocking(move || crate::xuntong::login_start(&app.state::<Runtime>()))
        .await
        .map_err(|e| msg(format!("登录异常：{e}")))?
}

#[tauri::command]
pub async fn xt_login_poll(app: AppHandle, device_code: String) -> AppResult<serde_json::Value> {
    tauri::async_runtime::spawn_blocking(move || crate::xuntong::login_poll(&app.state::<Runtime>(), &device_code))
        .await
        .map_err(|e| msg(format!("登录异常：{e}")))?
}

#[tauri::command]
pub async fn xt_account(app: AppHandle) -> AppResult<serde_json::Value> {
    tauri::async_runtime::spawn_blocking(move || crate::xuntong::account(&app.state::<Runtime>()))
        .await
        .map_err(|e| msg(format!("读取异常：{e}")))?
}

#[tauri::command]
pub async fn xt_logout(app: AppHandle) -> AppResult<()> {
    tauri::async_runtime::spawn_blocking(move || crate::xuntong::logout(&app.state::<Runtime>()))
        .await
        .map_err(|e| msg(format!("退出异常：{e}")))?
}

#[tauri::command]
pub async fn open_url(url: String) -> AppResult<()> {
    crate::xuntong::open_url(&url)
}
