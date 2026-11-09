use std::path::PathBuf;
use std::sync::Arc;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

use crate::ai;
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
use crate::state::AppState;
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
pub async fn close_project(state: State<'_, AppState>) -> AppResult<()> {
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
    evaltools::import_onnx(&state.current()?, &path, &model)
}

#[tauri::command]
pub async fn update_external(state: State<'_, AppState>, run_id: i64, model: evaltools::ExternalModel) -> AppResult<()> {
    evaltools::update_external(&state.current()?, run_id, &model)
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
pub async fn aug_preview(app: AppHandle, state: State<'_, AppState>, aug: serde_json::Value, img_size: u32) -> AppResult<Vec<String>> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || evaltools::aug_preview(&project, &app.state::<Runtime>(), &aug, img_size, 4))
        .await
        .map_err(|e| msg(format!("预览异常：{e}")))?
}

#[tauri::command]
pub async fn import_negatives(state: State<'_, AppState>, paths: Vec<String>) -> AppResult<usize> {
    evaltools::import_negatives(&state.current()?, &paths)
}

// ---------- 数据集版本 ----------

#[tauri::command]
pub async fn list_versions(state: State<'_, AppState>) -> AppResult<Vec<versions::VersionInfo>> {
    versions::list(&state.current()?)
}

#[tauri::command]
pub async fn create_version(
    state: State<'_, AppState>,
    name: String,
    note: String,
    ratios: [f64; 3],
) -> AppResult<versions::VersionInfo> {
    versions::create(&state.current()?, &name, &note, ratios)
}

#[tauri::command]
pub async fn rename_version(state: State<'_, AppState>, id: i64, name: String, note: String) -> AppResult<()> {
    versions::rename(&state.current()?, id, &name, &note)
}

#[tauri::command]
pub async fn delete_version(state: State<'_, AppState>, id: i64) -> AppResult<()> {
    versions::delete(&state.current()?, id)
}

#[tauri::command]
pub async fn diff_version(state: State<'_, AppState>, id: i64) -> AppResult<versions::VersionDiff> {
    versions::diff(&state.current()?, id)
}

#[tauri::command]
pub async fn restore_version(state: State<'_, AppState>, id: i64) -> AppResult<versions::RestoreResult> {
    versions::restore(&state.current()?, id)
}

#[tauri::command]
pub async fn export_version(app: AppHandle, state: State<'_, AppState>, id: i64, options: ExportOptions) -> AppResult<ExportResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let progress = progress_emitter(app.clone(), "export-progress");
        dataset::export_version(&project, id, &options, &progress)
    })
    .await
    .map_err(|e| msg(format!("导出异常：{e}")))?
}

/// 关键点项目的模板（点名 + 骨架），没有时返回默认的 COCO 人体 17 点
#[tauri::command]
pub async fn keypoint_template(state: State<'_, AppState>) -> AppResult<Template> {
    keypoints::get(&state.current()?)
}

#[tauri::command]
pub async fn save_keypoint_template(state: State<'_, AppState>, template: Template) -> AppResult<()> {
    keypoints::save(&state.current()?, &template)
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

#[tauri::command]
pub async fn export_dataset(app: AppHandle, state: State<'_, AppState>, options: ExportOptions) -> AppResult<ExportResult> {
    let project = state.current()?;
    tauri::async_runtime::spawn_blocking(move || {
        let progress = progress_emitter(app, "export-progress");
        dataset::export(&project, &options, &progress)
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
    image_id: i64,
    annotations: Vec<Annotation>,
) -> AppResult<String> {
    annotations::save(&*state.current()?, image_id, &annotations)
}

#[tauri::command]
pub async fn set_image_status(state: State<'_, AppState>, image_id: i64, status: String) -> AppResult<()> {
    annotations::set_status(&*state.current()?, image_id, &status)
}

// ---------- 类别 ----------

#[tauri::command]
pub async fn list_labels(state: State<'_, AppState>) -> AppResult<Vec<Label>> {
    annotations::list_labels(&*state.current()?)
}

#[tauri::command]
pub async fn add_label(state: State<'_, AppState>, name: String, color: String) -> AppResult<Label> {
    annotations::add_label(&*state.current()?, &name, &color)
}

#[tauri::command]
pub async fn update_label(
    state: State<'_, AppState>,
    id: i64,
    name: String,
    color: String,
    prompt: Option<String>,
) -> AppResult<()> {
    annotations::update_label(&*state.current()?, id, &name, &color, prompt.as_deref())
}

#[tauri::command]
pub async fn delete_label(state: State<'_, AppState>, id: i64) -> AppResult<usize> {
    annotations::delete_label(&*state.current()?, id)
}
