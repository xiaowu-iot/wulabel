mod ai;
mod annotations;
mod assist;
mod commands;
mod dataset;
mod dataset_import;
mod deploy;
mod db;
mod error;
mod evaltools;
mod feedback;
mod health;
mod images;
mod keypoints;
mod project;
mod protocol;
mod runtime;
mod state;
mod training;
mod versions;
mod video;

use std::path::PathBuf;
use std::sync::Arc;

use tauri::Manager;

/// Python worker 代码的位置：安装包里在资源目录，开发时在仓库的 worker/
fn worker_dir(app: &tauri::App) -> PathBuf {
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("..").join("worker");
    // 开发时直接用仓库里的代码，改 Python 不用重新编译
    if cfg!(debug_assertions) && dev.join("xlabel_worker").exists() {
        return dev;
    }
    if let Ok(res) = app.path().resource_dir() {
        let p = res.join("worker");
        if p.join("xlabel_worker").exists() {
            return p;
        }
    }
    dev
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(state::AppState::default())
        .manage(Arc::new(training::Jobs::default()))
        .manage(training::Predictor::default())
        .manage(video::Extractor::default())
        .manage(evaltools::FpTest::default())
        .manage(Arc::new(deploy::Live::default()))
        .setup(|app| {
            // 训练环境有好几 GB，放本机目录（Local），不放会漫游同步的 Roaming
            let rt = runtime::Runtime::new(app.path().app_config_dir()?, app.path().app_local_data_dir()?, worker_dir(app));
            app.manage(rt);
            Ok(())
        })
        .register_asynchronous_uri_scheme_protocol("xl", protocol::handle)
        .invoke_handler(tauri::generate_handler![
            commands::create_project,
            commands::open_project,
            commands::close_project,
            commands::current_project,
            commands::recent_projects,
            commands::remove_recent,
            commands::import_images,
            commands::extract_video,
            commands::probe_stream,
            commands::extract_stream,
            commands::stop_extract,
            commands::keypoint_template,
            commands::save_keypoint_template,
            commands::inspect_onnx,
            commands::import_onnx,
            commands::update_external,
            commands::evaluate_run,
            commands::fp_test,
            commands::stop_fp_test,
            commands::import_negatives,
            commands::list_versions,
            commands::create_version,
            commands::rename_version,
            commands::delete_version,
            commands::diff_version,
            commands::restore_version,
            commands::export_version,
            commands::sam_segment,
            commands::text_detect,
            commands::text_prelabel_all,
            commands::ai_models,
            commands::ai_detect,
            commands::ai_prelabel_all,
            commands::list_images,
            commands::project_stats,
            commands::last_import,
            commands::delete_images,
            commands::list_image_ids,
            commands::load_annotations,
            commands::save_annotations,
            commands::set_image_status,
            commands::list_labels,
            commands::add_label,
            commands::update_label,
            commands::delete_label,
            commands::split_summary,
            commands::assign_splits,
            commands::export_dataset,
            commands::import_dataset,
            commands::env_status,
            commands::save_settings,
            commands::env_install,
            commands::speed_test,
            commands::list_runs,
            commands::run_detail,
            commands::running_run,
            commands::start_training,
            commands::stop_training,
            commands::delete_run,
            commands::export_onnx,
            commands::predict_image,
            commands::prelabel_all,
            commands::reveal_path,
            commands::health_load,
            commands::health_run,
            commands::health_ignore,
            commands::append_annotation,
            commands::live_start,
            commands::live_stop,
            commands::live_conf,
            commands::live_snapshot,
            commands::export_formats,
            commands::build_service,
            feedback::submit_feedback,
        ])
        .build(tauri::generate_context!())
        .expect("程序启动失败")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                // 退出时结束推理进程和正在进行的训练，不留后台进程
                app.state::<training::Predictor>().shutdown();
                app.state::<Arc<deploy::Live>>().stop();
                app.state::<Arc<training::Jobs>>().kill_all();
            }
        });
}
