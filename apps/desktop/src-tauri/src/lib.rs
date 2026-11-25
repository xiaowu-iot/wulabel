mod ai;
mod annotations;
mod assist;
mod collab;
mod commands;
mod dataset;
mod dataset_import;
mod deploy;
mod dlprogress;
mod db;
mod error;
mod evaltools;
mod feedback;
mod health;
mod i18n;
mod images;
mod keypoints;
mod project;
mod protocol;
mod quality;
mod review;
mod report;
mod runtime;
mod share;
mod state;
mod team;
mod track;
mod training;
mod tunnel;
mod update;
mod versions;
mod video;
mod xuntong;

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
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(state::AppState::default())
        .manage(Arc::new(training::Jobs::default()))
        .manage(training::Predictor::default())
        .manage(video::Extractor::default())
        .manage(evaltools::FpTest::default())
        .manage(share::Share::default())
        .manage(Arc::new(deploy::Live::default()))
        .setup(|app| {
            // 训练环境有好几 GB，放本机目录（Local），不放会漫游同步的 Roaming
            let rt = runtime::Runtime::new(app.path().app_config_dir()?, app.path().app_local_data_dir()?, worker_dir(app));
            app.manage(rt);
            dlprogress::init(app.handle().clone());
            Ok(())
        })
        .on_window_event(|window, event| {
            // 训练、安装等还在进行时关主窗口：先拦下，前端弹确认框
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                if window.label() == "main" && commands::intercept_close(window.app_handle()) {
                    api.prevent_close();
                }
            }
        })
        .register_asynchronous_uri_scheme_protocol("xl", protocol::handle)
        .invoke_handler(tauri::generate_handler![
            commands::create_project,
            commands::open_project,
            commands::close_project,
            commands::current_project,
            commands::recent_projects,
            commands::remove_recent,
            commands::project_summary,
            commands::import_images,
            commands::stop_import,
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
            commands::aug_preview,
            commands::quality_scan,
            commands::export_report,
            commands::split_packages,
            commands::list_packages,
            commands::merge_package,
            commands::uncertain_images,
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
            commands::last_import_ids,
            commands::delete_images,
            commands::list_image_ids,
            commands::load_annotations,
            commands::save_annotations,
            commands::set_image_status,
            commands::set_status_many,
            commands::list_labels,
            commands::add_label,
            commands::update_label,
            commands::fill_label_prompts,
            commands::delete_label,
            commands::image_statuses,
            commands::merge_label,
            commands::stop_prelabel,
            commands::split_summary,
            commands::assign_splits,
            commands::list_split_images,
            commands::set_split,
            commands::export_dataset,
            commands::interpolate_frames,
            commands::track_frames,
            commands::import_dataset,
            commands::env_status,
            commands::save_settings,
            commands::env_install,
            commands::env_repair,
            commands::env_cancel,
            commands::busy_tasks,
            commands::close_ack,
            commands::quit_app,
            commands::speed_test,
            commands::list_runs,
            commands::run_detail,
            commands::running_run,
            commands::start_training,
            commands::resume_training,
            commands::stop_training,
            commands::delete_run,
            commands::export_onnx,
            commands::predict_image,
            commands::prelabel_all,
            commands::reveal_path,
            commands::health_load,
            commands::health_run,
            commands::health_ignore,
            commands::health_unignore,
            commands::health_status,
            commands::health_stop,
            commands::append_annotation,
            commands::live_start,
            commands::live_stop,
            commands::live_conf,
            commands::live_snapshot,
            commands::export_formats,
            commands::build_service,
            commands::share_status,
            commands::share_start,
            commands::share_stop,
            commands::share_save_options,
            commands::share_lock,
            commands::share_unlock,
            commands::share_review,
            commands::review_query,
            commands::review_ids,
            commands::review_members,
            commands::review_many,
            commands::review_undo,
            commands::team_members,
            commands::team_add_member,
            commands::team_update_member,
            commands::team_delete_member,
            commands::team_assign,
            commands::team_unassign,
            commands::team_stats,
            commands::team_export_csv,
            commands::team_activity,
            commands::ai_rules,
            commands::ai_save_rules,
            commands::ai_chat_rules,
            commands::ai_sync_rules,
            commands::ai_unbind_rules,
            commands::ai_rule_set_info,
            commands::ai_rules_platform_enabled,
            commands::xt_upload_model,
            commands::xt_models,
            commands::xt_pull_alarms,
            commands::xt_delete_model,
            commands::xt_login_start,
            commands::xt_login_poll,
            commands::xt_account,
            commands::xt_logout,
            commands::open_url,
            feedback::submit_feedback,
            feedback::feedback_logs,
            i18n::set_language,
            update::check_update,
            update::install_update,
            update::auto_update,
        ])
        .build(tauri::generate_context!())
        .expect("程序启动失败")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                // 退出时结束推理进程和正在进行的训练，不留后台进程
                app.state::<share::Share>().stop();
                app.state::<training::Predictor>().shutdown();
                app.state::<Arc<deploy::Live>>().stop();
                app.state::<Arc<training::Jobs>>().kill_all();
            }
        });
}
