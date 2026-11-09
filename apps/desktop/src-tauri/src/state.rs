use std::sync::Arc;

use parking_lot::RwLock;

use crate::error::{msg, AppResult};
use crate::project::Project;

#[derive(Default)]
pub struct AppState {
    project: RwLock<Option<Arc<Project>>>,
}

impl AppState {
    pub fn current(&self) -> AppResult<Arc<Project>> {
        self.project.read().clone().ok_or_else(|| msg("还没有打开项目"))
    }

    pub fn try_current(&self) -> Option<Arc<Project>> {
        self.project.read().clone()
    }

    pub fn set(&self, p: Option<Project>) {
        *self.project.write() = p.map(Arc::new);
    }
}
