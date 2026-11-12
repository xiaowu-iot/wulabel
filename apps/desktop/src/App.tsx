import { useEffect, useState, type ReactNode } from "react";
import { App as AntApp, Button, Space, Tooltip } from "antd";
import {
  AimOutlined,
  BarChartOutlined,
  CloseOutlined,
  CodeSandboxOutlined,
  FormOutlined,
  HeartOutlined,
  HomeOutlined,
  MessageOutlined,
  PictureOutlined,
  SettingOutlined,
  TeamOutlined,
} from "@ant-design/icons";
import { api, ShareStatus, taskName } from "./api";
import { listen } from "./transport";
import { PageKey, prefs, useApp } from "./store";
import Home from "./pages/Home";
import Data from "./pages/Data";
import Annotate from "./pages/Annotate";
import Train from "./pages/Train";
import Health from "./pages/Health";
import Eval from "./pages/Eval";
import Settings from "./pages/Settings";
import Deploy from "./pages/Deploy";
import Team from "./pages/Team";
import FeedbackModal from "./components/FeedbackModal";
import html2canvas from "html2canvas";

const NAV: { key: PageKey; label: string; icon: ReactNode; needProject: boolean }[] = [
  { key: "home", label: "项目", icon: <HomeOutlined />, needProject: false },
  { key: "data", label: "数据", icon: <PictureOutlined />, needProject: true },
  { key: "annotate", label: "标注", icon: <FormOutlined />, needProject: true },
  { key: "health", label: "体检", icon: <HeartOutlined />, needProject: true },
  { key: "train", label: "训练", icon: <BarChartOutlined />, needProject: true },
  { key: "eval", label: "评估", icon: <AimOutlined />, needProject: true },
  { key: "deploy", label: "部署", icon: <CodeSandboxOutlined />, needProject: true },
  { key: "team", label: "协作", icon: <TeamOutlined />, needProject: true },
];

const TITLES: Record<PageKey, string> = {
  home: "项目",
  data: "数据",
  annotate: "标注",
  health: "数据体检",
  train: "训练",
  eval: "评估",
  deploy: "部署",
  team: "实时协作",
  settings: "设置",
};

export default function App() {
  const { project, page, setPage, setProject, setTeam, team } = useApp();
  const { notification } = AntApp.useApp();

  // 启动时检查更新（设置里可关）。有新版本一天最多提醒一次，用户点了「不再提醒」的版本不再提。
  useEffect(() => {
    if (prefs.get("updateCheck", "1") !== "1") return;
    const timer = setTimeout(() => {
      api
        .checkUpdate()
        .then((u) => {
          if (!u.newer || prefs.get("updateSkip", "") === u.latest) return;
          const today = new Date().toDateString();
          if (prefs.get("updateNotified", "") === today) return;
          prefs.set("updateNotified", today);
          const key = "wulabel-update";
          notification.info({
            key,
            message: `Wulabel 有新版本 v${u.latest}`,
            description: (
              <div>
                <div>当前是 v{u.current}。{u.notes}</div>
                <Space style={{ marginTop: 10 }}>
                  <Button size="small" type="primary" onClick={() => u.url && api.openUrl(u.url).catch(() => {})}>
                    去下载
                  </Button>
                  <Button
                    size="small"
                    onClick={() => {
                      prefs.set("updateSkip", u.latest);
                      notification.destroy(key);
                    }}
                  >
                    这个版本不再提醒
                  </Button>
                </Space>
              </div>
            ),
            duration: 0,
            placement: "bottomRight",
          });
        })
        .catch(() => {});
    }, 3000);
    return () => clearTimeout(timer);
  }, [notification]);

  const [fbOpen, setFbOpen] = useState(false);
  const [fbShot, setFbShot] = useState<string | null>(null);
  const [shooting, setShooting] = useState(false);

  // 先截当前界面（此时反馈弹窗还没打开，不会被截进去），再打开弹窗
  const openFeedback = async () => {
    setShooting(true);
    try {
      const canvas = await html2canvas(document.body, {
        scale: Math.min(1, window.devicePixelRatio || 1),
        useCORS: true,
        logging: false,
        backgroundColor: null,
      });
      setFbShot(canvas.toDataURL("image/jpeg", 0.8));
    } catch {
      setFbShot(null);
    } finally {
      setShooting(false);
      setFbOpen(true);
    }
  };

  // 前端热重载后，从后端取回已打开的项目
  useEffect(() => {
    api.currentProject().then((p) => p && setProject(p)).catch(() => {});
  }, [setProject]);

  // 协作开没开（标注页据此加锁、显示审核打回）
  useEffect(() => {
    api
      .shareStatus()
      .then((s) => setTeam({ shareRunning: s.running }))
      .catch(() => {});
    const un = listen<ShareStatus | { running: false }>("share-status", (e) => setTeam({ shareRunning: e.payload.running }));
    return () => {
      un.then((f) => f());
    };
  }, [setTeam, project?.path]);

  const close = async () => {
    await api.closeProject();
    setProject(null);
  };

  return (
    <div className="app">
      <header className="top">
        <div className="logo">
          <b>◆</b> Wulabel <span className="dim" style={{ fontSize: 11, fontWeight: 400, marginLeft: 6 }}>by 小物学长</span>
        </div>
        <div className="crumb">
          {project ? (
            <>
              {project.name} <span className="dim">· {taskName(project.task)}</span> / <strong>{TITLES[page]}</strong>
            </>
          ) : (
            <strong>{TITLES[page]}</strong>
          )}
        </div>
        <div className="spacer" />
        <Button size="small" icon={<MessageOutlined />} loading={shooting} onClick={openFeedback} style={{ marginRight: 8 }}>
          反馈
        </Button>
        {project && (
          <Tooltip title={project.path}>
            <Button size="small" icon={<CloseOutlined />} onClick={close}>
              关闭项目
            </Button>
          </Tooltip>
        )}
      </header>

      <nav className="nav">
        {NAV.map((n) => (
          <button
            key={n.key}
            className={page === n.key ? "on" : ""}
            disabled={n.needProject && !project}
            onClick={() => setPage(n.key)}
          >
            {n.icon}
            {n.label}
            {n.key === "team" && team.shareRunning && <span className="presence-dot" style={{ position: "absolute", marginLeft: 22 }} />}
          </button>
        ))}
        <div className="gap" />
        <button className={page === "settings" ? "on" : ""} onClick={() => setPage("settings")}>
          <SettingOutlined />
          设置
        </button>
      </nav>

      <main>
        {page === "home" && <Home />}
        {page === "data" && project && <Data project={project} />}
        {page === "annotate" && project && <Annotate project={project} />}
        {page === "health" && project && <Health project={project} />}
        {page === "train" && project && <Train project={project} />}
        {page === "eval" && project && <Eval project={project} />}
        {page === "deploy" && project && <Deploy project={project} />}
        {page === "team" && project && <Team project={project} />}
        {page === "settings" && <Settings />}
      </main>

      <FeedbackModal open={fbOpen} shot={fbShot} onClose={() => setFbOpen(false)} />
    </div>
  );
}
