import { useEffect, useState, type ReactNode } from "react";
import { Button, Tooltip } from "antd";
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
} from "@ant-design/icons";
import { api, taskName } from "./api";
import { PageKey, useApp } from "./store";
import Home from "./pages/Home";
import Data from "./pages/Data";
import Annotate from "./pages/Annotate";
import Train from "./pages/Train";
import Health from "./pages/Health";
import Eval from "./pages/Eval";
import Settings from "./pages/Settings";
import Deploy from "./pages/Deploy";
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
];

const TITLES: Record<PageKey, string> = {
  home: "项目",
  data: "数据",
  annotate: "标注",
  health: "数据体检",
  train: "训练",
  eval: "评估",
  deploy: "部署",
  settings: "设置",
};

export default function App() {
  const { project, page, setPage, setProject } = useApp();

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
        {page === "train" && project && <Train />}
        {page === "eval" && project && <Eval project={project} />}
        {page === "deploy" && project && <Deploy project={project} />}
        {page === "settings" && <Settings />}
      </main>

      <FeedbackModal open={fbOpen} shot={fbShot} onClose={() => setFbOpen(false)} />
    </div>
  );
}
