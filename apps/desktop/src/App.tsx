import { useEffect, useRef, useState, type ReactNode } from "react";
import { App as AntApp, Button, Space, Tooltip } from "antd";
import {
  AimOutlined,
  AuditOutlined,
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
import { api, appApi, BusyTasks, busyList, OPEN_FEEDBACK_EVENT, ShareStatus, taskName, UpdateInfo } from "./api";
import { listen } from "./transport";
import { PageKey, prefs, useApp, useReview } from "./store";
import Home from "./pages/Home";
import Data from "./pages/Data";
import Annotate from "./pages/Annotate";
import Review from "./pages/Review";
import Train, { explainError, isPoor, metricNames, rateMetric } from "./pages/Train";
import Health from "./pages/Health";
import Eval from "./pages/Eval";
import Settings from "./pages/Settings";
import Deploy from "./pages/Deploy";
import Team from "./pages/Team";
import FeedbackModal from "./components/FeedbackModal";
import UpdateModal from "./components/UpdateModal";
import WhatsNew from "./components/WhatsNew";
import DownloadProgress from "./components/DownloadProgress";
import html2canvas from "html2canvas";
import { t } from "./i18n";

const NAV: { key: PageKey; label: string; icon: ReactNode; needProject: boolean }[] = [
  { key: "home", label: t("项目"), icon: <HomeOutlined />, needProject: false },
  { key: "data", label: t("数据"), icon: <PictureOutlined />, needProject: true },
  { key: "annotate", label: t("标注"), icon: <FormOutlined />, needProject: true },
  // 只在团队协作时显示
  { key: "review", label: t("审核"), icon: <AuditOutlined />, needProject: true },
  { key: "health", label: t("体检"), icon: <HeartOutlined />, needProject: true },
  { key: "train", label: t("训练"), icon: <BarChartOutlined />, needProject: true },
  { key: "eval", label: t("评估"), icon: <AimOutlined />, needProject: true },
  { key: "deploy", label: t("部署"), icon: <CodeSandboxOutlined />, needProject: true },
  { key: "team", label: t("协作"), icon: <TeamOutlined />, needProject: true },
];

const TITLES: Record<PageKey, string> = {
  home: t("项目"),
  data: t("数据"),
  annotate: t("标注"),
  review: t("审核"),
  health: t("数据体检"),
  train: t("训练"),
  eval: t("评估"),
  deploy: t("部署"),
  team: t("实时协作"),
  settings: t("设置"),
};

export default function App() {
  const { project, page, setPage, setProject, setTeam, team, openRun } = useApp();
  const review = useReview();
  // 不再协作了（成员都删了、停了共享）就离开审核页
  useEffect(() => {
    if (page === "review" && !review) setPage("annotate");
  }, [page, review, setPage]);
  const { notification, modal } = AntApp.useApp();
  const [upd, setUpd] = useState<UpdateInfo | null>(null);

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
            message: t("Wulabel 有新版本 v{version}", { version: u.latest }),
            description: (
              <div>
                <div>
                  {t("当前是 v{version}。", { version: u.current })}
                  {u.notes}
                </div>
                <Space style={{ marginTop: 10 }} wrap>
                  <Button
                    size="small"
                    type="primary"
                    onClick={() => {
                      notification.destroy(key);
                      setUpd(u);
                    }}
                  >
                    {t("立即更新")}
                  </Button>
                  <Button size="small" onClick={() => u.url && api.openUrl(u.url).catch(() => {})}>
                    {t("去下载")}
                  </Button>
                  <Button
                    size="small"
                    onClick={() => {
                      prefs.set("updateSkip", u.latest);
                      notification.destroy(key);
                    }}
                  >
                    {t("这个版本不再提醒")}
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

  // 别的页面（比如设置页安装失败）要打开反馈弹窗
  const feedbackRef = useRef(openFeedback);
  feedbackRef.current = openFeedback;
  useEffect(() => {
    const h = () => feedbackRef.current();
    window.addEventListener(OPEN_FEEDBACK_EVENT, h);
    return () => window.removeEventListener(OPEN_FEEDBACK_EVENT, h);
  }, []);

  // 训练、安装等还在进行时关窗口：后端先拦下，这里问一下要不要真的关
  const closeAsking = useRef(false);
  useEffect(() => {
    const un = listen<{ seq: number; busy: BusyTasks }>("app-close-blocked", (e) => {
      appApi.closeAck(e.payload.seq).catch(() => {});
      if (closeAsking.current) return;
      const busy = e.payload.busy;
      const list = busyList(busy);
      const onlyTraining = busy.training != null && list.length === 1;
      closeAsking.current = true;
      modal.confirm({
        title: onlyTraining
          ? t("训练还没结束，关闭软件会中断训练（之后可以在训练页点「继续训练」，从最后训完的一轮接着跑）。确定关闭吗？")
          : t("还有任务在进行，关闭软件会中断它们。确定关闭吗？"),
        content: onlyTraining ? undefined : (
          <ul style={{ margin: "4px 0 0", paddingLeft: 18 }}>
            {list.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
        ),
        okText: t("仍然关闭"),
        okButtonProps: { danger: true },
        cancelText: t("先不关"),
        onOk: () => appApi.quitApp().catch(() => {}),
        afterClose: () => {
          closeAsking.current = false;
        },
      });
    });
    return () => {
      un.then((f) => f());
    };
  }, [modal]);

  // 前端热重载后，从后端取回已打开的项目
  useEffect(() => {
    api.currentProject().then((p) => p && setProject(p)).catch(() => {});
  }, [setProject]);

  // 训练完成或失败：不管在哪个页面都提示，带上结论和下一步
  useEffect(() => {
    const un = listen<{ runId: number; event: { type?: string; status?: string } }>("train-event", async (e) => {
      const ev = e.payload.event;
      if (ev.type !== "finished" || (ev.status !== "done" && ev.status !== "failed")) return;
      const runId = e.payload.runId;
      const key = `train-done-${runId}`;
      const go = (p: PageKey) => {
        notification.destroy(key);
        // 带上这次训练，评估页、训练页直接打开它
        if (p === "annotate") setPage(p);
        else openRun(runId, p);
      };
      const d = await api.runDetail(runId).catch(() => null);
      if (ev.status === "failed") {
        const hint = d?.error ? explainError(d.error)?.hint : undefined;
        notification.error({
          key,
          message: t("训练失败"),
          description: (
            <div>
              {hint && <div style={{ marginBottom: 6 }}>{hint}</div>}
              <Button size="small" type="primary" onClick={() => go("train")}>
                {t("看错误")}
              </Button>
            </div>
          ),
          duration: 0,
          placement: "bottomRight",
        });
        return;
      }
      const task = useApp.getState().project?.task ?? "detect";
      const best = d?.run.bestMap50 ?? null;
      // 结果很差或者有醒目提示（比如没加载到预训练权重）时，先去看原因，不建议拿去预标注
      const poor = d != null && (isPoor(task, best) || d.warnings.length > 0);
      notification.success({
        key,
        message: t("训练完成"),
        description: (
          <div>
            {best != null && (
              <div>
                {t("最好 {metric} {v}（{rate}）", { metric: metricNames(task).m50, v: best.toFixed(3), rate: rateMetric(task, best).word })}
              </div>
            )}
            {poor && <div className="dim">{t("效果不理想，先去训练页看看原因。")}</div>}
            <Space style={{ marginTop: 6 }} wrap>
              {poor ? (
                <Button size="small" type="primary" onClick={() => go("train")}>
                  {t("看原因")}
                </Button>
              ) : (
                <Button size="small" type="primary" onClick={() => go("eval")}>
                  {t("去评估")}
                </Button>
              )}
              {poor ? (
                <Button size="small" onClick={() => go("eval")}>
                  {t("去评估")}
                </Button>
              ) : (
                <Button size="small" onClick={() => go("annotate")}>
                  {t("去预标注")}
                </Button>
              )}
              {!poor && (
                <Button size="small" onClick={() => go("train")}>
                  {t("看训练记录")}
                </Button>
              )}
            </Space>
          </div>
        ),
        duration: 0,
        placement: "bottomRight",
      });
    });
    return () => {
      un.then((f) => f());
    };
  }, [notification, setPage, openRun]);

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

  // 项目里有没有协作成员（没有就不用审核流程）
  useEffect(() => {
    if (!project || team.remote) return;
    api
      .teamMembers()
      .then((m) => setTeam({ hasMembers: m.length > 0 }))
      .catch(() => setTeam({ hasMembers: false }));
  }, [setTeam, project?.path, team.remote]); // eslint-disable-line react-hooks/exhaustive-deps

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
          {t("反馈")}
        </Button>
        {project && (
          <Tooltip title={project.path}>
            <Button size="small" icon={<CloseOutlined />} onClick={close}>
              {t("关闭项目")}
            </Button>
          </Tooltip>
        )}
      </header>

      <nav className="nav">
        {NAV.filter((n) => n.key !== "review" || review).map((n) => (
          <button
            key={n.key}
            className={page === n.key ? "on" : ""}
            disabled={n.needProject && !project}
            onClick={() => setPage(n.key)}
          >
            {n.icon}
            {n.label}
            {n.key === "team" && team.shareRunning && <span className="presence-dot" style={{ position: "absolute", marginLeft: 22 }} />}
            {n.key === "review" && !!project?.stats.labeled && <span className="nav-badge">{project.stats.labeled > 999 ? "999+" : project.stats.labeled}</span>}
          </button>
        ))}
        <div className="gap" />
        <button className={page === "settings" ? "on" : ""} onClick={() => setPage("settings")}>
          <SettingOutlined />
          {t("设置")}
        </button>
      </nav>

      <main>
        {page === "home" && <Home />}
        {page === "data" && project && <Data project={project} />}
        {page === "annotate" && project && <Annotate project={project} />}
        {page === "review" && project && review && <Review project={project} />}
        {page === "health" && project && <Health project={project} />}
        {page === "train" && project && <Train project={project} />}
        {page === "eval" && project && <Eval project={project} />}
        {page === "deploy" && project && <Deploy project={project} />}
        {page === "team" && project && <Team project={project} />}
        {page === "settings" && <Settings />}
      </main>

      <FeedbackModal open={fbOpen} shot={fbShot} onClose={() => setFbOpen(false)} />
      <UpdateModal info={upd} open={!!upd} onClose={() => setUpd(null)} />
      <WhatsNew />
      <DownloadProgress />
    </div>
  );
}
