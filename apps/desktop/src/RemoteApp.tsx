// 协作成员在浏览器里看到的界面：登录 → 标注（只有分给自己的图）
import { useCallback, useEffect, useState } from "react";
import { Alert, App, Button, Checkbox, Form, Input, Result, Spin, Tag, Tooltip } from "antd";
import { LockOutlined, LogoutOutlined, UserOutlined } from "@ant-design/icons";
import { api, errText, MemberStats, ROLE_NAMES, ShareEvent, taskName, Whoami } from "./api";
import { useApp } from "./store";
import { clearToken, connectLive, disconnectLive, getToken, listen, onAuthLost, setToken } from "./transport";
import Annotate from "./pages/Annotate";

type Phase = "loading" | "login" | "app" | "closed";

export default function RemoteApp() {
  const { message } = App.useApp();
  const project = useApp((s) => s.project);
  const setProject = useApp((s) => s.setProject);
  const setPage = useApp((s) => s.setPage);
  const team = useApp((s) => s.team);
  const setTeam = useApp((s) => s.setTeam);
  const [phase, setPhase] = useState<Phase>("loading");
  const [info, setInfo] = useState<{ project: string; hostName: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [live, setLive] = useState("connected");
  const [today, setToday] = useState<MemberStats | null>(null);

  const enter = useCallback(
    async (who: Whoami) => {
      setTeam({ remote: true, me: who.member, options: who.options });
      const p = await api.currentProject();
      if (p) setProject(p);
      setPage("annotate");
      connectLive();
      setPhase("app");
    },
    [setTeam, setProject, setPage],
  );

  const logout = useCallback(
    (why?: string) => {
      fetch("api/logout", { method: "POST", headers: { Authorization: `Bearer ${getToken()}` } }).catch(() => {});
      clearToken();
      disconnectLive();
      setTeam({ me: null });
      setProject(null);
      setPhase("login");
      if (why) message.warning(why);
    },
    [message, setTeam, setProject],
  );

  // 打开页面：取项目名；有令牌就直接进
  useEffect(() => {
    api
      .shareInfo()
      .then((i) => {
        setInfo(i);
        document.title = `${i.project} · Wulabel 协作`;
      })
      .catch(() => {});
    onAuthLost(() => logout("登录已失效，请重新登录"));
    if (!getToken()) {
      setPhase("login");
      return;
    }
    api
      .whoami()
      .then(enter)
      .catch(() => {
        clearToken();
        setPhase("login");
      });
    return () => onAuthLost(null);
  }, [enter, logout]);

  // 实时事件：被踢下线、协作结束、选项变化
  useEffect(() => {
    if (phase !== "app") return;
    const subs = [
      listen<ShareEvent>("share-event", (e) => {
        const ev = e.payload;
        if (ev.kind === "kick" && ev.memberId === team.me?.id) logout("发起人把你的账号停用或改了密码，请重新登录");
        else if (ev.kind === "closed") {
          disconnectLive();
          setPhase("closed");
        } else if (ev.kind === "options")
          setTeam({
            options: {
              allowAi: ev.allowAi ?? team.options.allowAi,
              watermark: ev.watermark ?? team.options.watermark,
              membersEditLabels: ev.membersEditLabels ?? team.options.membersEditLabels,
            },
          });
      }),
      listen<string>("live-status", (e) => setLive(e.payload)),
    ];
    return () => {
      subs.forEach((p) => p.then((un) => un()));
    };
  }, [phase, team.me?.id, team.options, logout, setTeam]);

  // 我今天的进度，每分钟刷新
  useEffect(() => {
    if (phase !== "app") return;
    const load = () =>
      api
        .myStats()
        .then((s) => setToday(s.today))
        .catch(() => {});
    load();
    const t = window.setInterval(load, 60000);
    return () => window.clearInterval(t);
  }, [phase]);

  const submit = async (v: { name: string; password: string; remember: boolean }) => {
    setBusy(true);
    try {
      const r = await api.login(v.name, v.password);
      setToken(r.token, v.remember);
      await enter(r);
    } catch (e) {
      message.error(errText(e));
    } finally {
      setBusy(false);
    }
  };

  if (phase === "loading") {
    return (
      <div className="remote-center">
        <Spin />
      </div>
    );
  }

  if (phase === "closed") {
    return (
      <div className="remote-center">
        <Result
          status="info"
          title="发起人已经停止了协作"
          subTitle="你的标注都已经保存在发起人那里。发起人重新开启后，刷新这个页面就能继续。"
          extra={<Button onClick={() => window.location.reload()}>刷新</Button>}
        />
      </div>
    );
  }

  if (phase === "login") {
    return (
      <div className="remote-center">
        <div className="remote-login">
          <div className="logo" style={{ fontSize: 20, marginBottom: 4 }}>
            <b>◆</b> Wulabel 协作
          </div>
          <div className="muted" style={{ marginBottom: 20 }}>
            {info ? (
              <>
                项目「{info.project}」· 发起人 {info.hostName}
              </>
            ) : (
              "连接发起人的 Wulabel…"
            )}
          </div>
          <Form layout="vertical" onFinish={submit} initialValues={{ remember: true }}>
            <Form.Item name="name" label="名字" rules={[{ required: true, message: "请填名字" }]}>
              <Input prefix={<UserOutlined />} placeholder="发起人给你设的名字" autoFocus />
            </Form.Item>
            <Form.Item name="password" label="密码" rules={[{ required: true, message: "请填密码" }]}>
              <Input.Password prefix={<LockOutlined />} />
            </Form.Item>
            <Form.Item name="remember" valuePropName="checked">
              <Checkbox>记住登录（自己的电脑才勾）</Checkbox>
            </Form.Item>
            <Button type="primary" htmlType="submit" block loading={busy}>
              登录
            </Button>
          </Form>
          <div className="dim" style={{ fontSize: 12, marginTop: 16 }}>
            没有账号或忘了密码，请联系发起人在 Wulabel 的「协作」页添加或重设。
          </div>
        </div>
      </div>
    );
  }

  const me = team.me;
  return (
    <div className="app remote">
      <header className="top">
        <div className="logo">
          <b>◆</b> Wulabel <span className="dim" style={{ fontSize: 11, fontWeight: 400, marginLeft: 6 }}>协作</span>
        </div>
        <div className="crumb">
          {project?.name} <span className="dim">· {project ? taskName(project.task) : ""}</span>
        </div>
        <div className="spacer" />
        {live !== "connected" && (
          <Tag color="warning" style={{ marginRight: 8 }}>
            实时连接断开，正在重连…
          </Tag>
        )}
        {today && (
          <Tooltip title={`分给我 ${me?.assigned ?? today.assigned} 张 · 今天被通过 ${today.passed}、被打回 ${today.rejected}`}>
            <span className="muted" style={{ fontSize: 12, marginRight: 12 }}>
              今天提交 <b>{today.submitted}</b> 张
            </span>
          </Tooltip>
        )}
        <span style={{ marginRight: 12, fontSize: 13 }}>
          {me?.name} <span className="dim">· {me ? ROLE_NAMES[me.role] : ""}</span>
        </span>
        <Button size="small" icon={<LogoutOutlined />} onClick={() => logout()}>
          退出
        </Button>
      </header>
      <main className="remote-main">
        {project ? (
          <Annotate project={project} />
        ) : (
          <Alert type="warning" showIcon message="读取项目失败，请刷新页面" style={{ margin: 24 }} />
        )}
      </main>
    </div>
  );
}
