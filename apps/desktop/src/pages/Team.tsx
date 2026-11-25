// 实时协作（发起人这边）：开启共享（局域网 / 在线）、成员、分配任务、在线状态、统计、操作记录
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  App,
  Button,
  Card,
  Checkbox,
  Empty,
  Input,
  InputNumber,
  Popconfirm,
  Radio,
  Segmented,
  Select,
  Space,
  Switch,
  Table,
  Tag,
  Tooltip,
  Typography,
  type InputRef,
} from "antd";
import { DownloadOutlined, PlusOutlined, ReloadOutlined, TeamOutlined } from "@ant-design/icons";
import { save } from "@tauri-apps/plugin-dialog";
import {
  ActivityRow,
  api,
  errText,
  Member,
  MemberStats,
  ProjectInfo,
  Role,
  ROLE_NAMES,
  ShareEvent,
  ShareOptions,
  ShareStatus,
} from "../api";
import { listen } from "../transport";
import { useApp } from "../store";
import { reviewMemberNext } from "./Review";
import { t } from "../i18n";
import { xuntongEnabled } from "../features";

const ACTIONS: Record<string, string> = {
  login: t("登录"),
  save: t("保存标注"),
  submit: t("提交"),
  pass: t("审核通过"),
  reject: t("打回"),
  label: t("类别"),
  assign: t("分配任务"),
  todo: t("改为未标注"),
  labeled: t("改为待审核"),
  reviewed: t("改为已通过"),
  skipped: t("跳过"),
};

const fmtTime = (ts: number) => new Date(ts * 1000).toLocaleString();
const fmtDur = (s: number) =>
  s < 60
    ? t("{n} 秒", { n: s })
    : s < 3600
      ? t("{n} 分钟", { n: Math.round(s / 60) })
      : t("{n} 小时", { n: (s / 3600).toFixed(1) });

/** 随机密码：去掉容易看错的字符 */
const randomPassword = () => {
  const cs = "abcdefghjkmnpqrstuvwxyz23456789";
  const a = new Uint32Array(8);
  crypto.getRandomValues(a);
  return Array.from(a, (n) => cs[n % cs.length]).join("");
};

function copy(text: string, ok: () => void) {
  navigator.clipboard.writeText(text).then(ok, () => {});
}

export default function Team({ project }: { project: ProjectInfo }) {
  const { message, modal } = App.useApp();
  const annotateImage = useApp((s) => s.annotateImage);
  const setPage = useApp((s) => s.setPage);
  const setTeam = useApp((s) => s.setTeam);
  const [status, setStatus] = useState<ShareStatus | null>(null);
  const [opts, setOpts] = useState<ShareOptions | null>(null);
  const [starting, setStarting] = useState(false);
  const [members, setMembers] = useState<Member[]>([]);
  const [stats, setStats] = useState<MemberStats[]>([]);
  const [range, setRange] = useState("today");
  const [acts, setActs] = useState<ActivityRow[]>([]);
  const [withSaves, setWithSaves] = useState(false);
  // 添加成员
  const [nm, setNm] = useState("");
  const [role, setRole] = useState<Role>("annotator");
  const [pw, setPw] = useState(randomPassword);
  const nameInput = useRef<InputRef>(null);
  // 分配
  const [assignIds, setAssignIds] = useState<number[]>([]);
  const [scope, setScope] = useState<"unassigned" | "todo" | "unreviewed">("unassigned");
  const [perPerson, setPerPerson] = useState<number | null>(null);

  const running = !!status?.running;

  const loadStatus = useCallback(async () => {
    const s = await api.shareStatus();
    setStatus(s);
    setOpts((o) => o ?? s.options);
    setTeam({ shareRunning: s.running });
  }, [setTeam]);

  const rangeFrom = useMemo(() => {
    const now = Date.now() / 1000;
    if (range === "today") {
      const d = new Date();
      d.setHours(0, 0, 0, 0);
      return Math.floor(d.getTime() / 1000);
    }
    if (range === "week") return Math.floor(now - 7 * 86400);
    if (range === "month") return Math.floor(now - 30 * 86400);
    return 0;
  }, [range]);

  const loadData = useCallback(async () => {
    try {
      const [m, st, a] = await Promise.all([
        api.teamMembers(),
        api.teamStats(rangeFrom, 4102444800),
        api.teamActivity(100, null, withSaves),
      ]);
      setMembers(m);
      setTeam({ hasMembers: m.length > 0 });
      setStats(st);
      setActs(a);
    } catch (e) {
      message.error(errText(e));
    }
  }, [rangeFrom, withSaves, message]);

  useEffect(() => {
    loadStatus().catch((e) => message.error(errText(e)));
  }, [loadStatus, message]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // 实时刷新：有人登录、提交、审核就刷新列表（攒一下，别刷太勤）
  const timer = useRef<number>();
  useEffect(() => {
    const later = () => {
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(loadData, 1500);
    };
    const subs = [
      listen<ShareStatus | { running: false }>("share-status", (e) => {
        const s = e.payload;
        if ("options" in s) {
          setStatus(s);
          setTeam({ shareRunning: s.running });
        } else {
          setStatus((old) => (old ? { ...old, running: false, online: [], relay: { state: "off", url: "", error: "" } } : old));
          setTeam({ shareRunning: false });
        }
      }),
      listen<ShareEvent>("share-event", (e) => {
        const ev = e.payload;
        if (ev.kind === "presence" && ev.online) setStatus((old) => (old ? { ...old, online: ev.online! } : old));
        if (ev.kind !== "lock") later();
      }),
    ];
    return () => {
      window.clearTimeout(timer.current);
      subs.forEach((p) => p.then((un) => un()));
    };
  }, [loadData, setTeam]);

  const setOpt = <K extends keyof ShareOptions>(k: K, v: ShareOptions[K]) => setOpts((o) => (o ? { ...o, [k]: v } : o));

  /** 开着共享时，开关类选项立即生效 */
  const setLive = async <K extends keyof ShareOptions>(k: K, v: ShareOptions[K]) => {
    if (!opts) return;
    const next = { ...opts, [k]: v };
    setOpts(next);
    try {
      await api.shareSaveOptions(next);
    } catch (e) {
      message.error(errText(e));
    }
  };

  const start = async () => {
    if (!opts) return;
    // 没有成员时开了也没人能登录，先拦下来
    if (members.filter((m) => !m.disabled).length === 0) {
      modal.info({
        title: t("还没有成员"),
        content: t("先添加成员（给每个人设名字和密码），再开启协作。"),
        okText: t("去添加成员"),
        onOk: () => {
          nameInput.current?.focus();
          nameInput.current?.input?.scrollIntoView({ behavior: "smooth", block: "center" });
        },
      });
      return;
    }
    setStarting(true);
    try {
      const s = await api.shareStart(opts);
      setStatus(s);
      setOpts(s.options);
      setTeam({ shareRunning: true });
      message.success(t("协作已开启"));
    } catch (e) {
      message.error(errText(e));
    } finally {
      setStarting(false);
    }
  };

  const stop = () =>
    modal.confirm({
      title: t("停止协作？"),
      content: t("成员会马上断开（已经保存的标注不受影响）。下次开启后他们刷新页面就能继续。"),
      okText: t("停止"),
      cancelText: t("取消"),
      onOk: async () => {
        await api.shareStop();
        await loadStatus();
      },
    });

  const addMember = async () => {
    if (!nm.trim()) {
      message.warning(t("请填名字"));
      return;
    }
    try {
      const m = await api.teamAddMember(nm.trim(), role, pw);
      // 协作开着时把访问地址也带上（外网中转连上了用外网地址，否则用局域网地址）
      const urls = !running ? [] : status?.relay?.state === "online" && status.relay.url ? [status.relay.url] : status?.lanUrls ?? [];
      const text = [t("账号：{name}　密码：{pw}", { name: m.name, pw }), ...(urls.length ? [t("访问地址：{url}", { url: urls.join(" / ") })] : [])].join("\n");
      modal.success({
        title: t("已添加{role}「{name}」", { role: ROLE_NAMES[m.role], name: m.name }),
        content: (
          <div>
            <div style={{ margin: "6px 0" }}>{t("把下面的账号密码和访问地址发给他（密码只显示这一次，忘了可以重设）：")}</div>
            <Typography.Paragraph copyable={{ text }} style={{ marginBottom: 0 }}>
              <code style={{ whiteSpace: "pre-wrap" }}>{text}</code>
            </Typography.Paragraph>
            {urls.length === 0 && <div className="dim" style={{ fontSize: 12, marginTop: 6 }}>{t("开启协作后会显示访问地址")}</div>}
          </div>
        ),
      });
      setNm("");
      setPw(randomPassword());
      loadData();
    } catch (e) {
      message.error(errText(e));
    }
  };

  const resetPassword = (m: Member) => {
    const npw = randomPassword();
    modal.confirm({
      title: t("重设「{name}」的密码？", { name: m.name }),
      content: (
        <div>
          {t("新密码：")}
          <code>{npw}</code>
          <div className="dim" style={{ fontSize: 12, marginTop: 6 }}>
            {t("确定后他会被退出登录，需要用新密码重新登录。")}
          </div>
        </div>
      ),
      okText: t("重设并复制"),
      cancelText: t("取消"),
      onOk: async () => {
        await api.teamUpdateMember(m.id, { password: npw });
        copy(t("账号：{name}　密码：{pw}", { name: m.name, pw: npw }), () => message.success(t("已复制新的账号密码")));
      },
    });
  };

  const patch = async (m: Member, p: Parameters<typeof api.teamUpdateMember>[1]) => {
    try {
      await api.teamUpdateMember(m.id, p);
      loadData();
    } catch (e) {
      message.error(errText(e));
    }
  };

  const assign = async () => {
    try {
      const r = await api.teamAssign({ memberIds: assignIds, scope, perPerson });
      message.success(
        t("已分配：{list}", { list: r.map((x) => t("{name} {count} 张", { name: x.name, count: x.count })).join(t("、")) }),
      );
      loadData();
    } catch (e) {
      message.error(errText(e));
    }
  };

  const exportCsv = async () => {
    const path = await save({
      title: t("导出工作量统计"),
      defaultPath: t("{name}_工作量_{date}.csv", { name: project.name, date: new Date().toISOString().slice(0, 10) }),
      filters: [{ name: "CSV", extensions: ["csv"] }],
    });
    if (!path) return;
    try {
      await api.teamExportCsv(rangeFrom, 4102444800, path);
      message.success(t("已导出，可以用 Excel 打开"));
    } catch (e) {
      message.error(errText(e));
    }
  };

  const online = useMemo(() => new Map((status?.online ?? []).map((p) => [p.memberId, p])), [status]);
  const annotators = members.filter((m) => !m.disabled && m.role === "annotator");

  if (!opts) return <div className="page" />;

  const relay = status?.relay;
  const shareUrl = relay?.state === "online" ? relay.url : "";

  return (
    <div className="page" style={{ maxWidth: 1100 }}>
      <h1>
        <TeamOutlined /> {t("实时协作")}
      </h1>
      <p className="desc">
        {t("几个人同时标同一个项目：你的电脑当主机，成员用浏览器打开地址、登录后就能标分给自己的图。一张图同一时间只有一个人能改，别人打开是只读，改动实时同步。")}
      </p>

      <Card size="small" title={t("共享")} style={{ marginBottom: 16 }}>
        {!running ? (
          <Space direction="vertical" style={{ width: "100%" }} size={12}>
            <Radio.Group value={opts.mode} onChange={(e) => setOpt("mode", e.target.value)}>
              <Radio.Button value="lan">{t("局域网")}</Radio.Button>
              <Radio.Button value="relay">{t("在线 · 自建中转")}</Radio.Button>
              {(xuntongEnabled() || opts.mode === "xuntong") && <Radio.Button value="xuntong">{t("在线 · 巡瞳官方中转")}</Radio.Button>}
            </Radio.Group>
            <div className="muted" style={{ fontSize: 12 }}>
              {opts.mode === "lan" && t("成员和你在同一个局域网（同一个公司网络、同一个 Wi-Fi），速度最快，数据不出公司。")}
              {opts.mode === "relay" &&
                t("成员在外网（外包、在家办公）。需要一台你们自己的云服务器装上 Wulabel 中转服务（仓库 relay/ 文件夹，有部署说明），数据只经过你们自己的服务器。")}
              {opts.mode === "xuntong" && t("成员在外网，又没有自己的服务器时用。用你的巡瞳 AI 账号（设置里的 API Key）连巡瞳的中转服务器。")}
            </div>
            {opts.mode === "xuntong" && (
              <Alert
                type="warning"
                showIcon
                message={t("经过巡瞳官方中转时，成员看到的图片和标注会经过巡瞳平台的服务器转发（只转发、不保存）。数据敏感的项目请用局域网或自建中转。")}
              />
            )}
            <Space wrap>
              <Input addonBefore={t("你的名字")} value={opts.hostName} onChange={(e) => setOpt("hostName", e.target.value)} style={{ width: 220 }} />
              <InputNumber addonBefore={t("端口")} min={1024} max={65535} value={opts.port} onChange={(v) => setOpt("port", Number(v ?? 8765))} style={{ width: 160 }} />
            </Space>
            {opts.mode === "relay" && (
              <Space wrap>
                <Input
                  addonBefore={t("中转地址")}
                  placeholder="https://relay.example.com/"
                  value={opts.relayUrl}
                  onChange={(e) => setOpt("relayUrl", e.target.value)}
                  style={{ width: 380 }}
                />
                <Input.Password
                  addonBefore={t("密钥")}
                  placeholder={t("部署中转时设的 --secret")}
                  value={opts.relaySecret}
                  onChange={(e) => setOpt("relaySecret", e.target.value)}
                  style={{ width: 300 }}
                />
              </Space>
            )}
            {opts.mode === "xuntong" && (
              <Input
                addonBefore={t("官方中转地址")}
                value={opts.xuntongRelayUrl}
                onChange={(e) => setOpt("xuntongRelayUrl", e.target.value)}
                style={{ width: 460 }}
              />
            )}
            <Space direction="vertical" size={4}>
              <Checkbox checked={opts.allowAi} onChange={(e) => setOpt("allowAi", e.target.checked)}>
                {t("成员可以用 AI 辅助（SAM 智能选取、模型预标注、文本提示，都在你的电脑上算；和 AI 聊识别规则会用你的巡瞳额度）")}
              </Checkbox>
              <Checkbox checked={opts.membersEditLabels} onChange={(e) => setOpt("membersEditLabels", e.target.checked)}>
                {t("成员可以新增、修改类别和识别说明（删除类别只有你能做）")}
              </Checkbox>
              <Checkbox checked={opts.watermark} onChange={(e) => setOpt("watermark", e.target.checked)}>
                {t("成员的画面叠加带他名字的水印（防截图外传，适合外包）")}
              </Checkbox>
            </Space>
            <Button type="primary" loading={starting} onClick={start}>
              {t("开启协作")}
            </Button>
          </Space>
        ) : (
          <Space direction="vertical" style={{ width: "100%" }} size={10}>
            <Space wrap>
              <Tag color="success">{t("协作中")}</Tag>
              <span className="muted">
                {opts.mode === "lan" ? t("局域网") : opts.mode === "relay" ? t("在线 · 自建中转") : t("在线 · 巡瞳官方中转")} ·{" "}
                {t("{n} 人在线", { n: online.size })}
              </span>
              <Button size="small" danger onClick={stop}>
                {t("停止协作")}
              </Button>
            </Space>
            {opts.mode !== "lan" && (
              <div>
                <span className="muted">{t("外网地址：")}</span>
                {relay?.state === "online" ? (
                  <>
                    <Typography.Text copyable={{ text: shareUrl }} strong>
                      {shareUrl}
                    </Typography.Text>
                  </>
                ) : relay?.state === "error" ? (
                  <Tag color="error">{relay.error}</Tag>
                ) : (
                  <Tag color="processing">{t("正在连接中转服务器…")}</Tag>
                )}
              </div>
            )}
            <div>
              <span className="muted">{t("局域网地址：")}</span>
              {status?.lanUrls.map((u) => (
                <Typography.Text key={u} copyable={{ text: u }} strong style={{ marginRight: 16 }}>
                  {u}
                </Typography.Text>
              ))}
            </div>
            <div className="dim" style={{ fontSize: 12 }}>
              {t("把地址和各自的账号密码发给成员，用 Chrome / Edge 打开即可，不用装软件。第一次开启时 Windows 可能弹出防火墙提示，请点「允许」。")}
            </div>
            <Space wrap>
              <Switch checked={opts.allowAi} onChange={(v) => setLive("allowAi", v)} /> {t("成员可用 AI 辅助")}
              <Switch checked={opts.membersEditLabels} onChange={(v) => setLive("membersEditLabels", v)} /> {t("成员可改类别")}
              <Switch checked={opts.watermark} onChange={(v) => setLive("watermark", v)} /> {t("水印")}
            </Space>
          </Space>
        )}
      </Card>

      <Card
        size="small"
        title={t("成员（{n}）", { n: members.length })}
        style={{ marginBottom: 16 }}
        extra={<Button size="small" type="text" icon={<ReloadOutlined />} onClick={loadData} />}
      >
        <Space wrap style={{ marginBottom: 12 }}>
          <Input ref={nameInput} placeholder={t("名字（登录用）")} value={nm} onChange={(e) => setNm(e.target.value)} style={{ width: 160 }} onPressEnter={addMember} />
          <Select<Role>
            value={role}
            onChange={setRole}
            style={{ width: 110 }}
            options={[
              { value: "annotator", label: t("标注员") },
              { value: "reviewer", label: t("审核员") },
            ]}
          />
          <Input
            addonBefore={t("密码")}
            value={pw}
            onChange={(e) => setPw(e.target.value)}
            style={{ width: 230 }}
            suffix={
              <Tooltip title={t("换一个随机密码")}>
                <ReloadOutlined onClick={() => setPw(randomPassword())} style={{ cursor: "pointer" }} />
              </Tooltip>
            }
          />
          <Button icon={<PlusOutlined />} onClick={addMember}>
            {t("添加")}
          </Button>
        </Space>
        <div className="dim" style={{ fontSize: 12, marginBottom: 8 }}>
          {t("标注员只能看到、修改分给自己的图；审核员能看全部图，负责审核通过或打回。你自己是管理员。")}
        </div>
        <Table<Member>
          size="small"
          rowKey="id"
          pagination={false}
          dataSource={members}
          locale={{ emptyText: <Empty description={t("还没有成员，在上面添加")} /> }}
          columns={[
            {
              title: t("名字"),
              dataIndex: "name",
              render: (n: string, m) => (
                <span style={{ opacity: m.disabled ? 0.5 : 1 }}>
                  {online.has(m.id) && <span className="presence-dot" />}
                  {n}
                  {m.disabled && <Tag style={{ marginLeft: 6 }}>{t("已停用")}</Tag>}
                </span>
              ),
            },
            {
              title: t("角色"),
              width: 110,
              render: (_, m) => (
                <Select<Role>
                  size="small"
                  value={m.role}
                  variant="borderless"
                  onChange={(v) => patch(m, { role: v })}
                  options={[
                    { value: "annotator", label: t("标注员") },
                    { value: "reviewer", label: t("审核员") },
                  ]}
                />
              ),
            },
            {
              title: t("在线"),
              width: 140,
              render: (_, m) => {
                const p = online.get(m.id);
                if (!p) return <span className="dim">{t("离线")}</span>;
                return p.imageId ? (
                  <a onClick={() => annotateImage(p.imageId!)}>{t("在看图 #{id}", { id: p.imageId })}</a>
                ) : (
                  <span className="muted">{t("在线")}</span>
                );
              },
            },
            {
              title: t("任务"),
              width: 200,
              render: (_, m) =>
                m.role === "annotator" ? (
                  <span>
                    {t("分配 {assigned} · 已交 {done}", { assigned: m.assigned, done: m.done })}
                    {m.rejected > 0 && (
                      <Tag color="warning" style={{ marginLeft: 6 }}>
                        {t("打回 {n}", { n: m.rejected })}
                      </Tag>
                    )}
                  </span>
                ) : (
                  <span className="dim">{t("看全部图")}</span>
                ),
            },
            {
              title: "",
              width: 320,
              render: (_, m) => (
                <Space size={0}>
                  {m.role === "annotator" && (
                    <Button
                      size="small"
                      type="link"
                      onClick={() => {
                        reviewMemberNext(m.id);
                        setPage("review");
                      }}
                    >
                      {t("看他的图")}
                    </Button>
                  )}
                  <Button size="small" type="link" onClick={() => resetPassword(m)}>
                    {t("重设密码")}
                  </Button>
                  <Button size="small" type="link" onClick={() => patch(m, { disabled: !m.disabled })}>
                    {m.disabled ? t("启用") : t("停用")}
                  </Button>
                  {m.assigned > m.done && (
                    <Popconfirm
                      title={t("收回他还没交的图？")}
                      description={t("收回的图变成未分配，可以再分给别人。")}
                      onConfirm={() =>
                        api
                          .teamUnassign(m.id, true)
                          .then((n) => {
                            message.success(t("收回了 {n} 张", { n }));
                            loadData();
                          })
                          .catch((e) => message.error(errText(e)))
                      }
                    >
                      <Button size="small" type="link">
                        {t("收回任务")}
                      </Button>
                    </Popconfirm>
                  )}
                  <Popconfirm
                    title={t("删除「{name}」？", { name: m.name })}
                    description={t("他没做完的图会收回；已经标的标注和记录保留。")}
                    onConfirm={() =>
                      api
                        .teamDeleteMember(m.id)
                        .then(loadData)
                        .catch((e) => message.error(errText(e)))
                    }
                  >
                    <Button size="small" type="link" danger>
                      {t("删除")}
                    </Button>
                  </Popconfirm>
                </Space>
              ),
            },
          ]}
        />
      </Card>

      <Card size="small" title={t("分配任务")} style={{ marginBottom: 16 }}>
        <Space direction="vertical" style={{ width: "100%" }}>
          <Select
            mode="multiple"
            placeholder={t("选择标注员")}
            style={{ width: "100%" }}
            value={assignIds}
            onChange={setAssignIds}
            options={annotators.map((m) => ({ value: m.id, label: t("{name}（已分配 {n}）", { name: m.name, n: m.assigned }) }))}
          />
          <Radio.Group value={scope} onChange={(e) => setScope(e.target.value)}>
            <Radio value="unassigned">{t("还没分给人的未标注图")}</Radio>
            <Radio value="todo">{t("全部未标注图（已分配的重新分）")}</Radio>
            <Radio value="unreviewed">{t("没审核过的图（未标注 + 待审核）")}</Radio>
          </Radio.Group>
          <Space>
            <InputNumber
              min={1}
              placeholder={t("不填就平均分完")}
              addonBefore={t("每人")}
              addonAfter={t("张")}
              value={perPerson}
              onChange={(v) => setPerPerson(v == null ? null : Number(v))}
              style={{ width: 240 }}
            />
            <Button type="primary" disabled={assignIds.length === 0} onClick={assign}>
              {t("分配")}
            </Button>
            <Button size="small" type="link" onClick={() => setAssignIds(annotators.map((m) => m.id))}>
              {t("选全部标注员")}
            </Button>
          </Space>
          <div className="dim" style={{ fontSize: 12 }}>{t("按图片顺序连续分，视频的连续帧会分给同一个人。分配后成员刷新列表就能看到。")}</div>
        </Space>
      </Card>

      <Card
        size="small"
        title={t("进度与工作量")}
        style={{ marginBottom: 16 }}
        extra={
          <Space>
            <Segmented
              size="small"
              value={range}
              onChange={(v) => setRange(v as string)}
              options={[
                { label: t("今天"), value: "today" },
                { label: t("近 7 天"), value: "week" },
                { label: t("近 30 天"), value: "month" },
                { label: t("全部"), value: "all" },
              ]}
            />
            <Button size="small" icon={<DownloadOutlined />} onClick={exportCsv}>
              {t("导出 CSV")}
            </Button>
          </Space>
        }
      >
        <Table<MemberStats>
          size="small"
          rowKey="memberId"
          pagination={false}
          dataSource={stats}
          locale={{ emptyText: <Empty description={t("这段时间还没有记录")} /> }}
          columns={[
            { title: t("成员"), dataIndex: "name" },
            { title: t("角色"), dataIndex: "role", width: 80, render: (r: Role) => ROLE_NAMES[r] },
            { title: t("分配"), dataIndex: "assigned", width: 70 },
            { title: t("提交张数"), dataIndex: "submitted", width: 90 },
            { title: t("标注对象"), dataIndex: "objects", width: 90 },
            { title: t("被通过"), dataIndex: "passed", width: 80 },
            { title: t("被打回"), dataIndex: "rejected", width: 80 },
            {
              title: t("通过率"),
              width: 80,
              render: (_, s) => (s.passed + s.rejected ? `${Math.round((s.passed * 100) / (s.passed + s.rejected))}%` : "—"),
            },
            { title: t("审核次数"), dataIndex: "reviews", width: 90 },
            { title: t("工作时长"), dataIndex: "activeSecs", width: 100, render: (v: number) => fmtDur(v) },
          ]}
        />
        <div className="dim" style={{ fontSize: 12, marginTop: 8 }}>
          {t("工作时长按两次操作间隔不超过 5 分钟累加估算。计件结算可以导出 CSV 用 Excel 处理。")}
        </div>
      </Card>

      <Card
        size="small"
        title={t("操作记录")}
        extra={
          <Checkbox checked={withSaves} onChange={(e) => setWithSaves(e.target.checked)}>
            {t("包含每次自动保存")}
          </Checkbox>
        }
      >
        <Table<ActivityRow>
          size="small"
          rowKey="id"
          dataSource={acts}
          pagination={{ pageSize: 15, size: "small" }}
          locale={{ emptyText: <Empty description={t("还没有记录")} /> }}
          columns={[
            { title: t("时间"), dataIndex: "time", width: 170, render: fmtTime },
            { title: t("成员"), dataIndex: "member", width: 120 },
            { title: t("操作"), dataIndex: "action", width: 100, render: (a: string) => ACTIONS[a] ?? a },
            {
              title: t("图片"),
              dataIndex: "imageId",
              width: 90,
              render: (id: number | null) => (id ? <a onClick={() => annotateImage(id)}>#{id}</a> : ""),
            },
            { title: t("说明"), render: (_, r) => r.detail || (r.objects ? t("{n} 个对象", { n: r.objects }) : "") },
          ]}
        />
      </Card>
    </div>
  );
}
