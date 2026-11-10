import { useCallback, useEffect, useRef, useState } from "react";
import { Alert, App, Button, Descriptions, Input, Modal, Progress, Radio, Select, Space, Spin, Tag, Typography } from "antd";
import { CheckCircleOutlined, FolderOpenOutlined, ReloadOutlined, ThunderboltOutlined, ToolOutlined, UserOutlined } from "@ant-design/icons";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import { AiModel, api, EnvStatus, errText, SpeedResult, XtAccount } from "../api";

interface DlProgress {
  file: string;
  downloaded: number;
  total: number;
  speed: number;
}

const fmtSize = (b: number) => (b >= 1 << 30 ? `${(b / 2 ** 30).toFixed(2)} GB` : `${(b / 2 ** 20).toFixed(1)} MB`);
const fmtSpeed = (b: number) => (b >= 1 << 20 ? `${(b / 2 ** 20).toFixed(1)} MB/s` : `${(b / 1024).toFixed(0)} KB/s`);
const fmtEta = (p: DlProgress) => {
  if (!p.speed || p.total <= p.downloaded) return "";
  const s = Math.round((p.total - p.downloaded) / p.speed);
  return s >= 60 ? ` · 剩余约 ${Math.floor(s / 60)} 分 ${s % 60} 秒` : ` · 剩余约 ${s} 秒`;
};

const VARIANTS = [
  { value: "cu130", label: "CUDA 13.0（需要较新的驱动）" },
  { value: "cu128", label: "CUDA 12.8（RTX 50 系列等新显卡）" },
  { value: "cu126", label: "CUDA 12.6" },
  { value: "cu118", label: "CUDA 11.8（较老的显卡和驱动）" },
  { value: "cpu", label: "CPU 版（没有 NVIDIA 显卡）" },
  { value: "mps", label: "macOS 版（Apple 芯片用 MPS 加速）" },
];

export default function Settings() {
  const { message, modal } = App.useApp();
  const [st, setSt] = useState<EnvStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [installing, setInstalling] = useState(false);
  const [variant, setVariant] = useState<string | null>(null);
  const [lines, setLines] = useState<string[]>([]);
  const [progress, setProgress] = useState<DlProgress | null>(null);
  const [python, setPython] = useState("");
  const [root, setRoot] = useState("");
  const [speed, setSpeed] = useState<SpeedResult | null>(null);
  const [testing, setTesting] = useState(false);
  const [aiUrl, setAiUrl] = useState("");
  const [aiKey, setAiKey] = useState("");
  const [aiTesting, setAiTesting] = useState(false);
  /** 巡瞳 AI 连接状态：一直显示在设置里，不只是一闪而过的提示 */
  const [aiStatus, setAiStatus] = useState<
    { state: "unset" } | { state: "testing" } | { state: "ok"; models: AiModel[]; at: Date } | { state: "error"; text: string; at: Date } | null
  >(null);
  const logBox = useRef<HTMLDivElement>(null);

  const load = useCallback(
    async (refresh: boolean) => {
      setLoading(true);
      try {
        const s = await api.envStatus(refresh);
        setSt(s);
        setPython(s.settings.python ?? "");
        setRoot(s.settings.envRoot ?? "");
        setAiUrl(s.settings.aiUrl ?? "");
        setAiKey(s.settings.aiKey ?? "");
        setInstalling(s.installing);
      } catch (e) {
        message.error(errText(e));
      } finally {
        setLoading(false);
      }
    },
    [message],
  );

  useEffect(() => {
    load(false);
  }, [load]);

  useEffect(() => {
    const un = listen<string>("env-log", (e) => {
      setLines((ls) => [...ls.slice(-400), e.payload]);
      requestAnimationFrame(() => logBox.current?.scrollTo(0, logBox.current.scrollHeight));
    });
    const up = listen<DlProgress>("env-progress", (e) => setProgress(e.payload));
    return () => {
      un.then((f) => f());
      up.then((f) => f());
    };
  }, []);

  const install = async () => {
    const v = variant ?? st?.variant ?? "cpu";
    setInstalling(true);
    setLines([]);
    setProgress(null);
    try {
      await api.envInstall(v);
      message.success("训练环境安装完成");
    } catch (e) {
      modal.error({ title: "安装没有完成", content: <pre style={{ whiteSpace: "pre-wrap", fontSize: 12 }}>{errText(e)}</pre>, width: 640 });
    } finally {
      setInstalling(false);
      setProgress(null);
      load(true);
    }
  };

  const runSpeedTest = async () => {
    const v = variant ?? st?.variant ?? "cpu";
    setTesting(true);
    setSpeed(null);
    try {
      const r = await api.speedTest(v);
      setSpeed(r);
      if (r.recommend !== (st?.settings.mirror ?? "china")) {
        await saveSettings({ mirror: r.recommend });
      }
      message.success(`推荐使用${r.recommend === "china" ? "国内镜像" : "官方源"}，已自动选择`);
    } catch (e) {
      message.error(errText(e));
    } finally {
      setTesting(false);
    }
  };

  const saveSettings = async (patch: Partial<EnvStatus["settings"]>) => {
    if (!st) return;
    try {
      await api.saveSettings({ ...st.settings, ...patch });
      await load(true);
    } catch (e) {
      message.error(errText(e));
    }
  };

  /** 调一次 /v1/models 验证 Key，结果常驻显示 */
  const testAi = useCallback(async () => {
    setAiStatus({ state: "testing" });
    try {
      const ms = await api.aiModels();
      setAiStatus({ state: "ok", models: ms, at: new Date() });
      // 账号名、套餐、额度（平台不支持时是 null）
      api
        .xtAccount()
        .then(setAccount)
        .catch(() => setAccount(null));
      return true;
    } catch (e) {
      setAiStatus({ state: "error", text: errText(e), at: new Date() });
      return false;
    }
  }, []);

  // 打开设置页时，填过 Key 就自动测一次
  const autoTested = useRef(false);
  useEffect(() => {
    if (!st || autoTested.current) return;
    autoTested.current = true;
    if (st.settings.aiKey) testAi();
    else setAiStatus({ state: "unset" });
  }, [st, testAi]);

  // ---------- 登录巡瞳账号 ----------

  const [account, setAccount] = useState<XtAccount | null>(null);
  const [showManual, setShowManual] = useState(false);
  const [login, setLogin] = useState<{ deviceCode: string; userCode: string; verifyUrl: string } | null>(null);
  const loginRun = useRef(0);

  const cancelLogin = () => {
    loginRun.current++;
    setLogin(null);
  };

  const startLogin = async () => {
    if (!st) return;
    const run = ++loginRun.current;
    try {
      // 高级里改了平台地址的，先存下来（登录走这个地址）
      if ((aiUrl.trim() || null) !== (st.settings.aiUrl ?? null)) {
        await api.saveSettings({ ...st.settings, aiUrl: aiUrl.trim() || null });
      }
      const c = await api.xtLoginStart();
      if (run !== loginRun.current) return;
      setLogin({ deviceCode: c.deviceCode, userCode: c.userCode, verifyUrl: c.verifyUrl });
      let interval = c.interval;
      const deadline = Date.now() + c.expiresIn * 1000;
      while (run === loginRun.current) {
        await new Promise((r) => setTimeout(r, interval * 1000));
        if (run !== loginRun.current) return;
        if (Date.now() > deadline) throw "登录码过期了，请重新点「登录巡瞳 AI 账号」";
        const r = await api.xtLoginPoll(c.deviceCode);
        if (r.status === "slow_down") interval += 5;
        else if (r.status === "expired") throw "登录码过期了，请重新点「登录巡瞳 AI 账号」";
        else if (r.status === "denied") throw "你在网页上选了拒绝";
        else if (r.status === "ok") {
          setLogin(null);
          await load(false);
          await testAi();
          message.success(`已登录巡瞳 AI${r.account?.name ? `：${r.account.name}` : ""}`);
          return;
        }
      }
    } catch (e) {
      if (run === loginRun.current) {
        setLogin(null);
        message.error(errText(e));
      }
    }
  };

  const logoutAi = () =>
    modal.confirm({
      title: "退出巡瞳 AI 账号？",
      content: "这台电脑上的 Key 会被删除（登录时自动创建的 Key 也会在平台上吊销）。巡瞳 AI 相关功能要重新登录才能用。",
      okText: "退出",
      cancelText: "取消",
      onOk: async () => {
        try {
          await api.xtLogout();
          await load(false);
          setAccount(null);
          setAiStatus({ state: "unset" });
        } catch (e) {
          message.error(errText(e));
        }
      },
    });

  /** 保存巡瞳 AI 设置并测试连接 */
  const saveAi = async () => {
    if (!st) return;
    setAiTesting(true);
    try {
      await api.saveSettings({ ...st.settings, aiUrl: aiUrl.trim() || null, aiKey: aiKey.trim() || null });
      await load(false);
      if (aiKey.trim()) {
        if (await testAi()) message.success("巡瞳 AI 连接成功");
        else message.error("巡瞳 AI 连接失败，原因见下方");
      } else {
        setAiStatus({ state: "unset" });
        message.success("已保存（没有填 API Key）");
      }
    } catch (e) {
      message.error(errText(e));
    } finally {
      setAiTesting(false);
    }
  };

  const info = st?.info as Record<string, string | number | boolean> | null | undefined;

  return (
    <div className="page" style={{ maxWidth: 1000 }}>
      <h1>设置</h1>
      <p className="desc">训练环境和下载源。标注、导入导出不需要训练环境。</p>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
          <h2 style={{ fontSize: 15, margin: 0 }}>
            训练环境{" "}
            {st && (st.ready ? <Tag color="success" icon={<CheckCircleOutlined />}>就绪</Tag> : <Tag color="warning">未就绪</Tag>)}
          </h2>
          <Button size="small" icon={<ReloadOutlined />} loading={loading} onClick={() => load(true)}>
            重新检查
          </Button>
        </div>
        {!st ? (
          <Spin />
        ) : (
          <>
            <Descriptions column={1} size="small" bordered labelStyle={{ width: 140 }}>
              <Descriptions.Item label="显卡">
                {st.gpu
                  ? `${st.gpu.name} · ${(st.gpu.vramMb / 1024).toFixed(0)} GB · 驱动 ${st.gpu.driver} · 支持 CUDA ${st.gpu.cuda}`
                  : st.variant === "mps"
                    ? "Mac：Apple 芯片用 MPS 加速训练（Intel 芯片的 Mac 只能用 CPU）"
                    : "没有检测到 NVIDIA 显卡，只能用 CPU 训练（会慢很多）"}
              </Descriptions.Item>
              <Descriptions.Item label="Python">{st.python ?? <span className="dim">未安装</span>}</Descriptions.Item>
              {info && (
                <Descriptions.Item label="PyTorch">
                  {info.torch ? (
                    `${info.torch} · ${info.cuda ? `显卡可用（${info.device}）` : info.mps ? "Apple 芯片 MPS 加速可用" : "CPU"}`
                  ) : (
                    <span style={{ color: "var(--bad)" }}>{String(info.error ?? "未安装")}</span>
                  )}
                </Descriptions.Item>
              )}
              {st.error && (
                <Descriptions.Item label="问题">
                  <span style={{ color: "var(--bad)", whiteSpace: "pre-wrap" }}>{st.error}</span>
                </Descriptions.Item>
              )}
            </Descriptions>

            {st.managed && (
              <div style={{ marginTop: 16 }}>
                <div className="row" style={{ marginBottom: 8 }}>
                  <span>PyTorch 版本</span>
                  <Select
                    style={{ width: 300 }}
                    value={variant ?? st.variant}
                    onChange={setVariant}
                    options={VARIANTS.filter((v) => (st.variant === "mps" ? v.value === "mps" : v.value !== "mps")).map((v) => ({
                      ...v,
                      label: v.value === st.variant ? `${v.label}（推荐）` : v.label,
                    }))}
                  />
                  <Button type="primary" icon={<ToolOutlined />} loading={installing} onClick={install}>
                    {st.ready ? "重新安装" : "一键安装"}
                  </Button>
                </div>
                <div className="dim" style={{ fontSize: 12, marginBottom: 8 }}>
                  GPU 版约 3GB，只需要一次。会自动下载 uv、Python 3.11、PyTorch。
                </div>
                <div className="row" style={{ gap: 8, flexWrap: "nowrap" }}>
                  <span style={{ whiteSpace: "nowrap" }}>安装位置</span>
                  <Input size="small" value={root} placeholder="默认：系统的本机数据目录（C 盘）" onChange={(e) => setRoot(e.target.value)} />
                  <Button
                    size="small"
                    icon={<FolderOpenOutlined />}
                    onClick={async () => {
                      const d = await open({ directory: true, title: "选择训练环境安装位置（建议空间大的盘）" });
                      if (typeof d === "string") setRoot(d);
                    }}
                  />
                  <Button size="small" disabled={(root.trim() || null) === (st.settings.envRoot || null)} onClick={() => saveSettings({ envRoot: root.trim() || null })}>
                    保存
                  </Button>
                </div>
                <div className="dim" style={{ fontSize: 12, marginTop: 4 }}>当前：{st.envDir}。改位置后需要重新安装。</div>
              </div>
            )}

            {installing && progress && progress.total > 0 && (
              <div style={{ marginTop: 12 }}>
                <Progress percent={Math.floor((progress.downloaded / progress.total) * 100)} status="active" />
                <div className="dim" style={{ fontSize: 12 }}>
                  {progress.file} · {fmtSize(progress.downloaded)} / {fmtSize(progress.total)} · {fmtSpeed(progress.speed)}
                  {fmtEta(progress)}
                </div>
              </div>
            )}

            {(installing || lines.length > 0) && (
              <div
                ref={logBox}
                style={{ marginTop: 12, height: 220, overflow: "auto", background: "#0e1013", borderRadius: 6, padding: 10, font: "12px/1.6 Consolas, monospace", color: "#b8c0c9", whiteSpace: "pre-wrap", userSelect: "text" }}
              >
                {lines.join("\n")}
              </div>
            )}
          </>
        )}
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
          <h2 style={{ fontSize: 15, margin: 0 }}>下载源</h2>
          <Button size="small" icon={<ThunderboltOutlined />} loading={testing} onClick={runSpeedTest}>
            测速并推荐
          </Button>
        </div>
        <Radio.Group value={st?.settings.mirror ?? "china"} onChange={(e) => saveSettings({ mirror: e.target.value })}>
          <Space direction="vertical">
            <Radio value="china">
              国内镜像（阿里云、清华、npmmirror，推荐国内使用）
              {speed?.china != null && <Tag style={{ marginLeft: 8 }} color={speed.recommend === "china" ? "success" : "default"}>{speed.china.toFixed(1)} MB/s{speed.recommend === "china" ? " · 更快" : ""}</Tag>}
              {speed != null && speed.china == null && <Tag style={{ marginLeft: 8 }}>连不上</Tag>}
            </Radio>
            <Radio value="official">
              官方源（PyPI、download.pytorch.org）
              {speed?.official != null && <Tag style={{ marginLeft: 8 }} color={speed.recommend === "official" ? "success" : "default"}>{speed.official.toFixed(1)} MB/s{speed.recommend === "official" ? " · 更快" : ""}</Tag>}
              {speed != null && speed.official == null && <Tag style={{ marginLeft: 8 }}>连不上</Tag>}
            </Radio>
          </Space>
        </Radio.Group>
        <div className="dim" style={{ fontSize: 12, marginTop: 8 }}>
          点「测速并推荐」会分别从两个源各拉几 MB 实测，自动勾选更快的。国内镜像也会让 SAM、文本提示模型从 hf-mirror.com 下载。
        </div>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <h2 style={{ fontSize: 15, marginTop: 0 }}>巡瞳 AI</h2>
        <p className="muted" style={{ marginTop: 0 }}>
          登录巡瞳 AI 账号后可以：标注页用巡瞳 AI 自动标注（专用模型批量标注很快，也可以用大模型或按项目类别识别）、和 AI 聊识别规则；部署页把模型上传到巡瞳平台；数据页拉取平台的告警图片；协作用「巡瞳官方中转」让外网成员连进来。识别、聊天会消耗账号的额度。
        </p>
        <Space direction="vertical" style={{ width: "100%" }}>
          {aiStatus?.state === "ok" ? (
            <Alert
              type="success"
              showIcon
              message={
                <span>
                  已登录{account?.name ? <b> {account.name}</b> : "，可以使用"}
                  {account?.plan && <Tag style={{ marginLeft: 8 }}>{account.plan}</Tag>}
                  <span className="dim" style={{ fontSize: 12, marginLeft: 8 }}>
                    {aiStatus.at.toLocaleTimeString()} 检查
                  </span>
                </span>
              }
              description={
                <div>
                  {account?.quota && account.quota.length > 0 && (
                    <div style={{ marginBottom: 6 }}>
                      {account.quota.map((q) => (
                        <span key={q.name} style={{ marginRight: 16 }}>
                          {q.name}：剩 <b>{q.total == null ? "不限" : Math.max(0, q.total - (q.used ?? 0))}</b>
                          {q.total != null && ` / ${q.total}`} {q.unit ?? ""}
                          {q.expiresAt ? <span className="dim" style={{ fontSize: 12 }}>（{new Date(q.expiresAt).toLocaleDateString()} 到期）</span> : null}
                        </span>
                      ))}
                    </div>
                  )}
                  {account?.plan === "未开通套餐" && (
                    <div style={{ marginBottom: 6, color: "var(--warn)" }}>账号还没有开通套餐，识别会失败，请先到巡瞳 AI 开通。</div>
                  )}
                  <div style={{ marginBottom: 4 }}>
                    可用的识别方式（在标注页「AI 辅助 → 巡瞳 AI」里选）：
                  </div>
                  <Space size={[4, 4]} wrap>
                    {aiStatus.models.map((m) => (
                      <Tag key={m.code} color={m.code === "classes" || m.code === "llm" ? "blue" : undefined}>
                        {m.name}
                      </Tag>
                    ))}
                  </Space>
                  <div style={{ marginTop: 8 }}>
                    <Space>
                      <Button size="small" onClick={testAi}>
                        刷新
                      </Button>
                      <Button size="small" onClick={logoutAi}>
                        退出登录
                      </Button>
                    </Space>
                  </div>
                </div>
              }
            />
          ) : (
            <>
              {aiStatus?.state === "testing" && <Alert type="info" showIcon message="正在连接巡瞳 AI…" />}
              {aiStatus?.state === "error" && (
                <Alert
                  type="error"
                  showIcon
                  message={`连接失败（${aiStatus.at.toLocaleTimeString()} 检查）`}
                  description={
                    <div>
                      <div>{aiStatus.text}</div>
                      <div className="dim" style={{ fontSize: 12, marginTop: 4 }}>
                        可以重新登录；手动填的 Key 请检查是否完整复制（以 sk-live- 开头）、在巡瞳控制台里是否启用、有没有设 IP 白名单。
                      </div>
                    </div>
                  }
                />
              )}
              <Space>
                <Button type="primary" icon={<UserOutlined />} loading={!!login} onClick={startLogin}>
                  登录巡瞳 AI 账号
                </Button>
                <span className="dim" style={{ fontSize: 12 }}>
                  在浏览器里登录巡瞳、点「允许」，Key 自动带回来，不用复制粘贴
                </span>
              </Space>
            </>
          )}
          <a style={{ fontSize: 12 }} onClick={() => setShowManual((v) => !v)}>
            {showManual ? "收起" : "高级：手动填 API Key、私有化部署的平台地址"}
          </a>
          {showManual && (
            <Space direction="vertical" style={{ width: "100%" }}>
              <Input addonBefore="平台地址" value={aiUrl} placeholder="留空就是 https://xuntongai.com（私有化部署时填自己的地址）" onChange={(e) => setAiUrl(e.target.value)} />
              <Input.Password addonBefore="API Key" value={aiKey} placeholder="sk-live-…（在巡瞳 AI 控制台的「API Key」页创建）" onChange={(e) => setAiKey(e.target.value)} />
              <Button loading={aiTesting} onClick={saveAi}>
                保存并测试连接
              </Button>
            </Space>
          )}
        </Space>
      </div>

      <Modal
        title="登录巡瞳 AI 账号"
        open={!!login}
        onCancel={cancelLogin}
        footer={<Button onClick={cancelLogin}>取消</Button>}
        width={460}
        destroyOnHidden
      >
        {login && (
          <div style={{ textAlign: "center", padding: "8px 0" }}>
            <div style={{ marginBottom: 12 }}>已在浏览器里打开巡瞳 AI，登录后点「允许」，这里会自动完成。</div>
            <div className="dim" style={{ fontSize: 12 }}>确认码（网页上显示的要和这个一样）</div>
            <div style={{ fontSize: 28, fontWeight: 700, letterSpacing: 4, margin: "4px 0 12px" }}>{login.userCode}</div>
            <Spin size="small" /> <span className="muted">等待你在浏览器里确认…</span>
            <div style={{ marginTop: 14, fontSize: 12 }}>
              浏览器没打开？
              <a onClick={() => api.openUrl(login.verifyUrl).catch(() => {})}>重新打开</a>
              ，或者复制地址：
              <Typography.Text copyable style={{ fontSize: 12 }}>
                {login.verifyUrl}
              </Typography.Text>
            </div>
          </div>
        )}
      </Modal>
      <div className="card">
        <h2 style={{ fontSize: 15, marginTop: 0 }}>使用已有的 Python 环境（高级）</h2>
        <p className="muted" style={{ marginTop: 0 }}>
          已经有装好 PyTorch 的环境可以直接指定，就不用再下载。需要 torch、torchvision、numpy、pillow，导出 ONNX 还需要 onnx。
        </p>
        <Space.Compact style={{ width: "100%" }}>
          <Input value={python} placeholder="例如 D:\envs\torch\Scripts\python.exe（留空则用软件自己安装的环境）" onChange={(e) => setPython(e.target.value)} />
          <Button
            icon={<FolderOpenOutlined />}
            onClick={async () => {
              const f = await open({ title: "选择 python.exe", filters: [{ name: "Python", extensions: ["exe", ""] }] });
              if (typeof f === "string") setPython(f);
            }}
          />
          <Button type="primary" onClick={() => saveSettings({ python: python.trim() || null })}>
            保存
          </Button>
        </Space.Compact>
        {st && !st.managed && (
          <Alert style={{ marginTop: 10 }} type="info" showIcon message="正在使用你指定的 Python 环境。清空后保存可改回软件自己管理的环境。" />
        )}
      </div>
    </div>
  );
}
