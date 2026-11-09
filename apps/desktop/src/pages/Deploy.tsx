import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  App,
  Button,
  Checkbox,
  Empty,
  Form,
  Input,
  InputNumber,
  Segmented,
  Select,
  Slider,
  Space,
  Switch,
  Tabs,
  Tag,
} from "antd";
import {
  CameraOutlined,
  CaretRightOutlined,
  DeleteOutlined,
  FolderOpenOutlined,
  ImportOutlined,
  PlusOutlined,
  StopOutlined,
} from "@ant-design/icons";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import { api, errText, FormatResult, Label, ProjectInfo, RunInfo } from "../api";
import { prefs, useApp } from "../store";
import { rgba } from "../components/AnnotCanvas";

const stamp = () => {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}`;
};
const join = (parent: string, name: string) => `${parent.replace(/[\\/]+$/, "")}${parent.includes("\\") ? "\\" : "/"}${name}`;

function FolderPicker({ value, onChange, title }: { value: string; onChange: (v: string) => void; title: string }) {
  return (
    <Space.Compact style={{ width: "100%" }}>
      <Input value={value} placeholder="选择存放位置" onChange={(e) => onChange(e.target.value)} />
      <Button
        icon={<FolderOpenOutlined />}
        onClick={async () => {
          const d = await open({ directory: true, title });
          if (typeof d === "string") onChange(d);
        }}
      />
    </Space.Compact>
  );
}

function LogBox({ lines }: { lines: string[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => ref.current?.scrollTo(0, ref.current.scrollHeight), [lines]);
  if (lines.length === 0) return null;
  return (
    <div ref={ref} className="logbox" style={{ height: 140, marginTop: 12 }}>
      {lines.join("\n")}
    </div>
  );
}

// ---------------- 实时验证 ----------------

interface FrameEv {
  jpg: string;
  w: number;
  h: number;
  boxes: [number, number, number, number, number, number][];
  ms: number;
  fps: number;
}

function LiveTab({ run, labels }: { run: RunInfo; labels: Label[] }) {
  const { message } = App.useApp();
  const refreshStats = useApp((s) => s.refreshStats);
  const [kind, setKind] = useState(prefs.get("liveKind", "rtsp"));
  const [url, setUrl] = useState(prefs.get("liveUrl", ""));
  const [file, setFile] = useState(prefs.get("liveFile", ""));
  const [conf, setConf] = useState(Number(prefs.get("liveConf", "0.4")));
  const [running, setRunning] = useState(false);
  const [starting, setStarting] = useState(false);
  const [classes, setClasses] = useState<string[]>([]);
  const [stat, setStat] = useState<{ w: number; h: number; fps: number; ms: number; counts: Record<string, number> } | null>(null);
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const busy = useRef(false);
  const classesRef = useRef<string[]>([]);
  classesRef.current = classes;

  const colorOf = useCallback(
    (name: string) => labels.find((l) => l.name === name)?.color ?? "#ffd43b",
    [labels],
  );
  const colorRef = useRef(colorOf);
  colorRef.current = colorOf;

  const drawFrame = (f: FrameEv) => {
    const cv = canvas.current;
    if (!cv || busy.current || !f.jpg) return;
    busy.current = true;
    const img = new Image();
    img.onload = () => {
      busy.current = false;
      const box = cv.parentElement!;
      const s = Math.min(box.clientWidth / f.w, box.clientHeight / f.h);
      const dpr = window.devicePixelRatio || 1;
      const W = Math.floor(f.w * s);
      const H = Math.floor(f.h * s);
      cv.style.width = `${W}px`;
      cv.style.height = `${H}px`;
      cv.width = Math.round(W * dpr);
      cv.height = Math.round(H * dpr);
      const ctx = cv.getContext("2d")!;
      ctx.drawImage(img, 0, 0, cv.width, cv.height);
      const k = (W * dpr) / f.w;
      ctx.lineWidth = 2 * dpr;
      ctx.font = `${12 * dpr}px "Microsoft YaHei UI", sans-serif`;
      for (const [x, y, w, h, c, sc] of f.boxes) {
        const name = classesRef.current[c] ?? String(c);
        const color = colorRef.current(name);
        ctx.strokeStyle = color;
        ctx.fillStyle = rgba(color, 0.12);
        ctx.fillRect(x * k, y * k, w * k, h * k);
        ctx.strokeRect(x * k, y * k, w * k, h * k);
        const text = `${name} ${sc.toFixed(2)}`;
        const tw = ctx.measureText(text).width + 8 * dpr;
        const ty = y * k > 18 * dpr ? y * k - 18 * dpr : y * k;
        ctx.fillStyle = color;
        ctx.fillRect(x * k, ty, tw, 18 * dpr);
        ctx.fillStyle = "#fff";
        ctx.fillText(text, x * k + 4 * dpr, ty + 13 * dpr);
      }
    };
    img.onerror = () => (busy.current = false);
    img.src = `data:image/jpeg;base64,${f.jpg}`;
  };

  useEffect(() => {
    const un = listen<{ kind: string; data: Record<string, unknown> }>("live", (e) => {
      const { kind: k, data } = e.payload;
      if (k === "open") {
        // 第一帧可能紧跟着到，先同步记下类别名
        classesRef.current = data.classes as string[];
        setClasses(data.classes as string[]);
        setStatus(`已连接 · ${data.width}×${data.height} · ${data.device === "cuda" ? "显卡推理" : "CPU 推理"}`);
      } else if (k === "frame") {
        const f = data as unknown as FrameEv;
        drawFrame(f);
        const counts: Record<string, number> = {};
        for (const b of f.boxes) {
          const n = classesRef.current[b[4]] ?? String(b[4]);
          counts[n] = (counts[n] ?? 0) + 1;
        }
        setStat({ w: f.w, h: f.h, fps: f.fps, ms: f.ms, counts });
      } else if (k === "log") {
        setStatus(String(data));
      } else if (k === "snapshot") {
        if (data.error) message.error(`截图失败：${data.error}`);
        else if (data.added) {
          message.success("已加入数据集（未标注）");
          refreshStats();
        } else message.info("这张和已有图片重复，没有重复加入");
      } else if (k === "ended") {
        setStatus("视频播放完毕");
      } else if (k === "error") {
        setError(String(data.msg ?? "出错了"));
      } else if (k === "stopped") {
        setRunning(false);
      }
    });
    return () => {
      un.then((f) => f());
      api.liveStop().catch(() => {});
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const source = kind === "rtsp" ? url : file;

  const start = async () => {
    setError(null);
    setStat(null);
    setStatus("正在连接…");
    setStarting(true);
    prefs.set("liveKind", kind);
    prefs.set("liveUrl", url);
    prefs.set("liveFile", file);
    try {
      await api.liveStart(run.id, source, conf);
      setRunning(true);
    } catch (e) {
      setError(errText(e));
      setStatus("");
    } finally {
      setStarting(false);
    }
  };

  return (
    <div>
      <div className="row" style={{ marginBottom: 12 }}>
        <Segmented
          value={kind}
          onChange={(v) => setKind(v as string)}
          options={[
            { value: "rtsp", label: "RTSP 摄像头" },
            { value: "file", label: "本地视频" },
          ]}
        />
        {kind === "rtsp" ? (
          <Input
            style={{ flex: 1, minWidth: 320 }}
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="rtsp://admin:密码@192.168.1.64:554/Streaming/Channels/101"
          />
        ) : (
          <Space.Compact style={{ flex: 1, minWidth: 320 }}>
            <Input value={file} onChange={(e) => setFile(e.target.value)} placeholder="选择 mp4 / avi / mkv 等视频文件" />
            <Button
              icon={<FolderOpenOutlined />}
              onClick={async () => {
                const f = await open({ title: "选择视频文件", filters: [{ name: "视频", extensions: ["mp4", "avi", "mkv", "mov", "flv", "ts", "h264", "264"] }] });
                if (typeof f === "string") setFile(f);
              }}
            />
          </Space.Compact>
        )}
        {running ? (
          <Button danger icon={<StopOutlined />} onClick={() => api.liveStop()}>
            停止
          </Button>
        ) : (
          <Button type="primary" icon={<CaretRightOutlined />} loading={starting} disabled={!source.trim()} onClick={start}>
            开始
          </Button>
        )}
      </div>
      <div className="row" style={{ marginBottom: 12 }}>
        <span className="muted">置信度</span>
        <Slider
          style={{ width: 220 }}
          min={0.05}
          max={0.95}
          step={0.05}
          value={conf}
          onChange={(v) => {
            setConf(v);
            prefs.set("liveConf", String(v));
            if (running) api.liveConf(v).catch(() => {});
          }}
        />
        <span style={{ width: 36 }}>{conf.toFixed(2)}</span>
        <Button icon={<CameraOutlined />} disabled={!running} onClick={() => api.liveSnapshot().catch((e) => message.error(errText(e)))}>
          截图加入数据集
        </Button>
        <span className="dim" style={{ fontSize: 12 }}>
          看到漏检、误报的画面就截下来，标注后再训练
        </span>
      </div>
      {error && <Alert type="error" showIcon style={{ marginBottom: 12 }} message={error} closable onClose={() => setError(null)} />}
      <div className="live-box">
        <canvas ref={canvas} style={{ display: stat ? "block" : "none" }} />
        {!stat && <div className="dim">{status || "填写 RTSP 地址或选择视频文件，点「开始」查看模型的实时检测效果"}</div>}
        {stat && (
          <div className="live-stat">
            <span>{status}</span>
            <span>{stat.fps.toFixed(1)} FPS · 推理 {stat.ms.toFixed(0)} ms</span>
            {Object.entries(stat.counts).map(([n, c]) => (
              <Tag key={n} color={colorOf(n)} style={{ marginInlineEnd: 0 }}>
                {n} {c}
              </Tag>
            ))}
          </div>
        )}
      </div>
      <div className="dim" style={{ fontSize: 12, marginTop: 8 }}>
        海康：rtsp://用户名:密码@IP:554/Streaming/Channels/101（主码流）或 102（子码流）；大华：rtsp://用户名:密码@IP:554/cam/realmonitor?channel=1&amp;subtype=0
      </div>
    </div>
  );
}

// ---------------- 导出模型 ----------------

const FORMATS = [
  { value: "onnx", name: "ONNX（动态尺寸）", desc: "通用格式，onnxruntime 在 Windows/Linux/ARM 上都能直接跑，任意输入尺寸" },
  { value: "onnx_static", name: "ONNX（固定尺寸）", desc: "输入固定为 [1,3,N,N]，给需要固定形状的推理框架用" },
  { value: "openvino", name: "OpenVINO", desc: "Intel CPU、核显、工控机加速，在软件里直接转换并验证（首次使用会自动安装 OpenVINO）" },
  { value: "tensorrt", name: "TensorRT 转换包", desc: "NVIDIA 显卡、Jetson。引擎和设备绑定，需在目标设备上运行脚本生成" },
  { value: "rknn", name: "RKNN 转换包", desc: "瑞芯微 RK3588/3568 等。官方工具只支持 Linux，需在 Ubuntu 上运行脚本转换" },
];

function ExportTab({ run, project }: { run: RunInfo; project: ProjectInfo }) {
  const { message } = App.useApp();
  const [formats, setFormats] = useState<string[]>(["onnx", "openvino"]);
  const [size, setSize] = useState(run.config?.imgSize ?? 640);
  const [parent, setParent] = useState(prefs.get("deployParent", ""));
  const [name, setName] = useState(`${project.name}_模型_${stamp()}`);
  const [busy, setBusy] = useState(false);
  const [lines, setLines] = useState<string[]>([]);
  const [results, setResults] = useState<FormatResult[]>([]);

  const go = async () => {
    if (!parent) {
      message.warning("请选择存放位置");
      return;
    }
    setBusy(true);
    setLines([]);
    setResults([]);
    prefs.set("deployParent", parent);
    const un = await listen<string>("deploy-log", (e) => setLines((l) => [...l.slice(-200), e.payload]));
    try {
      setResults(await api.exportFormats(run.id, formats, size, join(parent, name)));
      setName(`${project.name}_模型_${stamp()}`);
    } catch (e) {
      message.error(errText(e));
    } finally {
      un();
      setBusy(false);
    }
  };

  return (
    <div style={{ maxWidth: 820 }}>
      <Checkbox.Group value={formats} onChange={(v) => setFormats(v as string[])} style={{ width: "100%" }}>
        <Space direction="vertical" style={{ width: "100%" }}>
          {FORMATS.map((f) => (
            <Checkbox key={f.value} value={f.value}>
              <b>{f.name}</b> <span className="muted">· {f.desc}</span>
            </Checkbox>
          ))}
        </Space>
      </Checkbox.Group>
      <Alert
        type="warning"
        showIcon
        style={{ margin: "12px 0" }}
        message="内置的 Faster R-CNN 是两阶段模型，NPU（RKNN）和老版本 TensorRT 对它支持有限，转换可能失败或速度不理想；ONNX 和 OpenVINO 可以放心使用。"
      />
      <Form layout="vertical">
        <Form.Item label="固定尺寸（用于固定尺寸 ONNX、TensorRT、RKNN）">
          <Select style={{ width: 160 }} value={size} onChange={setSize} options={[320, 480, 640, 800, 960, 1280].map((v) => ({ value: v, label: `${v}×${v}` }))} />
        </Form.Item>
        <Form.Item label="导出到" extra={parent && join(parent, name)}>
          <FolderPicker value={parent} onChange={setParent} title="选择导出位置" />
          <Input style={{ marginTop: 8 }} value={name} onChange={(e) => setName(e.target.value)} addonBefore="文件夹名" />
        </Form.Item>
      </Form>
      <Button type="primary" loading={busy} disabled={formats.length === 0} onClick={go}>
        导出
      </Button>
      <LogBox lines={lines} />
      {results.length > 0 && (
        <div className="card" style={{ marginTop: 12 }}>
          {results.map((r) => (
            <div key={r.format} className="row" style={{ justifyContent: "space-between", padding: "6px 0", borderBottom: "1px solid var(--line)" }}>
              <span>
                <Tag color={r.ok ? "success" : "error"}>{r.ok ? "成功" : "失败"}</Tag>
                <b>{FORMATS.find((f) => f.value === r.format)?.name ?? r.format}</b> <span className="muted">{r.note}</span>
              </span>
              {r.path && (
                <Button size="small" icon={<FolderOpenOutlined />} onClick={() => api.revealPath(r.path)}>
                  打开
                </Button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------- 推理服务包 ----------------

interface StreamRow {
  name: string;
  url: string;
}

interface ServiceForm {
  streams: StreamRow[];
  interval: number;
  labels: string[];
  minScore: number;
  cooldown: number;
  hours: string;
  saveImage: boolean;
  webhookUrl: string;
  webhookImage: boolean;
  mqttHost: string;
  mqttPort: number;
  mqttTopic: string;
  mqttUser: string;
  mqttPass: string;
  port: number;
  gpu: boolean;
  reflux: boolean;
  refluxLow: number;
  refluxHigh: number;
}

const DEFAULT_FORM: ServiceForm = {
  streams: [{ name: "摄像头1", url: "" }],
  interval: 1,
  labels: [],
  minScore: 0.5,
  cooldown: 60,
  hours: "",
  saveImage: true,
  webhookUrl: "",
  webhookImage: false,
  mqttHost: "",
  mqttPort: 1883,
  mqttTopic: "xlabel/alarm",
  mqttUser: "",
  mqttPass: "",
  port: 8090,
  gpu: true,
  reflux: true,
  refluxLow: 0.3,
  refluxHigh: 0.5,
};

function ServiceTab({ run, project }: { run: RunInfo; project: ProjectInfo }) {
  const { message, modal } = App.useApp();
  const classes = run.config?.classes ?? [];
  const [f, setF] = useState<ServiceForm>(() => {
    try {
      return { ...DEFAULT_FORM, ...JSON.parse(prefs.get("serviceForm", "{}")) };
    } catch {
      return DEFAULT_FORM;
    }
  });
  const [parent, setParent] = useState(prefs.get("deployParent", ""));
  const [name, setName] = useState(`${project.name}_推理服务_${stamp()}`);
  const [busy, setBusy] = useState(false);
  const [lines, setLines] = useState<string[]>([]);

  const set = (patch: Partial<ServiceForm>) => {
    const next = { ...f, ...patch };
    setF(next);
    prefs.set("serviceForm", JSON.stringify(next));
  };
  const setStream = (i: number, patch: Partial<StreamRow>) => set({ streams: f.streams.map((s, j) => (j === i ? { ...s, ...patch } : s)) });

  const build = async () => {
    if (!parent) {
      message.warning("请选择存放位置");
      return;
    }
    if (f.hours && !/^\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}$/.test(f.hours.trim())) {
      message.warning("生效时段格式：07:00-19:00");
      return;
    }
    const config = {
      gpu: f.gpu,
      max_side: 1280,
      host: "0.0.0.0",
      port: f.port,
      interval: f.interval,
      streams: f.streams.filter((s) => s.url.trim()).map((s) => ({ name: s.name.trim() || s.url, url: s.url.trim() })),
      alert: { labels: f.labels, min_score: f.minScore, cooldown: f.cooldown, hours: f.hours.trim(), save_image: f.saveImage },
      webhook: { url: f.webhookUrl.trim(), with_image: f.webhookImage, headers: {} },
      mqtt: { host: f.mqttHost.trim(), port: f.mqttPort, topic: f.mqttTopic.trim(), username: f.mqttUser, password: f.mqttPass },
      reflux: { enabled: f.reflux, low: f.refluxLow, high: f.refluxHigh, interval: 60 },
    };
    setBusy(true);
    setLines([]);
    prefs.set("deployParent", parent);
    const un = await listen<string>("deploy-log", (e) => setLines((l) => [...l.slice(-200), e.payload]));
    try {
      const out = await api.buildService(run.id, config, join(parent, name));
      setName(`${project.name}_推理服务_${stamp()}`);
      modal.success({
        title: "推理服务包已生成",
        width: 560,
        content: (
          <div>
            <div style={{ wordBreak: "break-all" }}>{out}</div>
            <div className="muted" style={{ marginTop: 8 }}>
              整个文件夹拷到现场电脑：Windows 双击 run.bat，Linux 运行 sh run.sh。配置在 config.json，说明见 README.md。
            </div>
          </div>
        ),
        okText: "打开文件夹",
        onOk: () => api.revealPath(out),
      });
    } catch (e) {
      message.error(errText(e));
    } finally {
      un();
      setBusy(false);
    }
  };

  return (
    <Form layout="vertical" style={{ maxWidth: 900 }}>
      <div className="cards" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <div className="card">
          <h3 className="h3">视频流</h3>
          {f.streams.map((s, i) => (
            <div key={i} className="row" style={{ marginBottom: 8, flexWrap: "nowrap" }}>
              <Input style={{ width: 110 }} value={s.name} placeholder="名称" onChange={(e) => setStream(i, { name: e.target.value })} />
              <Input value={s.url} placeholder="rtsp://… 或视频文件路径" onChange={(e) => setStream(i, { url: e.target.value })} />
              <Button icon={<DeleteOutlined />} disabled={f.streams.length === 1} onClick={() => set({ streams: f.streams.filter((_, j) => j !== i) })} />
            </div>
          ))}
          <Button size="small" icon={<PlusOutlined />} onClick={() => set({ streams: [...f.streams, { name: `摄像头${f.streams.length + 1}`, url: "" }] })}>
            添加一路
          </Button>
          <Form.Item label="检测间隔（秒）" style={{ marginTop: 12, marginBottom: 0 }} extra="每路视频每隔多久检测一帧">
            <InputNumber min={0.1} max={60} step={0.5} value={f.interval} onChange={(v) => set({ interval: Number(v ?? 1) })} />
          </Form.Item>
          <div className="dim" style={{ fontSize: 12, marginTop: 8 }}>需要只检测画面某个区域时，生成后在 config.json 里给视频流加 region，见 README。</div>
        </div>

        <div className="card">
          <h3 className="h3">告警规则</h3>
          <Form.Item label="告警类别" extra="不选表示所有类别都告警" style={{ marginBottom: 10 }}>
            <Select mode="multiple" value={f.labels} onChange={(v) => set({ labels: v })} options={classes.map((c) => ({ value: c, label: c }))} placeholder="全部类别" />
          </Form.Item>
          <Space wrap>
            <Form.Item label="置信度阈值" style={{ marginBottom: 10 }}>
              <InputNumber min={0.05} max={0.99} step={0.05} value={f.minScore} onChange={(v) => set({ minScore: Number(v ?? 0.5) })} />
            </Form.Item>
            <Form.Item label="去重间隔（秒）" tooltip="同一路视频同一类别，这段时间内只告警一次" style={{ marginBottom: 10 }}>
              <InputNumber min={0} max={86400} value={f.cooldown} onChange={(v) => set({ cooldown: Number(v ?? 60) })} />
            </Form.Item>
            <Form.Item label="生效时段" style={{ marginBottom: 10 }}>
              <Input style={{ width: 130 }} value={f.hours} placeholder="全天" onChange={(e) => set({ hours: e.target.value })} />
            </Form.Item>
          </Space>
          <Checkbox checked={f.saveImage} onChange={(e) => set({ saveImage: e.target.checked })}>
            保存告警截图（带框）
          </Checkbox>
        </div>

        <div className="card">
          <h3 className="h3">告警推送</h3>
          <Form.Item label="HTTP Webhook" extra="告警时 POST JSON 到这个地址" style={{ marginBottom: 8 }}>
            <Input value={f.webhookUrl} placeholder="http://你的平台/api/alarm（不填则不推送）" onChange={(e) => set({ webhookUrl: e.target.value })} />
          </Form.Item>
          <Checkbox checked={f.webhookImage} onChange={(e) => set({ webhookImage: e.target.checked })} style={{ marginBottom: 12 }}>
            附带截图（base64）
          </Checkbox>
          <Form.Item label="MQTT" style={{ marginBottom: 0 }}>
            <Space.Compact style={{ width: "100%", marginBottom: 6 }}>
              <Input value={f.mqttHost} placeholder="服务器地址（不填则不推送）" onChange={(e) => set({ mqttHost: e.target.value })} />
              <InputNumber style={{ width: 90 }} value={f.mqttPort} onChange={(v) => set({ mqttPort: Number(v ?? 1883) })} />
            </Space.Compact>
            <Input value={f.mqttTopic} addonBefore="主题" style={{ marginBottom: 6 }} onChange={(e) => set({ mqttTopic: e.target.value })} />
            <Space.Compact style={{ width: "100%" }}>
              <Input value={f.mqttUser} placeholder="用户名（可选）" onChange={(e) => set({ mqttUser: e.target.value })} />
              <Input.Password value={f.mqttPass} placeholder="密码（可选）" onChange={(e) => set({ mqttPass: e.target.value })} />
            </Space.Compact>
          </Form.Item>
        </div>

        <div className="card">
          <h3 className="h3">服务</h3>
          <Space wrap>
            <Form.Item label="HTTP 接口端口" style={{ marginBottom: 10 }}>
              <InputNumber min={1} max={65535} value={f.port} onChange={(v) => set({ port: Number(v ?? 8090) })} />
            </Form.Item>
            <Form.Item label="用显卡推理" tooltip="需要现场电脑装 onnxruntime-gpu，否则自动用 CPU" style={{ marginBottom: 10 }}>
              <Switch checked={f.gpu} onChange={(v) => set({ gpu: v })} />
            </Form.Item>
          </Space>
          <Form.Item label="现场回流" extra="模型拿不准的画面存到 reflux/，拷回来导入标注，模型越用越准" style={{ marginBottom: 0 }}>
            <Space>
              <Switch checked={f.reflux} onChange={(v) => set({ reflux: v })} />
              <span className="muted">置信度在</span>
              <InputNumber size="small" min={0.05} max={0.95} step={0.05} value={f.refluxLow} disabled={!f.reflux} onChange={(v) => set({ refluxLow: Number(v ?? 0.3) })} />
              <span className="muted">～</span>
              <InputNumber size="small" min={0.05} max={0.99} step={0.05} value={f.refluxHigh} disabled={!f.reflux} onChange={(v) => set({ refluxHigh: Number(v ?? 0.5) })} />
              <span className="muted">之间</span>
            </Space>
          </Form.Item>
        </div>
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <Form.Item label="生成到" extra={parent && join(parent, name)} style={{ marginBottom: 12 }}>
          <FolderPicker value={parent} onChange={setParent} title="选择服务包存放位置" />
          <Input style={{ marginTop: 8 }} value={name} onChange={(e) => setName(e.target.value)} addonBefore="文件夹名" />
        </Form.Item>
        <Button type="primary" loading={busy} onClick={build}>
          生成推理服务包
        </Button>
        <span className="dim" style={{ marginLeft: 12, fontSize: 12 }}>
          只依赖 onnxruntime、OpenCV、numpy，现场电脑不需要 PyTorch 和显卡
        </span>
        <LogBox lines={lines} />
      </div>
    </Form>
  );
}

// ---------------- 现场回流 ----------------

function RefluxTab() {
  const { message } = App.useApp();
  const refreshStats = useApp((s) => s.refreshStats);
  const setPage = useApp((s) => s.setPage);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const pick = async () => {
    const d = await open({ directory: true, title: "选择推理服务包里的 reflux 文件夹" });
    if (typeof d !== "string") return;
    setBusy(true);
    try {
      const r = await api.importImages([d], true);
      await refreshStats();
      setResult(`导入 ${r.added} 张，跳过重复 ${r.duplicated} 张${r.failed.length ? `，${r.failed.length} 张读取失败` : ""}`);
    } catch (e) {
      message.error(errText(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ maxWidth: 720 }}>
      <div className="steps-v">
        <div><b>1</b> 推理服务在现场运行时，会把模型拿不准（置信度在设定区间内）的画面存到服务包的 <code>reflux/</code> 文件夹。</div>
        <div><b>2</b> 定期把 <code>reflux</code> 文件夹拷回来，在这里导入。</div>
        <div><b>3</b> 到「标注」页筛选「未标注」，用模型预标注后修正。</div>
        <div><b>4</b> 重新训练，再生成新的服务包替换现场的 model.onnx。模型会越用越准。</div>
      </div>
      <Button type="primary" icon={<ImportOutlined />} loading={busy} onClick={pick} style={{ marginTop: 12 }}>
        导入现场回流图片
      </Button>
      {result && (
        <Alert
          style={{ marginTop: 12 }}
          type="success"
          showIcon
          message={result}
          action={<Button size="small" onClick={() => setPage("annotate")}>去标注</Button>}
        />
      )}
    </div>
  );
}

// ---------------- 页面 ----------------

export default function Deploy({ project }: { project: ProjectInfo }) {
  const { message } = App.useApp();
  const setPage = useApp((s) => s.setPage);
  const [runs, setRuns] = useState<RunInfo[] | null>(null);
  const [runId, setRunId] = useState<number | null>(null);
  const [labels, setLabels] = useState<Label[]>([]);
  const [tab, setTab] = useState(prefs.get("deployTab", "live"));

  useEffect(() => {
    api.listRuns().then((rs) => {
      const ok = rs.filter((r) => r.hasWeights);
      setRuns(ok);
      setRunId(ok[0]?.id ?? null);
    }).catch((e) => message.error(errText(e)));
    api.listLabels().then(setLabels).catch(() => {});
  }, [message]);

  const run = useMemo(() => runs?.find((r) => r.id === runId) ?? null, [runs, runId]);

  if (runs && runs.length === 0) {
    return (
      <div className="empty-page">
        <Empty description="还没有训练好的模型。先训练一个模型，再到这里验证和部署。">
          <Button type="primary" onClick={() => setPage("train")}>去训练</Button>
        </Empty>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}>
        <div>
          <h1>部署</h1>
          <p className="desc" style={{ margin: 0 }}>先接摄像头看实际效果，再导出模型或生成推理服务包，拷到现场就能用。</p>
        </div>
        <Select
          style={{ width: 360 }}
          value={runId}
          onChange={setRunId}
          options={(runs ?? []).map((r) => ({ value: r.id, label: `模型：${r.name} · mAP50 ${r.bestMap50?.toFixed(3) ?? "—"}` }))}
        />
      </div>
      {run && (
        <Tabs
          activeKey={tab}
          onChange={(k) => {
            setTab(k);
            prefs.set("deployTab", k);
          }}
          items={[
            { key: "live", label: "实时验证", children: <LiveTab key={run.id} run={run} labels={labels} /> },
            { key: "export", label: "导出模型", children: <ExportTab key={run.id} run={run} project={project} /> },
            { key: "service", label: "推理服务包", children: <ServiceTab key={run.id} run={run} project={project} /> },
            { key: "reflux", label: "现场回流", children: <RefluxTab /> },
          ]}
        />
      )}
    </div>
  );
}
