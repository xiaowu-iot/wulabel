import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  App,
  Button,
  Checkbox,
  Empty,
  Form,
  Input,
  InputNumber,
  Progress,
  Segmented,
  Select,
  Slider,
  Space,
  Switch,
  Tabs,
  Tag,
  Typography,
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
import { api, BatchResult, DeployStep, errText, FormatResult, ImportProgress, isClsTask, Label, lastImportIds, ProjectInfo, RunInfo } from "../api";
import { prefs, useApp } from "../store";
import { t } from "../i18n";
import { xuntongEnabled } from "../features";
import { rgba } from "../components/AnnotCanvas";
import { XuntongUpload } from "../components/Xuntong";
import { fmtTime, metricNames, modelName, MODELS } from "./Train";

const stamp = () => {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
};
/** 把译文里的 {key} 换成一个元素（比如 <code>） */
const fill = (s: string, key: string, node: ReactNode) => {
  const i = s.indexOf(`{${key}}`);
  return i < 0 ? s : <>{s.slice(0, i)}{node}{s.slice(i + key.length + 2)}</>;
};
const VIDEO_EXTS = ["mp4", "avi", "mkv", "mov", "flv", "ts", "h264", "264"];
// 推理服务包的视频流：本地路径（不带 ://）最后一段没有扩展名，多半是填成了文件夹（Linux 的 /dev/video0 这类设备除外）
const looksLikeFolder = (url: string) => {
  const u = url.trim();
  if (!u || u.includes("://") || u.startsWith("/dev/")) return false;
  const last = u.replace(/[\\/]+$/, "").split(/[\\/]/).pop() ?? "";
  return /[\\/]$/.test(u) || !/\.[A-Za-z0-9]{1,5}$/.test(last);
};
const join = (parent: string, name: string) => `${parent.replace(/[\\/]+$/, "")}${parent.includes("\\") ? "\\" : "/"}${name}`;

function FolderPicker({ value, onChange, title }: { value: string; onChange: (v: string) => void; title: string }) {
  return (
    <Space.Compact style={{ width: "100%" }}>
      <Input value={value} placeholder={t("选择存放位置")} onChange={(e) => onChange(e.target.value)} />
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
  /** 分割：和 boxes 一一对应的多边形 [x1, y1, ...]（拿不到轮廓时为空） */
  polys?: number[][];
  /** 关键点：和 boxes 一一对应的 [x, y, v, ...] */
  kpts?: number[][];
  /** 分类：前几名 [类别序号, 概率] */
  tags?: [number, number][];
  ms: number;
  fps: number;
}

/** 训练记录对应的任务：按记录里的项目任务（多标签也是分类模型）；老记录没有时按模型名推断 */
const taskOf = (run: RunInfo) => {
  const t0 = run.task === "multilabel" ? "classify" : run.task;
  if (t0) return t0;
  const model = run.config?.model;
  return model?.startsWith("maskrcnn") ? "segment" : model?.startsWith("keypointrcnn") ? "pose" : model?.startsWith("cls_") ? "classify" : "detect";
};

/** 置信度输入框旁边的「用推荐值」（推荐值是评估页 F1 最高的阈值） */
function RecConf({ value, rec, onUse }: { value: number; rec: number | null; onUse: (v: number) => void }) {
  if (rec == null) return null;
  if (Math.abs(value - rec) < 0.001) return <span className="dim" style={{ fontSize: 12 }}>{t("（推荐值）")}</span>;
  return (
    <Button size="small" type="link" style={{ padding: 0 }} onClick={() => onUse(rec)}>
      {t("用推荐值 {v}", { v: rec.toFixed(2) })}
    </Button>
  );
}

/** 多标签分类：每个类别用评估时各自算出的推荐阈值（F1 最高）。返回 [开关, 改开关, 正在用的阈值表] */
function useClassConf(rec: Record<string, number> | null) {
  const [on, setOn] = useState(prefs.get("useClassConf", "1") === "1");
  const set = (v: boolean) => {
    setOn(v);
    prefs.set("useClassConf", v ? "1" : "0");
  };
  return [on, set, on && rec ? rec : null] as const;
}

function ClassConfToggle({ rec, on, onChange }: { rec: Record<string, number> | null; on: boolean; onChange: (v: boolean) => void }) {
  if (!rec) return null;
  const list = Object.entries(rec)
    .map(([n, c]) => `${n} ${c.toFixed(2)}`)
    .join(t("、"));
  return (
    <div>
      <Checkbox checked={on} onChange={(e) => onChange(e.target.checked)}>
        {t("每个类别用自己的推荐阈值")}
      </Checkbox>
      <div className="dim" style={{ fontSize: 12 }}>
        {on ? t("评估时按类别算出：{list}。其余类别用置信度阈值。", { list }) : t("所有类别用同一个置信度阈值。")}
      </div>
    </div>
  );
}

function LiveTab({ run, labels, recConf, classRec }: { run: RunInfo; labels: Label[]; recConf: number | null; classRec: Record<string, number> | null }) {
  const { message } = App.useApp();
  const refreshStats = useApp((s) => s.refreshStats);
  // 多标签分类：分数够阈值的类别都算（单标签只看第一名）
  const multi = useApp((s) => s.project?.task === "multilabel");
  const multiRef = useRef(multi);
  multiRef.current = multi;
  const [kind, setKind] = useState(prefs.get("liveKind", "rtsp"));
  const [url, setUrl] = useState(prefs.get("liveUrl", ""));
  const [file, setFile] = useState(prefs.get("liveFile", ""));
  const [conf, setConf] = useState(Number(prefs.get("liveConf", "0.4")));
  const [running, setRunning] = useState(false);
  const [starting, setStarting] = useState(false);
  const [classes, setClasses] = useState<string[]>([]);
  const [stat, setStat] = useState<{ w: number; h: number; fps: number; ms: number; counts: Record<string, number> } | null>(null);
  const [status, setStatus] = useState("");
  /** 启动阶段：prepare 准备环境 → model 加载模型 → open 连接视频源，收到第一帧前显示 */
  const [stage, setStage] = useState("");
  const [error, setError] = useState<string | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const busy = useRef(false);
  const classesRef = useRef<string[]>([]);
  classesRef.current = classes;
  const confRef = useRef(conf);
  confRef.current = conf;
  const [useCls, setUseCls, cc] = useClassConf(multi ? classRec : null);
  const ccRef = useRef(cc);
  ccRef.current = cc;
  /** 这个类别算不算有：多标签分类按类别阈值 */
  const thrOf = (name: string) => ccRef.current?.[name] ?? confRef.current;
  // 推理进程按最低的阈值过滤，各类别的阈值在这边判断
  const sendConf = (v: number) => (cc ? Math.min(v, ...Object.values(cc)) : v);
  // 默认用这个模型的推荐阈值，用户自己拖过就不再改
  const confTouched = useRef(false);
  useEffect(() => {
    if (recConf != null && !confTouched.current) setConf(recConf);
  }, [recConf]);
  const applyConf = (v: number) => {
    setConf(v);
    prefs.set("liveConf", String(v));
    if (running) api.liveConf(sendConf(v)).catch(() => {});
  };
  useEffect(() => {
    if (running) api.liveConf(sendConf(confRef.current)).catch(() => {});
    // 只在切换「每个类别用自己的推荐阈值」时同步
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cc]);

  const colorOf = useCallback(
    (name: string) => labels.find((l) => l.name === name)?.color ?? "#ffd43b",
    [labels],
  );
  const colorRef = useRef(colorOf);
  colorRef.current = colorOf;
  const skeleton = useRef<[number, number][]>([]);
  useEffect(() => {
    if (taskOf(run) === "pose") api.keypointTemplate().then((tp) => (skeleton.current = tp.skeleton)).catch(() => {});
  }, [run]);

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
      f.boxes.forEach(([x, y, w, h, c, sc], i) => {
        const name = classesRef.current[c] ?? String(c);
        const color = colorRef.current(name);
        ctx.strokeStyle = color;
        ctx.fillStyle = rgba(color, 0.12);
        const poly = f.polys?.[i];
        if (poly && poly.length >= 6) {
          // 分割：画轮廓，框只描细线
          ctx.beginPath();
          for (let j = 0; j < poly.length; j += 2) ctx.lineTo(poly[j] * k, poly[j + 1] * k);
          ctx.closePath();
          ctx.fillStyle = rgba(color, 0.3);
          ctx.fill();
          ctx.stroke();
          ctx.lineWidth = 1 * dpr;
          ctx.strokeRect(x * k, y * k, w * k, h * k);
          ctx.lineWidth = 2 * dpr;
        } else {
          ctx.fillRect(x * k, y * k, w * k, h * k);
          ctx.strokeRect(x * k, y * k, w * k, h * k);
        }
        const kp = f.kpts?.[i];
        if (kp) {
          const at = (j: number) => (kp[j * 3 + 2] > 0 ? [kp[j * 3] * k, kp[j * 3 + 1] * k] : null);
          for (const [a, b] of skeleton.current) {
            const pa = at(a);
            const pb = at(b);
            if (!pa || !pb) continue;
            ctx.beginPath();
            ctx.moveTo(pa[0], pa[1]);
            ctx.lineTo(pb[0], pb[1]);
            ctx.stroke();
          }
          ctx.fillStyle = "#fff";
          for (let j = 0; j * 3 < kp.length; j++) {
            const pt = at(j);
            if (!pt) continue;
            ctx.beginPath();
            ctx.arc(pt[0], pt[1], 3 * dpr, 0, Math.PI * 2);
            ctx.fill();
          }
        }
        const text = `${name} ${sc.toFixed(2)}`;
        const tw = ctx.measureText(text).width + 8 * dpr;
        const ty = y * k > 18 * dpr ? y * k - 18 * dpr : y * k;
        ctx.fillStyle = color;
        ctx.fillRect(x * k, ty, tw, 18 * dpr);
        ctx.fillStyle = "#fff";
        ctx.fillText(text, x * k + 4 * dpr, ty + 13 * dpr);
      });
      // 分类：整张画面的前几名写在左上角
      ctx.font = `${15 * dpr}px "Microsoft YaHei UI", sans-serif`;
      (f.tags ?? []).forEach(([c, p], i) => {
        const name = classesRef.current[c] ?? String(c);
        const text = `${name} ${p.toFixed(2)}`;
        const tw = ctx.measureText(text).width + 12 * dpr;
        const ty = (8 + i * 28) * dpr;
        const on = multiRef.current ? p >= thrOf(name) : i === 0;
        ctx.fillStyle = on ? colorRef.current(name) : "rgba(0,0,0,0.55)";
        ctx.fillRect(8 * dpr, ty, tw, 24 * dpr);
        ctx.fillStyle = "#fff";
        ctx.fillText(text, 14 * dpr, ty + 17 * dpr);
      });
    };
    img.onerror = () => (busy.current = false);
    img.src = `data:image/jpeg;base64,${f.jpg}`;
  };

  useEffect(() => {
    const un = listen<{ kind: string; data: Record<string, unknown> }>("live", (e) => {
      const { kind: k, data } = e.payload;
      if (k === "stage") {
        setStage(String(data.step ?? ""));
        setStatus("");
      } else if (k === "open") {
        setStage("");
        // 第一帧可能紧跟着到，先同步记下类别名
        classesRef.current = data.classes as string[];
        setClasses(data.classes as string[]);
        setStatus(t("已连接 · {w}×{h} · {dev}", { w: data.width as number, h: data.height as number, dev: data.device === "cuda" ? t("显卡推理") : t("CPU 推理") }));
      } else if (k === "frame") {
        const f = data as unknown as FrameEv;
        drawFrame(f);
        const counts: Record<string, number> = {};
        for (const b of f.boxes) {
          const n = classesRef.current[b[4]] ?? String(b[4]);
          counts[n] = (counts[n] ?? 0) + 1;
        }
        // 分类：只显示概率最高的类别；多标签分类显示分数够的全部类别
        for (const [c, sc] of (multiRef.current ? f.tags : f.tags?.slice(0, 1)) ?? []) {
          const name = classesRef.current[c] ?? String(c);
          if (sc >= (multiRef.current ? thrOf(name) : confRef.current)) counts[name] = 1;
        }
        setStat({ w: f.w, h: f.h, fps: f.fps, ms: f.ms, counts });
      } else if (k === "log") {
        setStatus(String(data));
      } else if (k === "snapshot") {
        if (data.error) message.error(t("截图失败：{e}", { e: String(data.error) }));
        else if (data.added) {
          message.success(t("已加入数据集（未标注）"));
          refreshStats();
        } else message.info(t("这张和已有图片重复，没有重复加入"));
      } else if (k === "ended") {
        setStatus(t("视频播放完毕"));
      } else if (k === "error") {
        setStage("");
        setError(String(data.msg ?? t("出错了")));
      } else if (k === "stopped") {
        setStage("");
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
    // 第一次用要装 OpenCV，装的过程（log）显示在阶段下面
    setStatus("");
    setStage("prepare");
    setStarting(true);
    prefs.set("liveKind", kind);
    prefs.set("liveUrl", url);
    prefs.set("liveFile", file);
    try {
      await api.liveStart(run.id, source, sendConf(conf));
      setRunning(true);
    } catch (e) {
      setError(errText(e));
      setStatus("");
      setStage("");
    } finally {
      setStarting(false);
    }
  };
  const stageNames: Record<string, string> = { prepare: t("准备运行环境…"), model: t("加载模型…"), open: t("连接视频源…") };
  const stageOrder = ["prepare", "model", "open"];

  return (
    <div>
      <div className="row" style={{ marginBottom: 12 }}>
        <Segmented
          value={kind}
          onChange={(v) => setKind(v as string)}
          options={[
            { value: "rtsp", label: t("视频流") },
            { value: "file", label: t("本地视频") },
          ]}
        />
        {kind === "rtsp" ? (
          <Input
            style={{ flex: 1, minWidth: 320 }}
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder={t("rtsp://…  或  https://…")}
          />
        ) : (
          <Space.Compact style={{ flex: 1, minWidth: 320 }}>
            <Input value={file} onChange={(e) => setFile(e.target.value)} placeholder={t("选择 mp4 / avi / mkv 等视频文件")} />
            <Button
              icon={<FolderOpenOutlined />}
              onClick={async () => {
                const f = await open({ title: t("选择视频文件"), filters: [{ name: t("视频"), extensions: VIDEO_EXTS }] });
                if (typeof f === "string") setFile(f);
              }}
            />
          </Space.Compact>
        )}
        {running ? (
          <Button danger icon={<StopOutlined />} onClick={() => api.liveStop()}>
            {t("停止")}
          </Button>
        ) : (
          <Button type="primary" icon={<CaretRightOutlined />} loading={starting} disabled={!source.trim()} onClick={start}>
            {t("开始")}
          </Button>
        )}
      </div>
      <div className="row" style={{ marginBottom: 12 }}>
        <span className="muted">{t("置信度")}</span>
        <Slider
          style={{ width: 220 }}
          min={0.05}
          max={0.95}
          step={0.05}
          value={conf}
          onChange={(v) => {
            confTouched.current = true;
            applyConf(v);
          }}
        />
        <span style={{ width: 36 }}>{conf.toFixed(2)}</span>
        <RecConf value={conf} rec={recConf} onUse={applyConf} />
        <Button icon={<CameraOutlined />} disabled={!running} onClick={() => api.liveSnapshot().catch((e) => message.error(errText(e)))}>
          {t("截图加入数据集")}
        </Button>
        <span className="dim" style={{ fontSize: 12 }}>
          {t("看到漏检、误报的画面就截下来，标注后再训练")}
        </span>
        {multi && classRec ? (
          <div style={{ flexBasis: "100%" }}>
            <ClassConfToggle rec={multi ? classRec : null} on={useCls} onChange={setUseCls} />
          </div>
        ) : null}
      </div>
      {error && <Alert type="error" showIcon style={{ marginBottom: 12 }} message={error} closable onClose={() => setError(null)} />}
      <div className="live-box">
        <canvas ref={canvas} style={{ display: stat ? "block" : "none" }} />
        {!stat && stage && !error ? (
          <div className="dim" style={{ width: 280, textAlign: "center" }}>
            <Progress percent={((stageOrder.indexOf(stage) + 1) / (stageOrder.length + 1)) * 100} status="active" showInfo={false} size="small" />
            <div>{stageNames[stage] ?? stage}</div>
            {status && <div style={{ fontSize: 12, wordBreak: "break-all" }}>{status}</div>}
          </div>
        ) : (
          !stat && <div className="dim">{status || t("填写视频流地址或选择视频文件，点「开始」查看模型的实时效果")}</div>
        )}
        {stat && (
          <div className="live-stat">
            <span>{status}</span>
            <span>{t("{fps} FPS · 推理 {ms} ms", { fps: stat.fps.toFixed(1), ms: stat.ms.toFixed(0) })}</span>
            {Object.entries(stat.counts).map(([n, c]) => (
              <Tag key={n} color={colorOf(n)} style={{ marginInlineEnd: 0 }}>
                {n} {c}
              </Tag>
            ))}
          </div>
        )}
      </div>
      <div className="dim" style={{ fontSize: 12, marginTop: 8 }}>
        {t("支持 RTSP 摄像头 / 直播流,以及 http(s) 的回放、点播地址(mp4、m3u8、flv)。")}
        <br />
        {t("海康：rtsp://用户名:密码@IP:554/Streaming/Channels/101（主码流）或 102（子码流）；大华：rtsp://用户名:密码@IP:554/cam/realmonitor?channel=1&subtype=0")}
      </div>
    </div>
  );
}

// ---------------- 导出模型 ----------------

const FORMATS = [
  { value: "onnx", name: t("ONNX（动态尺寸）"), desc: t("通用格式，onnxruntime 在 Windows/Linux/ARM 上都能直接跑，任意输入尺寸") },
  { value: "onnx_static", name: t("ONNX（固定尺寸）"), desc: t("输入固定为 [1,3,N,N]，给需要固定形状的推理框架用") },
  { value: "openvino", name: "OpenVINO", desc: t("Intel CPU、核显、工控机加速，在软件里直接转换并验证（首次使用会自动安装 OpenVINO）") },
  { value: "tensorrt", name: t("TensorRT 转换包"), desc: t("NVIDIA 显卡、Jetson。引擎和设备绑定，需在目标设备上运行脚本生成") },
  { value: "rknn", name: t("RKNN 转换包"), desc: t("瑞芯微 RK3588/3568 等。官方工具只支持 Linux，需在 Ubuntu 上运行脚本转换") },
];

/** 导出、生成服务包的步骤进度条（deploy-progress）；步数未知时只显示当前在做什么 */
/** 批量识别一个文件夹：导出表格，可以按识别结果把图片复制到各类别的文件夹 */
function BatchTab({ run, project, recConf, classRec }: { run: RunInfo; project: ProjectInfo; recConf: number | null; classRec: Record<string, number> | null }) {
  const { message } = App.useApp();
  const refreshStats = useApp((s) => s.refreshStats);
  const annotateQueue = useApp((s) => s.annotateQueue);
  const [adding, setAdding] = useState(false);
  const [added, setAdded] = useState<{ text: string; ids: number[] } | null>(null);
  const [input, setInput] = useState(prefs.get("batchInput", ""));
  const [output, setOutput] = useState(prefs.get("batchOutput", ""));
  const [conf, setConf] = useState(recConf ?? 0.5);
  const [sort, setSort] = useState(prefs.get("batchSort", "1") === "1");
  const [every, setEvery] = useState(Number(prefs.get("batchVideoEvery", "1")) || 1);
  const [minFrames, setMinFrames] = useState(Number(prefs.get("batchVideoMin", "1")) || 1);
  const [busy, setBusy] = useState(false);
  const [prog, setProg] = useState<ImportProgress | null>(null);
  const [res, setRes] = useState<BatchResult | null>(null);
  const isCls = isClsTask(project.task);
  const clsRec = project.task === "multilabel" ? classRec : null;
  const [useCls, setUseCls, cc] = useClassConf(clsRec);
  useEffect(() => {
    if (recConf != null) setConf(recConf);
  }, [recConf]);

  const start = async () => {
    if (!input || !output) {
      message.warning(t("先选要识别的文件夹和结果放在哪里"));
      return;
    }
    prefs.set("batchInput", input);
    prefs.set("batchOutput", output);
    prefs.set("batchSort", sort ? "1" : "0");
    prefs.set("batchVideoEvery", String(every));
    prefs.set("batchVideoMin", String(minFrames));
    setBusy(true);
    setRes(null);
    setAdded(null);
    setProg(null);
    const un = await listen<ImportProgress>("batch-progress", (e) => setProg(e.payload));
    try {
      const r = await api.batchPredict({ runId: run.id, input, output, conf, sort, classConf: cc ?? {}, videoEvery: every, videoMinFrames: minFrames });
      setRes(r);
      if (r.stopped)
        message.info(
          r.videos
            ? t("已停止，识别过的 {n} 张图片、{v} 个视频已写进表格", { n: r.images, v: r.videos })
            : t("已停止，识别过的 {n} 张已写进表格", { n: r.images }),
        );
    } catch (e) {
      message.error(errText(e));
    } finally {
      un();
      setBusy(false);
    }
  };

  // 拿不准的图加进项目：标好再训练，模型越用越准
  const addUncertain = async () => {
    if (!res) return;
    setAdding(true);
    setAdded(null);
    try {
      const r = await api.importImages(res.uncertain, true);
      await refreshStats();
      const text = t("导入 {added} 张，跳过重复 {dup} 张", { added: r.added, dup: r.duplicated });
      setAdded({ text, ids: r.added > 0 ? await lastImportIds().catch(() => []) : [] });
    } catch (e) {
      message.error(errText(e));
    } finally {
      setAdding(false);
    }
  };

  return (
    <div className="card" style={{ maxWidth: 760 }}>
      <p className="desc" style={{ marginTop: 0 }}>
        {isCls
          ? t("用这个模型把一个文件夹里的图片都识别一遍（包括子文件夹），结果导出成表格，Excel 直接打开；也可以按识别出的类别把图片复制到各自的文件夹。")
          : t("用这个模型把一个文件夹里的图片都检测一遍（包括子文件夹），每张图找到几个什么目标导出成表格，Excel 直接打开；也可以按找到的类别把图片复制到各自的文件夹。")}{" "}
        {t("视频也能识别：每个视频看一遍，表格里一个视频一行，识别出的每个类别存一张截图，按视频所在的子文件夹（比如单号）放好。")}
      </p>
      <Form layout="vertical" disabled={busy}>
        <Form.Item label={t("要识别的文件夹")}>
          <FolderPicker value={input} onChange={setInput} title={t("选择要识别的文件夹")} />
        </Form.Item>
        <Form.Item
          label={t("视频")}
          extra={t("视频按间隔取帧识别；一个类别在几帧里都检出才算有，可以减少偶尔一帧的误判。第一次识别视频会自动下载 ffmpeg（约 90MB）。")}
        >
          <Space wrap>
            <span>{t("每隔")}</span>
            <InputNumber min={0.2} max={60} step={0.5} value={every} onChange={(v) => setEvery(Number(v ?? 1))} style={{ width: 80 }} />
            <span>{t("秒看一帧，至少")}</span>
            <InputNumber min={1} max={50} step={1} precision={0} value={minFrames} onChange={(v) => setMinFrames(Number(v ?? 1))} style={{ width: 70 }} />
            <span>{t("帧检出才算有")}</span>
          </Space>
        </Form.Item>
        <Form.Item label={t("结果放在")} extra={t("不能放在要识别的文件夹里面。")}>
          <FolderPicker value={output} onChange={setOutput} title={t("选择结果放在哪个文件夹")} />
        </Form.Item>
        <Form.Item
          label={t("置信度阈值")}
          extra={
            project.task === "classify"
              ? t("最像的类别分数低于阈值时记为「拿不准」，留给人看。")
              : project.task === "multilabel"
                ? t("分数达到阈值的类别都算有；一个都没有时记为「拿不准」。")
                : t("分数低于阈值的目标不算。")
          }
        >
          <Space>
            <InputNumber min={0.05} max={0.95} step={0.05} value={conf} onChange={(v) => setConf(v ?? 0.5)} />
            <RecConf value={conf} rec={recConf} onUse={setConf} />
          </Space>
          <ClassConfToggle rec={clsRec} on={useCls} onChange={setUseCls} />
        </Form.Item>
        <Form.Item>
          <Checkbox checked={sort} onChange={(e) => setSort(e.target.checked)}>
            {project.task === "multilabel"
              ? t("按类别复制到各自的文件夹（一张图有几个类别就复制几份）")
              : isCls
                ? t("按类别复制到各自的文件夹")
                : t("按找到的类别复制到各自的文件夹（找到几种就复制几份）")}
          </Checkbox>
        </Form.Item>
      </Form>
      <Space>
        {!busy ? (
          <Button type="primary" icon={<CaretRightOutlined />} onClick={start}>
            {t("开始识别")}
          </Button>
        ) : (
          <Button danger icon={<StopOutlined />} onClick={() => api.batchPredictStop().catch(() => {})}>
            {t("停止")}
          </Button>
        )}
      </Space>
      {busy && (
        <div style={{ marginTop: 12 }}>
          <Progress size="small" percent={prog && prog.total ? Math.floor((prog.done / prog.total) * 100) : 0} status="active" />
          <div className="dim" style={{ fontSize: 12 }}>
            {prog ? t("已识别 {done}/{total} 张", { done: prog.done, total: prog.total }) : t("加载模型…")}
          </div>
        </div>
      )}
      {res && (
        <Alert
          style={{ marginTop: 12 }}
          type={res.failed ? "warning" : "success"}
          showIcon
          message={
            res.videos
              ? t("识别了 {n} 张图片、{v} 个视频（看了 {fr} 帧）", { n: res.images, v: res.videos, fr: res.frames }) +
                (res.failed ? t("，其中 {f} 个读不了或出错（表格里标为「识别出错」）", { f: res.failed }) : "")
              : res.failed
                ? t("识别了 {n} 张，其中 {f} 张读不了或出错（表格里标为「识别出错」）", { n: res.images, f: res.failed })
                : t("识别了 {n} 张", { n: res.images })
          }
          description={
            <div>
              <div style={{ margin: "4px 0 8px" }}>
                {res.counts.map(([name, n]) => (
                  <Tag key={name}>{res.videos ? t("{name} {n} 个", { name, n }) : t("{name} {n} 张", { name, n })}</Tag>
                ))}
              </div>
              {res.videos > 0 && (
                <div className="dim" style={{ fontSize: 12, marginBottom: 8 }}>
                  {t("视频的截图在结果文件夹里，按视频所在的子文件夹放，每个识别出的类别一张（分数最高的那一帧）。")}
                </div>
              )}
              <Space>
                <Button size="small" icon={<FolderOpenOutlined />} onClick={() => api.revealPath(res.output).catch(() => {})}>
                  {t("打开结果文件夹")}
                </Button>
                <Button size="small" onClick={() => api.revealPath(res.csv).catch(() => {})}>
                  {t("找到表格")}
                </Button>
              </Space>
              {res.uncertain.length > 0 && (
                <div style={{ marginTop: 10 }}>
                  <Space wrap>
                    <Button size="small" type="primary" icon={<ImportOutlined />} loading={adding} onClick={addUncertain}>
                      {isCls
                        ? t("把拿不准的 {n} 张加进项目去标", { n: res.uncertain.length })
                        : t("把没找到目标的 {n} 张加进项目去标", { n: res.uncertain.length })}
                    </Button>
                    {added && <span className="dim">{added.text}</span>}
                    {added && added.ids.length > 0 && (
                      <Button size="small" onClick={() => annotateQueue(added.ids, t("批量识别拿不准的图"))}>
                        {t("去标注这批")}
                      </Button>
                    )}
                  </Space>
                  <div className="dim" style={{ fontSize: 12, marginTop: 4 }}>
                    {t("标好后重新训练，模型会越用越准。")}
                  </div>
                </div>
              )}
            </div>
          }
        />
      )}
    </div>
  );
}

function StepBar({ step }: { step: DeployStep | null }) {
  if (!step) return null;
  const name = FORMATS.find((f) => f.value === step.name)?.name ?? step.name;
  return (
    <div style={{ marginTop: 12 }}>
      <Progress size="small" percent={step.total ? Math.floor((step.done / step.total) * 100) : 0} status="active" />
      <div className="dim" style={{ fontSize: 12 }}>
        {step.total > 0 && step.done < step.total ? t("第 {n} / {total} 步：{name}", { n: step.done + 1, total: step.total, name }) : name}
      </div>
    </div>
  );
}

/** 监听 deploy-progress，返回取消监听的函数 */
const listenStep = (set: (s: DeployStep) => void) => listen<DeployStep>("deploy-progress", (e) => set(e.payload));

function ExportTab({ run, project }: { run: RunInfo; project: ProjectInfo }) {
  const { message } = App.useApp();
  const builtin = MODELS.find((m) => m.value === run.config?.model);
  // 两阶段模型（Faster / Mask / Keypoint R-CNN）转 NPU、老 TensorRT 容易出问题；导入的外部模型不算
  const twoStage = run.engine !== "external" && !!builtin && !builtin.oneStage && builtin.task !== "classify";
  // 模型自己的输入尺寸（SSDLite 固定 320）
  const ownSize = builtin?.fixedSize ?? run.config?.imgSize ?? 640;
  const sizeOptions = [...new Set([...(builtin?.task === "classify" ? [160, 224, 256, 320, 384] : [320, 416, 480, 640, 800, 960, 1280]), ownSize])].sort((a, b) => a - b);
  // 默认只导出 ONNX；OpenVINO 要另外下载安装组件，需要时自己勾
  const [formats, setFormats] = useState<string[]>(["onnx"]);
  const [size, setSize] = useState(ownSize);
  const [parent, setParent] = useState(prefs.get("deployParent", ""));
  const [name, setName] = useState(t("{name}_模型_{time}", { name: project.name, time: stamp() }));
  const [busy, setBusy] = useState(false);
  const [lines, setLines] = useState<string[]>([]);
  const [results, setResults] = useState<FormatResult[]>([]);
  const [step, setStep] = useState<DeployStep | null>(null);

  const go = async () => {
    if (!parent) {
      message.warning(t("请选择存放位置"));
      return;
    }
    setBusy(true);
    setLines([]);
    setResults([]);
    setStep({ done: 0, total: 0, name: "" });
    prefs.set("deployParent", parent);
    const un = await listen<string>("deploy-log", (e) => setLines((l) => [...l.slice(-200), e.payload]));
    const unStep = await listenStep(setStep);
    try {
      setResults(await api.exportFormats(run.id, formats, size, join(parent, name)));
    } catch (e) {
      message.error(errText(e));
    } finally {
      // 不管成没成功都换一个新文件夹名，重试时不会报「输出目录不为空」
      setName(t("{name}_模型_{time}", { name: project.name, time: stamp() }));
      un();
      unStep();
      setStep(null);
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
      {twoStage && builtin ? (
        <Alert
          type="warning"
          showIcon
          style={{ margin: "12px 0" }}
          message={t("{name} 是两阶段模型，NPU（RKNN）和老版本 TensorRT 对它支持有限，转换可能失败或速度不理想；ONNX 和 OpenVINO 可以放心使用。", { name: builtin.name })}
        />
      ) : (
        <div style={{ height: 12 }} />
      )}
      <Form layout="vertical">
        <Form.Item label={t("固定尺寸（用于固定尺寸 ONNX、TensorRT、RKNN）")}>
          <Select
            style={{ width: 200 }}
            value={size}
            onChange={setSize}
            options={sizeOptions.map((v) => ({ value: v, label: v === ownSize ? t("{v}×{v}（训练时的尺寸）", { v }) : `${v}×${v}` }))}
          />
        </Form.Item>
        <Form.Item label={t("导出到")} extra={parent && join(parent, name)}>
          <FolderPicker value={parent} onChange={setParent} title={t("选择导出位置")} />
          <Input style={{ marginTop: 8 }} value={name} onChange={(e) => setName(e.target.value)} addonBefore={t("文件夹名")} />
        </Form.Item>
      </Form>
      <Button type="primary" loading={busy} disabled={formats.length === 0} onClick={go}>
        {t("导出")}
      </Button>
      <StepBar step={step} />
      <LogBox lines={lines} />
      {results.length > 0 && (
        <div className="card" style={{ marginTop: 12 }}>
          {results.map((r) => (
            <div key={r.format} className="row" style={{ justifyContent: "space-between", padding: "6px 0", borderBottom: "1px solid var(--line)" }}>
              <span>
                <Tag color={r.ok ? "success" : "error"}>{r.ok ? t("成功") : t("失败")}</Tag>
                <b>{FORMATS.find((f) => f.value === r.format)?.name ?? r.format}</b> <span className="muted">{r.note}</span>
              </span>
              {r.path && (
                <Button size="small" icon={<FolderOpenOutlined />} onClick={() => api.revealPath(r.path)}>
                  {t("打开")}
                </Button>
              )}
            </div>
          ))}
        </div>
      )}
      {xuntongEnabled() && <XuntongUpload run={run} project={project} />}
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
  streams: [{ name: t("摄像头{n}", { n: 1 }), url: "" }],
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

function ServiceTab({ run, project, recConf, classRec }: { run: RunInfo; project: ProjectInfo; recConf: number | null; classRec: Record<string, number> | null }) {
  const clsRec = project.task === "multilabel" ? classRec : null;
  const [useCls, setUseCls, cc] = useClassConf(clsRec);
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
  const [name, setName] = useState(t("{name}_推理服务_{time}", { name: project.name, time: stamp() }));
  const [busy, setBusy] = useState(false);
  const [lines, setLines] = useState<string[]>([]);
  const [step, setStep] = useState<DeployStep | null>(null);

  const set = (patch: Partial<ServiceForm>) => {
    const next = { ...f, ...patch };
    setF(next);
    prefs.set("serviceForm", JSON.stringify(next));
  };
  // 告警阈值默认用这个模型的推荐阈值，用户自己改过就不再动
  const scoreTouched = useRef(false);
  useEffect(() => {
    if (recConf != null && !scoreTouched.current) setF((o) => ({ ...o, minScore: recConf }));
  }, [recConf]);
  const setStream = (i: number, patch: Partial<StreamRow>) => set({ streams: f.streams.map((s, j) => (j === i ? { ...s, ...patch } : s)) });

  const build = async () => {
    if (!parent) {
      message.warning(t("请选择存放位置"));
      return;
    }
    if (f.hours && !/^\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}$/.test(f.hours.trim())) {
      message.warning(t("生效时段格式：07:00-19:00"));
      return;
    }
    const folders = f.streams.filter((s) => looksLikeFolder(s.url)).map((s) => s.name.trim() || s.url.trim());
    if (folders.length) {
      const go = await modal.confirm({
        title: t("视频流地址像是文件夹"),
        content: t("{names} 的地址没有文件扩展名。本地视频要填到视频文件本身，比如 1.mp4；填文件夹的话服务会一直连接失败。", { names: folders.join("、") }),
        okText: t("仍然生成"),
        cancelText: t("回去改"),
      });
      if (!go) return;
    }
    const config = {
      gpu: f.gpu,
      max_side: 1280,
      host: "0.0.0.0",
      port: f.port,
      interval: f.interval,
      streams: f.streams.filter((s) => s.url.trim()).map((s) => ({ name: s.name.trim() || s.url, url: s.url.trim() })),
      alert: { labels: f.labels, min_score: f.minScore, class_scores: cc ?? {}, cooldown: f.cooldown, hours: f.hours.trim(), save_image: f.saveImage },
      webhook: { url: f.webhookUrl.trim(), with_image: f.webhookImage, headers: {} },
      mqtt: { host: f.mqttHost.trim(), port: f.mqttPort, topic: f.mqttTopic.trim(), username: f.mqttUser, password: f.mqttPass },
      reflux: { enabled: f.reflux, low: f.refluxLow, high: f.refluxHigh, interval: 60 },
    };
    setBusy(true);
    setLines([]);
    setStep({ done: 0, total: 0, name: "" });
    prefs.set("deployParent", parent);
    const un = await listen<string>("deploy-log", (e) => setLines((l) => [...l.slice(-200), e.payload]));
    const unStep = await listenStep(setStep);
    try {
      const out = await api.buildService(run.id, config, join(parent, name));
      setName(t("{name}_推理服务_{time}", { name: project.name, time: stamp() }));
      modal.success({
        title: t("推理服务包已生成"),
        width: 560,
        content: (
          <div>
            <div style={{ wordBreak: "break-all" }}>{out}</div>
            <div className="muted" style={{ marginTop: 8 }}>
              {t("整个文件夹拷到现场电脑：Windows 双击 run.bat，Linux 运行 sh run.sh。配置在 config.json，说明见 README.md。")}
            </div>
          </div>
        ),
        okText: t("打开文件夹"),
        onOk: () => api.revealPath(out),
      });
    } catch (e) {
      message.error(errText(e));
    } finally {
      un();
      unStep();
      setStep(null);
      setBusy(false);
    }
  };

  return (
    <Form layout="vertical" style={{ maxWidth: 900 }}>
      <div className="cards" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <div className="card">
          <h3 className="h3">{t("视频流")}</h3>
          {f.streams.map((s, i) => (
            <div key={i} style={{ marginBottom: 8 }}>
              <div className="row" style={{ flexWrap: "nowrap" }}>
                <Input style={{ width: 110 }} value={s.name} placeholder={t("名称")} onChange={(e) => setStream(i, { name: e.target.value })} />
                <Space.Compact style={{ flex: 1 }}>
                  <Input
                    value={s.url}
                    status={looksLikeFolder(s.url) ? "warning" : undefined}
                    placeholder={t("rtsp:// / http(s):// 地址或视频文件路径")}
                    onChange={(e) => setStream(i, { url: e.target.value })}
                  />
                  <Button
                    icon={<FolderOpenOutlined />}
                    title={t("选择视频文件")}
                    onClick={async () => {
                      const v = await open({ title: t("选择视频文件"), filters: [{ name: t("视频"), extensions: VIDEO_EXTS }] });
                      if (typeof v === "string") setStream(i, { url: v });
                    }}
                  />
                </Space.Compact>
                <Button icon={<DeleteOutlined />} disabled={f.streams.length === 1} onClick={() => set({ streams: f.streams.filter((_, j) => j !== i) })} />
              </div>
              {looksLikeFolder(s.url) && (
                <Typography.Text type="warning" style={{ display: "block", fontSize: 12, marginTop: 4 }}>
                  {t("这像是一个文件夹。本地视频要填到视频文件本身（带 .mp4 等扩展名），填文件夹服务会一直连接失败。")}
                </Typography.Text>
              )}
            </div>
          ))}
          <Button size="small" icon={<PlusOutlined />} onClick={() => set({ streams: [...f.streams, { name: t("摄像头{n}", { n: f.streams.length + 1 }), url: "" }] })}>
            {t("添加一路")}
          </Button>
          <Form.Item label={t("检测间隔（秒）")} style={{ marginTop: 12, marginBottom: 0 }} extra={t("每路视频每隔多久检测一帧")}>
            <InputNumber min={0.1} max={60} step={0.5} value={f.interval} onChange={(v) => set({ interval: Number(v ?? 1) })} />
          </Form.Item>
          <div className="dim" style={{ fontSize: 12, marginTop: 8 }}>{t("需要只检测画面某个区域时，生成后在 config.json 里给视频流加 region，见 README。")}</div>
        </div>

        <div className="card">
          <h3 className="h3">{t("告警规则")}</h3>
          <Form.Item label={t("告警类别")} extra={t("不选表示所有类别都告警")} style={{ marginBottom: 10 }}>
            <Select mode="multiple" value={f.labels} onChange={(v) => set({ labels: v })} options={classes.map((c) => ({ value: c, label: c }))} placeholder={t("全部类别")} />
          </Form.Item>
          <Space wrap>
            <Form.Item label={t("置信度阈值")} style={{ marginBottom: 10 }}>
              <InputNumber
                min={0.05}
                max={0.99}
                step={0.05}
                value={f.minScore}
                onChange={(v) => {
                  scoreTouched.current = true;
                  set({ minScore: Number(v ?? 0.5) });
                }}
              />
              <div>
                <RecConf value={f.minScore} rec={recConf} onUse={(v) => set({ minScore: v })} />
              </div>
              <div style={{ maxWidth: 420 }}>
                <ClassConfToggle rec={clsRec} on={useCls} onChange={setUseCls} />
              </div>
            </Form.Item>
            <Form.Item label={t("去重间隔（秒）")} tooltip={t("同一路视频同一类别，这段时间内只告警一次")} style={{ marginBottom: 10 }}>
              <InputNumber min={0} max={86400} value={f.cooldown} onChange={(v) => set({ cooldown: Number(v ?? 60) })} />
            </Form.Item>
            <Form.Item label={t("生效时段")} style={{ marginBottom: 10 }}>
              <Input style={{ width: 130 }} value={f.hours} placeholder={t("全天")} onChange={(e) => set({ hours: e.target.value })} />
            </Form.Item>
          </Space>
          <Checkbox checked={f.saveImage} onChange={(e) => set({ saveImage: e.target.checked })}>
            {t("保存告警截图（带框）")}
          </Checkbox>
          {taskOf(run) === "classify" && (
            <div className="dim" style={{ fontSize: 12, marginTop: 8 }}>
              {t("分类模型按整张画面告警：某个类别的概率达到置信度阈值就告警，不按区域过滤。")}
            </div>
          )}
        </div>

        <div className="card">
          <h3 className="h3">{t("告警推送")}</h3>
          <Form.Item label="HTTP Webhook" extra={t("告警时 POST JSON 到这个地址")} style={{ marginBottom: 8 }}>
            <Input value={f.webhookUrl} placeholder={t("http://你的平台/api/alarm（不填则不推送）")} onChange={(e) => set({ webhookUrl: e.target.value })} />
          </Form.Item>
          <Checkbox checked={f.webhookImage} onChange={(e) => set({ webhookImage: e.target.checked })} style={{ marginBottom: 12 }}>
            {t("附带截图（base64）")}
          </Checkbox>
          <Form.Item label="MQTT" style={{ marginBottom: 0 }}>
            <Space.Compact style={{ width: "100%", marginBottom: 6 }}>
              <Input value={f.mqttHost} placeholder={t("服务器地址（不填则不推送）")} onChange={(e) => set({ mqttHost: e.target.value })} />
              <InputNumber style={{ width: 90 }} value={f.mqttPort} onChange={(v) => set({ mqttPort: Number(v ?? 1883) })} />
            </Space.Compact>
            <Input value={f.mqttTopic} addonBefore={t("主题@@MQTT")} style={{ marginBottom: 6 }} onChange={(e) => set({ mqttTopic: e.target.value })} />
            <Space.Compact style={{ width: "100%" }}>
              <Input value={f.mqttUser} placeholder={t("用户名（可选）")} onChange={(e) => set({ mqttUser: e.target.value })} />
              <Input.Password value={f.mqttPass} placeholder={t("密码（可选）")} onChange={(e) => set({ mqttPass: e.target.value })} />
            </Space.Compact>
          </Form.Item>
        </div>

        <div className="card">
          <h3 className="h3">{t("服务")}</h3>
          <Space wrap>
            <Form.Item label={t("HTTP 接口端口")} style={{ marginBottom: 10 }}>
              <InputNumber min={1} max={65535} value={f.port} onChange={(v) => set({ port: Number(v ?? 8090) })} />
            </Form.Item>
            <Form.Item label={t("用显卡推理")} tooltip={t("需要现场电脑装 onnxruntime-gpu，否则自动用 CPU")} style={{ marginBottom: 10 }}>
              <Switch checked={f.gpu} onChange={(v) => set({ gpu: v })} />
            </Form.Item>
          </Space>
          <Form.Item label={t("现场回流")} extra={t("模型拿不准的画面存到 reflux/，拷回来导入标注，模型越用越准")} style={{ marginBottom: 0 }}>
            <Space>
              <Switch checked={f.reflux} onChange={(v) => set({ reflux: v })} />
              <span className="muted">{t("置信度在")}</span>
              <InputNumber size="small" min={0.05} max={0.95} step={0.05} value={f.refluxLow} disabled={!f.reflux} onChange={(v) => set({ refluxLow: Number(v ?? 0.3) })} />
              <span className="muted">～</span>
              <InputNumber size="small" min={0.05} max={0.99} step={0.05} value={f.refluxHigh} disabled={!f.reflux} onChange={(v) => set({ refluxHigh: Number(v ?? 0.5) })} />
              <span className="muted">{t("之间")}</span>
            </Space>
          </Form.Item>
        </div>
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <Form.Item label={t("生成到")} extra={parent && join(parent, name)} style={{ marginBottom: 12 }}>
          <FolderPicker value={parent} onChange={setParent} title={t("选择服务包存放位置")} />
          <Input style={{ marginTop: 8 }} value={name} onChange={(e) => setName(e.target.value)} addonBefore={t("文件夹名")} />
        </Form.Item>
        <Button type="primary" loading={busy} onClick={build}>
          {t("生成推理服务包")}
        </Button>
        <span className="dim" style={{ marginLeft: 12, fontSize: 12 }}>
          {t("只依赖 onnxruntime、OpenCV、numpy，现场电脑不需要 PyTorch 和显卡")}
        </span>
        <StepBar step={step} />
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
  const annotateQueue = useApp((s) => s.annotateQueue);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [progress, setProgress] = useState<ImportProgress | null>(null);
  const [stopping, setStopping] = useState(false);
  /** 这次导入进来的图（去标注这批） */
  const [ids, setIds] = useState<number[]>([]);

  const pick = async () => {
    const d = await open({ directory: true, title: t("选择推理服务包里的 reflux 文件夹") });
    if (typeof d !== "string") return;
    setBusy(true);
    setResult(null);
    setIds([]);
    setStopping(false);
    setProgress({ done: 0, total: 0 });
    const un = await listen<ImportProgress>("import-progress", (e) => setProgress(e.payload));
    try {
      const r = await api.importImages([d], true);
      await refreshStats();
      const base = r.failed.length
        ? t("导入 {added} 张，跳过重复 {dup} 张，{failed} 张读取失败", { added: r.added, dup: r.duplicated, failed: r.failed.length })
        : t("导入 {added} 张，跳过重复 {dup} 张", { added: r.added, dup: r.duplicated });
      setResult(r.stopped ? t("已停止，{text}，还有 {n} 张没导入", { text: base, n: r.stopped }) : base);
      // 新导入的图在最近一批里；全是重复的就没有新的一批
      if (r.added > 0) setIds(await lastImportIds().catch(() => []));
    } catch (e) {
      message.error(errText(e));
    } finally {
      un();
      setProgress(null);
      setBusy(false);
    }
  };
  const stop = () => {
    setStopping(true);
    api.stopImport().catch((e) => message.error(errText(e)));
  };

  return (
    <div style={{ maxWidth: 720 }}>
      <div className="steps-v">
        <div><b>1</b> {fill(t("推理服务在现场运行时，会把模型拿不准（置信度在设定区间内）的画面存到服务包的 {dir} 文件夹。"), "dir", <code>reflux/</code>)}</div>
        <div><b>2</b> {fill(t("定期把 {dir} 文件夹拷回来，在这里导入。"), "dir", <code>reflux</code>)}</div>
        <div><b>3</b> {t("到「标注」页筛选「未标注」，用模型预标注后修正。")}</div>
        <div><b>4</b> {t("重新训练，再生成新的服务包替换现场的 model.onnx。模型会越用越准。")}</div>
      </div>
      <Space style={{ marginTop: 12 }}>
        <Button type="primary" icon={<ImportOutlined />} loading={busy} onClick={pick}>
          {t("导入现场回流图片")}
        </Button>
        {busy && (
          <Button danger loading={stopping} onClick={stop}>
            {t("停止")}
          </Button>
        )}
      </Space>
      {progress && (
        <div style={{ marginTop: 12 }}>
          <Progress percent={progress.total ? Math.floor((progress.done / progress.total) * 100) : 0} status="active" />
          <div className="dim" style={{ fontSize: 12 }}>
            {progress.total
              ? t("已处理 {done} / {total} 张（计算指纹、生成缩略图）", { done: progress.done, total: progress.total })
              : t("正在查找图片…")}
          </div>
        </div>
      )}
      {result && (
        <Alert
          style={{ marginTop: 12 }}
          type="success"
          showIcon
          message={result}
          action={
            ids.length > 0 ? (
              <Button size="small" type="primary" onClick={() => annotateQueue(ids, t("现场回流"))}>
                {t("去标注这批")}
              </Button>
            ) : (
              <Button size="small" onClick={() => setPage("annotate")}>{t("去标注")}</Button>
            )
          }
        />
      )}
    </div>
  );
}

// ---------------- 页面 ----------------

export default function Deploy({ project }: { project: ProjectInfo }) {
  const { message } = App.useApp();
  const setPage = useApp((s) => s.setPage);
  const mn = metricNames(project.task);
  const [runs, setRuns] = useState<RunInfo[] | null>(null);
  const [runId, setRunId] = useState<number | null>(null);
  const [labels, setLabels] = useState<Label[]>([]);
  const [tab, setTab] = useState(prefs.get("deployTab", "live"));
  const [recConf, setRecConf] = useState<number | null>(null);
  /** 每个类别的推荐阈值（多标签分类用） */
  const [classRec, setClassRec] = useState<Record<string, number> | null>(null);

  useEffect(() => {
    // 别的页带过来的那次训练优先；训练中的还没训完，不默认选
    const focus = useApp.getState().focusRunId;
    api.listRuns().then((rs) => {
      const ok = rs.filter((r) => r.hasWeights);
      setRuns(ok);
      const ready = ok.filter((r) => r.status !== "running");
      setRunId(ready.find((r) => r.id === focus)?.id ?? ready[0]?.id ?? null);
      useApp.getState().takeFocusRun();
    }).catch((e) => message.error(errText(e)));
    api.listLabels().then(setLabels).catch(() => {});
  }, [message]);

  // 评估页算出的推荐阈值（F1 最高），实时验证和推理服务包默认用它
  useEffect(() => {
    setRecConf(null);
    setClassRec(null);
    if (runId == null) return;
    let alive = true;
    api
      .runDetail(runId)
      .then((d) => {
        const sp = d.eval?.splits;
        const th = (sp?.val ?? sp?.test)?.thresholds;
        const c = th?.best.conf;
        if (alive && c != null) setRecConf(Math.max(0.05, Math.round(c * 100) / 100));
        const m: Record<string, number> = {};
        for (const pc of th?.perClass ?? []) {
          if (pc.conf != null && pc.name) m[pc.name] = Math.min(0.95, Math.max(0.05, Math.round(pc.conf * 100) / 100));
        }
        if (alive && Object.keys(m).length) setClassRec(m);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [runId]);

  const run = useMemo(() => runs?.find((r) => r.id === runId) ?? null, [runs, runId]);

  // 只有正在训练的记录时也当成还没有模型
  if (runs && !runs.some((r) => r.status !== "running")) {
    return (
      <div className="empty-page">
        <Empty description={t("还没有训练好的模型。先训练一个模型，再到这里验证和部署。")}>
          <Button type="primary" onClick={() => setPage("train")}>{t("去训练")}</Button>
        </Empty>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}>
        <div>
          <h1>{t("部署")}</h1>
          <p className="desc" style={{ margin: 0 }}>{t("先接摄像头看实际效果，再导出模型或生成推理服务包，拷到现场就能用。")}</p>
        </div>
        <Select
          style={{ width: 420 }}
          value={runId}
          onChange={setRunId}
          options={(runs ?? []).map((r) => ({
            value: r.id,
            disabled: r.status === "running",
            label: `${fmtTime(r.createdAt)} · ${modelName(r)} · ${r.status === "running" ? t("训练中") : `${mn.m50} ${r.bestMap50?.toFixed(3) ?? "—"}`}`,
          }))}
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
            { key: "live", label: t("实时验证"), children: <LiveTab key={run.id} run={run} labels={labels} recConf={recConf} classRec={classRec} /> },
            { key: "batch", label: t("批量识别"), children: <BatchTab key={run.id} run={run} project={project} recConf={recConf} classRec={classRec} /> },
            { key: "export", label: t("导出模型"), children: <ExportTab key={run.id} run={run} project={project} /> },
            { key: "service", label: t("推理服务包"), children: <ServiceTab key={run.id} run={run} project={project} recConf={recConf} classRec={classRec} /> },
            { key: "reflux", label: t("现场回流"), children: <RefluxTab /> },
          ]}
        />
      )}
    </div>
  );
}
