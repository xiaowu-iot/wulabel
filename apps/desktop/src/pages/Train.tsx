import { useCallback, useEffect, useRef, useState } from "react";
import { Alert, App, Button, Checkbox, Empty, Form, InputNumber, Popconfirm, Progress, Radio, Select, Space, Table, Tag, Tooltip } from "antd";
import {
  CaretRightOutlined,
  DeleteOutlined,
  ExportOutlined,
  FolderOpenOutlined,
  StopOutlined,
} from "@ant-design/icons";
import { listen } from "@tauri-apps/api/event";
import { api, EnvStatus, errText, ImportProgress, isClsTask, Label, ProjectInfo, RunDetail, RunInfo, RunStatus, SplitSummary, Task, TrainConfig, VersionInfo } from "../api";
import { prefs, useApp, useReview } from "../store";
import { t } from "../i18n";
import LineChart from "../components/LineChart";
import RunCompare from "../components/RunCompare";
import ImportModelModal from "../components/ImportModelModal";
import AugPanel from "../components/AugPanel";
import { DownloadEv, downloadPercent, downloadText } from "../components/DownloadProgress";
import SplitModal from "../components/SplitModal";

/** 内置模型，和 worker/xlabel_worker/models.py 的 MODELS 对应。oneStage：单阶段检测，部署到 TensorRT / NPU 更顺；
 *  yolox：YOLOX 系列（官方数据增强，增强设置里只有翻转生效）；defaultSize：选中时默认的输入尺寸 */
export const MODELS: {
  value: string;
  name: string;
  desc: string;
  task: Task;
  oneStage?: boolean;
  fixedSize?: number;
  yolox?: boolean;
  defaultSize?: number;
}[] = [
  { value: "yolox_nano", name: "YOLOX-Nano", desc: t("YOLO 系列最轻的一档，适合 CPU、手机和很弱的边缘设备；默认 416 输入"), task: "detect", oneStage: true, yolox: true, defaultSize: 416 },
  { value: "yolox_tiny", name: "YOLOX-Tiny", desc: t("很轻，适合 CPU 和 NPU 边缘盒子实时推理；默认 416 输入"), task: "detect", oneStage: true, yolox: true, defaultSize: 416 },
  { value: "yolox_s", name: "YOLOX-S", desc: t("速度和精度最均衡，大多数项目先用它"), task: "detect", oneStage: true, yolox: true, defaultSize: 640 },
  { value: "yolox_m", name: "YOLOX-M", desc: t("精度更高，建议有独立显卡（6GB 以上显存）"), task: "detect", oneStage: true, yolox: true, defaultSize: 640 },
  { value: "yolox_l", name: "YOLOX-L", desc: t("精度最高、最慢，建议 8GB 以上显存"), task: "detect", oneStage: true, yolox: true, defaultSize: 640 },
  { value: "frcnn_mobilenet", name: "Faster R-CNN MobileNetV3", desc: t("速度快、显存占用小，适合入门和没有独立显卡的电脑"), task: "detect" },
  { value: "frcnn_resnet50", name: "Faster R-CNN ResNet50 v2", desc: t("精度更高，训练和推理更慢，建议 6GB 以上显存"), task: "detect" },
  { value: "retinanet_r50", name: "RetinaNet ResNet50 v2", desc: t("单阶段，精度接近 Faster R-CNN；没有 RoIAlign，转 TensorRT 更顺"), task: "detect", oneStage: true },
  { value: "fcos_r50", name: "FCOS ResNet50", desc: t("单阶段、无锚框，结构简单，适合部署"), task: "detect", oneStage: true },
  { value: "ssdlite", name: "SSDLite MobileNetV3", desc: t("单阶段、很轻，适合 CPU 和 NPU 边缘设备实时推理；固定 320 输入，小目标一般"), task: "detect", oneStage: true, fixedSize: 320 },
  { value: "maskrcnn_r50", name: "Mask R-CNN ResNet50", desc: t("实例分割，速度一般，建议有 6GB 以上显存"), task: "segment" },
  { value: "maskrcnn_r50_v2", name: "Mask R-CNN ResNet50 v2", desc: t("实例分割，精度更高、更慢，建议 8GB 以上显存"), task: "segment" },
  { value: "keypointrcnn_r50", name: "Keypoint R-CNN ResNet50", desc: t("关键点检测；人体 17 点能用上 COCO 预训练，其他点数从头学关键点头"), task: "pose" },
  { value: "cls_mobilenet", name: "MobileNetV3 Large", desc: t("速度快、很轻，适合入门和边缘设备"), task: "classify" },
  { value: "cls_efficientnet", name: "EfficientNet-B0", desc: t("精度和速度比较均衡"), task: "classify" },
  { value: "cls_resnet50", name: "ResNet50", desc: t("精度更高，训练更慢，建议有独立显卡"), task: "classify" },
];

/** 各任务的默认训练参数 */
const TASK_DEFAULTS: Record<Task, Partial<TrainConfig>> = {
  detect: { model: "yolox_s", imgSize: 640, batch: 4, epochs: 50, lr: 0.01 },
  segment: { model: "maskrcnn_r50", imgSize: 640, batch: 2, epochs: 30, lr: 0.01 },
  pose: { model: "keypointrcnn_r50", imgSize: 640, batch: 2, epochs: 40, lr: 0.01 },
  classify: { model: "cls_mobilenet", imgSize: 224, batch: 32, epochs: 30, lr: 0.01 },
  multilabel: { model: "cls_mobilenet", imgSize: 224, batch: 32, epochs: 30, lr: 0.01 },
};

/** 模型列表里的任务：多标签分类用的也是分类模型 */
const modelTask = (task: Task): Task => (task === "multilabel" ? "classify" : task);

const isMac = navigator.userAgent.includes("Mac");

/** 按显存推荐模型。检测用 YOLOX：没有独立显卡用 Tiny，4GB 以上 S，8GB 以上 M；Mac（Apple 芯片）用 S */
const recommendModel = (task: Task, gpuMem: number) =>
  ({
    detect: isMac ? "yolox_s" : gpuMem >= 8000 ? "yolox_m" : gpuMem >= 3800 ? "yolox_s" : "yolox_tiny",
    segment: gpuMem >= 8000 ? "maskrcnn_r50_v2" : "maskrcnn_r50",
    pose: "keypointrcnn_r50",
    classify: gpuMem >= 6000 ? "cls_efficientnet" : "cls_mobilenet",
    multilabel: gpuMem >= 6000 ? "cls_efficientnet" : "cls_mobilenet",
  })[task];

/** 训练记录显示的模型名（导入的外部模型显示导入时起的名字） */
export const modelName = (r: RunInfo) =>
  r.engine === "external"
    ? t("{name}（外部）", { name: (r.config as { displayName?: string })?.displayName || t("外部模型") })
    : MODELS.find((m) => m.value === r.config?.model)?.name ?? String(r.config?.model ?? "");

/** 各任务主指标的叫法（分类没有 mAP，worker 把准确率放在 map50、宏 F1 放在 map；多标签分类 map50 是各类别 AP 的平均） */
export const metricNames = (task: Task) =>
  task === "classify"
    ? { m50: t("准确率"), m: t("宏平均 F1") }
    : task === "multilabel"
      ? { m50: "mAP", m: t("宏平均 F1") }
      : { m50: "mAP50", m: "mAP50-95" };

/** 主指标给个结论：检测类看 mAP50（<0.3 差、0.3~0.6 一般、>0.6 可用），分类看准确率（<0.6、0.6~0.85、>0.85） */
export const rateMetric = (task: Task, v: number) => {
  const [lo, hi] = task === "classify" ? [0.6, 0.85] : task === "multilabel" ? [0.5, 0.8] : [0.3, 0.6];
  if (v < lo) return { level: "bad" as const, word: t("差@@评价") };
  return v <= hi ? { level: "fair" as const, word: t("一般@@评价") } : { level: "good" as const, word: t("可用@@评价") };
};

/** 结果很差，多半什么都没学到（不建议拿去预标注） */
export const isPoor = (task: Task, v: number | null | undefined) => v == null || v < (task === "classify" ? 0.5 : task === "multilabel" ? 0.3 : 0.1);

/** 常见的训练错误换成人话。fix：能直接去改的地方（settings 换 PyTorch 版本、cpu 改用 CPU、data 去数据页） */
export const explainError = (err: string): { hint: string; fix?: "settings" | "cpu" | "data" } | null => {
  const e = err.toLowerCase();
  if (/no kernel image|cuda driver version is insufficient|found no nvidia driver|nvidia driver on your system is too old|invalid device function|cudnn_status_not_initialized|not compiled with cuda/.test(e))
    return { hint: t("显卡驱动和 PyTorch 版本对不上（或者显卡太老、太新）。去设置换一个 PyTorch 版本重新安装，或者在高级设置里改用 CPU。"), fix: "settings" };
  if (/out of memory|outofmemoryerror|can't allocate memory|memoryerror/.test(e))
    return { hint: t("显存或内存不够，批大小调到 1 也不行。在高级设置里调小输入尺寸，或者改用 CPU、换小一点的模型。"), fix: "cpu" };
  if (/no space left on device|errno 28|not enough space on the disk/.test(e)) return { hint: t("磁盘满了。清理一下项目所在的盘和训练环境所在的盘，再重新训练。") };
  if (/urlerror|urlopen error|connection (reset|refused|aborted)|timed out|getaddrinfo failed|name or service not known|max retries exceeded|remote end closed/.test(e))
    return { hint: t("网络问题，下载失败。检查网络或代理后重新训练。") };
  if (/cannot identify image file|(no such file or directory|filenotfounderror)[^\n]*\.(jpe?g|png|bmp|webp|tiff?)/.test(e))
    return { hint: t("图片文件找不到或打不开，可能被移动或删除了。到「数据」页检查一下。"), fix: "data" };
  return null;
};

/** 日志里要标出来的行（失败、错误、警告） */
const WARN_WORDS = ["失败", "错误", "没有可用", "error", "warning", "traceback", "failed", t("失败").toLowerCase()];
const isWarnLine = (l: string) => {
  const s = l.toLowerCase();
  return WARN_WORDS.some((w) => w && s.includes(w));
};

export const RUN_STATUS: Record<RunStatus, { text: string; color: string }> = {
  running: { text: t("训练中"), color: "processing" },
  done: { text: t("完成"), color: "success" },
  stopped: { text: t("已停止"), color: "warning" },
  failed: { text: t("失败"), color: "error" },
  interrupted: { text: t("已中断"), color: "default" },
};

export const fmtTime = (ts: number) => new Date(ts * 1000).toLocaleString();
const fmtDur = (s: number | null | undefined) => {
  if (s == null || !Number.isFinite(s)) return "—";
  if (s < 60) return t("{s} 秒", { s: Math.round(s) });
  if (s < 3600) return t("{m} 分 {s} 秒", { m: Math.floor(s / 60), s: Math.round(s % 60) });
  return t("{h} 小时 {m} 分", { h: Math.floor(s / 3600), m: Math.round((s % 3600) / 60) });
};

interface ProgressEv {
  epoch: number;
  epochs: number;
  iter: number;
  iters: number;
  loss: number;
  eta: number | null;
}

export default function Train({ project }: { project: ProjectInfo }) {
  const { message, modal } = App.useApp();
  const setPage = useApp((s) => s.setPage);
  const openRun = useApp((s) => s.openRun);
  const focusRunId = useApp((s) => s.focusRunId);
  const annotateQueue = useApp((s) => s.annotateQueue);
  const takeFocusRun = useApp((s) => s.takeFocusRun);
  const task = project.task;
  // 协作时才有审核；一个人标注时不出现「已审核」「只用已审核的图片」
  const review = useReview();
  const models = MODELS.filter((m) => m.task === modelTask(task));
  const mn = metricNames(task);
  const cfgKey = `trainConfig.${task}`;
  const [env, setEnv] = useState<EnvStatus | null>(null);
  const [split, setSplit] = useState<SplitSummary | null>(null);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [sel, setSel] = useState<number | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [prog, setProg] = useState<ProgressEv | null>(null);
  const [starting, setStarting] = useState(false);
  /** 开始训练前准备数据（导出数据集副本）的进度 */
  const [prep, setPrep] = useState<ImportProgress | null>(null);
  // 高级设置（输入尺寸、批大小、数据增强、设备）默认收起，记住用户上次的选择
  const [advOpen, setAdvOpen] = useState(prefs.get("trainAdv", "0") === "1");
  const toggleAdv = (v: boolean) => {
    setAdvOpen(v);
    prefs.set("trainAdv", v ? "1" : "0");
  };
  const [exporting, setExporting] = useState(false);
  /** 导出 ONNX 时 worker 报的最新一行 */
  const [onnxMsg, setOnnxMsg] = useState("");
  const [versions, setVersions] = useState<VersionInfo[]>([]);
  const [labels, setLabels] = useState<Label[]>([]);
  /** 标注全是预标注、人没改过的图 */
  const [rawIds, setRawIds] = useState<number[]>([]);
  const [compareIds, setCompareIds] = useState<number[]>([]);
  const [comparing, setComparing] = useState(false);
  const [importing, setImporting] = useState(false);
  const [splitOpen, setSplitOpen] = useState(false);
  const [showTrace, setShowTrace] = useState(false);
  /** 正在下载的预训练权重（训练开始前） */
  const [dl, setDl] = useState<DownloadEv | null>(null);
  const selRef = useRef<number | null>(null);
  selRef.current = sel;
  const logBox = useRef<HTMLDivElement>(null);

  const [cfg, setCfg] = useState<TrainConfig>(() => {
    // 每种任务单独记一份参数（分类 224px、检测 640px 等差别很大）；老版本只有一份 trainConfig，检测项目沿用它
    const saved = prefs.get(cfgKey, task === "detect" ? prefs.get("trainConfig", "") : "");
    const base: TrainConfig = {
      model: "frcnn_mobilenet",
      epochs: 30,
      imgSize: 640,
      batch: 4,
      lr: 0.01,
      device: "auto",
      onlyReviewed: false,
      ratios: [0.8, 0.2, 0],
      ...TASK_DEFAULTS[task],
    };
    try {
      return saved ? { ...base, ...JSON.parse(saved) } : base;
    } catch {
      return base;
    }
  });

  const loadRuns = useCallback(async () => {
    const rs = await api.listRuns();
    setRuns(rs);
    return rs;
  }, []);

  const loadDetail = useCallback(async (id: number) => {
    const d = await api.runDetail(id);
    // 用户往上翻着看日志时不拉到底，本来在底部才跟着滚
    const box = logBox.current;
    const atBottom = !box || box.scrollHeight - box.scrollTop - box.clientHeight < 40;
    if (selRef.current === id) setDetail(d);
    if (atBottom) requestAnimationFrame(() => logBox.current?.scrollTo(0, logBox.current.scrollHeight));
  }, []);

  // 从别处带着训练记录过来（比如训练失败的通知里点「看错误」）
  useEffect(() => {
    if (focusRunId == null || useApp.getState().page !== "train") return;
    setSel(focusRunId);
    takeFocusRun();
  }, [focusRunId, takeFocusRun]);

  useEffect(() => {
    (async () => {
      try {
        const [rs, sp] = await Promise.all([loadRuns(), api.splitSummary()]);
        setSplit(sp);
        const running = await api.runningRun();
        setSel((cur) => cur ?? running ?? rs[0]?.id ?? null);
      } catch (e) {
        message.error(errText(e));
      }
      api.envStatus(false).then(setEnv).catch(() => {});
      api.listVersions().then(setVersions).catch(() => {});
      api.listLabels().then(setLabels).catch(() => {});
      api.uneditedPrelabels().then(setRawIds).catch(() => {});
    })();
  }, [loadRuns, message]);

  useEffect(() => {
    setDetail(null);
    setProg(null);
    setShowTrace(false);
    setDl(null);
    if (sel != null) loadDetail(sel).catch((e) => message.error(errText(e)));
  }, [sel, loadDetail, message]);

  // 训练实时事件
  useEffect(() => {
    const un = listen<{ runId: number; event: Record<string, unknown> }>("train-event", (e) => {
      const { runId, event } = e.payload;
      if (runId !== selRef.current) return;
      const ty = event.type;
      if (ty === "progress") setProg(event as unknown as ProgressEv);
      else if (ty === "download") {
        const d = event as unknown as DownloadEv;
        setDl(d.finished || d.failed ? null : d);
      }
      else if (ty === "metric" || ty === "log" || ty === "error" || ty === "warn") loadDetail(runId);
      else if (ty === "batch") {
        message.warning(t("显存不够，批大小已从 {was} 自动调到 {batch}，重新开始训练", { was: event.was as number, batch: event.batch as number }));
        loadRuns();
        loadDetail(runId);
      }
      else if (ty === "finished") {
        setProg(null);
        loadRuns();
        loadDetail(runId);
        // 训练完成、失败的提示在 App 里统一弹（不在训练页也能看到）
      }
    });
    return () => {
      un.then((f) => f());
    };
  }, [loadDetail, loadRuns, message]);

  const update = (patch: Partial<TrainConfig>) => {
    const next = { ...cfg, ...patch };
    setCfg(next);
    prefs.set(cfgKey, JSON.stringify(next));
  };

  // 项目任务和当前选的模型对不上（比如分割项目却选了检测模型）时，自动切到对应任务的模型
  useEffect(() => {
    if (!MODELS.some((m) => m.value === cfg.model && m.task === modelTask(task))) {
      update({ ...TASK_DEFAULTS[task] });
    }
  }, [task]); // eslint-disable-line react-hooks/exhaustive-deps

  const picked = MODELS.find((m) => m.value === cfg.model);
  const sizeOptions = isClsTask(task) ? [128, 160, 224, 256, 320, 384] : [320, 416, 480, 640, 800, 960, 1280];
  /** 换模型：YOLOX 有自己的默认输入尺寸，从 YOLOX 换回其他模型时恢复 640 */
  const pickModel = (value: string) => {
    const next = MODELS.find((m) => m.value === value);
    // YOLOX 要的轮数多一些，少于 50 轮时提到 50
    if (next?.defaultSize) update({ model: value, imgSize: next.defaultSize, ...(next.yolox && cfg.epochs < 50 ? { epochs: 50 } : {}) });
    else if (picked?.defaultSize && !isClsTask(task)) update({ model: value, imgSize: 640 });
    else update({ model: value });
  };

  const doStart = async () => {
    setStarting(true);
    setPrep({ done: 0, total: 0 });
    const un = await listen<ImportProgress>("train-prepare", (e) => setPrep(e.payload));
    try {
      // 记住的版本可能已经被删掉了，那就用当前数据
      const id = await api.startTraining({ ...cfg, onlyReviewed: review && cfg.onlyReviewed, version: useVersion ? cfg.version : null, baseRun: baseRun ? baseRun.id : null });
      await loadRuns();
      setSel(id);
      setSplit(await api.splitSummary());
      message.success(t("已开始训练"));
    } catch (e) {
      modal.error({ title: t("没能开始训练"), content: errText(e) });
    } finally {
      un();
      setPrep(null);
      setStarting(false);
    }
  };

  const run = detail?.run;
  const logLines = detail?.log ?? [];
  const logHead = (detail?.logHead ?? []).filter((l) => l.trim());
  const warnMsgs = new Set((detail?.warnings ?? []).map((w) => w.msg));
  const errHint = detail?.error ? explainError(detail.error) : null;
  const running = runs.some((r) => r.status === "running");
  const trainable = split ? split.train + split.val + split.test + split.unset : 0;
  const metrics = detail?.metrics ?? [];
  const epochs = run?.config?.epochs ?? cfg.epochs;
  const gpuMem = env?.gpu?.vramMb ?? 0;
  const useVersion = cfg.version != null && versions.some((v) => v.id === cfg.version);
  // 可以接着训的：自己训练出来、有模型文件、模型属于这个任务的
  const baseRuns = runs.filter(
    (r) => r.engine !== "external" && r.hasWeights && r.status !== "running" && MODELS.some((m) => m.value === r.config?.model && m.task === modelTask(task)),
  );
  const baseRun = baseRuns.find((r) => r.id === cfg.baseRun) ?? null;
  const recommended = recommendModel(task, gpuMem);

  // 估一下训练集、验证集张数和步数（未划分的图按比例分；只用已审核时会更少，所以都说「约」）
  const ver = useVersion ? versions.find((v) => v.id === cfg.version) : undefined;
  const counts = ver ? { train: ver.train, val: ver.val, test: ver.test, unset: Math.max(0, ver.images - ver.train - ver.val - ver.test) } : split;
  const ratioSum = cfg.ratios[0] + cfg.ratios[1] + cfg.ratios[2] || 1;
  const trainN = counts ? counts.train + Math.round((counts.unset * cfg.ratios[0]) / ratioSum) : 0;
  const valN = counts ? counts.val + Math.round((counts.unset * cfg.ratios[1]) / ratioSum) : 0;
  // 和 worker 一样：图比批大小多时丢掉最后不满的一批
  const stepsPerEpoch = trainN > cfg.batch ? Math.floor(trainN / cfg.batch) : 1;
  const totalSteps = stepsPerEpoch * cfg.epochs;
  const fewData = trainN > 0 && (trainN < 20 || totalSteps < 300);
  const fewVal = trainN > 0 && valN <= 2;
  // 某个类别标得太少：分类按张数，检测类按框数（类别里的数是全部标注，所以只是大概）
  const clsTask = isClsTask(task);
  const fewCls = trainN > 0 ? labels.filter((l) => l.count < (clsTask ? 20 : 30)) : [];
  const fewClsText = fewCls
    .slice(0, 6)
    .map((l) => (l.count === 0 ? t("「{name}」还没标过", { name: l.name }) : clsTask ? t("「{name}」{n} 张", { name: l.name, n: l.count }) : t("「{name}」{n} 个框", { name: l.name, n: l.count })))
    .join("、") + (fewCls.length > 6 ? t(" 等 {n} 个类别", { n: fewCls.length }) : "");

  // 预标注没人改过的占了不少：没细看就确认的话，模型会把自己认错的也学进去
  const rawPct = trainable ? Math.round((rawIds.length / trainable) * 100) : 0;
  const rawMany = rawIds.length >= 10 && rawPct >= 30;
  const confirmRaw = async () => {
    try {
      await api.confirmPrelabels(rawIds);
      setRawIds([]);
    } catch (e) {
      message.error(errText(e));
    }
  };

  const start = () => {
    if (!fewData && !fewVal && !fewCls.length && !rawMany) {
      doStart();
      return;
    }
    // 凑够约 300 步要的轮数
    const needEpochs = Math.min(500, Math.ceil(300 / stepsPerEpoch));
    modal.confirm({
      title: fewData ? t("数据太少，这样训多半学不到东西") : fewCls.length ? t("有的类别标得太少") : rawMany ? t("预标注大多没人改过") : t("验证集太少"),
      width: 520,
      content: (
        <div>
          {fewData && (
            <>
              <p style={{ marginTop: 0 }}>
                {t("训练集约 {n} 张、批大小 {batch}，每轮约 {s} 步，一共只训约 {m} 步。步数太少时模型基本学不到东西，指标可能一直是 0。", { n: trainN, batch: cfg.batch, s: stepsPerEpoch, m: totalSteps })}
              </p>
              <ul style={{ paddingLeft: 18, margin: "0 0 8px" }}>
                <li>{t("多标一些图，每个类别至少几十张")}</li>
                {needEpochs > cfg.epochs && <li>{t("或者把训练轮数加大到 {x}（凑够约 300 步）", { x: needEpochs })}</li>}
                {task === "detect" && cfg.model !== "frcnn_mobilenet" && <li>{t("数据很少时可以先用 Faster R-CNN MobileNetV3，小数据上更容易学出来")}</li>}
              </ul>
            </>
          )}
          {fewCls.length > 0 && (
            <>
              <p style={{ marginTop: 0 }}>{t("这些类别标得太少，模型很难学会，识别时容易认不出：{list}", { list: fewClsText })}</p>
              <ul style={{ paddingLeft: 18, margin: "0 0 8px" }}>
                <li>{clsTask ? t("每个类别最好 50 张以上，至少 20 张") : t("每个类别最好 100 个框以上，至少 30 个")}</li>
                <li>{t("实在凑不够的，可以在「标注」页删掉这个类别，删的时候把标注合并到相近的类别")}</li>
              </ul>
            </>
          )}
          {rawMany && (
            <p style={{ marginTop: 0 }}>
              {t("{n} 张图（约 {pct}%）的标注全是预标注、没人改过。没细看就确认的话，模型会把自己认错的也学进去，越训越偏。建议先去「标注」页把这批翻一遍。", { n: rawIds.length, pct: rawPct })}
            </p>
          )}
          {fewVal && <p style={{ margin: 0 }}>{t("验证集只有 {n} 张，指标波动会很大、不可信。", { n: valN })}</p>}
        </div>
      ),
      okText: t("仍然开始"),
      cancelText: t("取消"),
      onOk: doStart,
    });
  };
  const descText = (review
    ? {
        detect: t("用已标注和已审核的图片训练检测模型。"),
        segment: t("用已标注和已审核的图片训练实例分割模型。"),
        pose: t("用已标注和已审核的图片训练关键点模型。"),
        classify: t("用已标注和已审核的图片训练图像分类模型。"),
        multilabel: t("用已标注和已审核的图片训练多标签分类模型（每个类别各自判断有没有）。"),
      }
    : {
        detect: t("用已标注的图片训练检测模型。"),
        segment: t("用已标注的图片训练实例分割模型。"),
        pose: t("用已标注的图片训练关键点模型。"),
        classify: t("用已标注的图片训练图像分类模型。"),
        multilabel: t("用已标注的图片训练多标签分类模型（每个类别各自判断有没有）。"),
      })[task];
  const fewText =
    isClsTask(task)
      ? t("图片较少，建议每个类别至少标 100 张图再训练。")
      : task === "pose"
        ? t("图片较少，建议每个类别至少标 100 个目标再训练。")
        : t("图片较少，建议每个类别至少标 100 个框再训练。");

  const doExport = async () => {
    if (!run) return;
    setExporting(true);
    setOnnxMsg(t("正在导出 ONNX…"));
    const un = await listen<string>("onnx-log", (e) => setOnnxMsg(e.payload));
    try {
      const p = await api.exportOnnx(run.id);
      await loadRuns();
      await loadDetail(run.id);
      modal.success({
        title: t("已导出 ONNX"),
        content: <div style={{ wordBreak: "break-all" }}>{p}<div className="muted">{t("同目录的 model.json 里有类别和输入输出说明。")}</div></div>,
        okText: t("好的"),
      });
    } catch (e) {
      message.error(errText(e));
    } finally {
      un();
      setExporting(false);
    }
  };

  /** 从断点继续训练：同一条记录，从最后训完的一轮接着跑 */
  const [resuming, setResuming] = useState(false);
  const resume = async () => {
    if (!run) return;
    setResuming(true);
    try {
      await api.resumeTraining(run.id);
      await loadRuns();
      await loadDetail(run.id);
      message.success(t("已从断点继续训练"));
    } catch (e) {
      modal.error({ title: t("没能继续训练"), content: errText(e) });
    } finally {
      setResuming(false);
    }
  };

  /** 删除勾选的几条训练记录（训练中的不能选） */
  const removeSelected = () => {
    const ids = compareIds.filter((id) => runs.some((r) => r.id === id && r.status !== "running"));
    if (!ids.length) return;
    modal.confirm({
      title: t("删除选中的 {n} 条训练记录？", { n: ids.length }),
      content: t("这些训练的模型文件、日志和数据集副本都会删除，无法恢复。"),
      okText: t("删除"),
      okButtonProps: { danger: true },
      cancelText: t("取消"),
      onOk: async () => {
        let failed = 0;
        for (const id of ids) {
          await api.deleteRun(id).catch(() => failed++);
        }
        setCompareIds([]);
        const rs = await loadRuns();
        if (sel != null && ids.includes(sel)) setSel(rs[0]?.id ?? null);
        if (failed) message.warning(t("有 {n} 条没删掉，可能正在使用", { n: failed }));
        else message.success(t("已删除 {n} 条训练记录", { n: ids.length }));
      },
    });
  };

  const remove = () => {
    if (!run) return;
    modal.confirm({
      title: t("删除训练记录 {name}？", { name: run.name }),
      content: t("模型文件、日志和这次训练用的数据集副本都会删除，无法恢复。"),
      okText: t("删除"),
      okButtonProps: { danger: true },
      cancelText: t("取消"),
      onOk: async () => {
        try {
          await api.deleteRun(run.id);
          const rs = await loadRuns();
          setSel(rs[0]?.id ?? null);
        } catch (e) {
          message.error(errText(e));
        }
      },
    });
  };

  return (
    <div className="page train-page">
      <div className="train-grid">
        <div>
          <h1>{t("训练")}</h1>
          <p className="desc">{descText}</p>
          {env && !env.ready && (
            <Alert
              type="warning"
              showIcon
              style={{ marginBottom: 12 }}
              message={t("还没有训练环境")}
              description={t("第一次训练前需要安装 PyTorch，约 3GB，只需要一次。")}
              action={<Button size="small" type="primary" onClick={() => setPage("settings")}>{t("去安装")}</Button>}
            />
          )}
          <div className="card" style={{ marginBottom: 12 }}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <span className="muted">{t("可训练图片")}</span>
              {trainable > 0 && (
                <Button size="small" onClick={() => setSplitOpen(true)}>
                  {t("查看和调整")}
                </Button>
              )}
            </div>
            <div className="kpi">{trainable}</div>
            {split && (
              <div className="dim" style={{ fontSize: 12 }}>
                {t("训练 {train} · 验证 {val} · 测试 {test}", { train: split.train, val: split.val, test: split.test })}
                {split.unset > 0 && ` · ${t("未划分 {n}（开始训练时按比例分配）", { n: split.unset })}`}
              </div>
            )}
            {review && cfg.onlyReviewed && !useVersion && (
              <div style={{ color: "var(--warn)", fontSize: 12, marginTop: 4 }}>
                {t("高级设置里勾了「只用审核通过的图片」，待审核的图这次不参与训练")}
              </div>
            )}
            {trainable > 0 && trainable < 50 && (
              <div style={{ color: "var(--warn)", fontSize: 12, marginTop: 6 }}>{fewText}</div>
            )}
          </div>
          <div className="card">
            <Form layout="vertical" size="small">
              <Form.Item
                label={t("起点")}
                tooltip={t("补了新数据后，接着上次训好的模型继续训，比从头训快、也更稳。类别有变化时，分类部分会重新学")}
              >
                <Select
                  style={{ width: "100%" }}
                  value={baseRun ? baseRun.id : 0}
                  onChange={(v) => {
                    const r = baseRuns.find((x) => x.id === v);
                    update(r ? { baseRun: r.id, model: String(r.config.model), imgSize: r.config.imgSize ?? cfg.imgSize } : { baseRun: null });
                  }}
                  options={[
                    { value: 0, label: t("从预训练权重开始（从头训）") },
                    ...baseRuns.map((r) => ({
                      value: r.id,
                      label: `${t("接着「{name}」继续训", { name: r.name })} · ${modelName(r)} · ${mn.m50} ${r.bestMap50?.toFixed(3) ?? "—"}`,
                    })),
                  ]}
                />
              </Form.Item>
              <Form.Item
                label={t("模型")}
                extra={
                  baseRun
                    ? t("接着训练时沿用那次的模型")
                    : t("标「推荐」的是按这台电脑推荐的（{hw}）", {
                        hw: isMac ? t("Apple 芯片") : gpuMem ? t("显卡 {n}GB 显存", { n: Math.round(gpuMem / 1024) }) : t("没有检测到 NVIDIA 显卡，CPU 训练选轻一点的"),
                      })
                }
              >
                <Radio.Group value={cfg.model} disabled={!!baseRun} onChange={(e) => pickModel(e.target.value)}>
                  <Space direction="vertical">
                    {models.map((m) => (
                      <Radio key={m.value} value={m.value}>
                        <b>{m.name}</b>
                        {m.value === recommended && <Tag color="blue" style={{ marginLeft: 6 }}>{t("推荐")}</Tag>}
                        {m.yolox && <Tag color="green" style={{ marginLeft: 6 }}>YOLO</Tag>}
                        {m.oneStage && <Tag style={{ marginLeft: 6 }}>{t("单阶段")}</Tag>}
                        <div className="dim" style={{ fontSize: 12 }}>{m.desc}</div>
                      </Radio>
                    ))}
                  </Space>
                </Radio.Group>
              </Form.Item>
              <Form.Item label={t("训练轮数")} tooltip={t("数据越少需要的轮数越多；一般 20~50，YOLOX 建议 50 以上")}>
                <InputNumber min={1} max={500} value={cfg.epochs} onChange={(v) => update({ epochs: Number(v ?? 30) })} />
              </Form.Item>
              <Form.Item label={t("训练数据")} tooltip={t("选一个数据集版本训练，方便对比不同数据的效果；版本在「数据 → 管理 → 数据集版本」里保存")}>
                <Select
                  style={{ width: "100%" }}
                  value={cfg.version != null && versions.some((v) => v.id === cfg.version) ? cfg.version : 0}
                  onChange={(v) => update({ version: v === 0 ? null : v })}
                  options={[
                    { value: 0, label: t("当前数据（{n} 张）", { n: trainable }) },
                    ...versions.map((v) => ({ value: v.id, label: t("版本「{name}」（{n} 张，{date}）", { name: v.name, n: v.images, date: new Date(v.createdAt * 1000).toLocaleDateString() }) })),
                  ]}
                />
              </Form.Item>
              <Button type="link" size="small" style={{ padding: 0, marginBottom: 10 }} onClick={() => toggleAdv(!advOpen)}>
                {advOpen ? t("收起高级设置 ▴") : t("高级设置：输入尺寸、批大小、数据增强、设备 ▾")}
              </Button>
              {advOpen && (
                <>
              <Space wrap>
                <Form.Item
                  label={t("输入尺寸")}
                  tooltip={
                    picked?.fixedSize
                      ? t("这个模型固定用 {size}×{size} 输入", { size: picked.fixedSize })
                      : isClsTask(task)
                        ? t("分类一般用 224；细节多的图可以加大")
                        : t("目标小（远处的人、安全帽）用大尺寸，速度会变慢")
                  }
                >
                  <Select
                    style={{ width: 100 }}
                    value={picked?.fixedSize ?? cfg.imgSize}
                    disabled={!!picked?.fixedSize}
                    onChange={(v) => update({ imgSize: v })}
                    options={sizeOptions.map((v) => ({ value: v, label: v }))}
                  />
                </Form.Item>
                <Form.Item label={t("批大小")} tooltip={t("显存不够报错时调小")}>
                  <InputNumber min={1} max={64} value={cfg.batch} onChange={(v) => update({ batch: Number(v ?? 4) })} />
                </Form.Item>
              </Space>
              {picked?.yolox ? (
                <Alert
                  style={{ marginBottom: 12 }}
                  type="info"
                  showIcon
                  message={t("YOLOX 用官方的数据增强")}
                  description={t("Mosaic 拼图、MixUp、随机缩放旋转、颜色抖动、水平翻转、多尺度训练都自动开启，最后约 10% 的轮数关掉强增强收尾。学习率按批大小自动换算。")}
                />
              ) : (
                <AugPanel value={cfg.aug} task={task} imgSize={picked?.fixedSize ?? cfg.imgSize} fixedSize={!!picked?.fixedSize} onChange={(aug) => update({ aug })} />
              )}
              <Form.Item label={t("设备")}>
                <Radio.Group value={cfg.device} onChange={(e) => update({ device: e.target.value })}>
                  <Radio value="auto">{t("自动（有显卡用显卡）")}</Radio>
                  <Radio value="cpu">{t("只用 CPU")}</Radio>
                </Radio.Group>
              </Form.Item>
              {review && (
                <Form.Item>
                  <Checkbox checked={cfg.onlyReviewed} onChange={(e) => update({ onlyReviewed: e.target.checked })}>
                    {t("只用审核通过的图片")}
                  </Checkbox>
                </Form.Item>
              )}
                </>
              )}
              {trainN > 0 && (
                <div className="dim" style={{ fontSize: 12, marginBottom: 6 }}>
                  {t("每轮约 {n} 步 · 共约 {m} 步", { n: stepsPerEpoch, m: totalSteps })}
                  {fewData && <span style={{ color: "var(--warn)" }}> · {t("步数太少，多半学不到东西")}</span>}
                  {fewVal && <div style={{ color: "var(--warn)" }}>{t("验证集只有 {n} 张，指标波动会很大、不可信。", { n: valN })}</div>}
                  {fewCls.length > 0 && <div style={{ color: "var(--warn)" }}>{t("这些类别标得太少，容易认不出：{list}", { list: fewClsText })}</div>}
                  {rawIds.length > 0 && (
                    <div style={{ color: rawMany ? "var(--warn)" : undefined }}>
                      {t("{n} 张图的标注全是预标注、没人改过", { n: rawIds.length })}
                      <Button type="link" size="small" style={{ padding: "0 4px", height: "auto", fontSize: 12 }} onClick={() => annotateQueue(rawIds, t("预标注没人改过的图"))}>
                        {t("去看这批")}
                      </Button>
                      <Popconfirm title={t("这 {n} 张都看过、没问题了？之后不再提醒。", { n: rawIds.length })} onConfirm={confirmRaw}>
                        <Button type="link" size="small" style={{ padding: "0 4px", height: "auto", fontSize: 12 }}>
                          {t("都看过了")}
                        </Button>
                      </Popconfirm>
                    </div>
                  )}
                </div>
              )}
              <Button
                type="primary"
                block
                icon={<CaretRightOutlined />}
                loading={starting}
                disabled={running || (!useVersion && trainable === 0) || (env != null && !env.ready)}
                onClick={start}
              >
                {running ? t("已有训练在进行") : t("开始训练")}
              </Button>
              {starting && prep && (
                <div style={{ marginTop: 8 }}>
                  <Progress size="small" percent={prep.total ? Math.floor((prep.done / prep.total) * 100) : 0} status="active" />
                  <div className="dim" style={{ fontSize: 12 }}>
                    {prep.total ? t("正在准备训练数据 {done}/{total}", { done: prep.done, total: prep.total }) : t("正在准备训练数据…")}
                  </div>
                </div>
              )}
            </Form>
          </div>
        </div>

        <div style={{ minWidth: 0 }}>
          {!run ? (
            <div className="card" style={{ padding: 48 }}>
              <Empty description={t("还没有训练记录")} />
            </div>
          ) : (
            <>
              {(detail?.warnings ?? []).map((w) => (
                <Alert
                  key={w.code}
                  type="warning"
                  showIcon
                  style={{ marginBottom: 12 }}
                  message={
                    w.code === "no_pretrained"
                      ? t("没下载到预训练权重，从头训练效果会差很多")
                      : w.code === "zero_metric"
                        ? t("{metric} 一直是 0，模型基本没学到东西", { metric: mn.m50 })
                        : w.msg
                  }
                  description={
                    w.code === "no_pretrained"
                      ? t("多半是网络问题（要能访问 github.com 或 download.pytorch.org）。检查网络或代理后重新训练；下载成功一次就会缓存，以后不用再下。")
                      : w.code === "zero_metric"
                        ? t("常见原因：图太少或训练步数太少（多标一些图、加大训练轮数）；标注或类别有问题（去体检看看）；没加载到预训练权重（看下面日志的关键信息）。")
                        : undefined
                  }
                  action={
                    w.code === "zero_metric" ? (
                      <Button size="small" onClick={() => setPage("health")}>
                        {t("去体检")}
                      </Button>
                    ) : undefined
                  }
                />
              ))}
              <div className="card" style={{ marginBottom: 12 }}>
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <div className="row">
                    <h2 style={{ fontSize: 15, margin: 0 }}>{run.name}</h2>
                    <Tag color={RUN_STATUS[run.status].color}>{RUN_STATUS[run.status].text}</Tag>
                    <span className="muted">
                      {modelName(run)} · {run.config?.imgSize}px ·{" "}
                      {run.config?.images
                        ? t("训练 {train} 张 / 验证 {val} 张 / {anns} 个标注", { train: run.config.images.train, val: run.config.images.val, anns: run.config.images.anns ?? run.config.images.boxes })
                        : ""}
                      {run.config?.batchRequested ? ` · ${t("批大小 {batch}（显存不够，从 {req} 自动调小）", { batch: run.config.batch, req: run.config.batchRequested })}` : ""}
                    </span>
                  </div>
                  <Space>
                    {run.status === "running" && (
                      <Popconfirm
                        title={t("停止训练？训完的轮次会保留，之后可以点「继续训练」从这里接着跑")}
                        okText={t("停止")}
                        cancelText={t("取消")}
                        okButtonProps={{ danger: true }}
                        onConfirm={() => api.stopTraining(run.id).then(() => message.info(t("正在停止，当前这批算完就停"))).catch((e) => message.error(errText(e)))}
                      >
                        <Button size="small" danger icon={<StopOutlined />}>
                          {t("停止")}
                        </Button>
                      </Popconfirm>
                    )}
                    {run.canResume && !running && (
                      <Tooltip title={t("从最后训完的一轮接着训练，曲线和日志接在后面")}>
                        <Button size="small" type="primary" icon={<CaretRightOutlined />} loading={resuming} onClick={resume}>
                          {t("继续训练")}
                        </Button>
                      </Tooltip>
                    )}
                    {run.hasWeights && (
                      <>
                        {/* 训练中的记录评估页不让选（还没训完） */}
                        {run.status !== "running" && <Button size="small" onClick={() => openRun(run.id, "eval")}>{t("看评估")}</Button>}
                        {run.engine === "torchvision" && (
                          <Button size="small" icon={<ExportOutlined />} loading={exporting} onClick={doExport}>
                            {run.hasOnnx ? t("重新导出 ONNX") : t("导出 ONNX")}
                          </Button>
                        )}
                      </>
                    )}
                    <Tooltip title={t("打开训练文件夹")}>
                      <Button size="small" icon={<FolderOpenOutlined />} onClick={() => api.revealPath(run.dir).catch((e) => message.error(errText(e)))} />
                    </Tooltip>
                    {run.status !== "running" && <Button size="small" icon={<DeleteOutlined />} onClick={remove} />}
                  </Space>
                </div>
                {exporting && (
                  <div style={{ marginTop: 12 }}>
                    {/* 导出没有分步，进度条只表示还在进行 */}
                    <Progress size="small" percent={100} status="active" showInfo={false} />
                    <div className="dim" style={{ fontSize: 12 }}>{onnxMsg}</div>
                  </div>
                )}
                {run.status === "running" && (
                  <div style={{ marginTop: 12 }}>
                    <Progress
                      percent={
                        !prog && dl
                          ? downloadPercent(dl)
                          : prog
                            ? Math.floor(((prog.epoch - 1 + prog.iter / prog.iters) / prog.epochs) * 100)
                            : Math.floor((metrics.length / epochs) * 100)
                      }
                      status="active"
                    />
                    <div className="dim" style={{ fontSize: 12 }}>
                      {prog
                        ? t("第 {epoch} / {epochs} 轮 · 本轮 {iter}/{iters} · loss {loss} · 预计剩余 {eta}", { epoch: prog.epoch, epochs: prog.epochs, iter: prog.iter, iters: prog.iters, loss: prog.loss.toFixed(4), eta: fmtDur(prog.eta) })
                        : dl
                          ? t("正在下载预训练权重 {name}：{progress}（只需下载一次，断了会接着下）", { name: dl.name, progress: downloadText(dl) })
                          : t("正在准备（首次会下载预训练权重）…")}
                    </div>
                  </div>
                )}
                {detail?.error && run.status !== "running" && (
                  <Alert
                    type="error"
                    style={{ marginTop: 12 }}
                    message={t("训练出错")}
                    description={
                      <div>
                        {errHint && <div style={{ fontWeight: 600, marginBottom: 6 }}>{errHint.hint}</div>}
                        {/* 先只显示错误本身，调用栈收起来 */}
                        <div style={{ whiteSpace: "pre-wrap", wordBreak: "break-all", fontSize: 12, maxHeight: 80, overflow: "auto" }}>{detail.error.split("\n\n")[0]}</div>
                        <a style={{ fontSize: 12 }} onClick={() => setShowTrace(!showTrace)}>
                          {showTrace ? t("收起详细信息") : t("详细信息")}
                        </a>
                        {showTrace && <pre style={{ whiteSpace: "pre-wrap", margin: "6px 0 0", fontSize: 12, maxHeight: 220, overflow: "auto", userSelect: "text" }}>{detail.error}</pre>}
                      </div>
                    }
                    action={
                      errHint?.fix === "settings" ? (
                        <Button size="small" onClick={() => setPage("settings")}>{t("去设置换 PyTorch 版本")}</Button>
                      ) : errHint?.fix === "cpu" ? (
                        <Button
                          size="small"
                          disabled={cfg.device === "cpu"}
                          onClick={() => {
                            update({ device: "cpu" });
                            toggleAdv(true);
                            message.success(t("已改成只用 CPU，可以重新开始训练"));
                          }}
                        >
                          {t("改用 CPU")}
                        </Button>
                      ) : errHint?.fix === "data" ? (
                        <Button size="small" onClick={() => setPage("data")}>{t("去数据页")}</Button>
                      ) : undefined
                    }
                  />
                )}
                <div className="cards" style={{ gridTemplateColumns: "1fr 1fr", marginTop: 12 }}>
                  <div>
                    <div className="muted">{t("损失 loss")}</div>
                    <LineChart xMax={epochs} yMin={0} series={[{ name: "loss", color: "#3d8bfd", values: metrics.map((m) => m.loss) }]} format={(v) => v.toFixed(2)} />
                  </div>
                  <div>
                    <div className="muted">
                      {t("验证集")} <span style={{ color: "#2fb67c" }}>{mn.m50}</span> / <span style={{ color: "#9b7bf2" }}>{mn.m}</span>
                      {run.bestMap50 != null && <span style={{ marginLeft: 8 }}>{t("最好 {v}", { v: run.bestMap50.toFixed(3) })}</span>}
                    </div>
                    <LineChart
                      xMax={epochs}
                      yMin={0}
                      yMax={1}
                      series={[
                        { name: mn.m50, color: "#2fb67c", values: metrics.map((m) => m.map50) },
                        { name: mn.m, color: "#9b7bf2", values: metrics.map((m) => m.map) },
                      ]}
                    />
                  </div>
                </div>
              </div>
              <div className="card" style={{ marginBottom: 12 }}>
                {logHead.length > 0 && (
                  <>
                    <div className="muted" style={{ marginBottom: 6 }}>{t("关键信息")} <span className="dim">· {t("日志开头的设备、数据量、预训练权重")}</span></div>
                    <div className="logbox" style={{ height: "auto", maxHeight: 110, marginBottom: 10 }}>
                      {logHead.map((l, i) => (
                        <div key={i} style={isWarnLine(l) || warnMsgs.has(l) ? { color: "var(--warn)" } : undefined}>{l}</div>
                      ))}
                    </div>
                  </>
                )}
                <div className="muted" style={{ marginBottom: 6 }}>{t("日志")}</div>
                <div ref={logBox} className="logbox">
                  {logLines.length === 0
                    ? "—"
                    : logLines.map((l, i) => (
                        <div key={i} style={{ minHeight: "1em", color: isWarnLine(l) || warnMsgs.has(l) ? "var(--warn)" : undefined }}>{l}</div>
                      ))}
                </div>
              </div>
            </>
          )}
          <div className="card">
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}>
              <span className="muted">{t("训练记录")} <span className="dim">· {t("勾选两次以上可以对比，也可以一起删除")}</span></span>
              <Space>
                <Tooltip title={t("导入别处训好的 YOLO 等 ONNX 模型，用来预标注、评估和部署")}>
                  <Button size="small" onClick={() => setImporting(true)}>
                    {t("导入外部模型")}
                  </Button>
                </Tooltip>
                <Tooltip title={compareIds.length > 8 ? t("最多同时对比 8 次训练") : undefined}>
                  <Button size="small" disabled={compareIds.length < 2 || compareIds.length > 8} onClick={() => setComparing(true)}>
                    {compareIds.length >= 2 ? t("对比所选（{n}）", { n: compareIds.length }) : t("对比所选")}
                  </Button>
                </Tooltip>
                <Button size="small" danger icon={<DeleteOutlined />} disabled={compareIds.length === 0} onClick={removeSelected}>
                  {compareIds.length ? t("删除所选（{n}）", { n: compareIds.length }) : t("删除所选")}
                </Button>
              </Space>
            </div>
            <Table<RunInfo>
              size="small"
              rowKey="id"
              dataSource={runs}
              pagination={false}
              rowSelection={{
                selectedRowKeys: compareIds,
                onChange: (keys) => setCompareIds(keys as number[]),
                getCheckboxProps: (r) => ({ disabled: r.status === "running" }),
              }}
              onRow={(r) => ({ onClick: () => setSel(r.id), style: { cursor: "pointer", background: r.id === sel ? "rgba(61,139,253,.12)" : undefined } })}
              columns={[
                { title: t("时间"), dataIndex: "createdAt", render: fmtTime },
                { title: t("模型"), render: (_, r) => modelName(r) },
                { title: t("数据"), render: (_, r) => r.config?.versionName ?? <span className="dim">{t("当前数据")}</span> },
                { title: t("轮数"), render: (_, r) => r.config?.epochs },
                { title: mn.m50, dataIndex: "bestMap50", render: (v: number | null) => (v == null ? "—" : v.toFixed(3)) },
                { title: t("状态"), dataIndex: "status", render: (s: RunStatus) => <Tag color={RUN_STATUS[s].color}>{RUN_STATUS[s].text}</Tag> },
              ]}
            />
          </div>
        </div>
      </div>
      <SplitModal
        project={project}
        review={review}
        open={splitOpen}
        ratios={cfg.ratios}
        onRatios={(ratios) => update({ ratios })}
        onClose={() => setSplitOpen(false)}
        onChanged={setSplit}
      />
      <ImportModelModal
        open={importing}
        onClose={() => setImporting(false)}
        onDone={async (id) => {
          await loadRuns();
          setSel(id);
        }}
      />
      <RunCompare
        open={comparing}
        task={task}
        runs={runs.filter((r) => compareIds.includes(r.id))}
        onClose={() => setComparing(false)}
      />
    </div>
  );
}
