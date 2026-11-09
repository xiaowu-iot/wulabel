import { useCallback, useEffect, useRef, useState } from "react";
import { Alert, App, Button, Checkbox, Empty, Form, InputNumber, Progress, Radio, Select, Space, Table, Tag, Tooltip } from "antd";
import {
  CaretRightOutlined,
  DeleteOutlined,
  ExportOutlined,
  FolderOpenOutlined,
  StopOutlined,
} from "@ant-design/icons";
import { listen } from "@tauri-apps/api/event";
import { api, EnvStatus, errText, ProjectInfo, RunDetail, RunInfo, RunStatus, SplitSummary, Task, TrainConfig, VersionInfo } from "../api";
import { prefs, useApp } from "../store";
import LineChart from "../components/LineChart";
import RunCompare from "../components/RunCompare";

/** 内置模型，和 worker/xlabel_worker/models.py 的 MODELS 对应。oneStage：单阶段检测，部署到 TensorRT / NPU 更顺 */
export const MODELS: { value: string; name: string; desc: string; task: Task; oneStage?: boolean; fixedSize?: number }[] = [
  { value: "frcnn_mobilenet", name: "Faster R-CNN MobileNetV3", desc: "速度快、显存占用小，适合入门和没有独立显卡的电脑", task: "detect" },
  { value: "frcnn_resnet50", name: "Faster R-CNN ResNet50 v2", desc: "精度更高，训练和推理更慢，建议 6GB 以上显存", task: "detect" },
  { value: "retinanet_r50", name: "RetinaNet ResNet50 v2", desc: "单阶段，精度接近 Faster R-CNN；没有 RoIAlign，转 TensorRT 更顺", task: "detect", oneStage: true },
  { value: "fcos_r50", name: "FCOS ResNet50", desc: "单阶段、无锚框，结构简单，适合部署", task: "detect", oneStage: true },
  { value: "ssdlite", name: "SSDLite MobileNetV3", desc: "单阶段、很轻，适合 CPU 和 NPU 边缘设备实时推理；固定 320 输入，小目标一般", task: "detect", oneStage: true, fixedSize: 320 },
  { value: "maskrcnn_r50", name: "Mask R-CNN ResNet50", desc: "实例分割，速度一般，建议有 6GB 以上显存", task: "segment" },
  { value: "maskrcnn_r50_v2", name: "Mask R-CNN ResNet50 v2", desc: "实例分割，精度更高、更慢，建议 8GB 以上显存", task: "segment" },
  { value: "keypointrcnn_r50", name: "Keypoint R-CNN ResNet50", desc: "关键点检测；人体 17 点能用上 COCO 预训练，其他点数从头学关键点头", task: "pose" },
  { value: "cls_mobilenet", name: "MobileNetV3 Large", desc: "速度快、很轻，适合入门和边缘设备", task: "classify" },
  { value: "cls_efficientnet", name: "EfficientNet-B0", desc: "精度和速度比较均衡", task: "classify" },
  { value: "cls_resnet50", name: "ResNet50", desc: "精度更高，训练更慢，建议有独立显卡", task: "classify" },
];

/** 各任务的默认训练参数 */
const TASK_DEFAULTS: Record<Task, Partial<TrainConfig>> = {
  detect: { model: "frcnn_mobilenet", imgSize: 640, batch: 4, epochs: 30, lr: 0.01 },
  segment: { model: "maskrcnn_r50", imgSize: 640, batch: 2, epochs: 30, lr: 0.01 },
  pose: { model: "keypointrcnn_r50", imgSize: 640, batch: 2, epochs: 40, lr: 0.01 },
  classify: { model: "cls_mobilenet", imgSize: 224, batch: 32, epochs: 30, lr: 0.01 },
};

/** 按显存推荐模型 */
const recommendModel = (task: Task, gpuMem: number) =>
  ({
    detect: gpuMem >= 6000 ? "frcnn_resnet50" : "frcnn_mobilenet",
    segment: gpuMem >= 8000 ? "maskrcnn_r50_v2" : "maskrcnn_r50",
    pose: "keypointrcnn_r50",
    classify: gpuMem >= 6000 ? "cls_efficientnet" : "cls_mobilenet",
  })[task];

/** 各任务主指标的叫法（分类没有 mAP，worker 把准确率放在 map50、宏 F1 放在 map） */
export const metricNames = (task: Task) =>
  task === "classify" ? { m50: "准确率", m: "宏平均 F1" } : { m50: "mAP50", m: "mAP50-95" };

export const RUN_STATUS: Record<RunStatus, { text: string; color: string }> = {
  running: { text: "训练中", color: "processing" },
  done: { text: "完成", color: "success" },
  stopped: { text: "已停止", color: "warning" },
  failed: { text: "失败", color: "error" },
  interrupted: { text: "已中断", color: "default" },
};

export const fmtTime = (ts: number) => new Date(ts * 1000).toLocaleString();
const fmtDur = (s: number | null | undefined) => {
  if (s == null || !Number.isFinite(s)) return "—";
  if (s < 60) return `${Math.round(s)} 秒`;
  if (s < 3600) return `${Math.floor(s / 60)} 分 ${Math.round(s % 60)} 秒`;
  return `${Math.floor(s / 3600)} 小时 ${Math.round((s % 3600) / 60)} 分`;
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
  const task = project.task;
  const models = MODELS.filter((m) => m.task === task);
  const mn = metricNames(task);
  const cfgKey = `trainConfig.${task}`;
  const [env, setEnv] = useState<EnvStatus | null>(null);
  const [split, setSplit] = useState<SplitSummary | null>(null);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [sel, setSel] = useState<number | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [prog, setProg] = useState<ProgressEv | null>(null);
  const [starting, setStarting] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [versions, setVersions] = useState<VersionInfo[]>([]);
  const [compareIds, setCompareIds] = useState<number[]>([]);
  const [comparing, setComparing] = useState(false);
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
    if (selRef.current === id) setDetail(d);
    requestAnimationFrame(() => logBox.current?.scrollTo(0, logBox.current.scrollHeight));
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const [rs, sp] = await Promise.all([loadRuns(), api.splitSummary()]);
        setSplit(sp);
        const running = await api.runningRun();
        setSel(running ?? rs[0]?.id ?? null);
      } catch (e) {
        message.error(errText(e));
      }
      api.envStatus(false).then(setEnv).catch(() => {});
      api.listVersions().then(setVersions).catch(() => {});
    })();
  }, [loadRuns, message]);

  useEffect(() => {
    setDetail(null);
    setProg(null);
    if (sel != null) loadDetail(sel).catch((e) => message.error(errText(e)));
  }, [sel, loadDetail, message]);

  // 训练实时事件
  useEffect(() => {
    const un = listen<{ runId: number; event: Record<string, unknown> }>("train-event", (e) => {
      const { runId, event } = e.payload;
      if (runId !== selRef.current) return;
      const t = event.type;
      if (t === "progress") setProg(event as unknown as ProgressEv);
      else if (t === "metric" || t === "log" || t === "error") loadDetail(runId);
      else if (t === "finished") {
        setProg(null);
        loadRuns();
        loadDetail(runId);
        const st = event.status as string;
        if (st === "done") message.success("训练完成");
        else if (st === "failed") message.error("训练失败，看下方日志");
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
    if (!MODELS.some((m) => m.value === cfg.model && m.task === task)) {
      update({ ...TASK_DEFAULTS[task] });
    }
  }, [task]); // eslint-disable-line react-hooks/exhaustive-deps

  const picked = MODELS.find((m) => m.value === cfg.model);
  const sizeOptions = task === "classify" ? [128, 160, 224, 256, 320, 384] : [320, 480, 640, 800, 960, 1280];

  const start = async () => {
    setStarting(true);
    try {
      // 记住的版本可能已经被删掉了，那就用当前数据
      const id = await api.startTraining({ ...cfg, version: useVersion ? cfg.version : null });
      await loadRuns();
      setSel(id);
      setSplit(await api.splitSummary());
      message.success("已开始训练");
    } catch (e) {
      modal.error({ title: "没能开始训练", content: errText(e) });
    } finally {
      setStarting(false);
    }
  };

  const run = detail?.run;
  const running = runs.some((r) => r.status === "running");
  const trainable = split ? split.train + split.val + split.test + split.unset : 0;
  const metrics = detail?.metrics ?? [];
  const epochs = run?.config?.epochs ?? cfg.epochs;
  const gpuMem = env?.gpu?.vramMb ?? 0;
  const useVersion = cfg.version != null && versions.some((v) => v.id === cfg.version);
  const recommended = recommendModel(task, gpuMem);
  const taskWord = { detect: "检测", segment: "实例分割", pose: "关键点", classify: "图像分类" }[task];
  const unitWord = task === "classify" ? "张图" : task === "pose" ? "个目标" : "个框";

  const doExport = async () => {
    if (!run) return;
    setExporting(true);
    try {
      const p = await api.exportOnnx(run.id);
      await loadRuns();
      await loadDetail(run.id);
      modal.success({
        title: "已导出 ONNX",
        content: <div style={{ wordBreak: "break-all" }}>{p}<div className="muted">同目录的 model.json 里有类别和输入输出说明。</div></div>,
        okText: "好的",
      });
    } catch (e) {
      message.error(errText(e));
    } finally {
      setExporting(false);
    }
  };

  const remove = () => {
    if (!run) return;
    modal.confirm({
      title: `删除训练记录 ${run.name}？`,
      content: "模型文件、日志和这次训练用的数据集副本都会删除，无法恢复。",
      okText: "删除",
      okButtonProps: { danger: true },
      cancelText: "取消",
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
          <h1>训练</h1>
          <p className="desc">用已标注和已审核的图片训练{taskWord}模型。</p>
          {env && !env.ready && (
            <Alert
              type="warning"
              showIcon
              style={{ marginBottom: 12 }}
              message="还没有训练环境"
              description="第一次训练前需要安装 PyTorch，约 3GB，只需要一次。"
              action={<Button size="small" type="primary" onClick={() => setPage("settings")}>去安装</Button>}
            />
          )}
          <div className="card" style={{ marginBottom: 12 }}>
            <div className="muted">可训练图片</div>
            <div className="kpi">{trainable}</div>
            {split && (
              <div className="dim" style={{ fontSize: 12 }}>
                训练 {split.train} · 验证 {split.val} · 测试 {split.test}
                {split.unset > 0 && ` · 未划分 ${split.unset}（开始训练时按比例分配）`}
              </div>
            )}
            {trainable > 0 && trainable < 50 && (
              <div style={{ color: "var(--warn)", fontSize: 12, marginTop: 6 }}>图片较少，建议每个类别至少标 100 {unitWord}再训练。</div>
            )}
          </div>
          <div className="card">
            <Form layout="vertical" size="small">
              <Form.Item label="模型">
                <Radio.Group value={cfg.model} onChange={(e) => update({ model: e.target.value })}>
                  <Space direction="vertical">
                    {models.map((m) => (
                      <Radio key={m.value} value={m.value}>
                        <b>{m.name}</b>
                        {m.value === recommended && <Tag color="blue" style={{ marginLeft: 6 }}>推荐</Tag>}
                        {m.oneStage && <Tag style={{ marginLeft: 6 }}>单阶段</Tag>}
                        <div className="dim" style={{ fontSize: 12 }}>{m.desc}</div>
                      </Radio>
                    ))}
                  </Space>
                </Radio.Group>
              </Form.Item>
              <Space wrap>
                <Form.Item label="训练轮数" tooltip="数据越少需要的轮数越多；一般 20~50">
                  <InputNumber min={1} max={500} value={cfg.epochs} onChange={(v) => update({ epochs: Number(v ?? 30) })} />
                </Form.Item>
                <Form.Item
                  label="输入尺寸"
                  tooltip={
                    picked?.fixedSize
                      ? `这个模型固定用 ${picked.fixedSize}×${picked.fixedSize} 输入`
                      : task === "classify"
                        ? "分类一般用 224；细节多的图可以加大"
                        : "目标小（远处的人、安全帽）用大尺寸，速度会变慢"
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
                <Form.Item label="批大小" tooltip="显存不够报错时调小">
                  <InputNumber min={1} max={64} value={cfg.batch} onChange={(v) => update({ batch: Number(v ?? 4) })} />
                </Form.Item>
              </Space>
              <Form.Item label="训练数据" tooltip="选一个数据集版本训练，方便对比不同数据的效果；版本在「数据 → 管理 → 数据集版本」里保存">
                <Select
                  style={{ width: "100%" }}
                  value={cfg.version != null && versions.some((v) => v.id === cfg.version) ? cfg.version : 0}
                  onChange={(v) => update({ version: v === 0 ? null : v })}
                  options={[
                    { value: 0, label: `当前数据（${trainable} 张）` },
                    ...versions.map((v) => ({ value: v.id, label: `版本「${v.name}」（${v.images} 张，${new Date(v.createdAt * 1000).toLocaleDateString()}）` })),
                  ]}
                />
              </Form.Item>
              <Form.Item label="设备">
                <Radio.Group value={cfg.device} onChange={(e) => update({ device: e.target.value })}>
                  <Radio value="auto">自动（有显卡用显卡）</Radio>
                  <Radio value="cpu">只用 CPU</Radio>
                </Radio.Group>
              </Form.Item>
              <Form.Item>
                <Checkbox checked={cfg.onlyReviewed} onChange={(e) => update({ onlyReviewed: e.target.checked })}>
                  只用已审核的图片
                </Checkbox>
              </Form.Item>
              <Button
                type="primary"
                block
                icon={<CaretRightOutlined />}
                loading={starting}
                disabled={running || (!useVersion && trainable === 0) || (env != null && !env.ready)}
                onClick={start}
              >
                {running ? "已有训练在进行" : "开始训练"}
              </Button>
            </Form>
          </div>
        </div>

        <div style={{ minWidth: 0 }}>
          {!run ? (
            <div className="card" style={{ padding: 48 }}>
              <Empty description="还没有训练记录" />
            </div>
          ) : (
            <>
              <div className="card" style={{ marginBottom: 12 }}>
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <div className="row">
                    <h2 style={{ fontSize: 15, margin: 0 }}>{run.name}</h2>
                    <Tag color={RUN_STATUS[run.status].color}>{RUN_STATUS[run.status].text}</Tag>
                    <span className="muted">
                      {MODELS.find((m) => m.value === run.config?.model)?.name ?? run.config?.model} · {run.config?.imgSize}px ·{" "}
                      {run.config?.images
                        ? `训练 ${run.config.images.train} 张 / 验证 ${run.config.images.val} 张 / ${run.config.images.anns ?? run.config.images.boxes} 个标注`
                        : ""}
                    </span>
                  </div>
                  <Space>
                    {run.status === "running" && (
                      <Button size="small" danger icon={<StopOutlined />} onClick={() => api.stopTraining(run.id).then(() => message.info("正在停止，当前这批算完就停")).catch((e) => message.error(errText(e)))}>
                        停止
                      </Button>
                    )}
                    {run.hasWeights && (
                      <>
                        <Button size="small" onClick={() => setPage("eval")}>看评估</Button>
                        <Button size="small" icon={<ExportOutlined />} loading={exporting} onClick={doExport}>
                          {run.hasOnnx ? "重新导出 ONNX" : "导出 ONNX"}
                        </Button>
                      </>
                    )}
                    <Tooltip title="打开训练文件夹">
                      <Button size="small" icon={<FolderOpenOutlined />} onClick={() => api.revealPath(run.dir).catch((e) => message.error(errText(e)))} />
                    </Tooltip>
                    {run.status !== "running" && <Button size="small" icon={<DeleteOutlined />} onClick={remove} />}
                  </Space>
                </div>
                {run.status === "running" && (
                  <div style={{ marginTop: 12 }}>
                    <Progress
                      percent={prog ? Math.floor(((prog.epoch - 1 + prog.iter / prog.iters) / prog.epochs) * 100) : Math.floor((metrics.length / epochs) * 100)}
                      status="active"
                    />
                    <div className="dim" style={{ fontSize: 12 }}>
                      {prog
                        ? `第 ${prog.epoch} / ${prog.epochs} 轮 · 本轮 ${prog.iter}/${prog.iters} · loss ${prog.loss.toFixed(4)} · 预计剩余 ${fmtDur(prog.eta)}`
                        : "正在准备（首次会下载预训练权重）…"}
                    </div>
                  </div>
                )}
                {detail?.error && run.status !== "running" && (
                  <Alert type="error" style={{ marginTop: 12 }} message="训练出错" description={<pre style={{ whiteSpace: "pre-wrap", margin: 0, fontSize: 12, maxHeight: 160, overflow: "auto" }}>{detail.error}</pre>} />
                )}
                <div className="cards" style={{ gridTemplateColumns: "1fr 1fr", marginTop: 12 }}>
                  <div>
                    <div className="muted">损失 loss</div>
                    <LineChart xMax={epochs} yMin={0} series={[{ name: "loss", color: "#3d8bfd", values: metrics.map((m) => m.loss) }]} format={(v) => v.toFixed(2)} />
                  </div>
                  <div>
                    <div className="muted">
                      验证集 <span style={{ color: "#2fb67c" }}>{mn.m50}</span> / <span style={{ color: "#9b7bf2" }}>{mn.m}</span>
                      {run.bestMap50 != null && <span style={{ marginLeft: 8 }}>最好 {run.bestMap50.toFixed(3)}</span>}
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
                <div className="muted" style={{ marginBottom: 6 }}>日志</div>
                <div ref={logBox} className="logbox">
                  {(detail?.log ?? []).join("\n") || "—"}
                </div>
              </div>
            </>
          )}
          <div className="card">
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}>
              <span className="muted">训练记录 <span className="dim">· 勾选两次以上可以对比</span></span>
              <Button size="small" disabled={compareIds.length < 2} onClick={() => setComparing(true)}>
                对比所选{compareIds.length >= 2 ? `（${compareIds.length}）` : ""}
              </Button>
            </div>
            <Table<RunInfo>
              size="small"
              rowKey="id"
              dataSource={runs}
              pagination={false}
              rowSelection={{
                selectedRowKeys: compareIds,
                onChange: (keys) => setCompareIds((keys as number[]).slice(0, 8)),
                getCheckboxProps: (r) => ({ disabled: r.status === "running" }),
              }}
              onRow={(r) => ({ onClick: () => setSel(r.id), style: { cursor: "pointer", background: r.id === sel ? "rgba(61,139,253,.12)" : undefined } })}
              columns={[
                { title: "时间", dataIndex: "createdAt", render: fmtTime },
                { title: "模型", render: (_, r) => MODELS.find((m) => m.value === r.config?.model)?.name ?? r.config?.model },
                { title: "数据", render: (_, r) => r.config?.versionName ?? <span className="dim">当前数据</span> },
                { title: "轮数", render: (_, r) => r.config?.epochs },
                { title: mn.m50, dataIndex: "bestMap50", render: (v: number | null) => (v == null ? "—" : v.toFixed(3)) },
                { title: "状态", dataIndex: "status", render: (s: RunStatus) => <Tag color={RUN_STATUS[s].color}>{RUN_STATUS[s].text}</Tag> },
              ]}
            />
          </div>
        </div>
      </div>
      <RunCompare
        open={comparing}
        task={task}
        runs={runs.filter((r) => compareIds.includes(r.id))}
        onClose={() => setComparing(false)}
      />
    </div>
  );
}
