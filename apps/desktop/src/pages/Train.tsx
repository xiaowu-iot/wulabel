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
import { api, EnvStatus, errText, RunDetail, RunInfo, RunStatus, SplitSummary, TrainConfig } from "../api";
import { prefs, useApp } from "../store";
import LineChart from "../components/LineChart";

export const MODELS = [
  { value: "frcnn_mobilenet", name: "Faster R-CNN MobileNetV3", desc: "速度快、显存占用小，适合入门和没有独立显卡的电脑" },
  { value: "frcnn_resnet50", name: "Faster R-CNN ResNet50 v2", desc: "精度更高，训练和推理更慢，建议 6GB 以上显存" },
];

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

export default function Train() {
  const { message, modal } = App.useApp();
  const setPage = useApp((s) => s.setPage);
  const [env, setEnv] = useState<EnvStatus | null>(null);
  const [split, setSplit] = useState<SplitSummary | null>(null);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [sel, setSel] = useState<number | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [prog, setProg] = useState<ProgressEv | null>(null);
  const [starting, setStarting] = useState(false);
  const [exporting, setExporting] = useState(false);
  const selRef = useRef<number | null>(null);
  selRef.current = sel;
  const logBox = useRef<HTMLDivElement>(null);

  const [cfg, setCfg] = useState<TrainConfig>(() => {
    const saved = prefs.get("trainConfig", "");
    const base: TrainConfig = { model: "frcnn_mobilenet", epochs: 30, imgSize: 640, batch: 4, lr: 0.01, device: "auto", onlyReviewed: false, ratios: [0.8, 0.2, 0] };
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
    prefs.set("trainConfig", JSON.stringify(next));
  };

  const start = async () => {
    setStarting(true);
    try {
      const id = await api.startTraining(cfg);
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
          <p className="desc">用已标注和已审核的图片训练检测模型。</p>
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
            {trainable > 0 && trainable < 50 && <div style={{ color: "var(--warn)", fontSize: 12, marginTop: 6 }}>图片较少，建议每个类别至少标 100 个框再训练。</div>}
          </div>
          <div className="card">
            <Form layout="vertical" size="small">
              <Form.Item label="模型">
                <Radio.Group value={cfg.model} onChange={(e) => update({ model: e.target.value })}>
                  <Space direction="vertical">
                    {MODELS.map((m) => (
                      <Radio key={m.value} value={m.value}>
                        <b>{m.name}</b>
                        {m.value === (gpuMem >= 6000 ? "frcnn_resnet50" : "frcnn_mobilenet") && <Tag color="blue" style={{ marginLeft: 6 }}>推荐</Tag>}
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
                <Form.Item label="输入尺寸" tooltip="目标小（远处的人、安全帽）用大尺寸，速度会变慢">
                  <Select
                    style={{ width: 100 }}
                    value={cfg.imgSize}
                    onChange={(v) => update({ imgSize: v })}
                    options={[320, 480, 640, 800, 960, 1280].map((v) => ({ value: v, label: v }))}
                  />
                </Form.Item>
                <Form.Item label="批大小" tooltip="显存不够报错时调小">
                  <InputNumber min={1} max={64} value={cfg.batch} onChange={(v) => update({ batch: Number(v ?? 4) })} />
                </Form.Item>
              </Space>
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
              <Button type="primary" block icon={<CaretRightOutlined />} loading={starting} disabled={running || trainable === 0 || (env != null && !env.ready)} onClick={start}>
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
                      {run.config?.images ? `训练 ${run.config.images.train} 张 / 验证 ${run.config.images.val} 张 / ${run.config.images.boxes} 个框` : ""}
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
                      验证集 <span style={{ color: "#2fb67c" }}>mAP50</span> / <span style={{ color: "#9b7bf2" }}>mAP50-95</span>
                      {run.bestMap50 != null && <span style={{ marginLeft: 8 }}>最好 {run.bestMap50.toFixed(3)}</span>}
                    </div>
                    <LineChart
                      xMax={epochs}
                      yMin={0}
                      yMax={1}
                      series={[
                        { name: "mAP50", color: "#2fb67c", values: metrics.map((m) => m.map50) },
                        { name: "mAP", color: "#9b7bf2", values: metrics.map((m) => m.map) },
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
            <div className="muted" style={{ marginBottom: 6 }}>训练记录</div>
            <Table<RunInfo>
              size="small"
              rowKey="id"
              dataSource={runs}
              pagination={false}
              onRow={(r) => ({ onClick: () => setSel(r.id), style: { cursor: "pointer", background: r.id === sel ? "rgba(61,139,253,.12)" : undefined } })}
              columns={[
                { title: "时间", dataIndex: "createdAt", render: fmtTime },
                { title: "模型", render: (_, r) => MODELS.find((m) => m.value === r.config?.model)?.name ?? r.config?.model },
                { title: "轮数", render: (_, r) => r.config?.epochs },
                { title: "mAP50", dataIndex: "bestMap50", render: (v: number | null) => (v == null ? "—" : v.toFixed(3)) },
                { title: "状态", dataIndex: "status", render: (s: RunStatus) => <Tag color={RUN_STATUS[s].color}>{RUN_STATUS[s].text}</Tag> },
              ]}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
