import { useEffect, useMemo, useState } from "react";
import { App, Button, Checkbox, Empty, InputNumber, Modal, Progress, Segmented, Select, Space, Tag } from "antd";
import { listen } from "@tauri-apps/api/event";
import { api, errText, ImportProgress, ProjectInfo, QualityReport, RunInfo, thumbUrl } from "../api";
import { useApp } from "../store";
import { modelName } from "../pages/Train";

const ISSUE: Record<string, { text: string; color: string }> = {
  blur: { text: "模糊", color: "orange" },
  dark: { text: "过暗", color: "purple" },
  bright: { text: "过曝", color: "gold" },
  flat: { text: "对比度低", color: "default" },
};

/** 图片质量筛查：找出模糊、过暗、过曝、对比度低的图，批量跳过或删除 */
export function QualityModal({ project, open, onClose, onChanged }: { project: ProjectInfo; open: boolean; onClose: () => void; onChanged: () => void }) {
  const { message, modal } = App.useApp();
  const annotateQueue = useApp((s) => s.annotateQueue);
  const refreshStats = useApp((s) => s.refreshStats);
  const [onlyTodo, setOnlyTodo] = useState(true);
  const [report, setReport] = useState<QualityReport | null>(null);
  const [progress, setProgress] = useState<ImportProgress | null>(null);
  const [kind, setKind] = useState("all");
  const [picked, setPicked] = useState<Set<number>>(new Set());

  useEffect(() => {
    if (open) {
      setReport(null);
      setPicked(new Set());
    }
  }, [open]);

  const run = async () => {
    setProgress({ done: 0, total: 0 });
    const un = await listen<ImportProgress>("quality-progress", (e) => setProgress(e.payload));
    try {
      const r = await api.qualityScan(onlyTodo);
      setReport(r);
      setPicked(new Set(r.items.map((i) => i.imageId)));
    } catch (e) {
      message.error(errText(e));
    } finally {
      un();
      setProgress(null);
    }
  };

  const shown = useMemo(() => (report?.items ?? []).filter((i) => kind === "all" || i.issues.includes(kind)), [report, kind]);
  const ids = shown.map((i) => i.imageId).filter((id) => picked.has(id));

  const skip = async () => {
    try {
      for (const id of ids) await api.setImageStatus(id, "skipped");
      message.success(`已把 ${ids.length} 张设为「跳过」，不参与训练`);
      await refreshStats();
      onChanged();
      setReport((r) => (r ? { ...r, items: r.items.filter((i) => !ids.includes(i.imageId)) } : r));
    } catch (e) {
      message.error(errText(e));
    }
  };

  const remove = () =>
    modal.confirm({
      title: `删除这 ${ids.length} 张图？`,
      content: "图片的标注会一起删除，无法恢复。复制进项目的图片副本会删掉，你电脑上的原图不受影响。",
      okText: "删除",
      okButtonProps: { danger: true },
      cancelText: "取消",
      onOk: async () => {
        try {
          await api.deleteImages("ids", ids);
          message.success(`已删除 ${ids.length} 张`);
          await refreshStats();
          onChanged();
          setReport((r) => (r ? { ...r, items: r.items.filter((i) => !ids.includes(i.imageId)) } : r));
        } catch (e) {
          message.error(errText(e));
        }
      },
    });

  return (
    <Modal title="图片质量筛查" open={open} onCancel={onClose} footer={null} width={900} destroyOnHidden>
      <p className="muted" style={{ marginTop: 0 }}>
        找出模糊、过暗、过曝、对比度太低的图。这些图标起来费劲、训练时也容易带偏模型，可以先跳过或删掉；也可以先过一眼再决定。
      </p>
      <Space style={{ marginBottom: 10 }}>
        <Checkbox checked={onlyTodo} onChange={(e) => setOnlyTodo(e.target.checked)}>
          只看未标注的图
        </Checkbox>
        <Button type="primary" onClick={run} loading={!!progress}>
          {report ? "重新筛查" : "开始筛查"}
        </Button>
      </Space>
      {progress && <Progress percent={progress.total ? Math.floor((progress.done / progress.total) * 100) : 0} size="small" />}
      {report && (
        <>
          <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
            <Segmented
              size="small"
              value={kind}
              onChange={(v) => setKind(v as string)}
              options={[
                { value: "all", label: `全部 ${report.items.length}` },
                ...Object.entries(ISSUE).map(([k, v]) => ({ value: k, label: `${v.text} ${report.items.filter((i) => i.issues.includes(k)).length}` })),
              ]}
            />
            <span className="dim" style={{ fontSize: 12 }}>
              查了 {report.checked} 张{report.failed ? `，${report.failed} 张读不出来` : ""}
            </span>
          </div>
          {shown.length === 0 ? (
            <Empty description="没有发现质量问题" />
          ) : (
            <>
              <div className="row" style={{ gap: 6, marginBottom: 8 }}>
                <span className="dim" style={{ fontSize: 12 }}>已选 {ids.length} 张（点图片取消选择）</span>
                <div className="spacer" />
                <Button size="small" onClick={() => annotateQueue(shown.map((i) => i.imageId), "质量问题")}>
                  逐张去看
                </Button>
                <Button size="small" disabled={!ids.length} onClick={skip}>
                  设为跳过
                </Button>
                <Button size="small" danger disabled={!ids.length} onClick={remove}>
                  删除
                </Button>
              </div>
              <div className="thumbs" style={{ maxHeight: 420, overflow: "auto" }}>
                {shown.map((it) => {
                  const on = picked.has(it.imageId);
                  return (
                    <div
                      key={it.imageId}
                      className="thumb"
                      style={{ opacity: on ? 1 : 0.4 }}
                      onClick={() =>
                        setPicked((s) => {
                          const n = new Set(s);
                          if (n.has(it.imageId)) n.delete(it.imageId);
                          else n.add(it.imageId);
                          return n;
                        })
                      }
                      title={`清晰度 ${it.sharpness.toFixed(0)} · 亮度 ${it.brightness.toFixed(0)} · 过曝 ${(it.overexposed * 100).toFixed(0)}% · 对比度 ${it.contrast.toFixed(0)}`}
                    >
                      <img src={thumbUrl(project.path, it.imageId)} loading="lazy" alt="" draggable={false} />
                      <span className="nm">{it.issues.map((k) => ISSUE[k]?.text ?? k).join("、")}</span>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </>
      )}
    </Modal>
  );
}

/** 主动学习挑图：用训练好的模型把未标注的图看一遍，挑出它最拿不准的，先标这些对模型提升最大 */
export function PickModal({ project, open, onClose }: { project: ProjectInfo; open: boolean; onClose: () => void }) {
  const { message } = App.useApp();
  const annotateQueue = useApp((s) => s.annotateQueue);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [runId, setRunId] = useState<number | null>(null);
  const [limit, setLimit] = useState(100);
  const [items, setItems] = useState<{ imageId: number; score: number; detail: string }[] | null>(null);
  const [progress, setProgress] = useState<ImportProgress | null>(null);

  useEffect(() => {
    if (!open) return;
    setItems(null);
    api
      .listRuns()
      .then((rs) => {
        const ok = rs.filter((r) => r.hasWeights);
        setRuns(ok);
        setRunId((cur) => (cur != null && ok.some((r) => r.id === cur) ? cur : ok[0]?.id ?? null));
      })
      .catch(() => {});
  }, [open]);

  const run = async () => {
    if (runId == null) return;
    setProgress({ done: 0, total: 0 });
    const un = await listen<ImportProgress>("prelabel-progress", (e) => setProgress(e.payload));
    try {
      setItems(await api.uncertainImages(runId, limit));
    } catch (e) {
      message.error(errText(e));
    } finally {
      un();
      setProgress(null);
    }
  };

  return (
    <Modal title="挑选最值得标的图" open={open} onCancel={onClose} footer={null} width={900} destroyOnHidden>
      <p className="muted" style={{ marginTop: 0 }}>
        用训练好的模型把「未标注」的图都看一遍，挑出模型最拿不准的。先标这些图，比随便标同样多的图对模型的提升大得多。
      </p>
      {runs.length === 0 ? (
        <Empty description="还没有训练好的模型，先训练一个再来挑" />
      ) : (
        <Space style={{ marginBottom: 10 }} wrap>
          <Select
            style={{ width: 360 }}
            value={runId}
            onChange={setRunId}
            options={runs.map((r) => ({ value: r.id, label: `${r.name} · ${modelName(r)}` }))}
          />
          <span>
            挑 <InputNumber min={10} max={2000} step={50} value={limit} onChange={(v) => setLimit(Number(v ?? 100))} /> 张
          </span>
          <Button type="primary" onClick={run} loading={!!progress}>
            开始挑选
          </Button>
        </Space>
      )}
      {progress && <Progress percent={progress.total ? Math.floor((progress.done / progress.total) * 100) : 0} size="small" />}
      {items && (
        <>
          <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
            <span className="dim" style={{ fontSize: 12 }}>按「拿不准」程度从高到低</span>
            <Button type="primary" size="small" disabled={!items.length} onClick={() => annotateQueue(items.map((i) => i.imageId), "模型挑选")}>
              按这个顺序去标注
            </Button>
          </div>
          {items.length === 0 ? (
            <Empty description="没有未标注的图" />
          ) : (
            <div className="thumbs" style={{ maxHeight: 420, overflow: "auto" }}>
              {items.map((it, i) => (
                <div key={it.imageId} className="thumb" title={it.detail}>
                  <img src={thumbUrl(project.path, it.imageId)} loading="lazy" alt="" draggable={false} />
                  <span className="nm">
                    {i + 1}. <Tag style={{ marginInlineEnd: 0 }}>{it.score.toFixed(2)}</Tag>
                  </span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </Modal>
  );
}
