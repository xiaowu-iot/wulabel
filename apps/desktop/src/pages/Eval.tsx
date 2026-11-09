import { useEffect, useMemo, useState } from "react";
import { App, Empty, Segmented, Select, Table, Tooltip } from "antd";
import { api, ClassEval, errText, ProjectInfo, RunDetail, RunInfo, thumbUrl } from "../api";
import { useApp } from "../store";
import { fmtTime, MODELS } from "./Train";

const pct = (v: number | null | undefined) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);

export default function Eval({ project }: { project: ProjectInfo }) {
  const { message } = App.useApp();
  const annotateImage = useApp((s) => s.annotateImage);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [sel, setSel] = useState<number | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [split, setSplit] = useState("val");
  const [errKind, setErrKind] = useState<"all" | "missed" | "extra" | "wrong">("all");

  useEffect(() => {
    api
      .listRuns()
      .then((rs) => {
        const ok = rs.filter((r) => r.hasWeights);
        setRuns(ok);
        setSel(ok[0]?.id ?? null);
      })
      .catch((e) => message.error(errText(e)));
  }, [message]);

  useEffect(() => {
    if (sel == null) return;
    api
      .runDetail(sel)
      .then((d) => {
        setDetail(d);
        const splits = Object.keys(d.eval?.splits ?? {});
        setSplit(splits.includes("val") ? "val" : splits[0] ?? "val");
      })
      .catch((e) => message.error(errText(e)));
  }, [sel, message]);

  const ev = detail?.eval?.splits?.[split];
  const classes = detail?.eval?.classes ?? [];
  const errors = useMemo(
    () => (ev?.errors ?? []).filter((e) => errKind === "all" || e[errKind] > 0).slice(0, 60),
    [ev, errKind],
  );

  if (runs.length === 0) {
    return (
      <div className="empty-page">
        <Empty description="还没有训练好的模型，先去「训练」页训练一个" />
      </div>
    );
  }

  const cmMax = Math.max(1, ...(ev?.confusion ?? []).flat());
  const names = [...classes, "背景"];

  return (
    <div className="page">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 14 }}>
        <div>
          <h1>评估</h1>
          <p className="desc" style={{ margin: 0 }}>
            用第 {detail?.eval?.bestEpoch ?? "—"} 轮的最好模型在{split === "val" ? "验证集" : "测试集"}上评估（{ev?.images ?? 0} 张）。置信度阈值 0.25，IoU 0.5。
          </p>
        </div>
        <div className="row">
          {Object.keys(detail?.eval?.splits ?? {}).length > 1 && (
            <Segmented value={split} onChange={(v) => setSplit(v as string)} options={Object.keys(detail!.eval!.splits).map((s) => ({ value: s, label: s === "val" ? "验证集" : "测试集" }))} />
          )}
          <Select
            style={{ width: 340 }}
            value={sel}
            onChange={setSel}
            options={runs.map((r) => ({
              value: r.id,
              label: `${fmtTime(r.createdAt)} · ${MODELS.find((m) => m.value === r.config?.model)?.name ?? r.config?.model} · mAP50 ${r.bestMap50?.toFixed(3) ?? "—"}`,
            }))}
          />
        </div>
      </div>

      {!ev ? (
        <div className="card" style={{ padding: 48 }}>
          <Empty description="这次训练没有评估结果（可能是中途停止的）" />
        </div>
      ) : (
        <>
          <div className="cards" style={{ gridTemplateColumns: "repeat(4, 1fr)", marginBottom: 12 }}>
            <Tooltip title="IoU 0.5 时各类别平均精度的均值，最常用的检测指标">
              <div className="card"><div className="muted">mAP50</div><div className="kpi">{ev.map50.toFixed(3)}</div></div>
            </Tooltip>
            <Tooltip title="IoU 0.5~0.95 的平均，对框的准确度要求更高">
              <div className="card"><div className="muted">mAP50-95</div><div className="kpi">{ev.map.toFixed(3)}</div></div>
            </Tooltip>
            <Tooltip title="模型报出来的框里，正确的比例。低说明误报多">
              <div className="card"><div className="muted">精确率</div><div className="kpi">{pct(ev.precision)}</div></div>
            </Tooltip>
            <Tooltip title="真实目标里，被找到的比例。低说明漏检多">
              <div className="card"><div className="muted">召回率</div><div className="kpi">{pct(ev.recall)}</div></div>
            </Tooltip>
          </div>

          <div className="cards" style={{ gridTemplateColumns: "1fr 1fr", marginBottom: 12 }}>
            <div className="card">
              <div className="muted" style={{ marginBottom: 8 }}>各类别</div>
              <Table<ClassEval>
                size="small"
                rowKey="index"
                pagination={false}
                dataSource={ev.perClass}
                columns={[
                  { title: "类别", dataIndex: "name" },
                  { title: "真实框", dataIndex: "gt" },
                  { title: "精确率", dataIndex: "precision", render: pct },
                  {
                    title: "召回率",
                    dataIndex: "recall",
                    render: (v: number) => <span style={{ color: v < 0.6 ? "var(--bad)" : undefined }}>{pct(v)}</span>,
                  },
                  {
                    title: "AP50",
                    dataIndex: "ap50",
                    render: (v: number | null) => <span style={{ color: v != null && v < 0.6 ? "var(--bad)" : undefined }}>{v == null ? "—" : v.toFixed(3)}</span>,
                  },
                ]}
              />
            </div>
            <div className="card" style={{ overflow: "auto" }}>
              <div className="muted" style={{ marginBottom: 8 }}>混淆矩阵 <span className="dim">（行：真实，列：预测）</span></div>
              <table className="cm">
                <thead>
                  <tr>
                    <th />
                    {names.map((n) => <th key={n}>{n}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {ev.confusion.map((row, i) => (
                    <tr key={i}>
                      <th>{names[i]}</th>
                      {row.map((v, j) => {
                        const a = v / cmMax;
                        const bg = v === 0 ? "transparent" : i === j ? `rgba(47,182,124,${0.15 + a * 0.75})` : `rgba(229,83,75,${0.15 + a * 0.75})`;
                        return (
                          <td key={j} style={{ background: bg }} title={`真实「${names[i]}」→ 预测「${names[j]}」：${v}`}>
                            {v || ""}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="dim" style={{ fontSize: 12, marginTop: 8 }}>
                最后一列「背景」是漏检，最后一行「背景」是误报。
              </div>
            </div>
          </div>

          <div className="card">
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 10 }}>
              <div className="muted">错例 <span className="dim">· 点图片去标注页检查和修改</span></div>
              <Segmented
                size="small"
                value={errKind}
                onChange={(v) => setErrKind(v as typeof errKind)}
                options={[
                  { value: "all", label: `全部 ${ev.errors.length}` },
                  { value: "missed", label: `有漏检 ${ev.errors.filter((e) => e.missed).length}` },
                  { value: "extra", label: `有误报 ${ev.errors.filter((e) => e.extra).length}` },
                  { value: "wrong", label: `类别错 ${ev.errors.filter((e) => e.wrong).length}` },
                ]}
              />
            </div>
            {errors.length === 0 ? (
              <Empty description="没有错例" />
            ) : (
              <div className="thumbs">
                {errors.map((e) => (
                  <div key={e.imageId} className="thumb" onClick={() => annotateImage(e.imageId)}>
                    <img src={thumbUrl(project.path, e.imageId)} loading="lazy" alt="" draggable={false} />
                    <span className="nm">
                      {e.missed > 0 && `漏 ${e.missed} `}
                      {e.extra > 0 && `误 ${e.extra} `}
                      {e.wrong > 0 && `错 ${e.wrong}`}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
