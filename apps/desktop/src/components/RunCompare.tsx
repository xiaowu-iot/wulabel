import { useEffect, useState } from "react";
import { App, Empty, Modal, Spin, Table } from "antd";
import { api, errText, RunDetail, RunInfo, Task } from "../api";
import LineChart from "./LineChart";
import { fmtTime, metricNames, modelName } from "../pages/Train";

const COLORS = ["#3d8bfd", "#e5534b", "#2fb67c", "#e7a33e", "#9b7bf2", "#14b8a6", "#f472b6", "#a3e635"];

const fmtDur = (s: number | null | undefined) => {
  if (s == null || !Number.isFinite(s)) return "—";
  if (s < 60) return `${Math.round(s)} 秒`;
  if (s < 3600) return `${Math.floor(s / 60)} 分`;
  return `${Math.floor(s / 3600)} 时 ${Math.round((s % 3600) / 60)} 分`;
};

/** 训练对比：参数、数据、指标曲线、各类别结果放在一起看 */
export default function RunCompare({ runs, task, open, onClose }: { runs: RunInfo[]; task: Task; open: boolean; onClose: () => void }) {
  const { message } = App.useApp();
  const [details, setDetails] = useState<RunDetail[] | null>(null);
  const mn = metricNames(task);

  useEffect(() => {
    if (!open || runs.length === 0) return;
    setDetails(null);
    Promise.all(runs.map((r) => api.runDetail(r.id)))
      .then(setDetails)
      .catch((e) => message.error(errText(e)));
  }, [open, runs, message]);

  const color = (i: number) => COLORS[i % COLORS.length];
  const label = (r: RunInfo) => r.name;
  const fmt = (v: number | null | undefined, pct = false) => (v == null ? "—" : pct ? `${(v * 100).toFixed(1)}%` : v.toFixed(3));

  // 参数对比：一行一个参数，一列一次训练；各次不一样的参数加粗
  const rows: { key: string; name: string; values: string[] }[] = details
    ? [
        { key: "model", name: "模型", values: details.map((d) => modelName(d.run)) },
        { key: "data", name: "训练数据", values: details.map((d) => d.run.config?.versionName ?? "当前数据") },
        {
          key: "images",
          name: "图片（训练 / 验证）",
          values: details.map((d) => (d.run.config?.images ? `${d.run.config.images.train} / ${d.run.config.images.val}` : "—")),
        },
        { key: "anns", name: "标注数", values: details.map((d) => String(d.run.config?.images?.anns ?? d.run.config?.images?.boxes ?? "—")) },
        { key: "epochs", name: "轮数", values: details.map((d) => String(d.run.config?.epochs ?? "—")) },
        { key: "size", name: "输入尺寸", values: details.map((d) => String(d.run.config?.imgSize ?? "—")) },
        { key: "batch", name: "批大小", values: details.map((d) => String(d.run.config?.batch ?? "—")) },
        { key: "lr", name: "学习率", values: details.map((d) => String(d.run.config?.lr ?? "—")) },
        { key: "best", name: `最好的${mn.m50}`, values: details.map((d) => fmt(d.run.bestMap50, task === "classify")) },
        {
          key: "bestm",
          name: `对应的${mn.m}`,
          values: details.map((d) => {
            const best = d.metrics.reduce<typeof d.metrics[number] | null>((a, m) => (a == null || m.map50 > a.map50 ? m : a), null);
            return fmt(best?.map);
          }),
        },
        { key: "time", name: "用时", values: details.map((d) => fmtDur(d.metrics[d.metrics.length - 1]?.time)) },
        { key: "at", name: "时间", values: details.map((d) => fmtTime(d.run.createdAt)) },
      ]
    : [];

  // 各类别：验证集上的 AP50（分类是 F1）
  const classes = details ? Array.from(new Set(details.flatMap((d) => d.eval?.classes ?? []))) : [];
  const perClass = classes.map((name) => {
    const vals = (details ?? []).map((d) => {
      const sp = d.eval?.splits?.val ?? Object.values(d.eval?.splits ?? {})[0];
      return sp?.perClass.find((c) => c.name === name)?.ap50 ?? null;
    });
    const best = Math.max(...vals.map((v) => v ?? -1));
    return { name, vals, best };
  });
  const maxEpochs = details ? Math.max(1, ...details.map((d) => d.run.config?.epochs ?? d.metrics.length)) : 1;

  return (
    <Modal title={`训练对比（${runs.length} 次）`} open={open} onCancel={onClose} footer={null} width={1000} destroyOnHidden>
      {!details ? (
        <div style={{ padding: 48, textAlign: "center" }}>
          <Spin />
        </div>
      ) : (
        <>
          <div className="row" style={{ gap: 14, flexWrap: "wrap", marginBottom: 10 }}>
            {details.map((d, i) => (
              <span key={d.run.id} className="row" style={{ gap: 6 }}>
                <span style={{ width: 12, height: 12, borderRadius: 3, background: color(i), display: "inline-block" }} />
                {label(d.run)}
              </span>
            ))}
          </div>
          <Table
            size="small"
            rowKey="key"
            pagination={false}
            dataSource={rows}
            columns={[
              { title: "", dataIndex: "name", width: 150 },
              ...details.map((d, i) => ({
                title: <span style={{ color: color(i) }}>{label(d.run)}</span>,
                key: String(d.run.id),
                render: (_: unknown, r: (typeof rows)[number]) => {
                  const differs = new Set(r.values).size > 1 && !["at", "time"].includes(r.key);
                  return <span style={{ fontWeight: differs ? 600 : undefined }}>{r.values[i]}</span>;
                },
              })),
            ]}
          />
          <div className="dim" style={{ fontSize: 12, margin: "6px 0 14px" }}>加粗的是这几次训练不一样的地方。</div>

          <div className="cards" style={{ gridTemplateColumns: "1fr 1fr", marginBottom: 14 }}>
            <div className="card">
              <div className="muted">验证集 {mn.m50}</div>
              <LineChart
                xMax={maxEpochs}
                yMin={0}
                yMax={1}
                series={details.map((d, i) => ({ name: label(d.run), color: color(i), values: d.metrics.map((m) => m.map50) }))}
              />
            </div>
            <div className="card">
              <div className="muted">训练损失 loss</div>
              <LineChart
                xMax={maxEpochs}
                yMin={0}
                series={details.map((d, i) => ({ name: label(d.run), color: color(i), values: d.metrics.map((m) => m.loss) }))}
              />
            </div>
          </div>

          <div className="muted" style={{ marginBottom: 6 }}>各类别（验证集 {task === "classify" ? "F1" : "AP50"}，每行最好的标绿）</div>
          {perClass.length === 0 ? (
            <Empty description="这几次训练还没有评估结果" />
          ) : (
            <Table
              size="small"
              rowKey="name"
              pagination={false}
              dataSource={perClass}
              columns={[
                { title: "类别", dataIndex: "name", width: 150 },
                ...details.map((d, i) => ({
                  title: <span style={{ color: color(i) }}>{label(d.run)}</span>,
                  key: String(d.run.id),
                  render: (_: unknown, r: (typeof perClass)[number]) => {
                    const v = r.vals[i];
                    const top = v != null && v === r.best && perClass.length > 0 && details.length > 1;
                    return <span style={{ color: top ? "var(--good, #2fb67c)" : undefined, fontWeight: top ? 600 : undefined }}>{fmt(v)}</span>;
                  },
                })),
              ]}
            />
          )}
        </>
      )}
    </Modal>
  );
}
