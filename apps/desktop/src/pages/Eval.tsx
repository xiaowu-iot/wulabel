import { useCallback, useEffect, useMemo, useState } from "react";
import { App, Button, Empty, Segmented, Select, Table, Tooltip } from "antd";
import { ReloadOutlined } from "@ant-design/icons";
import { listen } from "@tauri-apps/api/event";
import { api, ClassEval, errText, ProjectInfo, RunDetail, RunInfo, thumbUrl } from "../api";
import { useApp } from "../store";
import { fmtTime, metricNames, modelName } from "./Train";
import { FpTestPanel, ThresholdPanel } from "../components/EvalTools";

const pct = (v: number | null | undefined) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);

export default function Eval({ project }: { project: ProjectInfo }) {
  const { message } = App.useApp();
  const isCls = project.task === "classify";
  const mn = metricNames(project.task);
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

  const loadDetail = useCallback(
    (id: number) =>
      api
        .runDetail(id)
        .then((d) => {
          setDetail(d);
          const splits = Object.keys(d.eval?.splits ?? {});
          setSplit((cur) => (splits.includes(cur) ? cur : splits.includes("val") ? "val" : splits[0] ?? "val"));
        })
        .catch((e) => message.error(errText(e))),
    [message],
  );

  useEffect(() => {
    if (sel != null) loadDetail(sel);
  }, [sel, loadDetail]);

  // 用现在的数据重新评估（导入的外部模型第一次评估也走这里）
  const [evaluating, setEvaluating] = useState(false);
  const [evalLog, setEvalLog] = useState("");
  const reevaluate = async () => {
    if (sel == null) return;
    setEvaluating(true);
    setEvalLog("");
    const un = await listen<string>("eval-log", (e) => setEvalLog(e.payload));
    try {
      await api.evaluateRun(sel);
      await loadDetail(sel);
      setRuns(await api.listRuns().then((rs) => rs.filter((r) => r.hasWeights)));
      message.success("评估完成");
    } catch (e) {
      message.error(errText(e));
    } finally {
      un();
      setEvaluating(false);
    }
  };

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
  // 分类的混淆矩阵没有「背景」行列
  const names = isCls ? classes : [...classes, "背景"];

  return (
    <div className="page">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 14 }}>
        <div>
          <h1>评估</h1>
          <p className="desc" style={{ margin: 0 }}>
            {detail?.eval?.reevaluated
              ? `用 ${detail.eval.evaluatedAt ? fmtTime(detail.eval.evaluatedAt) : ""} 时的数据重新评估，`
              : `用第 ${detail?.eval?.bestEpoch ?? "—"} 轮的最好模型，`}
            在{split === "val" ? "验证集" : "测试集"}上（{ev?.images ?? 0} 张）。
            {isCls ? "每张图取模型分数最高的类别。" : "置信度阈值 0.25，IoU 0.5。"}
            {(project.task === "segment" || project.task === "pose") && "分割和关键点模型这里按目标框计算指标。"}
          </p>
        </div>
        <div className="row">
          <Tooltip title="用现在的已标注数据（验证集 / 测试集）重新算一遍指标。补了数据、或者导入的外部模型，都用这个">
            <Button icon={<ReloadOutlined />} loading={evaluating} onClick={reevaluate}>
              用当前数据评估
            </Button>
          </Tooltip>
          {Object.keys(detail?.eval?.splits ?? {}).length > 1 && (
            <Segmented value={split} onChange={(v) => setSplit(v as string)} options={Object.keys(detail!.eval!.splits).map((s) => ({ value: s, label: s === "val" ? "验证集" : "测试集" }))} />
          )}
          <Select
            style={{ width: 340 }}
            value={sel}
            onChange={setSel}
            options={runs.map((r) => ({
              value: r.id,
              label: `${fmtTime(r.createdAt)} · ${modelName(r)} · ${mn.m50} ${r.bestMap50?.toFixed(3) ?? "—"}`,
            }))}
          />
        </div>
      </div>

      {evaluating && evalLog && <div className="dim" style={{ fontSize: 12, marginBottom: 8 }}>{evalLog}</div>}
      {!ev ? (
        <div className="card" style={{ padding: 48 }}>
          <Empty
            description={
              detail?.run.engine === "external"
                ? "导入的外部模型还没有评估，点「用当前数据评估」算一下它在你数据上的表现"
                : "这次训练没有评估结果（可能是中途停止的），可以点「用当前数据评估」"
            }
          />
        </div>
      ) : (
        <>
          <div className="cards" style={{ gridTemplateColumns: "repeat(4, 1fr)", marginBottom: 12 }}>
            <Tooltip title={isCls ? "分对的图片占全部图片的比例" : "IoU 0.5 时各类别平均精度的均值，最常用的检测指标"}>
              <div className="card">
                <div className="muted">{mn.m50}</div>
                <div className="kpi">{isCls ? pct(ev.map50) : ev.map50.toFixed(3)}</div>
              </div>
            </Tooltip>
            <Tooltip title={isCls ? "各类别 F1 的平均，类别不均衡时比准确率更能反映效果" : "IoU 0.5~0.95 的平均，对框的准确度要求更高"}>
              <div className="card"><div className="muted">{mn.m}</div><div className="kpi">{ev.map.toFixed(3)}</div></div>
            </Tooltip>
            <Tooltip title={isCls ? "各类别精确率的平均：判成某类的图里，真是这一类的比例" : "模型报出来的框里，正确的比例。低说明误报多"}>
              <div className="card"><div className="muted">{isCls ? "平均精确率" : "精确率"}</div><div className="kpi">{pct(ev.precision)}</div></div>
            </Tooltip>
            <Tooltip title={isCls ? "各类别召回率的平均：某类的图里，被判对的比例" : "真实目标里，被找到的比例。低说明漏检多"}>
              <div className="card"><div className="muted">{isCls ? "平均召回率" : "召回率"}</div><div className="kpi">{pct(ev.recall)}</div></div>
            </Tooltip>
          </div>

          {ev.thresholds ? (
            <ThresholdPanel th={ev.thresholds} isCls={isCls} />
          ) : (
            <div className="dim" style={{ fontSize: 12, marginBottom: 12 }}>
              这次评估没有阈值曲线（旧版本训练的），点「用当前数据评估」就能算出推荐阈值。
            </div>
          )}

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
                  { title: isCls ? "图片数" : "真实框", dataIndex: "gt" },
                  { title: "精确率", dataIndex: "precision", render: pct },
                  {
                    title: "召回率",
                    dataIndex: "recall",
                    render: (v: number) => <span style={{ color: v < 0.6 ? "var(--bad)" : undefined }}>{pct(v)}</span>,
                  },
                  {
                    title: isCls ? "F1" : "AP50",
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
                {isCls ? "对角线是分对的，其他格子是把某类误判成另一类。" : "最后一列「背景」是漏检，最后一行「背景」是误报。"}
              </div>
            </div>
          </div>

          <div className="card">
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 10 }}>
              <div className="muted">错例 <span className="dim">· 点图片去标注页检查和修改</span></div>
              {isCls ? (
                <span className="dim" style={{ fontSize: 12 }}>分错 {ev.errors.length} 张，按模型把握从高到低（越靠前越可能是标错了）</span>
              ) : (
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
              )}
            </div>
            {errors.length === 0 ? (
              <Empty description="没有错例" />
            ) : (
              <div className="thumbs">
                {errors.map((e) => (
                  <div key={e.imageId} className="thumb" onClick={() => annotateImage(e.imageId)}>
                    <img src={thumbUrl(project.path, e.imageId)} loading="lazy" alt="" draggable={false} />
                    <span className="nm">
                      {isCls && e.gt != null && e.pred != null ? (
                        `标「${classes[e.gt] ?? e.gt}」→ 判「${classes[e.pred] ?? e.pred}」`
                      ) : (
                        <>
                          {e.missed > 0 && `漏 ${e.missed} `}
                          {e.extra > 0 && `误 ${e.extra} `}
                          {e.wrong > 0 && `错 ${e.wrong}`}
                        </>
                      )}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
      {detail && !isCls && <FpTestPanel run={detail.run} defaultConf={ev?.thresholds?.best.conf ?? 0.5} />}
    </div>
  );
}
