import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Alert, App, Button, Empty, Progress, Segmented, Select, Space, Table, Tooltip } from "antd";
import { FileTextOutlined, ReloadOutlined } from "@ant-design/icons";
import { listen } from "@tauri-apps/api/event";
import { save } from "@tauri-apps/plugin-dialog";
import { api, ClassEval, errText, EvalProgress, ProjectInfo, RunDetail, RunInfo, SplitEval, thumbUrl } from "../api";
import { Mark, useApp } from "../store";
import { fmtTime, metricNames, modelName, rateMetric } from "./Train";
import { FpTestPanel, ThresholdPanel } from "../components/EvalTools";
import { t } from "../i18n";

const pct = (v: number | null | undefined) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);

export default function Eval({ project }: { project: ProjectInfo }) {
  const { message } = App.useApp();
  const isCls = project.task === "classify";
  const mn = metricNames(project.task);
  const annotateQueue = useApp((s) => s.annotateQueue);
  const setPage = useApp((s) => s.setPage);
  const openRun = useApp((s) => s.openRun);
  const errBox = useRef<HTMLDivElement>(null);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [sel, setSel] = useState<number | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [split, setSplit] = useState("val");
  const [errKind, setErrKind] = useState<"all" | "missed" | "extra" | "wrong">("all");

  useEffect(() => {
    // 训练页点「看评估」带过来的那次训练优先；训练中的还没训完，不默认选
    const focus = useApp.getState().focusRunId;
    api
      .listRuns()
      .then((rs) => {
        const ok = rs.filter((r) => r.hasWeights);
        setRuns(ok);
        const ready = ok.filter((r) => r.status !== "running");
        setSel(ready.find((r) => r.id === focus)?.id ?? ready[0]?.id ?? null);
        useApp.getState().takeFocusRun();
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
  const [evalProg, setEvalProg] = useState<EvalProgress | null>(null);

  /** 导出交付报告（HTML，浏览器打开后可以打印成 PDF） */
  const exportReport = async () => {
    if (sel == null || !detail) return;
    const path = await save({
      title: t("保存交付报告"),
      defaultPath: `${project.name}_${detail.run.name}_${t("模型报告")}.html`,
      filters: [{ name: t("网页报告"), extensions: ["html"] }],
    });
    if (!path) return;
    try {
      await api.exportReport(sel, path);
      message.success({
        duration: 8,
        content: (
          <span>
            {t("报告已导出，用浏览器打开后可以「打印 → 另存为 PDF」。")}{" "}
            <a onClick={() => api.revealPath(path).catch(() => {})}>{t("打开所在文件夹")}</a>
          </span>
        ),
      });
    } catch (e) {
      message.error(errText(e));
    }
  };
  const reevaluate = async () => {
    if (sel == null) return;
    setEvaluating(true);
    setEvalLog("");
    setEvalProg(null);
    const un = await listen<string>("eval-log", (e) => setEvalLog(e.payload));
    const unProg = await listen<EvalProgress>("eval-progress", (e) => setEvalProg(e.payload));
    try {
      await api.evaluateRun(sel);
      await loadDetail(sel);
      setRuns(await api.listRuns().then((rs) => rs.filter((r) => r.hasWeights)));
      message.success(t("评估完成"));
    } catch (e) {
      message.error(errText(e));
    } finally {
      un();
      unProg();
      setEvaluating(false);
    }
  };
  /** 重新评估进度条下面的一行字：导出数据 → 加载模型 → 逐张评估 */
  const evalText = () => {
    const p = evalProg;
    if (!p) return evalLog || t("导出当前数据…");
    if (p.stage === "export") return t("正在整理评估数据 {done}/{total}", { done: p.done, total: p.total });
    if (p.stage === "load") return t("加载模型…");
    return p.stage === "test"
      ? t("正在评估测试集 {done}/{total} 张", { done: p.done, total: p.total })
      : t("正在评估验证集 {done}/{total} 张", { done: p.done, total: p.total });
  };

  const ev = detail?.eval?.splits?.[split];
  const classes = detail?.eval?.classes ?? [];
  const errors = useMemo(
    () => (ev?.errors ?? []).filter((e) => errKind === "all" || e[errKind] > 0).slice(0, 60),
    [ev, errKind],
  );

  /** 带着错例去标注页：只在这批图里翻，检测项目把对不上的框叠在图上 */
  const openErrors = (list: SplitEval["errors"], focus?: number) => {
    const name = (c: number | null) => (c == null ? "" : (classes[c] ?? String(c)));
    const marks: Record<number, Mark[]> = {};
    if (!isCls) {
      for (const e of list) {
        marks[e.imageId] = (e.boxes ?? []).map((b) =>
          b.k === "missed"
            ? { bbox: b.box, text: t("漏：{name}", { name: name(b.gt) }), color: "#e5534b" }
            : b.k === "extra"
              ? { bbox: b.box, text: t("误：{name} {score}", { name: name(b.pred), score: (b.score ?? 0).toFixed(2) }), color: "#e8590c" }
              : { bbox: b.box, text: t("错：标「{gt}」→ 判「{pred}」", { gt: name(b.gt), pred: name(b.pred) }), color: "#ae3ec9" },
        );
      }
    }
    annotateQueue(
      list.map((e) => e.imageId),
      t("评估错例"),
      { focus, marks: isCls ? undefined : marks },
    );
  };
  const filtered = (ev?.errors ?? []).filter((e) => errKind === "all" || e[errKind] > 0);
  // 以前的评估结果没有记下具体的框，要重新评估一次
  const noBoxes = !isCls && !!ev?.errors.length && !ev.errors.some((e) => e.boxes);

  // 只有正在训练的记录时也当成还没有模型
  if (!runs.some((r) => r.status !== "running")) {
    return (
      <div className="empty-page">
        <Empty description={t("还没有训练好的模型，先去「训练」页训练一个")} />
      </div>
    );
  }

  const cmMax = Math.max(1, ...(ev?.confusion ?? []).flat());
  // 分类的混淆矩阵没有「背景」行列
  const names = isCls ? classes : [...classes, t("背景")];
  // 一句话结论：按主指标打分，点出最差的类别
  const rate = ev ? rateMetric(project.task, ev.map50) : null;
  const worst = ev && ev.perClass.length > 1
    ? [...ev.perClass].filter((c) => c.gt > 0 && c.ap50 != null).sort((a, b) => (a.ap50 ?? 0) - (b.ap50 ?? 0))[0]
    : undefined;
  const showErrors = (kind: typeof errKind) => {
    setErrKind(kind);
    errBox.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  const descVars = {
    time: detail?.eval?.evaluatedAt ? fmtTime(detail.eval.evaluatedAt) : "",
    epoch: detail?.eval?.bestEpoch ?? "—",
    n: ev?.images ?? 0,
  };

  return (
    <div className="page">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 14 }}>
        <div>
          <h1>{t("评估")}</h1>
          <p className="desc" style={{ margin: 0 }}>
            {detail?.eval?.reevaluated
              ? split === "val"
                ? t("用 {time} 时的数据重新评估，在验证集上（{n} 张）。", descVars)
                : t("用 {time} 时的数据重新评估，在测试集上（{n} 张）。", descVars)
              : split === "val"
                ? t("用第 {epoch} 轮的最好模型，在验证集上（{n} 张）。", descVars)
                : t("用第 {epoch} 轮的最好模型，在测试集上（{n} 张）。", descVars)}
            {isCls ? t("每张图取模型分数最高的类别。") : t("置信度阈值 0.25，IoU 0.5。")}
            {(project.task === "segment" || project.task === "pose") && t("分割和关键点模型这里按目标框计算指标。")}
          </p>
        </div>
        <div className="row">
          <Tooltip title={t("用现在的已标注数据（验证集 / 测试集）重新算一遍指标。补了数据、或者导入的外部模型，都用这个")}>
            <Button icon={<ReloadOutlined />} loading={evaluating} onClick={reevaluate}>
              {t("用当前数据评估")}
            </Button>
          </Tooltip>
          <Tooltip title={t("把数据、训练过程、评估、推荐阈值、误报测试整理成一份报告，交给客户验收")}>
            <Button icon={<FileTextOutlined />} disabled={!detail?.eval} onClick={exportReport}>
              {t("导出报告")}
            </Button>
          </Tooltip>
          {Object.keys(detail?.eval?.splits ?? {}).length > 1 && (
            <Segmented value={split} onChange={(v) => setSplit(v as string)} options={Object.keys(detail!.eval!.splits).map((s) => ({ value: s, label: s === "val" ? t("验证集") : t("测试集") }))} />
          )}
          <Select
            style={{ width: 340 }}
            value={sel}
            onChange={setSel}
            options={runs.map((r) => ({
              value: r.id,
              disabled: r.status === "running",
              label: `${fmtTime(r.createdAt)} · ${modelName(r)} · ${r.status === "running" ? t("训练中") : `${mn.m50} ${r.bestMap50?.toFixed(3) ?? "—"}`}`,
            }))}
          />
        </div>
      </div>

      {evaluating && (
        <div style={{ marginBottom: 8 }}>
          <Progress
            size="small"
            percent={evalProg && evalProg.total ? Math.floor((evalProg.done / evalProg.total) * 100) : 0}
            status="active"
          />
          <div className="dim" style={{ fontSize: 12 }}>{evalText()}</div>
        </div>
      )}
      {!ev ? (
        <div className="card" style={{ padding: 48 }}>
          <Empty
            description={
              detail?.run.engine === "external"
                ? t("导入的外部模型还没有评估，点「用当前数据评估」算一下它在你数据上的表现")
                : t("这次训练没有评估结果（可能是中途停止的），可以点「用当前数据评估」")
            }
          />
        </div>
      ) : (
        <>
          {rate && (
            <Alert
              type={rate.level === "good" ? "success" : rate.level === "fair" ? "info" : "warning"}
              showIcon
              style={{ marginBottom: 12 }}
              message={
                <span>
                  {t("结论：{rate}（{metric} {v}）", { rate: rate.word, metric: mn.m50, v: isCls ? pct(ev.map50) : ev.map50.toFixed(3) })}
                  {ev.images < 10 && ` · ${split === "val" ? t("验证集太少，指标波动大") : t("测试集太少，指标波动大")}`}
                  {worst && worst.ap50 != null && ` · ${t("最差的类别「{name}」（{metric} {v}）", { name: worst.name, metric: isCls ? "F1" : "AP50", v: worst.ap50.toFixed(3) })}`}
                </span>
              }
              action={
                <Space size={6} wrap>
                  {ev.errors.length > 0 && (
                    <Button size="small" onClick={() => showErrors(!isCls && ev.errors.some((e) => e.missed > 0) ? "missed" : "all")}>
                      {isCls ? t("看分错的图") : ev.errors.some((e) => e.missed > 0) ? t("看漏检的错例") : t("看错例")}
                    </Button>
                  )}
                  {ev.errors.length > 0 && (
                    <Button size="small" onClick={() => openErrors(ev.errors)}>
                      {t("把错例拿去标注")}
                    </Button>
                  )}
                  {rate.level !== "good" ? (
                    <Button size="small" onClick={() => setPage("train")}>
                      {t("补数据后再训 / 换大模型")}
                    </Button>
                  ) : (
                    sel != null && (
                      <Button size="small" type="primary" onClick={() => openRun(sel, "deploy")}>
                        {t("去部署")}
                      </Button>
                    )
                  )}
                </Space>
              }
            />
          )}
          <div className="cards" style={{ gridTemplateColumns: "repeat(4, 1fr)", marginBottom: 12 }}>
            <Tooltip title={isCls ? t("分对的图片占全部图片的比例") : t("IoU 0.5 时各类别平均精度的均值，最常用的检测指标")}>
              <div className="card">
                <div className="muted">{mn.m50}</div>
                <div className="kpi">{isCls ? pct(ev.map50) : ev.map50.toFixed(3)}</div>
              </div>
            </Tooltip>
            <Tooltip title={isCls ? t("各类别 F1 的平均，类别不均衡时比准确率更能反映效果") : t("IoU 0.5~0.95 的平均，对框的准确度要求更高")}>
              <div className="card"><div className="muted">{mn.m}</div><div className="kpi">{ev.map.toFixed(3)}</div></div>
            </Tooltip>
            <Tooltip title={isCls ? t("各类别精确率的平均：判成某类的图里，真是这一类的比例") : t("模型报出来的框里，正确的比例。低说明误报多")}>
              <div className="card"><div className="muted">{isCls ? t("平均精确率") : t("精确率")}</div><div className="kpi">{pct(ev.precision)}</div></div>
            </Tooltip>
            <Tooltip title={isCls ? t("各类别召回率的平均：某类的图里，被判对的比例") : t("真实目标里，被找到的比例。低说明漏检多")}>
              <div className="card"><div className="muted">{isCls ? t("平均召回率") : t("召回率")}</div><div className="kpi">{pct(ev.recall)}</div></div>
            </Tooltip>
          </div>

          {ev.thresholds ? (
            <ThresholdPanel th={ev.thresholds} isCls={isCls} />
          ) : (
            <div className="dim" style={{ fontSize: 12, marginBottom: 12 }}>
              {t("这次评估没有阈值曲线（旧版本训练的），点「用当前数据评估」就能算出推荐阈值。")}
            </div>
          )}

          <div className="cards" style={{ gridTemplateColumns: "1fr 1fr", marginBottom: 12 }}>
            <div className="card">
              <div className="muted" style={{ marginBottom: 8 }}>{t("各类别")}</div>
              <Table<ClassEval>
                size="small"
                rowKey="index"
                pagination={false}
                dataSource={ev.perClass}
                columns={[
                  { title: t("类别"), dataIndex: "name" },
                  { title: isCls ? t("图片数") : t("真实框"), dataIndex: "gt" },
                  { title: t("精确率"), dataIndex: "precision", render: pct },
                  {
                    title: t("召回率"),
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
              <div className="muted" style={{ marginBottom: 8 }}>{t("混淆矩阵")} <span className="dim">{t("（行：真实，列：预测）")}</span></div>
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
                          <td key={j} style={{ background: bg }} title={t("真实「{gt}」→ 预测「{pred}」：{n}", { gt: names[i], pred: names[j], n: v })}>
                            {v || ""}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="dim" style={{ fontSize: 12, marginTop: 8 }}>
                {isCls ? t("对角线是分对的，其他格子是把某类误判成另一类。") : t("最后一列「背景」是漏检，最后一行「背景」是误报。")}
              </div>
            </div>
          </div>

          <div className="card" ref={errBox}>
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 10 }}>
              <div className="muted">
                {t("错例")} <span className="dim">· {t("点图片去标注页检查和修改")}</span>
                {!isCls && (
                  <div className="dim" style={{ fontSize: 12, marginTop: 2 }}>
                    {t("漏：标了框但模型没找到 · 误：模型多报、你没标 · 错：位置对上了但类别不一样")}
                    {noBoxes && <div style={{ color: "var(--warn)" }}>{t("点右上角「用当前数据评估」重新评估一次，点进错例就能在图上看到具体是哪个框")}</div>}
                  </div>
                )}
              </div>
              {isCls ? (
                <span className="dim" style={{ fontSize: 12 }}>{t("分错 {n} 张，按模型把握从高到低（越靠前越可能是标错了）", { n: ev.errors.length })}</span>
              ) : (
                <Segmented
                  size="small"
                  value={errKind}
                  onChange={(v) => setErrKind(v as typeof errKind)}
                  options={[
                    { value: "all", label: t("全部 {n}", { n: ev.errors.length }) },
                    { value: "missed", label: t("有漏检 {n}", { n: ev.errors.filter((e) => e.missed).length }) },
                    { value: "extra", label: t("有误报 {n}", { n: ev.errors.filter((e) => e.extra).length }) },
                    { value: "wrong", label: t("类别错 {n}", { n: ev.errors.filter((e) => e.wrong).length }) },
                  ]}
                />
              )}
            </div>
            {errors.length === 0 ? (
              <Empty description={t("没有错例")} />
            ) : (
              <div className="thumbs">
                {errors.map((e) => (
                  <div key={e.imageId} className="thumb" onClick={() => openErrors(filtered, e.imageId)}>
                    <img src={thumbUrl(project.path, e.imageId)} loading="lazy" alt="" draggable={false} />
                    <span className="nm">
                      {isCls && e.gt != null && e.pred != null ? (
                        t("标「{gt}」→ 判「{pred}」", { gt: classes[e.gt] ?? e.gt, pred: classes[e.pred] ?? e.pred })
                      ) : (
                        <>
                          {e.missed > 0 && t("漏 {n}", { n: e.missed }) + " "}
                          {e.extra > 0 && t("误 {n}", { n: e.extra }) + " "}
                          {e.wrong > 0 && t("错 {n}", { n: e.wrong })}
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
      {/* 换了模型要重新开始，别留着上一个模型的测试结果 */}
      {detail && !isCls && <FpTestPanel key={detail.run.id} run={detail.run} defaultConf={ev?.thresholds?.best.conf ?? 0.5} />}
    </div>
  );
}
