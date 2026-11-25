import { useEffect, useMemo, useRef, useState } from "react";
import { Alert, App, Button, Empty, Progress, Select, Space, Tag, Tooltip } from "antd";
import { CheckOutlined, EyeInvisibleOutlined, FormOutlined, HeartOutlined, PlusOutlined } from "@ant-design/icons";
import { listen } from "@tauri-apps/api/event";
import { api, errText, HealthProgress, HealthReport, Issue, IssueKind, ProjectInfo, RunInfo, thumbUrl } from "../api";
import { Mark, useApp, useReview } from "../store";
import { statusTag } from "../status";
import { fmtTime } from "./Train";
import { t } from "../i18n";

type Filter = IssueKind | "dup_image";

/** cls：分类项目也有的检查（其余都是框相关的，分类项目不显示） */
const KINDS: { key: Filter; name: string; color: string; tip: string; model?: boolean; cls?: boolean }[] = [
  { key: "missing", name: t("疑似漏标"), color: "#e5534b", tip: t("模型很确定这里有目标，但没有标注"), model: true },
  { key: "wrong_class", name: t("疑似类别错"), color: "#e5534b", tip: t("框的位置对上了，但标注的类别和模型判断的不一样"), model: true, cls: true },
  { key: "loose_box", name: t("框不准"), color: "#e7a33e", tip: t("类别一致，但框和模型预测的重合度低，可能画偏或大小不对"), model: true },
  { key: "extra_box", name: t("疑似多余框"), color: "#e7a33e", tip: t("标了框，但模型在这个位置什么都没看到"), model: true },
  { key: "tiny_box", name: t("极小框"), color: "#9b7bf2", tip: t("短边小于 8 像素，模型几乎学不到") },
  { key: "dup_box", name: t("重复框"), color: "#9b7bf2", tip: t("同一张图里同类别的两个框几乎完全重叠") },
  { key: "dup_image", name: t("重复图片"), color: "#3d8bfd", tip: t("画面几乎一样的图片，会让评估结果虚高"), cls: true },
  { key: "no_label", name: t("无类别"), color: "#9aa1ab", tip: t("框没有类别，不会参与训练") },
];


const PAGE = 48;

/** 体检进度下面的一行字 */
function stageText(r: HealthProgress) {
  if (r.stopping) return t("正在停止…");
  if (r.stage === "hash") return t("查重：计算指纹 {done} / {total} 张", { done: r.done, total: r.total });
  if (r.stage === "compare") return t("查重：两两比对");
  if (r.stage === "model") return t("模型检查 {done} / {total} 张", { done: r.done, total: r.total });
  return t("准备中…");
}

/** 缩略图上叠加问题框 */
function IssueThumb({ path, issue, dims, color }: { path: string; issue: Issue; dims?: [number, number]; color: string }) {
  return (
    <div className="thumb" style={{ cursor: "default" }}>
      <img src={thumbUrl(path, issue.imageId)} loading="lazy" alt="" draggable={false} />
      {dims && issue.bbox && (
        <svg viewBox={`0 0 ${dims[0]} ${dims[1]}`} preserveAspectRatio="xMidYMid meet" style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
          <rect
            x={issue.bbox[0]}
            y={issue.bbox[1]}
            width={issue.bbox[2]}
            height={issue.bbox[3]}
            fill="none"
            stroke={color}
            strokeWidth={3}
            vectorEffect="non-scaling-stroke"
            strokeDasharray="6 3"
          />
        </svg>
      )}
    </div>
  );
}

export default function Health({ project }: { project: ProjectInfo }) {
  const { message, modal } = App.useApp();
  const annotateImage = useApp((s) => s.annotateImage);
  const annotateQueue = useApp((s) => s.annotateQueue);
  const refreshStats = useApp((s) => s.refreshStats);
  const [report, setReport] = useState<HealthReport | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [runId, setRunId] = useState<number | null>(null);
  const [running, setRunning] = useState<HealthProgress | null>(null);
  // 这个页面发起的体检被点了停止（结束时提示「已停止」而不是报错）
  const stopping = useRef(false);
  const [filter, setFilter] = useState<Filter | null>(null);
  const [limit, setLimit] = useState(PAGE);
  // 每组重复图片保留哪张（key 是组里 id 连起来），没选的用排在第一张的（后端按标注完整程度排好了）
  const [keep, setKeep] = useState<Record<string, number>>({});
  const [showChanged, setShowChanged] = useState(false);
  const [busy, setBusy] = useState(false);
  const review = useReview();
  const cls = project.task === "classify";
  const kinds = useMemo(() => (cls ? KINDS.filter((k) => k.cls) : KINDS), [cls]);

  useEffect(() => {
    api.healthLoad().then((r) => {
      setReport(r);
      setLoaded(true);
    }).catch((e) => message.error(errText(e)));
    api.listRuns().then((rs) => {
      const ok = rs.filter((r) => r.hasWeights);
      setRuns(ok);
      setRunId(ok[0]?.id ?? null);
    }).catch(() => {});
  }, [message]);

  // 体检在后台跑：离开页面再回来能接着看进度；跑完（不管是不是这次打开页面时发起的）都刷新报告
  useEffect(() => {
    let alive = true;
    let finished = false;
    const uns = [
      listen<HealthProgress>("health-progress", (e) => {
        if (alive) setRunning(e.payload);
      }),
      listen<boolean>("health-done", () => {
        finished = true;
        if (!alive) return;
        setRunning(null);
        api.healthLoad().then((r) => {
          if (alive) setReport(r);
        }).catch(() => {});
      }),
    ];
    // 先挂好监听再查，免得中间刚好跑完漏掉
    Promise.all(uns)
      .then(() => api.healthStatus())
      .then((s) => {
        if (alive && s && !finished) setRunning(s);
      })
      .catch(() => {});
    return () => {
      alive = false;
      uns.forEach((p) => p.then((un) => un()));
    };
  }, []);

  const labelName = useMemo(() => {
    const m = new Map((report?.classes ?? []).map((c) => [c.labelId, c.name]));
    return (id: number | null) => (id != null && m.get(id)) || t("未分类");
  }, [report]);

  // 体检后标注改过的图上的问题：不算数，默认收起
  const changed = useMemo(() => new Set(report?.changed ?? []), [report]);

  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const i of report?.issues ?? []) if (!changed.has(i.key)) c[i.kind] = (c[i.kind] ?? 0) + 1;
    c.dup_image = report?.dupGroups.length ?? 0;
    return c;
  }, [report, changed]);

  // 默认选中第一个有问题的类型
  useEffect(() => {
    if (report && filter == null) {
      setFilter((kinds.find((k) => counts[k.key]) ?? kinds[0]).key);
    }
  }, [report, counts, filter, kinds]);

  const start = async () => {
    stopping.current = false;
    setRunning({ stage: "", done: 0, total: 0, stopping: false });
    try {
      const r = await api.healthRun(runId);
      setReport(r);
      setFilter(null);
      setLimit(PAGE);
      setKeep({});
      setShowChanged(false);
      message.success(t("体检完成，发现 {n} 处问题", { n: r.issues.length + r.dupGroups.length }));
    } catch (e) {
      if (stopping.current) message.info(t("已停止体检"));
      else modal.error({ title: t("体检没有完成"), content: errText(e) });
    } finally {
      // 被拒绝（已经有体检在跑）时接着显示那次的进度
      setRunning(await api.healthStatus().catch(() => null));
    }
  };

  const stop = () => {
    stopping.current = true;
    setRunning((r) => (r ? { ...r, stopping: true } : r));
    api.healthStop().catch((e) => message.error(errText(e)));
  };

  const drop = (keys: string[]) =>
    setReport((r) => (r ? { ...r, issues: r.issues.filter((i) => !keys.includes(i.key)) } : r));

  const ignoreKeys = async (keys: string[]) => {
    try {
      await api.healthIgnore(keys);
      setReport((r) => (r ? { ...r, issues: r.issues.filter((i) => !keys.includes(i.key)), ignored: (r.ignored ?? 0) + keys.length } : r));
    } catch (e) {
      message.error(errText(e));
    }
  };

  const unignore = async () => {
    try {
      await api.healthUnignore();
      setReport(await api.healthLoad());
    } catch (e) {
      message.error(errText(e));
    }
  };

  const addBox = (i: Issue) =>
    api.appendAnnotation(i.imageId, { labelId: i.predLabelId, type: "bbox", data: i.bbox!, score: null, source: "manual" });

  // 加上后这张图的标注变了，下次打开体检页会算「已修改」，不用再记忽略
  const accept = async (i: Issue) => {
    if (!i.bbox || i.predLabelId == null) return;
    try {
      await addBox(i);
      drop([i.key]);
      refreshStats();
      message.success(t("已加为「{name}」", { name: labelName(i.predLabelId) }));
    } catch (e) {
      message.error(errText(e));
    }
  };

  const acceptAll = (items: Issue[]) => {
    const todo = items.filter((i) => i.bbox && i.predLabelId != null);
    modal.confirm({
      title: t("把这 {n} 处疑似漏标都加为标注？", { n: todo.length }),
      content: t("按模型预测的框和类别直接加上，之后可以在标注页修改。"),
      okText: t("全部加上"),
      cancelText: t("取消"),
      onOk: async () => {
        setBusy(true);
        const done: string[] = [];
        try {
          for (const i of todo) {
            await addBox(i);
            done.push(i.key);
          }
        } catch (e) {
          message.error(errText(e));
        } finally {
          drop(done);
          setBusy(false);
          refreshStats();
          if (done.length) message.success(t("已加上 {n} 个标注", { n: done.length }));
        }
      },
    });
  };

  const ignoreAll = (items: Issue[], name: string) => {
    modal.confirm({
      title: t("忽略这 {n} 处「{name}」？", { n: items.length, name }),
      content: t("以后体检不再提示，可以随时恢复。"),
      okText: t("全部忽略"),
      cancelText: t("取消"),
      onOk: () => ignoreKeys(items.map((i) => i.key)),
    });
  };

  // 去标注页时带上这一类问题的所有图，翻图只在这些图里翻
  const go = (i: Issue) => {
    const name = KINDS.find((x) => x.key === i.kind)?.name ?? "";
    const marks: Record<number, Mark[]> = {};
    // 点的这一处先放，同一张图有好几处问题时高亮点的那处
    for (const x of [i, ...list]) {
      if (x.bbox && !marks[x.imageId]) marks[x.imageId] = [{ bbox: x.bbox, text: t("体检：{text}", { text: name }), color: "#ffd43b", fg: "#1b1e23" }];
    }
    const ids = [...new Set(list.some((x) => x.key === i.key) ? list.map((x) => x.imageId) : [i.imageId])];
    annotateQueue(ids, name || t("体检"), { focus: i.imageId, marks });
  };

  const keepOf = (g: number[]) => {
    const k = keep[g.join(",")];
    return k != null && g.includes(k) ? k : g[0];
  };

  const removeDups = async (extra: number[], left: (g: number[]) => boolean) => {
    try {
      const n = await api.deleteImages("ids", extra);
      setReport((r) => (r ? { ...r, dupGroups: r.dupGroups.filter(left) } : r));
      refreshStats();
      message.success(t("删掉了 {n} 张重复图片", { n }));
    } catch (e) {
      message.error(errText(e));
    }
  };

  const dedupe = (group: number[]) => {
    const extra = group.filter((id) => id !== keepOf(group));
    modal.confirm({
      title: t("删除这组里的 {n} 张重复图片？", { n: extra.length }),
      content: t("保留绿框那张。图片的标注会一起删除，你电脑上的原图不受影响。"),
      okText: t("删除"),
      okButtonProps: { danger: true },
      cancelText: t("取消"),
      onOk: () => removeDups(extra, (g) => g !== group),
    });
  };

  const dedupeAll = () => {
    if (!report) return;
    const extra = report.dupGroups.flatMap((g) => g.filter((id) => id !== keepOf(g)));
    modal.confirm({
      title: t("每组只保留一张，共删除 {n} 张重复图片？", { n: extra.length }),
      content: t("每组保留绿框那张（默认留标注最完整的）。图片的标注会一起删除，你电脑上的原图不受影响。"),
      okText: t("删除"),
      okButtonProps: { danger: true },
      cancelText: t("取消"),
      onOk: () => removeDups(extra, () => false),
    });
  };

  const kind = kinds.find((k) => k.key === filter);
  const all = (report?.issues ?? []).filter((i) => i.kind === filter);
  const open = all.filter((i) => !changed.has(i.key));
  const nChanged = all.length - open.length;
  const list = showChanged ? [...open, ...all.filter((i) => changed.has(i.key))] : open;
  const maxBoxes = Math.max(1, ...(report?.classes ?? []).map((c) => c.boxes));
  const maxSize = Math.max(1, ...(report?.sizes ?? []).map((s) => s[1]));

  return (
    <div className="page">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 14 }}>
        <div>
          <h1>{t("数据体检")}</h1>
          <p className="desc" style={{ margin: 0 }}>
            {report
              ? report.runName
                ? t("上次体检：{time} · {images} 张图、{boxes} 个框 · 用模型 {run} 检查", {
                    time: fmtTime(report.createdAt),
                    images: report.imagesChecked,
                    boxes: report.boxesChecked,
                    run: report.runName,
                  })
                : t("上次体检：{time} · {images} 张图、{boxes} 个框 · 只做了规则检查", {
                    time: fmtTime(report.createdAt),
                    images: report.imagesChecked,
                    boxes: report.boxesChecked,
                  })
              : t("检查标注里的常见问题，找出漏标、错标、重复图片等，点一下就能去改。")}
          </p>
        </div>
        <Space>
          <Select
            style={{ width: 300 }}
            value={runId ?? 0}
            onChange={(v) => setRunId(v === 0 ? null : v)}
            options={[
              ...runs.map((r) => ({ value: r.id, label: t("用模型检查：{name} · mAP50 {map}", { name: r.name, map: r.bestMap50?.toFixed(3) ?? "—" }) })),
              { value: 0, label: t("只做规则检查（不用模型）") },
            ]}
          />
          <Button type="primary" icon={<HeartOutlined />} loading={!!running} onClick={start}>
            {report ? t("重新体检") : t("开始体检")}
          </Button>
        </Space>
      </div>

      {running && (
        <div className="card" style={{ marginBottom: 12 }}>
          <div className="row" style={{ gap: 12, flexWrap: "nowrap" }}>
            <Progress style={{ flex: 1, margin: 0 }} percent={running.total ? Math.floor((running.done / running.total) * 100) : 0} status="active" />
            <Button size="small" danger loading={running.stopping} onClick={stop}>
              {t("停止")}
            </Button>
          </div>
          <div className="dim" style={{ fontSize: 12 }}>{stageText(running)}</div>
        </div>
      )}

      {runs.length === 0 && (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 12 }}
          message={t("还没有训练好的模型，现在只能做规则检查。训练一个模型后再体检，可以找出漏标、错标和框画偏的地方。")}
        />
      )}

      {!report ? (
        loaded && (
          <div className="card" style={{ padding: 48 }}>
            <Empty description={t("还没有体检过")}>
              <Button type="primary" onClick={start} loading={!!running}>
                {t("开始体检")}
              </Button>
            </Empty>
          </div>
        )
      ) : (
        <>
          <div className="cards" style={{ gridTemplateColumns: `repeat(${kinds.length}, 1fr)`, marginBottom: 12 }}>
            {kinds.map((k) => {
              const n = counts[k.key] ?? 0;
              const off = k.model && !report.runId;
              return (
                <Tooltip key={k.key} title={off ? t("需要用模型检查") : k.tip}>
                  <div
                    className="card click"
                    style={{ padding: 12, borderColor: filter === k.key ? k.color : undefined, opacity: off ? 0.45 : 1 }}
                    onClick={() => {
                      setFilter(k.key);
                      setLimit(PAGE);
                    }}
                  >
                    <div className="muted" style={{ fontSize: 12 }}>{k.name}</div>
                    <div className="kpi" style={{ color: n ? k.color : "var(--dim)" }}>{off ? "—" : n}</div>
                  </div>
                </Tooltip>
              );
            })}
          </div>

          <div className="cards" style={{ gridTemplateColumns: "minmax(0, 1.7fr) minmax(280px, 1fr)", alignItems: "start" }}>
            <div className="card">
              <div className="row" style={{ justifyContent: "space-between", marginBottom: 10 }}>
                <h2 style={{ fontSize: 14, margin: 0 }}>
                  {kind?.name} <span className="dim">· {filter === "dup_image" ? t("{n} 组", { n: report.dupGroups.length }) : t("{n} 处", { n: open.length })}</span>
                </h2>
                <span className="dim" style={{ fontSize: 12 }}>
                  {kind?.tip}
                  {!!report.ignored && (
                    <>
                      {" · "}
                      {t("已忽略 {n} 条", { n: report.ignored })} <a onClick={unignore}>{t("恢复")}</a>
                    </>
                  )}
                </span>
              </div>

              {filter !== "dup_image" && open.length > 0 && (
                <div className="row" style={{ gap: 8, marginBottom: 10 }}>
                  {filter === "missing" && (
                    <Button size="small" icon={<PlusOutlined />} loading={busy} onClick={() => acceptAll(open)}>
                      {t("全部加为标注（{n} 处）", { n: open.length })}
                    </Button>
                  )}
                  <Button size="small" icon={<EyeInvisibleOutlined />} disabled={busy} onClick={() => ignoreAll(open, kind?.name ?? "")}>
                    {t("全部忽略（{n} 处）", { n: open.length })}
                  </Button>
                </div>
              )}

              {filter === "dup_image" ? (
                report.dupGroups.length === 0 ? (
                  <Empty description={t("没有重复图片")} image={<CheckOutlined style={{ fontSize: 40, color: "var(--ok)" }} />} />
                ) : (
                  <>
                    <div className="row" style={{ gap: 8, marginBottom: 10 }}>
                      <Button size="small" danger onClick={dedupeAll}>
                        {t("每组只保留一张（删除 {n} 张）", { n: report.dupGroups.reduce((s, g) => s + g.length - 1, 0) })}
                      </Button>
                      <span className="dim" style={{ fontSize: 12 }}>
                        {t("绿框是要保留的图，默认留标注最完整的那张；点别的图可以换，双击去标注页看。")}
                      </span>
                    </div>
                    {report.dupGroups.slice(0, limit).map((g) => (
                      <div key={g.join(",")} className="row" style={{ marginBottom: 10, flexWrap: "nowrap", alignItems: "stretch" }}>
                        <div className="thumbs" style={{ flex: 1, gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))" }}>
                          {g.map((id) => {
                            const kept = keepOf(g) === id;
                            const info = report.imgInfo?.[String(id)];
                            const st = info ? statusTag(info[0], review) : null;
                            return (
                              <div
                                key={id}
                                className="thumb"
                                title={info ? t("#{id} · {n} 个标注", { id, n: info[1] }) : `#${id}`}
                                onClick={() => setKeep((m) => ({ ...m, [g.join(",")]: id }))}
                                onDoubleClick={() => annotateImage(id)}
                                style={{ outline: kept ? "2px solid var(--ok)" : undefined }}
                              >
                                <img src={thumbUrl(project.path, id)} loading="lazy" alt="" draggable={false} />
                                {st && (
                                  <Tag className="st" color={st.color}>
                                    {st.text}
                                  </Tag>
                                )}
                                <span className="nm">{kept ? t("保留") : `#${id}`}</span>
                              </div>
                            );
                          })}
                        </div>
                        <Button size="small" style={{ alignSelf: "center" }} onClick={() => dedupe(g)}>
                          {t("删除其余 {n} 张", { n: g.length - 1 })}
                        </Button>
                      </div>
                    ))}
                  </>
                )
              ) : list.length === 0 ? (
                <Empty
                  description={kind?.model && !report.runId ? t("选一个训练好的模型再体检，才能检查这一项") : t("没有发现这类问题")}
                  image={<CheckOutlined style={{ fontSize: 40, color: "var(--ok)" }} />}
                />
              ) : (
                <div className="issue-grid">
                  {list.slice(0, limit).map((i) => {
                    const done = changed.has(i.key);
                    return (
                      <div key={i.key} className="issue" style={done ? { opacity: 0.55 } : undefined}>
                        <IssueThumb path={project.path} issue={i} dims={report.dims[String(i.imageId)]} color={kind?.color ?? "#fff"} />
                        <div className="issue-text">
                          {done && <Tag style={{ marginRight: 4 }}>{t("已修改")}</Tag>}
                          {i.detail}
                        </div>
                        <div className="row" style={{ gap: 4 }}>
                          <Button size="small" icon={<FormOutlined />} onClick={() => go(i)}>
                            {t("去修改")}
                          </Button>
                          {i.kind === "missing" && !done && (
                            <Tooltip title={t("直接加一个「{name}」框", { name: labelName(i.predLabelId) })}>
                              <Button size="small" icon={<PlusOutlined />} onClick={() => accept(i)}>
                                {t("加为标注")}
                              </Button>
                            </Tooltip>
                          )}
                          <Tooltip title={t("不是问题，以后体检不再提示")}>
                            <Button size="small" type="text" icon={<EyeInvisibleOutlined />} onClick={() => ignoreKeys([i.key])} />
                          </Tooltip>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
              {filter !== "dup_image" && nChanged > 0 && (
                <div className="dim" style={{ fontSize: 12, marginTop: 10 }}>
                  {t("另有 {n} 处所在的图体检后改过标注，可能已经改好了", { n: nChanged })}{" "}
                  <a onClick={() => setShowChanged(!showChanged)}>{showChanged ? t("收起") : t("显示")}</a>
                </div>
              )}
              {((filter === "dup_image" ? report.dupGroups.length : list.length) > limit) && (
                <div style={{ textAlign: "center", marginTop: 10 }}>
                  <Button onClick={() => setLimit(limit + PAGE)}>{t("显示更多")}</Button>
                </div>
              )}
            </div>

            <div>
              <div className="card" style={{ marginBottom: 12 }}>
                <h2 style={{ fontSize: 14, marginTop: 0 }}>{t("建议")}</h2>
                {report.suggestions.length === 0 ? (
                  <div className="dim">{t("数据看起来不错。")}</div>
                ) : (
                  report.suggestions.map((s, i) => (
                    <div key={i} className="suggest">{s}</div>
                  ))
                )}
              </div>
              <div className="card" style={{ marginBottom: 12 }}>
                <h2 style={{ fontSize: 14, marginTop: 0 }}>
                  {t("类别分布")} <span className="dim" style={{ fontWeight: 400 }}>{cls ? t("（图片数）") : t("（框数 · 图片数）")}</span>
                </h2>
                {report.classes.length === 0 && <div className="dim">{t("还没有类别")}</div>}
                {report.classes.map((c) => (
                  <div key={c.labelId} style={{ marginBottom: 8 }}>
                    <div className="row" style={{ justifyContent: "space-between" }}>
                      <span>{c.name}</span>
                      <span className="muted" style={{ color: c.boxes * 5 < maxBoxes ? "var(--warn)" : undefined }}>
                        {cls ? c.images : `${c.boxes} · ${c.images}`}
                      </span>
                    </div>
                    <div style={{ height: 6, background: "var(--panel2)", borderRadius: 3, overflow: "hidden" }}>
                      <div style={{ width: `${(c.boxes / maxBoxes) * 100}%`, height: "100%", background: c.color }} />
                    </div>
                  </div>
                ))}
              </div>
              {!cls && (
                <div className="card">
                  <h2 style={{ fontSize: 14, marginTop: 0 }}>{t("框大小分布")} <span className="dim" style={{ fontWeight: 400 }}>{t("（短边像素）")}</span></h2>
                  <div className="sizes">
                    {report.sizes.map(([name, n], i) => (
                      <Tooltip key={name} title={t("{range} 像素：{n} 个框", { range: name, n })}>
                        <div className="size-col">
                          <div className="size-bar" style={{ height: `${(n / maxSize) * 100}%`, background: i === 0 ? "var(--warn)" : "var(--accent)" }} />
                          <span>{name}</span>
                        </div>
                      </Tooltip>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
