import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { App, Button, ColorPicker, Empty, Input, Modal, Progress, Segmented, Select, Slider, Space, Switch, Tag, Tooltip } from "antd";
import { listen } from "@tauri-apps/api/event";
import {
  BorderOutlined,
  CheckOutlined,
  CopyOutlined,
  DeleteOutlined,
  EditOutlined,
  ExpandOutlined,
  LeftOutlined,
  PlusOutlined,
  RedoOutlined,
  RightOutlined,
  SelectOutlined,
  StepForwardOutlined,
  ThunderboltOutlined,
  UndoOutlined,
} from "@ant-design/icons";
import { api, Annotation, errText, ImageAnnotations, ImageStatus, imageUrl, Label, ProjectInfo, RunInfo, thumbUrl } from "../api";
import { prefs, useApp } from "../store";
import AnnotCanvas, { Box, RectXYWH, Tool } from "../components/AnnotCanvas";

const PALETTE = ["#3d8bfd", "#e5534b", "#2fb67c", "#e7a33e", "#9b7bf2", "#14b8a6", "#f472b6", "#a3e635", "#f97316", "#60a5fa"];
const FILTERS = [
  { label: "全部", value: "all" },
  { label: "未标注", value: "todo" },
  { label: "已标注", value: "labeled" },
  { label: "已审核", value: "reviewed" },
];
const STATUS_TEXT: Record<ImageStatus, { text: string; color?: string }> = {
  todo: { text: "未标注" },
  labeled: { text: "已标注", color: "processing" },
  reviewed: { text: "已审核", color: "success" },
  skipped: { text: "已跳过" },
};
const SOURCE_TEXT: Record<string, string> = { manual: "手工", model: "模型", sam: "SAM", ai: "AI" };
const HISTORY_MAX = 200;

let keySeq = 0;
const newKey = () => `b${++keySeq}`;

const toBox = (a: Annotation): Box => ({
  key: newKey(),
  labelId: a.labelId,
  x: a.data[0],
  y: a.data[1],
  w: a.data[2],
  h: a.data[3],
  source: a.source,
  score: a.score,
});

const toAnn = (b: Box): Annotation => ({
  labelId: b.labelId,
  type: "bbox",
  data: [b.x, b.y, b.w, b.h],
  score: b.score ?? null,
  source: b.source,
});

type SaveState = "saved" | "dirty" | "saving" | "error";

export default function Annotate({ project }: { project: ProjectInfo }) {
  const { message, modal } = App.useApp();
  const refreshStats = useApp((s) => s.refreshStats);
  const setPage = useApp((s) => s.setPage);
  const takeFocusImage = useApp((s) => s.takeFocusImage);
  const highlight = useApp((s) => s.highlight);

  const [labels, setLabels] = useState<Label[]>([]);
  const [curLabel, setCurLabel] = useState<number | null>(null);
  const [filter, setFilter] = useState("all");
  const [ids, setIds] = useState<number[] | null>(null);
  const [idx, setIdx] = useState(0);
  const [meta, setMeta] = useState<ImageAnnotations | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [tool, setTool] = useState<Tool>("rect");
  const [saveState, setSaveState] = useState<SaveState>("saved");
  const [fitSignal, setFitSignal] = useState(0);
  const [showNames, setShowNames] = useState(true);
  const [scale, setScale] = useState(1);
  const [newLabel, setNewLabel] = useState("");
  const [editing, setEditing] = useState<Label | null>(null);
  const [histLen, setHistLen] = useState({ past: 0, future: 0 });

  const boxesRef = useRef<Box[]>([]);
  const metaRef = useRef<ImageAnnotations | null>(null);
  const othersRef = useRef<Annotation[]>([]); // 非矩形框的标注原样保留
  const history = useRef<{ past: Box[][]; future: Box[][] }>({ past: [], future: [] });
  const dirty = useRef(false);
  const timer = useRef<number>();

  const labelMap = useMemo(() => new Map(labels.map((l) => [l.id, l])), [labels]);
  const colorOf = useCallback((id: number | null) => (id != null && labelMap.get(id)?.color) || "#9aa1ab", [labelMap]);
  const nameOf = useCallback((id: number | null) => (id != null && labelMap.get(id)?.name) || "未分类", [labelMap]);

  // ---------- 保存 ----------

  const flush = useCallback(async () => {
    window.clearTimeout(timer.current);
    const m = metaRef.current;
    if (!dirty.current || !m) return;
    dirty.current = false;
    setSaveState("saving");
    try {
      const status = await api.saveAnnotations(m.id, [...boxesRef.current.map(toAnn), ...othersRef.current]);
      if (metaRef.current?.id === m.id) {
        metaRef.current = { ...metaRef.current, status };
        setMeta(metaRef.current);
      }
      setSaveState(dirty.current ? "dirty" : "saved");
    } catch (e) {
      dirty.current = true;
      setSaveState("error");
      message.error(`保存失败：${errText(e)}`);
    }
  }, [message]);

  const schedule = useCallback(() => {
    dirty.current = true;
    setSaveState("dirty");
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(flush, 400);
  }, [flush]);

  const setAll = (next: Box[]) => {
    boxesRef.current = next;
    setBoxes(next);
  };

  /** 修改标注都走这里：记历史、自动保存 */
  const commit = useCallback(
    (next: Box[]) => {
      const h = history.current;
      h.past.push(boxesRef.current);
      if (h.past.length > HISTORY_MAX) h.past.shift();
      h.future = [];
      setHistLen({ past: h.past.length, future: 0 });
      setAll(next);
      schedule();
    },
    [schedule],
  );

  const undo = useCallback(() => {
    const h = history.current;
    const prev = h.past.pop();
    if (!prev) return;
    h.future.push(boxesRef.current);
    setHistLen({ past: h.past.length, future: h.future.length });
    setAll(prev);
    setSelected(null);
    schedule();
  }, [schedule]);

  const redo = useCallback(() => {
    const h = history.current;
    const next = h.future.pop();
    if (!next) return;
    h.past.push(boxesRef.current);
    setHistLen({ past: h.past.length, future: h.future.length });
    setAll(next);
    setSelected(null);
    schedule();
  }, [schedule]);

  // 离开页面前保存
  useEffect(
    () => () => {
      flush().then(refreshStats);
    },
    [flush, refreshStats],
  );

  // ---------- 类别 ----------

  const loadLabels = useCallback(async () => {
    const ls = await api.listLabels();
    setLabels(ls);
    setCurLabel((cur) => (cur != null && ls.some((l) => l.id === cur) ? cur : ls[0]?.id ?? null));
  }, []);

  useEffect(() => {
    loadLabels().catch((e) => message.error(errText(e)));
  }, [loadLabels, message]);

  const addLabel = async () => {
    const name = newLabel.trim();
    if (!name) return;
    try {
      const l = await api.addLabel(name, PALETTE[labels.length % PALETTE.length]);
      setNewLabel("");
      await loadLabels();
      setCurLabel(l.id);
    } catch (e) {
      message.error(errText(e));
    }
  };

  const saveLabelEdit = async () => {
    if (!editing) return;
    try {
      await api.updateLabel(editing.id, editing.name, editing.color);
      setEditing(null);
      await loadLabels();
    } catch (e) {
      message.error(errText(e));
    }
  };

  const removeLabel = (l: Label) => {
    modal.confirm({
      title: `删除类别「${l.name}」？`,
      content: l.count > 0 ? `这个类别下的 ${l.count} 个标注会一起删除，无法恢复。` : "这个类别还没有标注。",
      okText: "删除",
      okButtonProps: { danger: true },
      cancelText: "取消",
      onOk: async () => {
        try {
          await flush();
          await api.deleteLabel(l.id);
          await loadLabels();
          await reloadCurrent();
          await refreshStats();
        } catch (e) {
          message.error(errText(e));
        }
      },
    });
  };

  /** 选类别：同时把选中的框改成这个类别 */
  const pickLabel = (id: number) => {
    setCurLabel(id);
    if (selected) {
      const b = boxesRef.current.find((x) => x.key === selected);
      if (b && b.labelId !== id)
        commit(boxesRef.current.map((x) => (x.key === selected ? { ...x, labelId: id, source: "manual", score: null } : x)));
    }
  };

  // ---------- 图片列表与翻图 ----------

  useEffect(() => {
    let alive = true;
    (async () => {
      await flush();
      const all = await api.listImageIds(filter === "all" ? null : filter);
      if (!alive) return;
      let start = 0;
      const focus = takeFocusImage();
      if (focus != null && all.includes(focus)) start = all.indexOf(focus);
      else if (filter === "all") {
        // 默认从第一张未标注的图开始
        const todo = await api.listImageIds("todo");
        if (todo.length) start = Math.max(0, all.indexOf(todo[0]));
      }
      setIds(all);
      setIdx(start);
    })().catch((e) => message.error(errText(e)));
    return () => {
      alive = false;
    };
  }, [filter, flush, takeFocusImage, message]);

  const applyLoaded = (m: ImageAnnotations) => {
    metaRef.current = m;
    setMeta(m);
    othersRef.current = m.annotations.filter((a) => a.type !== "bbox");
    setAll(m.annotations.filter((a) => a.type === "bbox").map(toBox));
    history.current = { past: [], future: [] };
    setHistLen({ past: 0, future: 0 });
    setSelected(null);
  };

  const reloadCurrent = async () => {
    const id = metaRef.current?.id;
    if (id != null) applyLoaded(await api.loadAnnotations(id));
  };

  const imageId = ids?.[idx];
  useEffect(() => {
    if (imageId === undefined) {
      metaRef.current = null;
      setMeta(null);
      setAll([]);
      return;
    }
    let alive = true;
    (async () => {
      await flush();
      const m = await api.loadAnnotations(imageId);
      if (alive) applyLoaded(m);
    })().catch((e) => message.error(errText(e)));
    return () => {
      alive = false;
    };
  }, [imageId, flush, message]); // eslint-disable-line react-hooks/exhaustive-deps

  const go = useCallback(
    (i: number) => {
      if (!ids || i < 0 || i >= ids.length) return;
      setIdx(i);
      refreshStats();
      loadLabels();
    },
    [ids, refreshStats, loadLabels],
  );

  const setStatus = async (status: ImageStatus, next: boolean) => {
    const m = metaRef.current;
    if (!m) return;
    try {
      await flush();
      await api.setImageStatus(m.id, status);
      metaRef.current = { ...m, status };
      setMeta(metaRef.current);
      if (next) go(idx + 1);
      else refreshStats();
    } catch (e) {
      message.error(errText(e));
    }
  };

  // ---------- 编辑 ----------

  const onCreate = (r: RectXYWH) => {
    if (curLabel == null) {
      message.warning("先在右侧添加一个类别");
      return;
    }
    const b: Box = { key: newKey(), labelId: curLabel, ...r, source: "manual" };
    commit([...boxesRef.current, b]);
    setSelected(b.key);
  };

  // 人改过的模型框算人工确认
  const onUpdate = (key: string, r: RectXYWH) =>
    commit(boxesRef.current.map((b) => (b.key === key ? { ...b, ...r, source: "manual", score: null } : b)));

  // ---------- AI 预标注 ----------

  const [models, setModels] = useState<RunInfo[]>([]);
  const [modelId, setModelId] = useState<number | null>(null);
  const [conf, setConf] = useState(Number(prefs.get("prelabelConf", "0.4")));
  const [predicting, setPredicting] = useState(false);
  const [batch, setBatch] = useState<{ done: number; total: number } | null>(null);

  useEffect(() => {
    api
      .listRuns()
      .then((rs) => {
        const ok = rs.filter((r) => r.hasWeights);
        setModels(ok);
        setModelId(ok[0]?.id ?? null);
      })
      .catch(() => {});
  }, []);

  const predictCurrent = async () => {
    const m = metaRef.current;
    if (!m || modelId == null) return;
    setPredicting(true);
    try {
      const preds = await api.predictImage(modelId, m.id, conf);
      await loadLabels();
      // 和已有框重叠很大的不重复添加
      const iou = (a: RectXYWH, b: RectXYWH) => {
        const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
        const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
        const inter = ix * iy;
        return inter / (a.w * a.h + b.w * b.h - inter);
      };
      const fresh = preds
        .filter((p) => !boxesRef.current.some((b) => iou(b, p) > 0.7))
        .map((p) => ({ key: newKey(), labelId: p.labelId, x: p.x, y: p.y, w: p.w, h: p.h, source: "model", score: p.score }));
      if (fresh.length === 0) {
        message.info(preds.length ? "模型找到的目标都已经标过了" : "模型没有找到目标，可以调低置信度试试");
      } else {
        commit([...boxesRef.current, ...fresh]);
        message.success(`添加了 ${fresh.length} 个预标注框（虚线），检查后按 Enter 通过`);
      }
    } catch (e) {
      message.error(errText(e));
    } finally {
      setPredicting(false);
    }
  };

  const predictAll = () => {
    if (modelId == null) return;
    modal.confirm({
      title: "批量预标注",
      content: "对所有「未标注」且还没有框的图片运行模型。预标注的图片仍然算未标注，需要你逐张检查确认后才会用于训练。",
      okText: "开始",
      cancelText: "取消",
      onOk: async () => {
        await flush();
        setBatch({ done: 0, total: 0 });
        const un = await listen<{ done: number; total: number }>("prelabel-progress", (e) => setBatch(e.payload));
        try {
          const r = await api.prelabelAll(modelId, conf);
          await loadLabels();
          await reloadCurrent();
          message.success(`已预标注 ${r.images} 张图片，共 ${r.boxes} 个框`);
        } catch (e) {
          message.error(errText(e));
        } finally {
          un();
          setBatch(null);
        }
      },
    });
  };

  const deleteBox = (key: string | null) => {
    if (!key) return;
    commit(boxesRef.current.filter((b) => b.key !== key));
    setSelected(null);
  };

  const copyPrev = async () => {
    if (!ids || idx === 0 || !metaRef.current) return;
    try {
      const prev = await api.loadAnnotations(ids[idx - 1]);
      const { width: W, height: H } = metaRef.current;
      const copied = prev.annotations
        .filter((a) => a.type === "bbox")
        .map(toBox)
        .map((b) => ({ ...b, x: Math.min(b.x, W - 1), y: Math.min(b.y, H - 1) }))
        .map((b) => ({ ...b, w: Math.min(b.w, W - b.x), h: Math.min(b.h, H - b.y), source: "manual" }));
      if (copied.length === 0) {
        message.info("上一张没有标注");
        return;
      }
      commit([...boxesRef.current, ...copied]);
      message.success(`已复制上一张的 ${copied.length} 个框`);
    } catch (e) {
      message.error(errText(e));
    }
  };

  // ---------- 快捷键 ----------

  const keyHandler = useRef<(e: KeyboardEvent) => void>(() => {});
  keyHandler.current = (e: KeyboardEvent) => {
    const t = e.target as HTMLElement;
    if (t?.tagName === "INPUT" || t?.tagName === "TEXTAREA" || t?.closest?.(".ant-modal")) return;
    const k = e.key.toLowerCase();
    const ctrl = e.ctrlKey || e.metaKey;
    // 焦点在按钮上时 Enter 由按钮自己处理，避免触发两次
    if (t?.tagName === "BUTTON" && k === "enter") return;
    if (ctrl && k === "z" && !e.shiftKey) undo();
    else if (ctrl && (k === "y" || (k === "z" && e.shiftKey))) redo();
    else if (ctrl && k === "d") copyPrev();
    else if (ctrl) return;
    else if (k === "v") setTool("select");
    else if (k === "r") setTool("rect");
    else if (/^[1-9]$/.test(k)) {
      const l = labels[Number(k) - 1];
      if (l) pickLabel(l.id);
    } else if (k === "delete" || k === "backspace") deleteBox(selected);
    else if (k === "d" || k === "arrowright") go(idx + 1);
    else if (k === "a" || k === "arrowleft") go(idx - 1);
    else if (k === "enter") setStatus("reviewed", true);
    else if (k === "f") setFitSignal((n) => n + 1);
    else if (k === "h") setShowNames((v) => !v);
    else if (k === "p" && modelId != null && !predicting) predictCurrent();
    else if (k === "escape") setSelected(null);
    else return;
    e.preventDefault();
  };
  useEffect(() => {
    const h = (e: KeyboardEvent) => keyHandler.current(e);
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  // ---------- 渲染 ----------

  if (ids && ids.length === 0 && filter === "all") {
    return (
      <div className="empty-page">
        <Empty description="项目里还没有图片">
          <Button type="primary" onClick={() => setPage("data")}>
            去导入图片
          </Button>
        </Empty>
      </div>
    );
  }

  const stripStart = Math.max(0, idx - 12);
  const strip = (ids ?? []).slice(stripStart, stripStart + 25);
  const curColor = colorOf(curLabel);
  const saveText = { saved: "已保存", dirty: "未保存", saving: "保存中…", error: "保存失败" }[saveState];

  return (
    <div className="annot">
      <div className="annot-top">
        <Button size="small" icon={<LeftOutlined />} disabled={idx === 0} onClick={() => go(idx - 1)} />
        <span style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          <strong>{meta?.name ?? "—"}</strong>{" "}
          <span className="muted">
            · {ids?.length ? idx + 1 : 0} / {ids?.length ?? 0}
            {meta && ` · ${meta.width}×${meta.height}`}
          </span>
        </span>
        <Button size="small" icon={<RightOutlined />} disabled={!ids || idx >= ids.length - 1} onClick={() => go(idx + 1)} />
        {meta && <Tag color={STATUS_TEXT[meta.status].color}>{STATUS_TEXT[meta.status].text}</Tag>}
        <Segmented size="small" value={filter} options={FILTERS} onChange={(v) => setFilter(v as string)} />
        <div className="spacer" />
        <span className="dim" style={{ fontSize: 12 }}>
          {Math.round(scale * 100)}%
        </span>
        <span style={{ fontSize: 12, color: saveState === "error" ? "var(--bad)" : "var(--dim)" }}>{saveText}</span>
      </div>

      <div className="annot-tools">
        <Tooltip title="选择 / 调整框 (V)" placement="right">
          <button className={tool === "select" ? "on" : ""} onClick={() => setTool("select")}>
            <SelectOutlined />
            <span className="k">V</span>
          </button>
        </Tooltip>
        <Tooltip title="画矩形框 (R)" placement="right">
          <button className={tool === "rect" ? "on" : ""} onClick={() => setTool("rect")}>
            <BorderOutlined />
            <span className="k">R</span>
          </button>
        </Tooltip>
        <hr />
        <Tooltip title="撤销 (Ctrl+Z)" placement="right">
          <button disabled={histLen.past === 0} onClick={undo}>
            <UndoOutlined />
          </button>
        </Tooltip>
        <Tooltip title="重做 (Ctrl+Y)" placement="right">
          <button disabled={histLen.future === 0} onClick={redo}>
            <RedoOutlined />
          </button>
        </Tooltip>
        <Tooltip title="复制上一张的框 (Ctrl+D)" placement="right">
          <button disabled={idx === 0} onClick={copyPrev}>
            <CopyOutlined />
          </button>
        </Tooltip>
        <Tooltip title="适应窗口 (F)" placement="right">
          <button onClick={() => setFitSignal((n) => n + 1)}>
            <ExpandOutlined />
          </button>
        </Tooltip>
      </div>

      <div className="annot-stage">
        {meta && imageId !== undefined ? (
          <AnnotCanvas
            src={imageUrl(project.path, imageId)}
            imgW={meta.width}
            imgH={meta.height}
            boxes={boxes}
            colorOf={colorOf}
            nameOf={nameOf}
            selected={selected}
            tool={tool}
            drawColor={curColor}
            showNames={showNames}
            fitSignal={fitSignal}
            onSelect={setSelected}
            onCreate={onCreate}
            onUpdate={onUpdate}
            onScale={setScale}
            highlight={highlight && highlight.imageId === imageId ? highlight : null}
          />
        ) : (
          ids?.length === 0 && <Empty description="没有符合筛选条件的图片" />
        )}
        <div className="annot-hint">
          R 画框 · V 选择 · 1–9 切换类别 · Del 删除 · A/D 翻图 · Enter 审核通过 · P 模型预标注 · 滚轮缩放 · 空格或右键拖动 · H 隐藏类别名
        </div>
      </div>

      <aside className="annot-side">
        <section>
          <h3>AI 预标注</h3>
          {models.length === 0 ? (
            <div className="dim" style={{ fontSize: 12 }}>
              标注几十张后去「训练」页训练一个模型，就能用它自动标注剩下的图片。
            </div>
          ) : (
            <>
              <Select
                size="small"
                style={{ width: "100%", marginBottom: 6 }}
                value={modelId}
                onChange={setModelId}
                options={models.map((r) => ({ value: r.id, label: `${r.name} · mAP50 ${r.bestMap50?.toFixed(3) ?? "—"}` }))}
              />
              <div className="row" style={{ gap: 6, marginBottom: 6, flexWrap: "nowrap" }}>
                <span className="dim" style={{ fontSize: 12, whiteSpace: "nowrap" }}>置信度</span>
                <Slider
                  style={{ flex: 1, margin: "0 4px" }}
                  min={0.05}
                  max={0.95}
                  step={0.05}
                  value={conf}
                  onChange={(v) => {
                    setConf(v);
                    prefs.set("prelabelConf", String(v));
                  }}
                />
                <span style={{ fontSize: 12, width: 30 }}>{conf.toFixed(2)}</span>
              </div>
              <div className="row" style={{ gap: 6 }}>
                <Button size="small" type="primary" icon={<ThunderboltOutlined />} loading={predicting} onClick={predictCurrent}>
                  标注本图 (P)
                </Button>
                <Button size="small" onClick={predictAll} disabled={!!batch}>
                  批量预标注
                </Button>
              </div>
              {batch && (
                <Progress size="small" percent={batch.total ? Math.floor((batch.done / batch.total) * 100) : 0} style={{ marginTop: 6 }} />
              )}
            </>
          )}
        </section>

        <section>
          <h3>类别</h3>
          {labels.length === 0 && <div className="dim" style={{ marginBottom: 8 }}>先添加要标注的类别</div>}
          {labels.map((l, i) => (
            <div key={l.id} className={l.id === curLabel ? "lbl on" : "lbl"} onClick={() => pickLabel(l.id)}>
              <span className="sw" style={{ background: l.color }} />
              <span className="n">{l.name}</span>
              <span className="dim" style={{ fontSize: 11 }}>
                {l.count}
              </span>
              {i < 9 && <kbd>{i + 1}</kbd>}
              <span className="ops">
                <EditOutlined
                  onClick={(e) => {
                    e.stopPropagation();
                    setEditing({ ...l });
                  }}
                />
                <DeleteOutlined
                  onClick={(e) => {
                    e.stopPropagation();
                    removeLabel(l);
                  }}
                />
              </span>
            </div>
          ))}
          <Space.Compact style={{ width: "100%", marginTop: 8 }}>
            <Input
              size="small"
              placeholder="新类别名，回车添加"
              value={newLabel}
              maxLength={40}
              onChange={(e) => setNewLabel(e.target.value)}
              onPressEnter={addLabel}
            />
            <Button size="small" icon={<PlusOutlined />} onClick={addLabel} />
          </Space.Compact>
        </section>

        <section>
          <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}>
            <h3 style={{ margin: 0 }}>本图对象 <span className="dim">{boxes.length}</span></h3>
            <span className="row" style={{ gap: 6, fontSize: 12 }}>
              <span className="dim">类别名</span>
              <Switch size="small" checked={showNames} onChange={setShowNames} />
            </span>
          </div>
          {boxes.length === 0 && <div className="dim">还没有框，按 R 在图上拖动画框</div>}
          {boxes.map((b, i) => (
            <div
              key={b.key}
              className={b.key === selected ? "obj on" : "obj"}
              onClick={() => {
                setSelected(b.key);
                setTool("select");
              }}
            >
              <span className="sw" style={{ background: colorOf(b.labelId) }} />
              <span className="n">
                {i + 1}. {nameOf(b.labelId)}
              </span>
              <span className="dim" style={{ fontSize: 11 }}>
                {Math.round(b.w)}×{Math.round(b.h)}
              </span>
              {b.source !== "manual" && <Tag style={{ marginInlineEnd: 0 }}>{SOURCE_TEXT[b.source] ?? b.source}</Tag>}
              <DeleteOutlined
                className="del"
                onClick={(e) => {
                  e.stopPropagation();
                  deleteBox(b.key);
                }}
              />
            </div>
          ))}
        </section>

        <section>
          <h3>审核</h3>
          <div className="row" style={{ gap: 6 }}>
            <Button size="small" icon={<CheckOutlined />} onClick={() => setStatus("reviewed", true)}>
              通过并下一张
            </Button>
            <Button size="small" icon={<StepForwardOutlined />} onClick={() => setStatus("skipped", true)}>
              跳过
            </Button>
            {meta && (meta.status === "reviewed" || meta.status === "skipped") && (
              <Button size="small" onClick={() => setStatus(boxes.length ? "labeled" : "todo", false)}>
                撤回
              </Button>
            )}
          </div>
          <div className="dim" style={{ fontSize: 12, marginTop: 6 }}>
            「跳过」的图不参与训练，可用于模糊、无关的图片。
          </div>
        </section>
      </aside>

      <div className="annot-strip">
        {strip.map((id, i) => (
          <div key={id} className={id === imageId ? "t on" : "t"} onClick={() => go(stripStart + i)}>
            <img src={thumbUrl(project.path, id)} alt="" draggable={false} />
          </div>
        ))}
      </div>

      <Modal
        title="编辑类别"
        open={!!editing}
        onCancel={() => setEditing(null)}
        onOk={saveLabelEdit}
        okText="保存"
        cancelText="取消"
        width={380}
      >
        {editing && (
          <Space style={{ marginTop: 12 }}>
            <ColorPicker
              value={editing.color}
              disabledAlpha
              presets={[{ label: "常用", colors: PALETTE }]}
              onChange={(c) => setEditing({ ...editing, color: c.toHexString() })}
            />
            <Input
              value={editing.name}
              maxLength={40}
              onChange={(e) => setEditing({ ...editing, name: e.target.value })}
              onPressEnter={saveLabelEdit}
              style={{ width: 240 }}
            />
          </Space>
        )}
      </Modal>
    </div>
  );
}
