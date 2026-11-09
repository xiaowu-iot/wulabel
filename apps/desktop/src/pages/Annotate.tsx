import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { App, Button, ColorPicker, Empty, Input, Modal, Progress, Segmented, Select, Slider, Space, Switch, Tag, Tooltip } from "antd";
import { listen } from "@tauri-apps/api/event";
import {
  AimOutlined,
  BorderOutlined,
  CheckOutlined,
  CopyOutlined,
  DeleteOutlined,
  EditOutlined,
  ExpandOutlined,
  HighlightOutlined,
  LeftOutlined,
  PlusOutlined,
  RedoOutlined,
  RightOutlined,
  SelectOutlined,
  StepForwardOutlined,
  ThunderboltOutlined,
  UndoOutlined,
} from "@ant-design/icons";
import {
  AiModel,
  api,
  Annotation,
  errText,
  ImageAnnotations,
  ImageStatus,
  imageUrl,
  KeypointData,
  KeypointTemplate,
  Label,
  PredResult,
  PredTag,
  ProjectInfo,
  RunInfo,
  SamResult,
  thumbUrl,
} from "../api";
import { prefs, useApp } from "../store";
import AnnotCanvas, { Box, Poly, RectXYWH, Tool } from "../components/AnnotCanvas";

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
const SOURCE_TEXT: Record<string, string> = { manual: "手工", model: "模型", sam: "SAM", ai: "巡瞳", text: "文本" };
const HISTORY_MAX = 200;

/** COCO 人体 17 点（和后端 keypoints.rs 的默认模板一致） */
const COCO_PERSON: KeypointTemplate = {
  name: "COCO 人体 17 点",
  points: ["鼻子", "左眼", "右眼", "左耳", "右耳", "左肩", "右肩", "左肘", "右肘", "左腕", "右腕", "左髋", "右髋", "左膝", "右膝", "左踝", "右踝"],
  skeleton: [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
    [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
  ],
};

let keySeq = 0;
const newKey = () => `b${++keySeq}`;

/** 关键点目标：目标框 + 关键点 [x, y, v, ...] */
interface Pose {
  key: string;
  labelId: number | null;
  x: number;
  y: number;
  w: number;
  h: number;
  kpts: number[];
  source: string;
  score?: number | null;
}

/** 图像分类：整图一个类别 */
interface ClsTag {
  labelId: number;
  source: string;
  score?: number | null;
}

const toBox = (a: Annotation): Box => {
  const d = a.data as number[];
  return { key: newKey(), labelId: a.labelId, x: d[0], y: d[1], w: d[2], h: d[3], source: a.source, score: a.score };
};

const toAnn = (b: Box): Annotation => ({
  labelId: b.labelId,
  type: "bbox",
  data: [b.x, b.y, b.w, b.h],
  score: b.score ?? null,
  source: b.source,
});

const toPoly = (a: Annotation): Poly => ({
  key: newKey(),
  labelId: a.labelId,
  points: a.data as number[],
  source: a.source,
  score: a.score,
});

const polyToAnn = (pl: Poly): Annotation => ({
  labelId: pl.labelId,
  type: "polygon",
  data: pl.points,
  score: pl.score ?? null,
  source: pl.source,
});

const toPose = (a: Annotation): Pose => {
  const d = a.data as KeypointData;
  return {
    key: newKey(),
    labelId: a.labelId,
    x: d.box[0],
    y: d.box[1],
    w: d.box[2],
    h: d.box[3],
    kpts: [...d.kpts],
    source: a.source,
    score: a.score,
  };
};

const poseToAnn = (p: Pose): Annotation => ({
  labelId: p.labelId,
  type: "keypoints",
  data: { box: [p.x, p.y, p.w, p.h], kpts: p.kpts },
  score: p.score ?? null,
  source: p.source,
});

const tagToAnn = (t: ClsTag): Annotation => ({ labelId: t.labelId, type: "tag", data: [], score: t.score ?? null, source: t.source });

const polyBox = (pts: number[]): RectXYWH => {
  const xs = pts.filter((_, i) => i % 2 === 0);
  const ys = pts.filter((_, i) => i % 2 === 1);
  const x = Math.min(...xs);
  const y = Math.min(...ys);
  return { x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y };
};

const iou = (a: RectXYWH, b: RectXYWH) => {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const inter = ix * iy;
  return inter / (a.w * a.h + b.w * b.h - inter || 1);
};

const r2 = (v: number) => Math.round(v * 100) / 100;

type SaveState = "saved" | "dirty" | "saving" | "error";
type Snap = { boxes: Box[]; polys: Poly[]; poses: Pose[]; tag: ClsTag | null };
type AiSrc = "model" | "text" | "ai";

export default function Annotate({ project }: { project: ProjectInfo }) {
  const { message, modal } = App.useApp();
  const refreshStats = useApp((s) => s.refreshStats);
  const setPage = useApp((s) => s.setPage);
  const takeFocusImage = useApp((s) => s.takeFocusImage);
  const highlight = useApp((s) => s.highlight);

  const task = project.task;
  const isSeg = task === "segment";
  const isCls = task === "classify";
  const isPose = task === "pose";
  const drawTool: Tool = isSeg ? "polygon" : "rect";

  const [labels, setLabels] = useState<Label[]>([]);
  const [curLabel, setCurLabel] = useState<number | null>(null);
  const [filter, setFilter] = useState("all");
  const [ids, setIds] = useState<number[] | null>(null);
  const [idx, setIdx] = useState(0);
  const [meta, setMeta] = useState<ImageAnnotations | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [polys, setPolys] = useState<Poly[]>([]);
  const [poses, setPoses] = useState<Pose[]>([]);
  const [tag, setTag] = useState<ClsTag | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [tool, setTool] = useState<Tool>(isCls ? "select" : drawTool);
  const [drawingPoly, setDrawingPoly] = useState(false);
  const [saveState, setSaveState] = useState<SaveState>("saved");
  const [fitSignal, setFitSignal] = useState(0);
  const [showNames, setShowNames] = useState(true);
  const [scale, setScale] = useState(1);
  const [newLabel, setNewLabel] = useState("");
  const [editing, setEditing] = useState<Label | null>(null);
  const [histLen, setHistLen] = useState({ past: 0, future: 0 });
  // 图像分类：选完类别自动下一张
  const [autoNext, setAutoNext] = useState(prefs.get("clsAutoNext", "1") === "1");
  const [tagHints, setTagHints] = useState<PredTag[]>([]);
  // 关键点
  const [template, setTemplate] = useState<KeypointTemplate | null>(null);
  const [placing, setPlacing] = useState<{ key: string; idx: number; single?: boolean } | null>(null);
  const [tplEdit, setTplEdit] = useState<{ name: string; points: string; skeleton: string } | null>(null);
  // SAM
  const [samPoints, setSamPoints] = useState<number[][]>([]);
  const [samBox, setSamBox] = useState<RectXYWH | null>(null);
  const [samPreview, setSamPreview] = useState<SamResult | null>(null);
  const [samBusy, setSamBusy] = useState(false);
  const samUsed = useRef(false);

  const boxesRef = useRef<Box[]>([]);
  const polysRef = useRef<Poly[]>([]);
  const posesRef = useRef<Pose[]>([]);
  const tagRef = useRef<ClsTag | null>(null);
  const metaRef = useRef<ImageAnnotations | null>(null);
  const othersRef = useRef<Annotation[]>([]); // 不认识的标注类型原样保留
  const history = useRef<{ past: Snap[]; future: Snap[] }>({ past: [], future: [] });
  const dirty = useRef(false);
  const timer = useRef<number>();

  const labelMap = useMemo(() => new Map(labels.map((l) => [l.id, l])), [labels]);
  const colorOf = useCallback((id: number | null) => (id != null && labelMap.get(id)?.color) || "#9aa1ab", [labelMap]);
  const nameOf = useCallback((id: number | null) => (id != null && labelMap.get(id)?.name) || "未分类", [labelMap]);
  const nk = template?.points.length ?? 0;

  // ---------- 保存 ----------

  const flush = useCallback(async () => {
    window.clearTimeout(timer.current);
    const m = metaRef.current;
    if (!dirty.current || !m) return;
    dirty.current = false;
    setSaveState("saving");
    try {
      const status = await api.saveAnnotations(m.id, [
        ...boxesRef.current.map(toAnn),
        ...polysRef.current.map(polyToAnn),
        ...posesRef.current.map(poseToAnn),
        ...(tagRef.current ? [tagToAnn(tagRef.current)] : []),
        ...othersRef.current,
      ]);
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

  /** 直接替换当前图的标注（不记历史、不保存） */
  const setSnap = (s: Partial<Snap>) => {
    if (s.boxes) {
      boxesRef.current = s.boxes;
      setBoxes(s.boxes);
    }
    if (s.polys) {
      polysRef.current = s.polys;
      setPolys(s.polys);
    }
    if (s.poses) {
      posesRef.current = s.poses;
      setPoses(s.poses);
    }
    if ("tag" in s) {
      tagRef.current = s.tag ?? null;
      setTag(s.tag ?? null);
    }
  };
  const snap = (): Snap => ({ boxes: boxesRef.current, polys: polysRef.current, poses: posesRef.current, tag: tagRef.current });

  /** 修改标注：记历史、自动保存 */
  const change = (s: Partial<Snap>) => {
    const h = history.current;
    h.past.push(snap());
    if (h.past.length > HISTORY_MAX) h.past.shift();
    h.future = [];
    setHistLen({ past: h.past.length, future: 0 });
    setSnap(s);
    schedule();
  };
  const commit = (next: Box[]) => change({ boxes: next });
  const commitPoly = (next: Poly[]) => change({ polys: next });
  const commitPose = (next: Pose[]) => change({ poses: next });
  const commitTag = (t: ClsTag | null) => change({ tag: t });

  const undo = () => {
    const h = history.current;
    const prev = h.past.pop();
    if (!prev) return;
    h.future.push(snap());
    setHistLen({ past: h.past.length, future: h.future.length });
    setSnap(prev);
    setSelected(null);
    setPlacing(null);
    schedule();
  };

  const redo = () => {
    const h = history.current;
    const next = h.future.pop();
    if (!next) return;
    h.past.push(snap());
    setHistLen({ past: h.past.length, future: h.future.length });
    setSnap(next);
    setSelected(null);
    setPlacing(null);
    schedule();
  };

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

  // 关键点模板
  const loadTemplate = useCallback(async () => {
    if (isPose) setTemplate(await api.keypointTemplate());
  }, [isPose]);
  useEffect(() => {
    loadTemplate().catch((e) => message.error(errText(e)));
  }, [loadTemplate, message]);

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
      await api.updateLabel(editing.id, editing.name, editing.color, editing.prompt ?? "");
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

  /** 选类别：分类项目直接给整图打这个类别；其他任务同时把选中的目标改成这个类别 */
  const pickLabel = (id: number) => {
    setCurLabel(id);
    if (isCls) {
      if (tagRef.current?.labelId !== id || tagRef.current.source !== "manual") commitTag({ labelId: id, source: "manual" });
      if (autoNext && ids && idx < ids.length - 1) go(idx + 1);
      return;
    }
    if (!selected) return;
    const relabel = <T extends { key: string; labelId: number | null }>(arr: T[]) =>
      arr.map((x) => (x.key === selected ? { ...x, labelId: id, source: "manual", score: null } : x));
    if (boxesRef.current.some((x) => x.key === selected && x.labelId !== id)) commit(relabel(boxesRef.current));
    else if (polysRef.current.some((x) => x.key === selected && x.labelId !== id)) commitPoly(relabel(polysRef.current));
    else if (posesRef.current.some((x) => x.key === selected && x.labelId !== id)) commitPose(relabel(posesRef.current));
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

  const clearSam = () => {
    setSamPoints([]);
    setSamBox(null);
    setSamPreview(null);
  };

  const applyLoaded = (m: ImageAnnotations) => {
    metaRef.current = m;
    setMeta(m);
    const known = ["bbox", "polygon", "keypoints", "tag"];
    othersRef.current = m.annotations.filter((a) => !known.includes(a.type));
    const t = m.annotations.find((a) => a.type === "tag" && a.labelId != null);
    setSnap({
      boxes: m.annotations.filter((a) => a.type === "bbox").map(toBox),
      polys: m.annotations.filter((a) => a.type === "polygon").map(toPoly),
      poses: m.annotations.filter((a) => a.type === "keypoints").map(toPose),
      tag: t ? { labelId: t.labelId as number, source: t.source, score: t.score } : null,
    });
    history.current = { past: [], future: [] };
    setHistLen({ past: 0, future: 0 });
    setSelected(null);
    setPlacing(null);
    setTagHints([]);
    clearSam();
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
      setSnap({ boxes: [], polys: [], poses: [], tag: null });
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

  const needLabel = () => {
    if (curLabel == null) {
      message.warning("先在右侧添加一个类别");
      return true;
    }
    return false;
  };

  /** 新建关键点目标：先有框，然后按模板顺序逐个点关键点 */
  const createPose = (r: RectXYWH, source = "manual", score: number | null = null, labelId = curLabel) => {
    if (!template) return;
    const p: Pose = { key: newKey(), labelId, ...r, kpts: new Array(nk * 3).fill(0), source, score };
    commitPose([...posesRef.current, p]);
    setSelected(p.key);
    setPlacing({ key: p.key, idx: 0 });
  };

  const onCreate = (r: RectXYWH) => {
    if (needLabel()) return;
    if (isPose) {
      createPose(r);
      return;
    }
    const b: Box = { key: newKey(), labelId: curLabel, ...r, source: "manual" };
    commit([...boxesRef.current, b]);
    setSelected(b.key);
  };

  // 人改过的模型框算人工确认；关键点目标的框也在这里调整
  const onUpdate = (key: string, r: RectXYWH) => {
    if (posesRef.current.some((p) => p.key === key)) {
      commitPose(
        posesRef.current.map((p) => {
          if (p.key !== key) return p;
          // 整体拖动（大小没变）时关键点跟着框一起移动；调整大小时关键点不动
          const moved = Math.abs(r.w - p.w) < 0.01 && Math.abs(r.h - p.h) < 0.01;
          const dx = r.x - p.x;
          const dy = r.y - p.y;
          const kpts = moved
            ? p.kpts.map((v, i) => (p.kpts[i - (i % 3) + 2] > 0 ? (i % 3 === 0 ? v + dx : i % 3 === 1 ? v + dy : v) : v))
            : p.kpts;
          return { ...p, ...r, kpts, source: "manual", score: null };
        }),
      );
      return;
    }
    commit(boxesRef.current.map((b) => (b.key === key ? { ...b, ...r, source: "manual", score: null } : b)));
  };

  const onCreatePoly = (points: number[]) => {
    if (needLabel()) return;
    const pl: Poly = { key: newKey(), labelId: curLabel, points, source: "manual" };
    commitPoly([...polysRef.current, pl]);
    setSelected(pl.key);
  };

  const onUpdatePoly = (key: string, points: number[]) =>
    commitPoly(polysRef.current.map((pl) => (pl.key === key ? { ...pl, points, source: "manual", score: null } : pl)));

  // ---------- 关键点 ----------

  const setKpt = (key: string, i: number, x: number, y: number, v: number) =>
    commitPose(
      posesRef.current.map((p) => {
        if (p.key !== key) return p;
        const k = [...p.kpts];
        while (k.length < nk * 3) k.push(0);
        k[i * 3] = v > 0 ? x : 0;
        k[i * 3 + 1] = v > 0 ? y : 0;
        k[i * 3 + 2] = v;
        return { ...p, kpts: k, source: "manual", score: null };
      }),
    );

  /** 放好当前点，转到下一个点（单点重放时放完就结束） */
  const advance = (cur: { key: string; idx: number; single?: boolean }) => {
    if (cur.single || cur.idx + 1 >= nk) setPlacing(null);
    else setPlacing({ key: cur.key, idx: cur.idx + 1 });
  };

  const onPlaceKpt = (x: number, y: number, occluded: boolean) => {
    if (!placing) return;
    if (!posesRef.current.some((p) => p.key === placing.key)) {
      setPlacing(null);
      return;
    }
    setKpt(placing.key, placing.idx, x, y, occluded ? 1 : 2);
    advance(placing);
  };

  const skipKpt = () => {
    if (!placing) return;
    setKpt(placing.key, placing.idx, 0, 0, 0);
    advance(placing);
  };

  const backKpt = () => {
    if (!placing || placing.idx === 0) return;
    setKpt(placing.key, placing.idx - 1, 0, 0, 0);
    setPlacing({ key: placing.key, idx: placing.idx - 1 });
  };

  const onMoveKpt = (key: string, i: number, x: number, y: number) => {
    const p = posesRef.current.find((q) => q.key === key);
    const v = p?.kpts[i * 3 + 2] ?? 2;
    setKpt(key, i, x, y, v > 0 ? v : 2);
  };

  const openTplEdit = () => {
    const t = template ?? COCO_PERSON;
    setTplEdit({
      name: t.name,
      points: t.points.join("\n"),
      skeleton: t.skeleton.map(([a, b]) => `${a + 1}-${b + 1}`).join("\n"),
    });
  };

  const saveTpl = async () => {
    if (!tplEdit) return;
    const points = tplEdit.points.split("\n").map((s) => s.trim()).filter(Boolean);
    const skeleton: [number, number][] = [];
    for (const line of tplEdit.skeleton.split("\n").map((s) => s.trim()).filter(Boolean)) {
      const m = /^(\d+)\s*[-–—,，\s]\s*(\d+)$/.exec(line);
      if (!m) {
        message.error(`连线「${line}」格式不对，应该写成 1-2`);
        return;
      }
      skeleton.push([Number(m[1]) - 1, Number(m[2]) - 1]);
    }
    try {
      await api.saveKeypointTemplate({ name: tplEdit.name, points, skeleton });
      setTplEdit(null);
      await loadTemplate();
      message.success("模板已保存");
    } catch (e) {
      message.error(errText(e));
    }
  };

  // ---------- AI 辅助（我的模型 / 文本提示 / 巡瞳 AI）----------

  const [models, setModels] = useState<RunInfo[]>([]);
  const [modelId, setModelId] = useState<number | null>(null);
  const [conf, setConf] = useState(Number(prefs.get("prelabelConf", "0.4")));
  const [predicting, setPredicting] = useState(false);
  const [batch, setBatch] = useState<{ done: number; total: number } | null>(null);
  const [aiSrc, setAiSrc] = useState<AiSrc>(isCls ? "model" : ((prefs.get("aiSrc", "model") as AiSrc) || "model"));
  const [aiModels, setAiModels] = useState<AiModel[] | null>(null);
  const [aiModel, setAiModel] = useState(prefs.get("aiModel", ""));
  const [aiError, setAiError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listRuns()
      .then((rs) => {
        // 只列和项目任务对得上的模型
        const ok = rs.filter((r) => r.hasWeights);
        setModels(ok);
        setModelId(ok[0]?.id ?? null);
      })
      .catch(() => {});
  }, []);

  // 切到巡瞳 AI 时拉一次可用模型
  useEffect(() => {
    if (aiSrc !== "ai" || aiModels) return;
    api
      .aiModels()
      .then((ms) => {
        setAiModels(ms);
        setAiError(null);
        if (!ms.some((m) => m.code === aiModel)) setAiModel(ms[0]?.code ?? "");
      })
      .catch((e) => setAiError(errText(e)));
  }, [aiSrc, aiModels, aiModel]);

  const pickSrc = (v: AiSrc) => {
    setAiSrc(v);
    prefs.set("aiSrc", v);
  };

  /** 把一次预测结果放进当前图（和已有标注重叠很大的不重复添加）。source：model / text / ai */
  const applyPreds = (preds: PredResult, source: string) => {
    const tagWord = source === "model" ? "模型" : source === "ai" ? "巡瞳 AI" : "文本提示";
    if (isCls) {
      setTagHints(preds.tags.slice(0, 3));
      const top = preds.tags[0];
      if (!top) {
        message.info("模型没有给出结果");
      } else if (tagRef.current) {
        message.info(`${tagWord}认为是「${nameOf(top.labelId)}」（${top.score.toFixed(2)}），已有类别不覆盖，可在下方点选`);
      } else if (top.score >= conf) {
        commitTag({ labelId: top.labelId, source, score: top.score });
        message.success(`${tagWord}认为是「${nameOf(top.labelId)}」（${top.score.toFixed(2)}），检查后按 Enter 通过`);
      } else {
        message.info(`${tagWord}最像「${nameOf(top.labelId)}」，但只有 ${top.score.toFixed(2)}，低于置信度，没有采用`);
      }
      return;
    }
    const none = () => message.info(`${tagWord}没有找到目标，可以调低置信度试试`);
    const allDone = () => message.info(`${tagWord}找到的目标都已经标过了`);
    if (isPose) {
      // 关键点模型给完整的点；其他来源只有框，加成空的目标，再手工点关键点
      const have = posesRef.current.map((p) => ({ x: p.x, y: p.y, w: p.w, h: p.h }));
      const fromPoses = preds.poses.map((p) => ({ labelId: p.labelId, r: { x: p.box[0], y: p.box[1], w: p.box[2], h: p.box[3] }, kpts: p.kpts, score: p.score }));
      const fromBoxes = preds.boxes.map((b) => ({ labelId: b.labelId, r: { x: b.x, y: b.y, w: b.w, h: b.h }, kpts: new Array(nk * 3).fill(0), score: b.score }));
      const cands = fromPoses.length ? fromPoses : fromBoxes;
      if (cands.length === 0) return none();
      const fresh = cands
        .filter((c) => !have.some((h) => iou(h, c.r) > 0.7))
        .map((c) => ({ key: newKey(), labelId: c.labelId, ...c.r, kpts: c.kpts, source, score: c.score }));
      if (fresh.length === 0) return allDone();
      commitPose([...posesRef.current, ...fresh]);
      message.success(`添加了 ${fresh.length} 个目标（虚线）${fromPoses.length ? "" : "，选中后点「重放关键点」逐个标点"}`);
      return;
    }
    if (isSeg) {
      // 只有框（没有 SAM 时）就先转成矩形轮廓，再手工细修
      const cands = preds.polys.length
        ? preds.polys.map((p) => ({ labelId: p.labelId, points: p.points, score: p.score }))
        : preds.boxes.map((b) => ({ labelId: b.labelId, points: [b.x, b.y, b.x + b.w, b.y, b.x + b.w, b.y + b.h, b.x, b.y + b.h], score: b.score }));
      if (cands.length === 0) return none();
      const have = polysRef.current.map((pl) => polyBox(pl.points));
      const fresh = cands
        .filter((p) => !have.some((b) => iou(b, polyBox(p.points)) > 0.7))
        .map((p) => ({ key: newKey(), labelId: p.labelId, points: p.points, source, score: p.score }));
      if (fresh.length === 0) return allDone();
      commitPoly([...polysRef.current, ...fresh]);
      message.success(`添加了 ${fresh.length} 个预标注轮廓（虚线），检查后按 Enter 通过`);
      return;
    }
    if (preds.boxes.length === 0) return none();
    const fresh = preds.boxes
      .filter((p) => !boxesRef.current.some((b) => iou(b, p) > 0.7))
      .map((p) => ({ key: newKey(), labelId: p.labelId, x: p.x, y: p.y, w: p.w, h: p.h, source, score: p.score }));
    if (fresh.length === 0) return allDone();
    commit([...boxesRef.current, ...fresh]);
    message.success(`添加了 ${fresh.length} 个预标注框（虚线），检查后按 Enter 通过`);
  };

  const predictCurrent = async () => {
    const m = metaRef.current;
    if (!m || predicting) return;
    if (aiSrc === "model" && modelId == null) return;
    if (aiSrc === "ai" && !aiModel) return;
    setPredicting(true);
    try {
      const preds =
        aiSrc === "model"
          ? await api.predictImage(modelId as number, m.id, conf)
          : aiSrc === "text"
            ? await api.textDetect(m.id, conf)
            : await api.aiDetect(m.id, aiModel, conf);
      await loadLabels();
      applyPreds(preds, aiSrc);
    } catch (e) {
      message.error(errText(e));
    } finally {
      setPredicting(false);
    }
  };

  const predictAll = () => {
    if (aiSrc === "model" && modelId == null) return;
    const what = { model: "模型", text: "文本提示", ai: "巡瞳 AI" }[aiSrc];
    modal.confirm({
      title: `批量预标注（${what}）`,
      content: (
        <>
          对所有「未标注」且还没有标注的图片运行{what}。预标注的图片仍然算未标注，需要你逐张检查确认后才会用于训练。
          {aiSrc === "ai" && <div style={{ marginTop: 8, color: "var(--warn)" }}>巡瞳 AI 每张图计 1 次调用，会消耗你的套餐额度。</div>}
        </>
      ),
      okText: "开始",
      cancelText: "取消",
      onOk: async () => {
        await flush();
        setBatch({ done: 0, total: 0 });
        const un = await listen<{ done: number; total: number }>("prelabel-progress", (e) => setBatch(e.payload));
        try {
          const r =
            aiSrc === "model"
              ? await api.prelabelAll(modelId as number, conf)
              : aiSrc === "text"
                ? await api.textPrelabelAll(conf)
                : await api.aiPrelabelAll(aiModel, conf);
          await loadLabels();
          await reloadCurrent();
          const unit = isCls ? "个类别" : isSeg ? "个轮廓" : isPose ? "个目标" : "个框";
          message.success(`已预标注 ${r.images} 张图片，共 ${r.boxes} ${unit}`);
        } catch (e) {
          message.error(errText(e));
        } finally {
          un();
          setBatch(null);
        }
      },
    });
  };

  // ---------- SAM 交互分割 ----------

  const runSam = async (points: number[][], box: RectXYWH | null) => {
    const m = metaRef.current;
    if (!m || (points.length === 0 && !box)) {
      setSamPreview(null);
      return;
    }
    setSamBusy(true);
    try {
      const r = await api.samSegment(m.id, points, box ? [box.x, box.y, box.x + box.w, box.y + box.h] : null);
      samUsed.current = true;
      if (metaRef.current?.id === m.id) setSamPreview(r.points.length >= 6 ? r : null);
      if (r.points.length < 6) message.info("这里没分出目标，换个位置点，或者框一下试试");
    } catch (e) {
      message.error(errText(e));
    } finally {
      setSamBusy(false);
    }
  };

  const onSamClick = (x: number, y: number, negative: boolean) => {
    const pts = [...samPoints, [x, y, negative ? 0 : 1]];
    setSamPoints(pts);
    runSam(pts, samBox);
  };

  const onSamBox = (r: RectXYWH) => {
    setSamBox(r);
    setSamPoints([]);
    runSam([], r);
  };

  /** 采用 SAM 结果：分割项目得到多边形，检测项目得到贴边的框，关键点项目得到新目标 */
  const acceptSam = () => {
    const r = samPreview;
    if (!r || needLabel()) return;
    const rect = { x: r2(r.box[0]), y: r2(r.box[1]), w: r2(r.box[2]), h: r2(r.box[3]) };
    if (isSeg) {
      const pl: Poly = { key: newKey(), labelId: curLabel, points: r.points.map(r2), source: "sam", score: r.score };
      commitPoly([...polysRef.current, pl]);
      setSelected(pl.key);
    } else if (isPose) {
      createPose(rect, "sam", r.score);
    } else {
      const b: Box = { key: newKey(), labelId: curLabel, ...rect, source: "sam", score: r.score };
      commit([...boxesRef.current, b]);
      setSelected(b.key);
    }
    clearSam();
  };

  // 切走 SAM 工具时丢掉没采用的结果
  useEffect(() => {
    if (tool !== "sam") clearSam();
  }, [tool]);

  // ---------- 删除、复制上一张 ----------

  const deleteBox = (key: string | null) => {
    if (isCls) {
      if (tagRef.current) commitTag(null);
      return;
    }
    if (!key) return;
    if (boxesRef.current.some((b) => b.key === key)) commit(boxesRef.current.filter((b) => b.key !== key));
    else if (polysRef.current.some((pl) => pl.key === key)) commitPoly(polysRef.current.filter((pl) => pl.key !== key));
    else if (posesRef.current.some((p) => p.key === key)) commitPose(posesRef.current.filter((p) => p.key !== key));
    if (placing?.key === key) setPlacing(null);
    setSelected(null);
  };

  const copyPrev = async () => {
    if (!ids || idx === 0 || !metaRef.current) return;
    try {
      const prev = await api.loadAnnotations(ids[idx - 1]);
      const { width: W, height: H } = metaRef.current;
      const cx = (v: number) => Math.min(Math.max(v, 0), W);
      const cy = (v: number) => Math.min(Math.max(v, 0), H);
      if (isCls) {
        const t = prev.annotations.find((a) => a.type === "tag" && a.labelId != null);
        if (!t) {
          message.info("上一张还没有类别");
          return;
        }
        commitTag({ labelId: t.labelId as number, source: "manual" });
        message.success(`已沿用上一张的类别「${nameOf(t.labelId)}」`);
        return;
      }
      if (isPose) {
        const copied = prev.annotations
          .filter((a) => a.type === "keypoints")
          .map(toPose)
          .map((p) => {
            const x = Math.min(p.x, W - 1);
            const y = Math.min(p.y, H - 1);
            const kpts = p.kpts.map((v, i) => (i % 3 === 0 ? cx(v) : i % 3 === 1 ? cy(v) : v));
            return { ...p, key: newKey(), x, y, w: Math.min(p.w, W - x), h: Math.min(p.h, H - y), kpts, source: "manual", score: null };
          });
        if (copied.length === 0) {
          message.info("上一张没有关键点目标");
          return;
        }
        commitPose([...posesRef.current, ...copied]);
        message.success(`已复制上一张的 ${copied.length} 个目标`);
        return;
      }
      if (isSeg) {
        const copied = prev.annotations
          .filter((a) => a.type === "polygon")
          .map(toPoly)
          .map((pl) => ({
            ...pl,
            key: newKey(),
            points: pl.points.map((v, i) => (i % 2 === 0 ? cx(v) : cy(v))),
            source: "manual",
            score: null,
          }));
        if (copied.length === 0) {
          message.info("上一张没有标注");
          return;
        }
        commitPoly([...polysRef.current, ...copied]);
        message.success(`已复制上一张的 ${copied.length} 个轮廓`);
        return;
      }
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
    // 正在画多边形时，Enter/Esc/退格等交给画布处理，别触发翻图、审核等快捷键
    if (drawingPoly) return;
    const k = e.key.toLowerCase();
    const ctrl = e.ctrlKey || e.metaKey;
    // 放置关键点：N 跳过这个点、退格回到上一个点、Esc / Enter 结束
    if (placing && !ctrl) {
      if (k === "n") skipKpt();
      else if (k === "backspace") backKpt();
      else if (k === "escape" || k === "enter") setPlacing(null);
      else if (k !== " ") return;
      e.preventDefault();
      return;
    }
    // SAM：Enter 采用、Esc 放弃、退格撤掉最后一个点
    if (tool === "sam" && (samPoints.length > 0 || samBox) && !ctrl) {
      if (k === "enter") acceptSam();
      else if (k === "escape") clearSam();
      else if (k === "backspace") {
        const pts = samPoints.slice(0, -1);
        setSamPoints(pts);
        runSam(pts, samBox);
      } else return;
      e.preventDefault();
      return;
    }
    // 焦点在按钮上时 Enter 由按钮自己处理，避免触发两次
    if (t?.tagName === "BUTTON" && k === "enter") return;
    if (ctrl && k === "z" && !e.shiftKey) undo();
    else if (ctrl && (k === "y" || (k === "z" && e.shiftKey))) redo();
    else if (ctrl && k === "d") copyPrev();
    else if (ctrl) return;
    else if (k === "v" && !isCls) setTool("select");
    else if (k === "r" && !isCls) setTool(drawTool);
    else if (k === "s" && !isCls) setTool("sam");
    else if (/^[1-9]$/.test(k)) {
      const l = labels[Number(k) - 1];
      if (l) pickLabel(l.id);
    } else if (k === "delete" || k === "backspace") deleteBox(selected);
    else if (k === "d" || k === "arrowright") go(idx + 1);
    else if (k === "a" || k === "arrowleft") go(idx - 1);
    else if (k === "enter") setStatus("reviewed", true);
    else if (k === "f") setFitSignal((n) => n + 1);
    else if (k === "h") setShowNames((v) => !v);
    else if (k === "p") predictCurrent();
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
  const kptCount = (p: Pose) => p.kpts.filter((_, i) => i % 3 === 2).filter((v) => v > 0).length;

  // 右侧「本图对象」：框、多边形、关键点目标合并成一个列表，连续编号
  const objects = [
    ...boxes.map((b) => ({ key: b.key, labelId: b.labelId, source: b.source, desc: `${Math.round(b.w)}×${Math.round(b.h)}` })),
    ...polys.map((pl) => ({ key: pl.key, labelId: pl.labelId, source: pl.source, desc: `${pl.points.length / 2} 点` })),
    ...poses.map((p) => ({ key: p.key, labelId: p.labelId, source: p.source, desc: `${kptCount(p)}/${nk} 点` })),
  ];
  const selPose = poses.find((p) => p.key === selected) ?? null;
  const placingPose = placing ? poses.find((p) => p.key === placing.key) : null;
  const toolLabel = isSeg ? "画多边形轮廓 (R)" : isPose ? "画目标框，再逐个点关键点 (R)" : "画矩形框 (R)";
  const hint = isCls
    ? "1–9 选类别" + (autoNext ? "（选完自动下一张）" : "") + " · A/D 翻图 · Enter 审核通过 · Del 清除类别 · P 模型预测 · Ctrl+D 沿用上一张"
    : `${isSeg ? "R 画轮廓（逐点点击，双击或 Enter 闭合，退格撤点）" : isPose ? "R 画目标框后逐个点关键点" : "R 画框"} · S 智能选取 · V 选择 · 1–9 切换类别 · Del 删除 · A/D 翻图 · Enter 审核通过 · P 预标注 · 滚轮缩放 · 空格或右键拖动 · H 隐藏类别名`;
  const srcOptions = [
    { label: "我的模型", value: "model" },
    ...(isCls ? [] : [{ label: "文本提示", value: "text" }, { label: "巡瞳 AI", value: "ai" }]),
  ];
  const canPredict = aiSrc === "model" ? modelId != null : aiSrc === "ai" ? !!aiModel && !aiError : labels.length > 0;

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
        {!isCls && (
          <>
            <Tooltip title="选择 / 调整 (V)" placement="right">
              <button className={tool === "select" ? "on" : ""} onClick={() => setTool("select")}>
                <SelectOutlined />
                <span className="k">V</span>
              </button>
            </Tooltip>
            <Tooltip title={toolLabel} placement="right">
              <button className={tool === drawTool ? "on" : ""} onClick={() => setTool(drawTool)}>
                {isSeg ? <HighlightOutlined /> : <BorderOutlined />}
                <span className="k">R</span>
              </button>
            </Tooltip>
            <Tooltip
              title="智能选取 SAM (S)：点目标自动出轮廓，Shift+点击排除，拖动框选；Enter 采用、Esc 放弃"
              placement="right"
            >
              <button className={tool === "sam" ? "on" : ""} onClick={() => setTool("sam")}>
                <AimOutlined />
                <span className="k">S</span>
              </button>
            </Tooltip>
            <hr />
          </>
        )}
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
        <Tooltip title={isCls ? "沿用上一张的类别 (Ctrl+D)" : "复制上一张的标注 (Ctrl+D)"} placement="right">
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
            boxes={isCls ? [] : [...boxes, ...poses.map((p) => ({ key: p.key, labelId: p.labelId, x: p.x, y: p.y, w: p.w, h: p.h, source: p.source, score: p.score }))]}
            polys={polys}
            colorOf={colorOf}
            nameOf={nameOf}
            selected={selected}
            tool={isCls ? "select" : tool}
            drawColor={curColor}
            showNames={showNames}
            fitSignal={fitSignal}
            onSelect={setSelected}
            onCreate={onCreate}
            onUpdate={onUpdate}
            onCreatePoly={onCreatePoly}
            onUpdatePoly={onUpdatePoly}
            onDrawing={setDrawingPoly}
            onScale={setScale}
            highlight={highlight && highlight.imageId === imageId ? highlight : null}
            kpts={poses.map((p) => ({ key: p.key, labelId: p.labelId, kpts: p.kpts }))}
            skeleton={template?.skeleton}
            placing={placing}
            onPlaceKpt={onPlaceKpt}
            onMoveKpt={onMoveKpt}
            sam={{ points: samPoints, box: samBox, preview: samPreview?.points ?? null }}
            onSamClick={onSamClick}
            onSamBox={onSamBox}
            badge={
              isCls
                ? tag
                  ? {
                      text: nameOf(tag.labelId) + (tag.source !== "manual" && tag.score != null ? ` · ${tag.score.toFixed(2)}` : ""),
                      color: colorOf(tag.labelId),
                    }
                  : { text: "未分类 · 按 1–9 选类别", color: "rgba(60,64,72,.9)" }
                : null
            }
          />
        ) : (
          ids?.length === 0 && <Empty description="没有符合筛选条件的图片" />
        )}
        {placing && placingPose && template && (
          <div className="place-hint">
            点击 <b>{template.points[placing.idx]}</b>（{placing.idx + 1}/{nk}） · Shift+点击 = 被遮挡 · N 跳过 · 退格 回退 · Esc 结束
          </div>
        )}
        {tool === "sam" && !isCls && (
          <div className="place-hint">
            {samBusy
              ? samUsed.current
                ? "SAM 计算中…"
                : "SAM 准备中…（首次使用要安装组件、下载模型，可能需要几分钟）"
              : samPreview
                ? `已选出目标 · Enter 采用为「${nameOf(curLabel)}」 · 继续点击修正（Shift+点击排除） · Esc 放弃`
                : "点击目标自动选出轮廓 · 也可以拖动框选 · Shift+点击排除区域"}
          </div>
        )}
        <div className="annot-hint">{hint}</div>
      </div>

      <aside className="annot-side">
        <section>
          <h3>AI 辅助</h3>
          {srcOptions.length > 1 && (
            <Segmented size="small" block value={aiSrc} options={srcOptions} onChange={(v) => pickSrc(v as AiSrc)} style={{ marginBottom: 8 }} />
          )}
          {aiSrc === "model" && models.length === 0 && (
            <div className="dim" style={{ fontSize: 12 }}>
              标注几十张后去「训练」页训练一个模型，就能用它自动标注剩下的图片。
            </div>
          )}
          {aiSrc === "model" && models.length > 0 && (
            <Select
              size="small"
              style={{ width: "100%", marginBottom: 6 }}
              value={modelId}
              onChange={setModelId}
              options={models.map((r) => ({
                value: r.id,
                label: `${r.name} · ${isCls ? "准确率" : "mAP50"} ${r.bestMap50?.toFixed(3) ?? "—"}`,
              }))}
            />
          )}
          {aiSrc === "text" && (
            <div className="dim" style={{ fontSize: 12, marginBottom: 6 }}>
              按类别名在图里找目标（开放词汇检测{isSeg ? " + SAM 出轮廓" : ""}）。模型只懂英文：中文类别请在类别的
              <EditOutlined style={{ margin: "0 2px" }} />
              里填「英文提示词」，如 安全帽 → helmet。首次使用需要下载模型。
              {labels.some((l) => !l.prompt && /[^\x00-\x7f]/.test(l.name)) && (
                <div style={{ color: "var(--warn)", marginTop: 4 }}>
                  还没填英文提示词：{labels.filter((l) => !l.prompt && /[^\x00-\x7f]/.test(l.name)).map((l) => l.name).join("、")}
                </div>
              )}
            </div>
          )}
          {aiSrc === "ai" &&
            (aiError ? (
              <div className="dim" style={{ fontSize: 12, marginBottom: 6 }}>
                {aiError}
                <div className="row" style={{ gap: 6, marginTop: 6 }}>
                  <Button size="small" onClick={() => setPage("settings")}>
                    去设置
                  </Button>
                  <Button size="small" onClick={() => setAiModels(null)}>
                    重试
                  </Button>
                </div>
              </div>
            ) : (
              <>
                <Select
                  size="small"
                  style={{ width: "100%", marginBottom: 6 }}
                  value={aiModel || undefined}
                  placeholder={aiModels ? "选择识别模型" : "正在读取可用模型…"}
                  loading={!aiModels}
                  onChange={(v) => {
                    setAiModel(v);
                    prefs.set("aiModel", v);
                  }}
                  options={(aiModels ?? []).map((m) => ({ value: m.code, label: m.name }))}
                />
                <div className="dim" style={{ fontSize: 12, marginBottom: 6 }}>
                  {aiModels?.find((m) => m.code === aiModel)?.description ?? ""}
                  {isSeg && "（识别出的框会再用 SAM 出轮廓）"}
                </div>
              </>
            ))}
          {(aiSrc !== "model" || models.length > 0) && (
            <>
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
                <Button size="small" type="primary" icon={<ThunderboltOutlined />} loading={predicting} disabled={!canPredict} onClick={predictCurrent}>
                  {isCls ? "预测本图 (P)" : "标注本图 (P)"}
                </Button>
                <Button size="small" onClick={predictAll} disabled={!!batch || !canPredict}>
                  批量预标注
                </Button>
              </div>
            </>
          )}
          {batch && <Progress size="small" percent={batch.total ? Math.floor((batch.done / batch.total) * 100) : 0} style={{ marginTop: 6 }} />}
          {isCls && tagHints.length > 0 && (
            <div style={{ marginTop: 8 }}>
              <div className="dim" style={{ fontSize: 12, marginBottom: 4 }}>模型的候选（点选采用）</div>
              {tagHints.map((h) => (
                <div key={h.labelId} className="lbl" onClick={() => commitTag({ labelId: h.labelId, source: "model", score: h.score })}>
                  <span className="sw" style={{ background: colorOf(h.labelId) }} />
                  <span className="n">{nameOf(h.labelId)}</span>
                  <span className="dim" style={{ fontSize: 11 }}>{h.score.toFixed(2)}</span>
                </div>
              ))}
            </div>
          )}
        </section>

        <section>
          <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}>
            <h3 style={{ margin: 0 }}>类别</h3>
            {isCls && (
              <span className="row" style={{ gap: 6, fontSize: 12 }}>
                <span className="dim">选完下一张</span>
                <Switch
                  size="small"
                  checked={autoNext}
                  onChange={(v) => {
                    setAutoNext(v);
                    prefs.set("clsAutoNext", v ? "1" : "0");
                  }}
                />
              </span>
            )}
          </div>
          {labels.length === 0 && <div className="dim" style={{ marginBottom: 8 }}>先添加要{isCls ? "区分" : "标注"}的类别</div>}
          {labels.map((l, i) => {
            const on = isCls ? tag?.labelId === l.id : l.id === curLabel;
            return (
              <div key={l.id} className={on ? "lbl on" : "lbl"} onClick={() => pickLabel(l.id)}>
                <span className="sw" style={{ background: l.color }} />
                <span className="n">
                  {l.name}
                  {l.prompt && <span className="dim" style={{ fontSize: 11 }}> · {l.prompt}</span>}
                </span>
                <span className="dim" style={{ fontSize: 11 }}>
                  {l.count}
                </span>
                {i < 9 && <kbd>{i + 1}</kbd>}
                <span className="ops">
                  <EditOutlined
                    onClick={(e) => {
                      e.stopPropagation();
                      setEditing({ ...l, prompt: l.prompt ?? "" });
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
            );
          })}
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

        {isPose && template && (
          <section>
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}>
              <h3 style={{ margin: 0 }}>
                关键点模板 <span className="dim">{nk} 点</span>
              </h3>
              <Button size="small" type="link" onClick={openTplEdit} style={{ padding: 0 }}>
                编辑
              </Button>
            </div>
            <div className="dim" style={{ fontSize: 12 }}>{template.name}</div>
          </section>
        )}

        {isCls ? (
          <section>
            <h3>本图类别</h3>
            {tag ? (
              <div className="obj on">
                <span className="sw" style={{ background: colorOf(tag.labelId) }} />
                <span className="n">{nameOf(tag.labelId)}</span>
                {tag.source !== "manual" && (
                  <Tag style={{ marginInlineEnd: 0 }}>
                    {SOURCE_TEXT[tag.source] ?? tag.source}
                    {tag.score != null && ` ${tag.score.toFixed(2)}`}
                  </Tag>
                )}
                <DeleteOutlined className="del" onClick={() => commitTag(null)} />
              </div>
            ) : (
              <div className="dim">还没有类别，点上面的类别或按 1–9</div>
            )}
          </section>
        ) : (
          <section>
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}>
              <h3 style={{ margin: 0 }}>本图对象 <span className="dim">{objects.length}</span></h3>
              <span className="row" style={{ gap: 6, fontSize: 12 }}>
                <span className="dim">类别名</span>
                <Switch size="small" checked={showNames} onChange={setShowNames} />
              </span>
            </div>
            {objects.length === 0 && (
              <div className="dim">
                {isSeg ? "还没有轮廓，按 R 在目标边缘逐点点击，或按 S 智能选取" : isPose ? "还没有目标，按 R 画框后逐个点关键点" : "还没有框，按 R 在图上拖动画框"}
              </div>
            )}
            {objects.map((o, i) => (
              <div
                key={o.key}
                className={o.key === selected ? "obj on" : "obj"}
                onClick={() => {
                  setSelected(o.key);
                  setTool("select");
                }}
              >
                <span className="sw" style={{ background: colorOf(o.labelId) }} />
                <span className="n">
                  {i + 1}. {nameOf(o.labelId)}
                </span>
                <span className="dim" style={{ fontSize: 11 }}>
                  {o.desc}
                </span>
                {o.source !== "manual" && <Tag style={{ marginInlineEnd: 0 }}>{SOURCE_TEXT[o.source] ?? o.source}</Tag>}
                <DeleteOutlined
                  className="del"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteBox(o.key);
                  }}
                />
              </div>
            ))}
          </section>
        )}

        {isPose && selPose && template && (
          <section>
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}>
              <h3 style={{ margin: 0 }}>
                关键点 <span className="dim">{kptCount(selPose)}/{nk}</span>
              </h3>
              <Button size="small" type="link" style={{ padding: 0 }} onClick={() => setPlacing({ key: selPose.key, idx: 0 })}>
                重放关键点
              </Button>
            </div>
            <div className="dim" style={{ fontSize: 12, marginBottom: 4 }}>点名字重新放这个点；点状态切换「可见 / 被遮挡」</div>
            <div style={{ maxHeight: 260, overflow: "auto" }}>
              {template.points.map((name, i) => {
                const v = selPose.kpts[i * 3 + 2] ?? 0;
                const on = placing?.key === selPose.key && placing.idx === i;
                return (
                  <div key={i} className={on ? "kpt-row on" : "kpt-row"} onClick={() => setPlacing({ key: selPose.key, idx: i, single: true })}>
                    <span className="dim" style={{ width: 20 }}>{i + 1}</span>
                    <span style={{ flex: 1 }}>{name}</span>
                    <span
                      className="st"
                      title={v === 2 ? "可见（点击改成被遮挡）" : v === 1 ? "被遮挡（点击改成可见）" : "还没标"}
                      onClick={(e) => {
                        if (v === 0) return;
                        e.stopPropagation();
                        setKpt(selPose.key, i, selPose.kpts[i * 3], selPose.kpts[i * 3 + 1], v === 2 ? 1 : 2);
                      }}
                    >
                      {v === 2 ? "●" : v === 1 ? "○" : "—"}
                    </span>
                    {v > 0 && (
                      <DeleteOutlined
                        style={{ fontSize: 11 }}
                        onClick={(e) => {
                          e.stopPropagation();
                          setKpt(selPose.key, i, 0, 0, 0);
                        }}
                      />
                    )}
                  </div>
                );
              })}
            </div>
          </section>
        )}

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
              <Button size="small" onClick={() => setStatus(objects.length || tag ? "labeled" : "todo", false)}>
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
        width={420}
      >
        {editing && (
          <>
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
                style={{ width: 280 }}
              />
            </Space>
            {!isCls && (
              <div style={{ marginTop: 12 }}>
                <div className="dim" style={{ fontSize: 12, marginBottom: 4 }}>
                  英文提示词（「文本提示」自动标注用；可写多个，用逗号隔开，如 helmet, hard hat）
                </div>
                <Input
                  value={editing.prompt ?? ""}
                  maxLength={120}
                  placeholder="例如 helmet"
                  onChange={(e) => setEditing({ ...editing, prompt: e.target.value })}
                  onPressEnter={saveLabelEdit}
                />
              </div>
            )}
          </>
        )}
      </Modal>

      <Modal
        title="关键点模板"
        open={!!tplEdit}
        onCancel={() => setTplEdit(null)}
        onOk={saveTpl}
        okText="保存"
        cancelText="取消"
        width={520}
      >
        {tplEdit && (
          <>
            <div className="dim" style={{ fontSize: 12, margin: "8px 0" }}>
              点的顺序就是标注时点击的顺序。已经有关键点标注后只能改名字和连线，不能增减点数。
            </div>
            <Input
              value={tplEdit.name}
              onChange={(e) => setTplEdit({ ...tplEdit, name: e.target.value })}
              placeholder="模板名"
              style={{ marginBottom: 8 }}
            />
            <div className="row" style={{ gap: 8, alignItems: "flex-start" }}>
              <div style={{ flex: 1 }}>
                <div className="dim" style={{ fontSize: 12, marginBottom: 4 }}>点名（每行一个）</div>
                <Input.TextArea rows={12} value={tplEdit.points} onChange={(e) => setTplEdit({ ...tplEdit, points: e.target.value })} />
              </div>
              <div style={{ width: 150 }}>
                <div className="dim" style={{ fontSize: 12, marginBottom: 4 }}>连线（每行一对序号，如 1-2）</div>
                <Input.TextArea rows={12} value={tplEdit.skeleton} onChange={(e) => setTplEdit({ ...tplEdit, skeleton: e.target.value })} />
              </div>
            </div>
            <Button
              size="small"
              style={{ marginTop: 8 }}
              onClick={() =>
                setTplEdit({
                  name: COCO_PERSON.name,
                  points: COCO_PERSON.points.join("\n"),
                  skeleton: COCO_PERSON.skeleton.map(([a, b]) => `${a + 1}-${b + 1}`).join("\n"),
                })
              }
            >
              使用 COCO 人体 17 点
            </Button>
          </>
        )}
      </Modal>
    </div>
  );
}
