import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Alert, App, Button, ColorPicker, Empty, Input, InputNumber, Modal, Popover, Progress, Segmented, Select, Slider, Space, Switch, Tag, Tooltip } from "antd";
import { listen, sendLive } from "../transport";
import {
  AimOutlined,
  BorderOutlined,
  CheckOutlined,
  CopyOutlined,
  DeleteOutlined,
  EditOutlined,
  ExpandOutlined,
  FastForwardOutlined,
  HighlightOutlined,
  LeftOutlined,
  LoadingOutlined,
  MessageOutlined,
  NodeIndexOutlined,
  PlusOutlined,
  RedoOutlined,
  RightOutlined,
  RollbackOutlined,
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
  isClsTask,
  ImageTeamInfo,
  ImportProgress,
  imageUrl,
  KeypointData,
  KeypointTemplate,
  Label,
  PredResult,
  PredTag,
  ProjectInfo,
  RunInfo,
  SamResult,
  ShareEvent,
  thumbUrl,
} from "../api";
import { prefs, useApp, useReview } from "../store";
import { statusTag } from "../status";
import AnnotCanvas, { Box, Poly, RectXYWH, Tool } from "../components/AnnotCanvas";
import AiRulesDrawer from "../components/AiRulesDrawer";
import { t } from "../i18n";
import { xuntongEnabled } from "../features";

const PALETTE = ["#3d8bfd", "#e5534b", "#2fb67c", "#e7a33e", "#9b7bf2", "#14b8a6", "#f472b6", "#a3e635", "#f97316", "#60a5fa"];
const FILTERS = [
  { label: t("全部"), value: "all" },
  { label: t("未标注"), value: "todo" },
  // 只在团队协作（审核流程）时用：标注员提交的待审核，审核员通过的已通过
  { label: t("待审核"), value: "labeled" },
  { label: t("已通过"), value: "reviewed" },
  { label: t("已跳过"), value: "skipped" },
];
/** 一个人标注（不用审核）时的筛选：已标注包含已审核 */
const SOLO_FILTERS = [
  { label: t("全部"), value: "all" },
  { label: t("未标注"), value: "todo" },
  { label: t("已标注"), value: "done" },
  { label: t("已跳过"), value: "skipped" },
];
const SOURCE_TEXT: Record<string, string> = { manual: t("手工"), model: t("模型"), sam: "SAM", ai: t("巡瞳"), text: t("文本"), interp: t("插值"), track: t("跟踪") };
const HISTORY_MAX = 200;
// 底部缩略图条高度（像素）
const STRIP_H_DEFAULT = 86;
const clampStrip = (h: number) => (Number.isFinite(h) ? Math.min(320, Math.max(60, Math.round(h))) : STRIP_H_DEFAULT);

/** COCO 人体 17 点（和后端 keypoints.rs 的默认模板一致） */
const COCO_PERSON: KeypointTemplate = {
  name: t("COCO 人体 17 点"),
  points: [t("鼻子"), t("左眼"), t("右眼"), t("左耳"), t("右耳"), t("左肩"), t("右肩"), t("左肘"), t("右肘"), t("左腕"), t("右腕"), t("左髋"), t("右髋"), t("左膝"), t("右膝"), t("左踝"), t("右踝")],
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

/** 图像分类：整图的类别（单标签只有一个，多标签可以有好几个） */
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
type Snap = { boxes: Box[]; polys: Poly[]; poses: Pose[]; tags: ClsTag[] };
type AiSrc = "model" | "text" | "ai";

export default function Annotate({ project }: { project: ProjectInfo }) {
  const { message, modal } = App.useApp();
  const refreshStats = useApp((s) => s.refreshStats);
  const setPage = useApp((s) => s.setPage);
  const takeFocusImage = useApp((s) => s.takeFocusImage);
  const highlight = useApp((s) => s.highlight);
  const queue = useApp((s) => s.queue);
  const clearQueue = useApp((s) => s.clearQueue);
  // 实时协作：浏览器里的成员（remote），或者发起人正在共享
  const team = useApp((s) => s.team);
  const remote = team.remote;
  const myId = remote ? (team.me?.id ?? -1) : 0;
  const isAnnotator = remote && team.me?.role === "annotator";
  const canReview = !remote || team.me?.role === "reviewer";
  const collab = remote || team.shareRunning;
  // 协作时才有审核流程；一个人标注时标完就算数
  const review = useReview();
  const canEditLabels = !remote || team.options.membersEditLabels;
  const aiAllowed = !remote || team.options.allowAi;
  /** 别人正在改这张图时是他的名字（只读） */
  const [lockHolder, setLockHolder] = useState<string | null>(null);
  const [teamInfo, setTeamInfo] = useState<ImageTeamInfo | null>(null);
  const [idsVersion, setIdsVersion] = useState(0);
  /** 正在填打回原因（null 表示没在打回） */
  const [rejecting, setRejecting] = useState<string | null>(null);
  const [rulesOpen, setRulesOpen] = useState(false);
  const readOnlyRef = useRef<string | null>(null);

  const task = project.task;
  const isSeg = task === "segment";
  const isCls = isClsTask(task);
  // 多标签分类：1–9 勾选 / 取消，一张图可以有好几个类别
  const multi = task === "multilabel";
  const isPose = task === "pose";
  const drawTool: Tool = isSeg ? "polygon" : "rect";

  const [labels, setLabels] = useState<Label[]>([]);
  const [curLabel, setCurLabel] = useState<number | null>(null);
  const [filter, setFilter] = useState(queue ? "queue" : "all");
  // 那一批被退出了（比如去看一张不在批里的图），筛选回到「全部」
  useEffect(() => {
    if (filter === "queue" && !queue) setFilter("all");
  }, [filter, queue]);
  // 体检问题、评估错例的提示框，挡着看图时可以先藏起来
  const [showMarks, setShowMarks] = useState(true);
  const [ids, setIds] = useState<number[] | null>(null);
  const [idx, setIdx] = useState(0);
  /** 每张图的状态（缩略图状态点、跳到下一张未标注用）；拿不到时是空的 */
  const [statusMap, setStatusMap] = useState<Map<number, ImageStatus>>(() => new Map());
  /** 翻过最后一张时的提示（已标注、未标注张数） */
  const [endInfo, setEndInfo] = useState<{ done: number; todo: number; firstTodo: number } | null>(null);
  // 顶栏序号：点开输入序号跳转
  const [jumpOpen, setJumpOpen] = useState(false);
  const [jumpTo, setJumpTo] = useState<number | null>(null);
  const [meta, setMeta] = useState<ImageAnnotations | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [polys, setPolys] = useState<Poly[]>([]);
  const [poses, setPoses] = useState<Pose[]>([]);
  const [tags, setTags] = useState<ClsTag[]>([]);
  // 单标签分类只有这一个
  const tag = tags[0] ?? null;
  const [selected, setSelected] = useState<string | null>(null);
  const [tool, setTool] = useState<Tool>(isCls ? "select" : drawTool);
  const [drawingPoly, setDrawingPoly] = useState(false);
  const [saveState, setSaveState] = useState<SaveState>("saved");
  const [fitSignal, setFitSignal] = useState(0);
  const [showNames, setShowNames] = useState(true);
  // 只显示某一类的框（一类一类核对时用，配合数据页按类别筛选）；null 显示全部
  const [onlyLabel, setOnlyLabel] = useState<number | null>(null);
  const shown = (labelId: number | null | undefined) => onlyLabel == null || labelId === onlyLabel;
  const [keysOpen, setKeysOpen] = useState(false);
  const [scale, setScale] = useState(1);
  const [newLabel, setNewLabel] = useState("");
  const [editing, setEditing] = useState<Label | null>(null);
  const [histLen, setHistLen] = useState({ past: 0, future: 0 });
  // 图像分类：选完类别自动下一张
  const [autoNext, setAutoNext] = useState(prefs.get("clsAutoNext", "1") === "1");
  // 底部缩略图条的高度，拖它上边缘调整，双击恢复默认
  const [stripH, setStripH] = useState(() => clampStrip(Number(prefs.get("stripH", String(STRIP_H_DEFAULT)))));
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
  const tagsRef = useRef<ClsTag[]>([]);
  const metaRef = useRef<ImageAnnotations | null>(null);
  const othersRef = useRef<Annotation[]>([]); // 不认识的标注类型原样保留
  const history = useRef<{ past: Snap[]; future: Snap[] }>({ past: [], future: [] });
  const dirty = useRef(false);
  const timer = useRef<number>();

  const labelMap = useMemo(() => new Map(labels.map((l) => [l.id, l])), [labels]);
  const colorOf = useCallback((id: number | null) => (id != null && labelMap.get(id)?.color) || "#9aa1ab", [labelMap]);
  const nameOf = useCallback((id: number | null) => (id != null && labelMap.get(id)?.name) || t("未分类"), [labelMap]);
  const nk = template?.points.length ?? 0;

  // ---------- 保存 ----------

  /** 本地记下某张图的新状态（没读到状态表时不记） */
  const markStatus = useCallback(
    (id: number, s: ImageStatus) => setStatusMap((mp) => (mp.has(id) && mp.get(id) !== s ? new Map(mp).set(id, s) : mp)),
    [],
  );

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
        ...tagsRef.current.map(tagToAnn),
        ...othersRef.current,
      ]);
      markStatus(m.id, status);
      if (metaRef.current?.id === m.id) {
        metaRef.current = { ...metaRef.current, status };
        setMeta(metaRef.current);
      }
      setSaveState(dirty.current ? "dirty" : "saved");
    } catch (e) {
      dirty.current = true;
      setSaveState("error");
      message.error(t("保存失败：{err}", { err: errText(e) }));
    }
  }, [message, markStatus]);

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
    if (s.tags) {
      tagsRef.current = s.tags;
      setTags(s.tags);
    }
  };
  const snap = (): Snap => ({ boxes: boxesRef.current, polys: polysRef.current, poses: posesRef.current, tags: tagsRef.current });

  /** 修改标注：记历史、自动保存 */
  const change = (s: Partial<Snap>) => {
    if (readOnlyRef.current) {
      message.warning(t("「{name}」正在标这张图，现在只能看", { name: readOnlyRef.current }));
      return;
    }
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
  const commitTag = (t: ClsTag | null) => change({ tags: t ? [t] : [] });
  /** 多标签：有就去掉，没有就加上 */
  const toggleTag = (id: number) => {
    const cur = tagsRef.current;
    change({ tags: cur.some((x) => x.labelId === id) ? cur.filter((x) => x.labelId !== id) : [...cur, { labelId: id, source: "manual" }] });
  };
  /** 多标签：加上一个类别（已有就不动） */
  const addTag = (x: ClsTag) => {
    if (!tagsRef.current.some((y) => y.labelId === x.labelId)) change({ tags: [...tagsRef.current, x] });
  };

  const undo = () => {
    if (readOnlyRef.current) return;
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
    if (readOnlyRef.current) return;
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
      await api.updateLabel(editing.id, editing.name, editing.color, editing.prompt ?? "", editing.description ?? "");
      setEditing(null);
      await loadLabels();
    } catch (e) {
      message.error(errText(e));
    }
  };

  const removeLabel = (l: Label) => {
    // 有标注时可以先合并到另一个类别，标注不丢
    const others = labels.filter((x) => x.id !== l.id);
    const canMerge = l.count > 0 && others.length > 0;
    let target: number | null = null;
    const okProps = () =>
      target != null
        ? { okText: t("合并"), okButtonProps: { danger: false } }
        : { okText: canMerge ? t("直接删除") : t("删除"), okButtonProps: { danger: true } };
    const dlg = modal.confirm({
      title: t("删除类别「{name}」？", { name: l.name }),
      content:
        l.count > 0 ? (
          <>
            <div>{t("这个类别下的 {n} 个标注会一起删除，无法恢复。", { n: l.count })}</div>
            {canMerge && (
              <div style={{ marginTop: 10 }}>
                <div className="dim" style={{ fontSize: 12, marginBottom: 4 }}>
                  {t("也可以把这些标注合并到另一个类别，标注不会丢：")}
                </div>
                <Select
                  allowClear
                  style={{ width: "100%" }}
                  placeholder={t("合并到…")}
                  options={others.map((x) => ({ value: x.id, label: x.name }))}
                  onChange={(v: number | undefined) => {
                    target = v ?? null;
                    dlg.update(okProps());
                  }}
                />
              </div>
            )}
          </>
        ) : (
          t("这个类别还没有标注。")
        ),
      ...okProps(),
      cancelText: t("取消"),
      onOk: async () => {
        const to = target;
        try {
          await flush();
          if (to != null) {
            const n = await api.mergeLabel(l.id, to);
            message.success(t("已把 {n} 个标注合并到「{name}」", { n, name: nameOf(to) }));
          } else {
            await api.deleteLabel(l.id);
          }
          await loadLabels();
          await reloadCurrent();
          await loadStatuses();
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
    if (multi) {
      toggleTag(id);
      return;
    }
    if (isCls) {
      const cur = tagsRef.current[0];
      if (cur?.labelId !== id || cur.source !== "manual") commitTag({ labelId: id, source: "manual" });
      if (autoNext && ids) go(nextIndex());
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

  /** 重新读每张图的状态；读不到（比如发起人的版本较旧）就不显示状态点 */
  const loadStatuses = useCallback(async () => {
    const st = await api.imageStatuses().catch(() => null);
    setStatusMap(new Map<number, ImageStatus>(st ?? []));
  }, []);

  useEffect(() => {
    let alive = true;
    (async () => {
      await flush();
      await loadStatuses();
      let all: number[];
      if (filter === "queue" && queue) {
        // 指定的一批图：按给定顺序，去掉已经删除的
        const exist = new Set(await api.listImageIds(null));
        all = queue.ids.filter((id) => exist.has(id));
      } else {
        all = await api.listImageIds(filter === "all" || filter === "queue" ? null : filter);
      }
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
  }, [filter, flush, loadStatuses, takeFocusImage, message, queue, idsVersion]);

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
    // 单标签分类只取第一个；多标签都要（同一个类别只留一个）
    const tagList: ClsTag[] = [];
    for (const a of m.annotations) {
      if (a.type !== "tag" || a.labelId == null || tagList.some((x) => x.labelId === a.labelId)) continue;
      tagList.push({ labelId: a.labelId, source: a.source, score: a.score });
      if (!multi) break;
    }
    setSnap({
      boxes: m.annotations.filter((a) => a.type === "bbox").map(toBox),
      polys: m.annotations.filter((a) => a.type === "polygon").map(toPoly),
      poses: m.annotations.filter((a) => a.type === "keypoints").map(toPose),
      tags: tagList,
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
      setSnap({ boxes: [], polys: [], poses: [], tags: [] });
      return;
    }
    let alive = true;
    readOnlyRef.current = null;
    (async () => {
      await flush();
      const m = await api.loadAnnotations(imageId);
      if (alive) applyLoaded(m);
      // 加锁（协作时别人正在改就只读），顺便取分配、打回信息
      const r = await api.lockImage(imageId).catch(() => null);
      if (alive && r) {
        readOnlyRef.current = r.holder;
        setLockHolder(r.holder);
        setTeamInfo(r.info);
      }
    })().catch((e) => message.error(errText(e)));
    return () => {
      alive = false;
    };
  }, [imageId, flush, message]); // eslint-disable-line react-hooks/exhaustive-deps

  // 协作：每 30 秒续一次锁、报告在看哪张
  useEffect(() => {
    if (!collab || imageId === undefined) return;
    if (remote) sendLive({ type: "view", imageId });
    const timer = window.setInterval(() => {
      if (remote) sendLive({ type: "view", imageId });
      else
        api
          .lockImage(imageId)
          .then((r) => {
            if (metaRef.current?.id !== imageId) return;
            readOnlyRef.current = r.holder;
            setLockHolder(r.holder);
          })
          .catch(() => {});
    }, 30000);
    return () => window.clearInterval(timer);
  }, [collab, remote, imageId]);

  // 离开标注页时放开锁
  useEffect(
    () => () => {
      api.unlockImage().catch(() => {});
    },
    [],
  );

  // 协作的实时事件：别人改了这张图就刷新；锁放开了就接手；类别、分配变了就重新读
  useEffect(() => {
    if (!collab) return;
    const relock = (cur: number, notify: boolean) =>
      api
        .lockImage(cur)
        .then((r) => {
          if (metaRef.current?.id !== cur) return;
          const was = readOnlyRef.current;
          readOnlyRef.current = r.holder;
          setLockHolder(r.holder);
          setTeamInfo(r.info);
          if (notify && was && !r.holder) message.info(t("对方已经离开这张图，现在可以编辑了"));
        })
        .catch(() => {});
    const un = listen<ShareEvent>("share-event", (e) => {
      const ev = e.payload;
      if (ev.kind === "labels") {
        loadLabels().catch(() => {});
        return;
      }
      if (ev.kind === "assign") {
        if (remote) setIdsVersion((n) => n + 1);
        return;
      }
      const cur = metaRef.current?.id;
      // 审核页批量通过 / 打回 / 撤销：刷新缩略图上的状态；正在看的这张也重新读
      if (ev.kind === "status-batch") {
        loadStatuses().catch(() => {});
        if (cur != null && ev.ids?.includes(cur) && ev.memberId !== myId) {
          if (readOnlyRef.current || !dirty.current) {
            api
              .loadAnnotations(cur)
              .then((m) => {
                if (metaRef.current?.id === cur) applyLoaded(m);
              })
              .catch(() => {});
          }
          relock(cur, false);
          if (ev.action === "reject") message.warning(t("「{name}」打回了这张图", { name: ev.by }));
        }
        return;
      }
      if (cur == null || ev.imageId !== cur) return;
      if ((ev.kind === "ann" || ev.kind === "status") && ev.memberId !== myId) {
        if (readOnlyRef.current || !dirty.current) {
          api
            .loadAnnotations(cur)
            .then((m) => {
              if (metaRef.current?.id === cur) applyLoaded(m);
            })
            .catch(() => {});
        }
        relock(cur, false);
        if (ev.kind === "status" && ev.action === "reject") message.warning(t("「{name}」打回了这张图", { name: ev.by }));
      }
      if (ev.kind === "lock" && ev.holder == null && readOnlyRef.current) relock(cur, true);
    });
    return () => {
      un.then((f) => f());
    };
  }, [collab, remote, myId, loadLabels, message]); // eslint-disable-line react-hooks/exhaustive-deps

  /** 翻过最后一张：提示这一批标完了，可以去训练，或者接着标未标注的 */
  const showEnd = async () => {
    await flush();
    // 只在「全部」「未标注」和指定的一批图里提示；已标注、已跳过等筛选翻到头就停
    if (!(filter === "all" || filter === "todo" || filter === "queue")) return;
    let done = 0;
    let todo = 0;
    let firstTodo = -1;
    // 按当前列表里每张图的最新状态算（「这一批」就是现在翻的这些图）
    const st = await api.imageStatuses().catch(() => null);
    if (st && st.length && ids) {
      const mp = new Map<number, ImageStatus>(st);
      setStatusMap(mp);
      ids.forEach((id, i) => {
        const s = mp.get(id);
        if (s === "todo") {
          todo++;
          if (firstTodo < 0) firstTodo = i;
        } else if (s === "labeled" || s === "reviewed") done++;
      });
    } else {
      await refreshStats();
      const s = useApp.getState().project?.stats;
      done = (s?.labeled ?? 0) + (s?.reviewed ?? 0);
      todo = s?.todo ?? 0;
    }
    setEndInfo({ done, todo, firstTodo });
  };

  const go = (i: number) => {
    if (!ids || i < 0) return;
    if (i >= ids.length) {
      if (ids.length > 0 && !endInfo) showEnd().catch((e) => message.error(errText(e)));
      return;
    }
    setIdx(i);
    refreshStats();
    loadLabels();
  };

  /** 标完一张后去哪：「全部」「未标注」或指定的一批图里，跳到后面第一张未标注的，没有就下一张。
   *  审核员审图时照旧下一张 */
  const nextIndex = () => {
    if (ids && (!review || isAnnotator) && (filter === "all" || filter === "todo" || filter === "queue")) {
      for (let j = idx + 1; j < ids.length; j++) if (statusMap.get(ids[j]) === "todo") return j;
    }
    return idx + 1;
  };

  /** Shift+D：下一张未标注，后面没有了就提示这一批标完了 */
  const goNextTodo = () => {
    if (!ids) return;
    // 没读到状态时退回普通的下一张
    if (statusMap.size === 0) return go(idx + 1);
    for (let j = idx + 1; j < ids.length; j++) if (statusMap.get(ids[j]) === "todo") return go(j);
    go(ids.length);
  };

  /** 改当前图的状态，成功返回 true */
  const setStatus = async (status: ImageStatus, next: boolean): Promise<boolean> => {
    const m = metaRef.current;
    if (!m) return false;
    if (readOnlyRef.current) {
      message.warning(t("「{name}」正在标这张图，现在只能看", { name: readOnlyRef.current }));
      return false;
    }
    // 标注员的「通过」是提交，等审核
    const real: ImageStatus = isAnnotator && status === "reviewed" ? "labeled" : status;
    try {
      await flush();
      await api.setImageStatus(m.id, status);
      if (real === "labeled" || real === "reviewed") setTeamInfo((ti) => (ti ? { ...ti, rejectReason: "" } : ti));
      metaRef.current = { ...m, status: real };
      setMeta(metaRef.current);
      markStatus(m.id, real);
      if (next) go(nextIndex());
      else refreshStats();
      return true;
    } catch (e) {
      message.error(errText(e));
      return false;
    }
  };

  /** 撤回「没有目标」：那张图改回原来的状态，并跳回去 */
  const undoNegative = async (id: number, prev: ImageStatus) => {
    try {
      await api.setImageStatus(id, prev);
      markStatus(id, prev);
      if (metaRef.current?.id === id) {
        metaRef.current = { ...metaRef.current, status: prev };
        setMeta(metaRef.current);
      }
      const i = ids ? ids.indexOf(id) : -1;
      if (i >= 0) {
        setEndInfo(null);
        setIdx(i);
      }
      refreshStats();
    } catch (e) {
      message.error(errText(e));
    }
  };

  /** 不用审核时的「完成并下一张」：有标注的图本来就算已标注，直接下一张；
   *  没有目标的图记成「没有目标」（存为已审核，作为负样本参与训练） */
  const finish = async () => {
    if (review) return setStatus("reviewed", true);
    const m = metaRef.current;
    if (!m) return;
    // 分类项目每张图要有一个类别，没有「没有目标」这回事
    if (isCls && !tag) {
      message.warning(t("先选一个类别，或按 X 跳过这张"));
      return;
    }
    if (objects.length === 0 && !tag) {
      const prev = m.status;
      const ok = await setStatus("reviewed", true);
      // 误按了 Enter 可以撤回
      if (ok && prev !== "reviewed") {
        const key = `neg-${m.id}`;
        message.open({
          key,
          type: "success",
          duration: 5,
          content: (
            <span>
              {t("已记为「没有目标」")}
              <a
                style={{ marginLeft: 8 }}
                onClick={() => {
                  message.destroy(key);
                  undoNegative(m.id, prev);
                }}
              >
                {t("撤回")}
              </a>
            </span>
          ),
        });
      }
      return;
    }
    // 批量预标注的图还是未标注（等人确认），看过按完成就算已标注；跳过的图同理
    if (m.status === "skipped" || m.status === "todo") return setStatus("labeled", true);
    await flush();
    go(nextIndex());
  };

  // ---------- 编辑 ----------

  const needLabel = () => {
    if (curLabel == null) {
      message.warning(t("先在右侧添加一个类别"));
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
        message.error(t("连线「{line}」格式不对，应该写成 1-2", { line }));
        return;
      }
      skeleton.push([Number(m[1]) - 1, Number(m[2]) - 1]);
    }
    try {
      await api.saveKeypointTemplate({ name: tplEdit.name, points, skeleton });
      setTplEdit(null);
      await loadTemplate();
      message.success(t("模板已保存"));
    } catch (e) {
      message.error(errText(e));
    }
  };

  // ---------- AI 辅助（我的模型 / 文本提示 / 巡瞳 AI）----------

  // 文本提示模型的分数普遍比训练出的模型低，单独记一个置信度（默认 0.2）
  const confKey = (src: AiSrc) => (src === "text" ? "textConf" : "prelabelConf");
  const confDefault = (src: AiSrc) => (src === "text" ? "0.2" : "0.4");
  const [models, setModels] = useState<RunInfo[]>([]);
  const [modelId, setModelId] = useState<number | null>(null);
  const [aiSrc, setAiSrc] = useState<AiSrc>(() => {
    let saved = (prefs.get("aiSrc", "model") as AiSrc) || "model";
    if (saved === "ai" && !xuntongEnabled()) saved = "model";
    return isCls || (team.remote && saved === "ai") ? "model" : saved;
  });
  const [conf, setConf] = useState(() => Number(prefs.get(confKey(aiSrc), confDefault(aiSrc))));
  const [predicting, setPredicting] = useState(false);
  const [batch, setBatch] = useState<{ done: number; total: number } | null>(null);
  const [aiModels, setAiModels] = useState<AiModel[] | null>(null);
  const [aiModel, setAiModel] = useState(prefs.get("aiModel", ""));
  const [aiError, setAiError] = useState<string | null>(null);
  // 辅助模型准备过程（安装组件、下载模型）的最新一行日志
  const [assistLog, setAssistLog] = useState("");

  useEffect(() => {
    const un = listen<string>("assist-log", (e) => setAssistLog(e.payload));
    return () => {
      un.then((f) => f());
    };
  }, []);

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
    setConf(Number(prefs.get(confKey(v), confDefault(v))));
  };

  // 用文本提示时，给还没填英文提示词的中文类别按对照表自动补上（协作成员没有权限，忽略失败）
  useEffect(() => {
    if (aiSrc !== "text" || remote) return;
    api
      .fillLabelPrompts()
      .then((n) => (n > 0 ? loadLabels() : undefined))
      .catch(() => {});
  }, [aiSrc, remote, labels.length]); // eslint-disable-line react-hooks/exhaustive-deps

  /** 把一次预测结果放进当前图（和已有标注重叠很大的不重复添加）。source：model / text / ai */
  const applyPreds = (preds: PredResult, source: string) => {
    const tagWord = source === "model" ? t("模型") : source === "ai" ? t("巡瞳 AI") : t("文本提示");
    if (multi) {
      setTagHints(preds.tags.slice(0, 8));
      const sure = preds.tags.filter((x) => x.score >= conf);
      const names = sure.map((x) => nameOf(x.labelId)).join("、");
      const top = preds.tags[0];
      if (!top) {
        message.info(t("模型没有给出结果"));
      } else if (!sure.length) {
        message.info(t("{src}没有分数达到置信度的类别，最像「{name}」（{score}）", { src: tagWord, name: nameOf(top.labelId), score: top.score.toFixed(2) }));
      } else if (tagsRef.current.length) {
        message.info(t("{src}认为有：{names}，已有类别不覆盖，可在下方点选", { src: tagWord, names }));
      } else {
        change({ tags: sure.map((x) => ({ labelId: x.labelId, source, score: x.score })) });
        message.success(t("{src}认为有：{names}，检查后按 Enter 通过", { src: tagWord, names }));
      }
      return;
    }
    if (isCls) {
      setTagHints(preds.tags.slice(0, 3));
      const top = preds.tags[0];
      if (!top) {
        message.info(t("模型没有给出结果"));
      } else if (tagsRef.current.length) {
        message.info(
          t("{src}认为是「{name}」（{score}），已有类别不覆盖，可在下方点选", { src: tagWord, name: nameOf(top.labelId), score: top.score.toFixed(2) }),
        );
      } else if (top.score >= conf) {
        commitTag({ labelId: top.labelId, source, score: top.score });
        message.success(t("{src}认为是「{name}」（{score}），检查后按 Enter 通过", { src: tagWord, name: nameOf(top.labelId), score: top.score.toFixed(2) }));
      } else {
        message.info(
          t("{src}最像「{name}」，但只有 {score}，低于置信度，没有采用", { src: tagWord, name: nameOf(top.labelId), score: top.score.toFixed(2) }),
        );
      }
      return;
    }
    const none = () => message.info(t("{src}没有找到目标，可以调低置信度试试", { src: tagWord }));
    const allDone = () => message.info(t("{src}找到的目标都已经标过了", { src: tagWord }));
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
      message.success(
        fromPoses.length
          ? t("添加了 {n} 个目标（虚线）", { n: fresh.length })
          : t("添加了 {n} 个目标（虚线），选中后点「重放关键点」逐个标点", { n: fresh.length }),
      );
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
      message.success(t("添加了 {n} 个预标注轮廓（虚线），检查后按 Enter 通过", { n: fresh.length }));
      return;
    }
    if (preds.boxes.length === 0) return none();
    const fresh = preds.boxes
      .filter((p) => !boxesRef.current.some((b) => iou(b, p) > 0.7))
      .map((p) => ({ key: newKey(), labelId: p.labelId, x: p.x, y: p.y, w: p.w, h: p.h, source, score: p.score }));
    if (fresh.length === 0) return allDone();
    commit([...boxesRef.current, ...fresh]);
    message.success(t("添加了 {n} 个预标注框（虚线），检查后按 Enter 通过", { n: fresh.length }));
  };

  const predictCurrent = async () => {
    const m = metaRef.current;
    if (!m || predicting) return;
    if (aiSrc === "model" && modelId == null) return;
    if (aiSrc === "ai" && !aiModel) return;
    setPredicting(true);
    setAssistLog("");
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
    const what = { model: t("模型"), text: t("文本提示"), ai: t("巡瞳 AI") }[aiSrc];
    modal.confirm({
      title: t("批量预标注（{what}）", { what }),
      content: (
        <>
          {t("对所有「未标注」且还没有标注的图片运行{what}。预标注的图片仍然算未标注，需要你逐张检查确认后才会用于训练。", { what })}
          {aiSrc === "ai" && <div style={{ marginTop: 8, color: "var(--warn)" }}>{t("巡瞳 AI 每张图计 1 次调用，会消耗你的套餐额度。")}</div>}
        </>
      ),
      okText: t("开始"),
      cancelText: t("取消"),
      // 不等跑完就关掉对话框，进度和「停止」按钮在右侧
      onOk: () => {
        runPrelabel();
      },
    });
  };

  // 批量预标注可以中途停止，已经标好的保留
  const batchStop = useRef(false);
  const [batchStopping, setBatchStopping] = useState(false);
  const stopBatch = () => {
    batchStop.current = true;
    setBatchStopping(true);
    api.stopPrelabel().catch(() => {});
  };

  const runPrelabel = async () => {
    batchStop.current = false;
    setBatchStopping(false);
    setBatch({ done: 0, total: 0 });
    const un = await listen<{ done: number; total: number }>("prelabel-progress", (e) => setBatch(e.payload));
    try {
      await flush();
      const r =
        aiSrc === "model"
          ? await api.prelabelAll(modelId as number, conf)
          : aiSrc === "text"
            ? await api.textPrelabelAll(conf)
            : await api.aiPrelabelAll(aiModel, conf);
      await loadLabels();
      // 预标注只改还没有标注的图：当前图没标、也没有没保存的改动时才重新读，免得冲掉正在改的
      const empty = !boxesRef.current.length && !polysRef.current.length && !posesRef.current.length && !tagsRef.current.length;
      if (empty && !dirty.current) await reloadCurrent();
      const v = { images: r.images, n: r.boxes };
      message.success(
        batchStop.current
          ? t("已停止，预标注了 {n} 张", { n: r.images })
          : isCls
            ? t("已预标注 {images} 张图片，共 {n} 个类别", v)
            : isSeg
              ? t("已预标注 {images} 张图片，共 {n} 个轮廓", v)
              : isPose
                ? t("已预标注 {images} 张图片，共 {n} 个目标", v)
                : t("已预标注 {images} 张图片，共 {n} 个框", v),
      );
    } catch (e) {
      message.error(errText(e));
    } finally {
      un();
      setBatch(null);
      setBatchStopping(false);
    }
  };

  // ---------- SAM 交互分割 ----------

  const runSam = async (points: number[][], box: RectXYWH | null) => {
    const m = metaRef.current;
    if (!m || (points.length === 0 && !box)) {
      setSamPreview(null);
      return;
    }
    setSamBusy(true);
    setAssistLog("");
    try {
      const r = await api.samSegment(m.id, points, box ? [box.x, box.y, box.x + box.w, box.y + box.h] : null);
      samUsed.current = true;
      if (metaRef.current?.id === m.id) setSamPreview(r.points.length >= 6 ? r : null);
      if (r.points.length < 6) message.info(t("这里没分出目标，换个位置点，或者框一下试试"));
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
      if (tagsRef.current.length) commitTag(null);
      return;
    }
    if (!key) return;
    if (boxesRef.current.some((b) => b.key === key)) commit(boxesRef.current.filter((b) => b.key !== key));
    else if (polysRef.current.some((pl) => pl.key === key)) commitPoly(polysRef.current.filter((pl) => pl.key !== key));
    else if (posesRef.current.some((p) => p.key === key)) commitPose(posesRef.current.filter((p) => p.key !== key));
    if (placing?.key === key) setPlacing(null);
    setSelected(null);
  };

  // ---------- 视频标注：关键帧插值、往后跟踪 ----------

  const [videoBusy, setVideoBusy] = useState<"interp" | "track" | null>(null);
  const [trackN, setTrackN] = useState(() => Number(prefs.get("trackCount", "30")) || 30);
  const [trackProg, setTrackProg] = useState<ImportProgress | null>(null);
  const canVideo = !isCls && !isSeg && !remote;

  const interpolate = async () => {
    const m = metaRef.current;
    if (!m || videoBusy) return;
    setVideoBusy("interp");
    try {
      await flush();
      const r = await api.interpolateFrames(m.id);
      message.success(
        r.unmatched
          ? t("已在中间 {frames} 帧上插值 {objects} 个目标，{unmatched} 个目标两头对不上，没有插值", { frames: r.frames, objects: r.objects, unmatched: r.unmatched })
          : t("已在中间 {frames} 帧上插值 {objects} 个目标", { frames: r.frames, objects: r.objects }),
      );
      refreshStats();
    } catch (e) {
      message.error(errText(e));
    } finally {
      setVideoBusy(null);
    }
  };

  const trackForward = async () => {
    const m = metaRef.current;
    if (!m || videoBusy) return;
    prefs.set("trackCount", String(trackN));
    setVideoBusy("track");
    setTrackProg(null);
    const un = await listen<ImportProgress>("track-progress", (e) => setTrackProg(e.payload));
    try {
      await flush();
      const r = await api.trackFrames(m.id, trackN);
      message.success(
        t("跟踪了 {frames} 帧", { frames: r.frames }) +
          (r.lost ? t("，{n} 个目标中途跟丢", { n: r.lost }) : "") +
          (r.hitKeyframe ? t("，碰到后面手动标过的帧停下") : "") +
          t("。已跳到最后一帧，框偏了就拖正，再接着跟踪"),
      );
      refreshStats();
      if (r.lastId != null && ids) {
        const i = ids.indexOf(r.lastId);
        if (i >= 0) go(i);
      }
    } catch (e) {
      message.error(errText(e));
    } finally {
      un();
      setVideoBusy(null);
      setTrackProg(null);
    }
  };

  const copyPrev = async () => {
    if (!ids || idx === 0 || !metaRef.current) return;
    try {
      const prev = await api.loadAnnotations(ids[idx - 1]);
      const { width: W, height: H } = metaRef.current;
      const cx = (v: number) => Math.min(Math.max(v, 0), W);
      const cy = (v: number) => Math.min(Math.max(v, 0), H);
      if (multi) {
        const got: ClsTag[] = [];
        for (const a of prev.annotations) {
          if (a.type === "tag" && a.labelId != null && !got.some((x) => x.labelId === a.labelId)) got.push({ labelId: a.labelId, source: "manual" });
        }
        if (!got.length) {
          message.info(t("上一张还没有类别"));
          return;
        }
        change({ tags: got });
        message.success(t("已沿用上一张的类别「{name}」", { name: got.map((x) => nameOf(x.labelId)).join("、") }));
        return;
      }
      if (isCls) {
        const tg = prev.annotations.find((a) => a.type === "tag" && a.labelId != null);
        if (!tg) {
          message.info(t("上一张还没有类别"));
          return;
        }
        commitTag({ labelId: tg.labelId as number, source: "manual" });
        message.success(t("已沿用上一张的类别「{name}」", { name: nameOf(tg.labelId) }));
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
          message.info(t("上一张没有关键点目标"));
          return;
        }
        commitPose([...posesRef.current, ...copied]);
        message.success(t("已复制上一张的 {n} 个目标", { n: copied.length }));
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
          message.info(t("上一张没有标注"));
          return;
        }
        commitPoly([...polysRef.current, ...copied]);
        message.success(t("已复制上一张的 {n} 个轮廓", { n: copied.length }));
        return;
      }
      const copied = prev.annotations
        .filter((a) => a.type === "bbox")
        .map(toBox)
        .map((b) => ({ ...b, x: Math.min(b.x, W - 1), y: Math.min(b.y, H - 1) }))
        .map((b) => ({ ...b, w: Math.min(b.w, W - b.x), h: Math.min(b.h, H - b.y), source: "manual" }));
      if (copied.length === 0) {
        message.info(t("上一张没有标注"));
        return;
      }
      commit([...boxesRef.current, ...copied]);
      message.success(t("已复制上一张的 {n} 个框", { n: copied.length }));
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
    else if ((k === "d" || k === "arrowright") && e.shiftKey) goNextTodo();
    else if (k === "d" || k === "arrowright") go(idx + 1);
    else if (k === "a" || k === "arrowleft") go(idx - 1);
    else if (k === "enter") finish();
    else if (k === "x" && !e.repeat) setStatus("skipped", true); // 按住不放不连续跳
    else if (k === "f") setFitSignal((n) => n + 1);
    else if (k === "h") setShowNames((v) => !v);
    else if (k === "p") predictCurrent();
    else if (e.key === "?" || (k === "/" && e.shiftKey)) setKeysOpen(true);
    else if (k === "escape") setSelected(null);
    else return;
    e.preventDefault();
  };
  useEffect(() => {
    const h = (e: KeyboardEvent) => keyHandler.current(e);
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  /** 审核打回：图退回未标注，原因显示给标注员 */
  const reject = async () => {
    const m = metaRef.current;
    if (!m || rejecting === null) return;
    try {
      await flush();
      await api.reviewImage(m.id, false, rejecting);
      metaRef.current = { ...m, status: "todo" };
      setMeta(metaRef.current);
      markStatus(m.id, "todo");
      setRejecting(null);
      message.success(t("已打回，图回到标注员的任务里"));
      go(idx + 1);
    } catch (e) {
      message.error(errText(e));
    }
  };

  // ---------- 渲染 ----------

  if (ids && ids.length === 0 && filter === "all") {
    return (
      <div className="empty-page">
        {remote ? (
          <Empty description={isAnnotator ? t("还没有分给你的图。发起人分配后这里会自动出现") : t("项目里还没有图片")} />
        ) : (
          <Empty description={t("项目里还没有图片")}>
            <Button type="primary" onClick={() => setPage("data")}>
              {t("去导入图片")}
            </Button>
          </Empty>
        )}
      </div>
    );
  }

  const stripStart = Math.max(0, idx - 12);
  const strip = (ids ?? []).slice(stripStart, stripStart + 25);
  /** 缩略图上的状态点：未标注不显示；不用审核时已审核也算已标注（蓝） */
  const dotClass = (id: number) => {
    const s = statusMap.get(id);
    if (s === "labeled" || (s === "reviewed" && !review)) return "dot labeled";
    if (s === "reviewed") return "dot reviewed";
    if (s === "skipped") return "dot skipped";
    return null;
  };
  /** 顶栏输入序号跳转 */
  const jump = () => {
    if (ids && jumpTo != null && jumpTo >= 1 && jumpTo <= ids.length) go(Math.round(jumpTo) - 1);
    setJumpOpen(false);
  };
  const curColor = colorOf(curLabel);
  const saveText = { saved: t("已保存"), dirty: t("未保存"), saving: t("保存中…"), error: t("保存失败") }[saveState];
  const kptCount = (p: Pose) => p.kpts.filter((_, i) => i % 3 === 2).filter((v) => v > 0).length;

  // 右侧「本图对象」：框、多边形、关键点目标合并成一个列表，连续编号
  const objects = [
    ...boxes.map((b) => ({ key: b.key, labelId: b.labelId, source: b.source, desc: `${Math.round(b.w)}×${Math.round(b.h)}` })),
    ...polys.map((pl) => ({ key: pl.key, labelId: pl.labelId, source: pl.source, desc: t("{n} 点", { n: pl.points.length / 2 }) })),
    ...poses.map((p) => ({ key: p.key, labelId: p.labelId, source: p.source, desc: t("{n}/{total} 点", { n: kptCount(p), total: nk }) })),
  ];
  const selPose = poses.find((p) => p.key === selected) ?? null;
  const placingPose = placing ? poses.find((p) => p.key === placing.key) : null;
  const toolLabel = isSeg ? t("画多边形轮廓 (R)") : isPose ? t("画目标框，再逐个点关键点 (R)") : t("画矩形框 (R)");
  const enterHint = review ? t("Enter 审核通过") : t("Enter 完成并下一张");
  const hint = multi
    ? t("1–9 勾选类别（可以多选） · A/D 翻图 · {enter} · Del 清除类别 · P 模型预测 · Ctrl+D 沿用上一张 · ? 全部快捷键", { enter: enterHint })
    : isCls
    ? t("1–9 选类别{auto} · A/D 翻图 · {enter} · Del 清除类别 · P 模型预测 · Ctrl+D 沿用上一张 · ? 全部快捷键", {
        auto: autoNext ? t("（选完自动下一张）") : "",
        enter: enterHint,
      })
    : t("{draw} · S 智能选取 · V 选择 · 1–9 切换类别 · Del 删除 · A/D 翻图 · {enter} · P 预标注 · 滚轮缩放 · 空格或右键拖动 · H 隐藏类别名 · ? 全部快捷键", {
        enter: enterHint,
        draw: isSeg ? t("R 画轮廓（逐点点击，双击或 Enter 闭合，退格撤点）") : isPose ? t("R 画目标框后逐个点关键点") : t("R 画框"),
      });
  // 点名要加粗，先整句翻译再从 {name} 处分开
  const placeText = t("点击 {name}（{n}/{total}） · Shift+点击 = 被遮挡 · N 跳过 · 退格 回退 · Esc 结束", {
    n: (placing?.idx ?? 0) + 1,
    total: nk,
  }).split("{name}");
  // 中间有个编辑图标，先整句翻译再从 {icon} 处分开
  const textHelp = (
    isSeg
      ? t("按类别名在图里找目标（开放词汇检测 + SAM 出轮廓）。模型只懂英文，常见的中文类别会自动对应英文（如 安全帽 → helmet，显示在类别名后面）；对不上或想改的，在类别的{icon}里填「英文提示词」。首次使用需要下载模型。")
      : t("按类别名在图里找目标（开放词汇检测）。模型只懂英文，常见的中文类别会自动对应英文（如 安全帽 → helmet，显示在类别名后面）；对不上或想改的，在类别的{icon}里填「英文提示词」。首次使用需要下载模型。")
  ).split("{icon}");
  const srcOptions = [
    { label: t("我的模型"), value: "model" },
    // 协作成员不能直接用巡瞳 AI（用的是发起人的额度），可以用「识别规则」和 AI 聊
    ...(isCls ? [] : [{ label: t("文本提示"), value: "text" }, ...(remote || !xuntongEnabled() ? [] : [{ label: t("巡瞳 AI"), value: "ai" }])]),
  ];
  const filterOpts = [
    ...(review ? FILTERS : SOLO_FILTERS).map((f) => (f.value === "all" && isAnnotator ? { ...f, label: t("我的全部") } : f)),
    ...(collab ? [{ label: t("被打回"), value: "rejected" }] : []),
  ];
  const canPredict = aiSrc === "model" ? modelId != null : aiSrc === "ai" ? !!aiModel && !aiError : labels.length > 0;

  const startStripResize = (e: { clientY: number; preventDefault: () => void }) => {
    e.preventDefault();
    const y0 = e.clientY;
    const h0 = stripH;
    let h = h0;
    const move = (ev: MouseEvent) => {
      h = clampStrip(h0 - (ev.clientY - y0));
      setStripH(h);
    };
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      document.body.style.cursor = "";
      prefs.set("stripH", String(h));
    };
    document.body.style.cursor = "row-resize";
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  };

  return (
    <div className="annot" style={{ gridTemplateRows: `42px minmax(0, 1fr) ${stripH}px` }}>
      <div className="annot-top">
        <Button size="small" icon={<LeftOutlined />} disabled={idx === 0} onClick={() => go(idx - 1)} />
        <span style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          <strong>{meta?.name ?? "—"}</strong>{" "}
          <span className="muted">
            ·{" "}
            <Popover
              trigger="click"
              open={jumpOpen}
              onOpenChange={(o) => {
                setJumpOpen(o);
                if (o) setJumpTo(idx + 1);
              }}
              content={
                <Space.Compact>
                  <InputNumber
                    size="small"
                    autoFocus
                    min={1}
                    max={ids?.length || 1}
                    precision={0}
                    value={jumpTo}
                    onChange={(v) => setJumpTo(typeof v === "number" ? v : null)}
                    onPressEnter={jump}
                    style={{ width: 110 }}
                  />
                  <Button size="small" type="primary" onClick={jump}>
                    {t("跳转")}
                  </Button>
                </Space.Compact>
              }
            >
              <span className="annot-jump" title={t("点击输入序号跳转")}>
                {ids?.length ? idx + 1 : 0} / {ids?.length ?? 0}
              </span>
            </Popover>
            {meta && ` · ${meta.width}×${meta.height}`}
          </span>
        </span>
        <Button size="small" icon={<RightOutlined />} disabled={!ids || idx >= ids.length - 1} onClick={() => go(idx + 1)} />
        {meta && (() => {
          // 不用审核时：已审核的图显示成已标注；没有目标的显示「没有目标」
          const st = review || meta.status !== "reviewed" || objects.length || tag ? statusTag(meta.status, review) : { text: t("没有目标"), color: undefined };
          return <Tag color={st.color}>{st.text}</Tag>;
        })()}
        <Segmented
          size="small"
          value={filter}
          options={queue ? [...filterOpts, { label: `${queue.name} ${queue.ids.length}`, value: "queue" }] : filterOpts}
          onChange={(v) => setFilter(v as string)}
        />
        {queue?.from === "review" && (
          <Button size="small" onClick={() => setPage("review")}>
            {t("返回审核")}
          </Button>
        )}
        {filter === "queue" && queue?.marks && (
          <Button size="small" type={showMarks ? "default" : "text"} onClick={() => setShowMarks((v) => !v)}>
            {showMarks ? t("隐藏提示框") : t("显示提示框")}
          </Button>
        )}
        {queue && (
          <Tooltip title={t("不再显示这批图")}>
            <Button
              size="small"
              type="text"
              onClick={() => {
                clearQueue();
                setFilter("all");
              }}
            >
              ×
            </Button>
          </Tooltip>
        )}
        <div className="spacer" />
        {!remote && teamInfo?.assignee && (
          <span className="dim" style={{ fontSize: 12 }}>
            {t("分配给 {name}", { name: teamInfo.assignee })}
          </span>
        )}
        <span className="dim" style={{ fontSize: 12 }}>
          {Math.round(scale * 100)}%
        </span>
        <span style={{ fontSize: 12, color: saveState === "error" ? "var(--bad)" : "var(--dim)" }}>{saveText}</span>
      </div>

      <div className="annot-tools">
        {!isCls && (
          <>
            <Tooltip title={t("选择 / 调整 (V)")} placement="right">
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
              title={t("智能选取 SAM (S)：点目标自动出轮廓，Shift+点击排除，拖动框选；Enter 采用、Esc 放弃")}
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
        <Tooltip title={t("撤销 (Ctrl+Z)")} placement="right">
          <button disabled={histLen.past === 0} onClick={undo}>
            <UndoOutlined />
          </button>
        </Tooltip>
        <Tooltip title={t("重做 (Ctrl+Y)")} placement="right">
          <button disabled={histLen.future === 0} onClick={redo}>
            <RedoOutlined />
          </button>
        </Tooltip>
        <Tooltip title={isCls ? t("沿用上一张的类别 (Ctrl+D)") : t("复制上一张的标注 (Ctrl+D)")} placement="right">
          <button disabled={idx === 0} onClick={copyPrev}>
            <CopyOutlined />
          </button>
        </Tooltip>
        {canVideo && (
          <>
            <Tooltip
              title={t("关键帧插值：先在前面某一帧和这一帧都标好框，点这里把中间的帧自动补上（按类别和位置对应同一个目标）")}
              placement="right"
            >
              <button disabled={idx === 0 || !!videoBusy || !objects.length} onClick={interpolate}>
                {videoBusy === "interp" ? <LoadingOutlined /> : <NodeIndexOutlined />}
              </button>
            </Tooltip>
            <Popover
              placement="right"
              trigger="click"
              title={t("往后跟踪")}
              content={
                <div style={{ width: 240 }}>
                  <div className="muted" style={{ fontSize: 12, marginBottom: 8 }}>
                    {t("从这一帧的框出发，自动跟到后面的帧（同一文件夹按文件名排序）。目标被挡住或跟丢就停；碰到手动标过的帧也停，不会覆盖人工标注。")}
                  </div>
                  <Space.Compact style={{ width: "100%" }}>
                    <InputNumber min={1} max={500} value={trackN} addonBefore={t("帧数")} onChange={(v) => setTrackN(Number(v ?? 30))} style={{ width: 150 }} />
                    <Button type="primary" loading={videoBusy === "track"} disabled={!objects.length || (!!videoBusy && videoBusy !== "track")} onClick={trackForward}>
                      {t("跟踪")}
                    </Button>
                  </Space.Compact>
                  {videoBusy === "track" && trackProg && (
                    <Progress size="small" percent={trackProg.total ? Math.floor((trackProg.done / trackProg.total) * 100) : 0} />
                  )}
                </div>
              }
            >
              <Tooltip title={t("往后跟踪：从这一帧的框自动跟到后面的帧")} placement="right">
                <button disabled={!!videoBusy && videoBusy !== "track"}>
                  {videoBusy === "track" ? <LoadingOutlined /> : <FastForwardOutlined />}
                </button>
              </Tooltip>
            </Popover>
          </>
        )}
        <Tooltip title={t("适应窗口 (F)")} placement="right">
          <button onClick={() => setFitSignal((n) => n + 1)}>
            <ExpandOutlined />
          </button>
        </Tooltip>
      </div>

      <div className="annot-stage">
        {(lockHolder || teamInfo?.rejectReason) && (
          <div className="annot-banner">
            {lockHolder && <Alert type="info" showIcon message={t("「{name}」正在标这张图，你现在只能看，他的改动会实时显示", { name: lockHolder })} />}
            {teamInfo?.rejectReason && (
              <Alert
                type="warning"
                showIcon
                message={t("被打回：{reason}", { reason: teamInfo.rejectReason })}
                description={
                  teamInfo.reviewedBy
                    ? isAnnotator
                      ? t("审核：{name}。改好后点「提交并下一张」", { name: teamInfo.reviewedBy })
                      : t("审核：{name}。改好后点「通过并下一张」", { name: teamInfo.reviewedBy })
                    : undefined
                }
              />
            )}
          </div>
        )}
        {remote && team.options.watermark && team.me && (
          <div className="annot-watermark">
            {Array.from({ length: 24 }, (_, i) => (
              <span key={i} style={{ left: `${(i % 4) * 26 + 2}%`, top: `${Math.floor(i / 4) * 17 + 4}%` }}>
                {team.me!.name} · {new Date().toLocaleDateString()}
              </span>
            ))}
          </div>
        )}
        {meta && imageId !== undefined ? (
          <AnnotCanvas
            src={imageUrl(project.path, imageId)}
            imgW={meta.width}
            imgH={meta.height}
            boxes={
              isCls
                ? []
                : [...boxes, ...poses.map((p) => ({ key: p.key, labelId: p.labelId, x: p.x, y: p.y, w: p.w, h: p.h, source: p.source, score: p.score }))].filter((b) =>
                    shown(b.labelId),
                  )
            }
            polys={polys.filter((x) => shown(x.labelId))}
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
            marks={
              showMarks
                ? highlight && highlight.imageId === imageId
                  ? [{ bbox: highlight.bbox, text: t("体检：{text}", { text: highlight.text }), color: "#ffd43b", fg: "#1b1e23" }]
                  : (filter === "queue" && imageId != null && queue?.marks?.[imageId]) || undefined
                : undefined
            }
            kpts={poses.filter((p) => shown(p.labelId)).map((p) => ({ key: p.key, labelId: p.labelId, kpts: p.kpts }))}
            skeleton={template?.skeleton}
            placing={placing}
            onPlaceKpt={onPlaceKpt}
            onMoveKpt={onMoveKpt}
            sam={{ points: samPoints, box: samBox, preview: samPreview?.points ?? null }}
            onSamClick={onSamClick}
            onSamBox={onSamBox}
            badge={
              multi
                ? tags.length
                  ? {
                      text: tags.map((x) => nameOf(x.labelId) + (x.source !== "manual" && x.score != null ? ` ${x.score.toFixed(2)}` : "")).join(" · "),
                      color: colorOf(tags[0].labelId),
                    }
                  : { text: t("未分类 · 按 1–9 勾选类别（可以多选）"), color: "rgba(60,64,72,.9)" }
                : isCls
                ? tag
                  ? {
                      text: nameOf(tag.labelId) + (tag.source !== "manual" && tag.score != null ? ` · ${tag.score.toFixed(2)}` : ""),
                      color: colorOf(tag.labelId),
                    }
                  : { text: t("未分类 · 按 1–9 选类别"), color: "rgba(60,64,72,.9)" }
                : null
            }
          />
        ) : (
          ids?.length === 0 && <Empty description={t("没有符合筛选条件的图片")} />
        )}
        {placing && placingPose && template && (
          <div className="place-hint">
            {placeText[0]}
            <b>{template.points[placing.idx]}</b>
            {placeText[1]}
          </div>
        )}
        {tool === "sam" && !isCls && (
          <div className="place-hint">
            {samBusy
              ? samUsed.current
                ? t("SAM 计算中…")
                : t("SAM 准备中…（首次使用要安装组件、下载模型，可能需要几分钟）") + (assistLog ? ` · ${assistLog.slice(0, 60)}` : "")
              : samPreview
                ? t("已选出目标 · Enter 采用为「{name}」 · 继续点击修正（Shift+点击排除） · Esc 放弃", { name: nameOf(curLabel) })
                : t("点击目标自动选出轮廓 · 也可以拖动框选 · Shift+点击排除区域")}
          </div>
        )}
        <div className="annot-hint">{hint}</div>
      </div>

      <Modal title={t("快捷键")} open={keysOpen} onCancel={() => setKeysOpen(false)} footer={null} width={560}>
        {(
          [
            [
              review ? t("翻图和审核") : t("翻图"),
              [
                ["A / ←", t("上一张")],
                ["D / →", t("下一张")],
                ["Shift + D / Shift + →", t("下一张未标注")],
                ["Enter", review ? t("审核通过（并跳到下一张）") : t("完成并下一张（没有目标的图记为负样本）")],
                ["X", t("跳过（不参与训练）")],
                ["P", t("用选中的模型预测当前图")],
                ["Ctrl + D", t("沿用上一张的标注")],
              ],
            ],
            multi
              ? [t("分类"), [["1–9", t("勾选 / 取消类别（可以多选）")], ["Del", t("清除这张图的全部类别")]]]
              : isCls
              ? [t("分类"), [["1–9", autoNext ? t("选类别（选完自动下一张）") : t("选类别")], ["Del", t("清除这张图的类别")]]]
              : [
                  t("画框和编辑"),
                  [
                    ["V", t("选择、移动、调整大小")],
                    ["R", isSeg ? t("画多边形轮廓：逐点点击，双击或 Enter 闭合，退格撤一个点") : isPose ? t("画目标框，再逐个点关键点") : t("画矩形框")],
                    ["S", t("智能选取（SAM）：点一下出轮廓，Enter 确认、Esc 清除、退格撤点")],
                    ["1–9", t("切换当前类别（选中框时改它的类别）")],
                    [t("Del / 退格"), t("删除选中的标注")],
                    ["Esc", t("取消选择")],
                    ...(isPose ? [["N", t("关键点：跳过当前点（看不见）")]] : []),
                  ],
                ],
            [
              t("视图和撤销"),
              [
                [t("滚轮"), t("缩放")],
                [t("空格 + 拖动 / 右键拖动"), t("移动画面")],
                ["F", t("适应窗口")],
                ["H", t("隐藏 / 显示类别名")],
                ["Ctrl + Z", t("撤销")],
                ["Ctrl + Y / Ctrl + Shift + Z", t("重做")],
                ["?", t("打开这个面板")],
              ],
            ],
          ] as [string, string[][]][]
        ).map(([group, rows]) => (
          <div key={group} style={{ marginBottom: 14 }}>
            <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>
              {group}
            </div>
            {rows.map(([k, v]) => (
              <div key={k} className="row" style={{ flexWrap: "nowrap", gap: 12, padding: "3px 0", alignItems: "flex-start" }}>
                <kbd style={{ minWidth: 150, fontFamily: "inherit", fontSize: 12, color: "var(--text)" }}>{k}</kbd>
                <span style={{ fontSize: 13 }}>{v}</span>
              </div>
            ))}
          </div>
        ))}
      </Modal>

      <aside className="annot-side">
        {aiAllowed && (
        <section>
          <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}>
            <h3 style={{ margin: 0 }}>{t("AI 辅助")}</h3>
            {!isCls && xuntongEnabled() && (
              <Tooltip title={t("告诉巡瞳 AI 每个类别具体指什么；可以和 AI 聊，它帮你整理规则、在这张图上试识别")}>
                <Button size="small" type="link" icon={<MessageOutlined />} style={{ padding: 0 }} onClick={() => setRulesOpen(true)}>
                  {t("识别规则")}
                </Button>
              </Tooltip>
            )}
          </div>
          {srcOptions.length > 1 && (
            <Segmented size="small" block value={aiSrc} options={srcOptions} onChange={(v) => pickSrc(v as AiSrc)} style={{ marginBottom: 8 }} />
          )}
          {aiSrc === "model" && models.length === 0 && (
            <div className="dim" style={{ fontSize: 12 }}>
              {t("标注几十张后去「训练」页训练一个模型，就能用它自动标注剩下的图片。")}
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
                label: `${r.name} · ${multi ? "mAP" : isCls ? t("准确率") : "mAP50"} ${r.bestMap50?.toFixed(3) ?? "—"}`,
              }))}
            />
          )}
          {aiSrc === "text" && (
            <div className="dim" style={{ fontSize: 12, marginBottom: 6 }}>
              {textHelp[0]}
              <EditOutlined style={{ margin: "0 2px" }} />
              {textHelp[1]}
              {labels.some((l) => !l.prompt && /[^\x00-\x7f]/.test(l.name)) && (
                <div style={{ color: "var(--warn)", marginTop: 4 }}>
                  {t("还没填英文提示词：{names}", {
                    names: labels
                      .filter((l) => !l.prompt && /[^\x00-\x7f]/.test(l.name))
                      .map((l) => l.name)
                      .join(t("、")),
                  })}
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
                    {t("去设置")}
                  </Button>
                  <Button size="small" onClick={() => setAiModels(null)}>
                    {t("重试")}
                  </Button>
                </div>
              </div>
            ) : (
              <>
                <Select
                  size="small"
                  style={{ width: "100%", marginBottom: 6 }}
                  value={aiModel || undefined}
                  placeholder={aiModels ? t("选择识别模型") : t("正在读取可用模型…")}
                  loading={!aiModels}
                  onChange={(v) => {
                    setAiModel(v);
                    prefs.set("aiModel", v);
                  }}
                  options={(aiModels ?? []).map((m) => ({ value: m.code, label: m.name }))}
                />
                <div className="dim" style={{ fontSize: 12, marginBottom: 6 }}>
                  {aiModels?.find((m) => m.code === aiModel)?.description ?? ""}
                  {isSeg && t("（识别出的框会再用 SAM 出轮廓）")}
                </div>
              </>
            ))}
          {(aiSrc !== "model" || models.length > 0) && (
            <>
              <div className="row" style={{ gap: 6, marginBottom: 6, flexWrap: "nowrap" }}>
                <span className="dim" style={{ fontSize: 12, whiteSpace: "nowrap" }}>{t("置信度")}</span>
                <Slider
                  style={{ flex: 1, margin: "0 4px" }}
                  min={0.05}
                  max={0.95}
                  step={0.05}
                  value={conf}
                  onChange={(v) => {
                    setConf(v);
                    prefs.set(confKey(aiSrc), String(v));
                  }}
                />
                <span style={{ fontSize: 12, width: 30 }}>{conf.toFixed(2)}</span>
              </div>
              {predicting && aiSrc !== "model" && (
                <div className="dim" style={{ fontSize: 12, marginBottom: 6 }}>
                  {assistLog ? assistLog.slice(0, 80) : aiSrc === "text" ? t("首次使用要安装组件、下载模型，可能需要几分钟…") : t("正在请求巡瞳 AI…")}
                </div>
              )}
              <div className="row" style={{ gap: 6 }}>
                <Button size="small" type="primary" icon={<ThunderboltOutlined />} loading={predicting} disabled={!canPredict} onClick={predictCurrent}>
                  {isCls ? t("预测本图 (P)") : t("标注本图 (P)")}
                </Button>
                {!remote && (
                  <Button size="small" onClick={predictAll} disabled={!!batch || !canPredict}>
                    {t("批量预标注")}
                  </Button>
                )}
              </div>
            </>
          )}
          {batch && (
            <div className="row" style={{ gap: 6, marginTop: 6, flexWrap: "nowrap" }}>
              <Progress size="small" percent={batch.total ? Math.floor((batch.done / batch.total) * 100) : 0} style={{ flex: 1, margin: 0 }} />
              <Button size="small" danger loading={batchStopping} onClick={stopBatch}>
                {t("停止")}
              </Button>
            </div>
          )}
          {isCls && tagHints.length > 0 && (
            <div style={{ marginTop: 8 }}>
              <div className="dim" style={{ fontSize: 12, marginBottom: 4 }}>{t("模型的候选（点选采用）")}</div>
              {tagHints.map((h) => (
                <div
                  key={h.labelId}
                  className={multi && tags.some((x) => x.labelId === h.labelId) ? "lbl on" : "lbl"}
                  onClick={() => (multi ? addTag : commitTag)({ labelId: h.labelId, source: "model", score: h.score })}
                >
                  <span className="sw" style={{ background: colorOf(h.labelId) }} />
                  <span className="n">{nameOf(h.labelId)}</span>
                  <span className="dim" style={{ fontSize: 11 }}>{h.score.toFixed(2)}</span>
                </div>
              ))}
            </div>
          )}
        </section>
        )}

        <section>
          <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}>
            <h3 style={{ margin: 0 }}>{t("类别")}</h3>
            {isCls && !multi && (
              <span className="row" style={{ gap: 6, fontSize: 12 }}>
                <span className="dim">{t("选完下一张")}</span>
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
          {labels.length === 0 && <div className="dim" style={{ marginBottom: 8 }}>{isCls ? t("先添加要区分的类别") : t("先添加要标注的类别")}</div>}
          {labels.map((l, i) => {
            const on = multi ? tags.some((x) => x.labelId === l.id) : isCls ? tag?.labelId === l.id : l.id === curLabel;
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
                  {canEditLabels && (
                    <EditOutlined
                      onClick={(e) => {
                        e.stopPropagation();
                        setEditing({ ...l, prompt: l.prompt ?? "" });
                      }}
                    />
                  )}
                  {!remote && (
                    <DeleteOutlined
                      onClick={(e) => {
                        e.stopPropagation();
                        removeLabel(l);
                      }}
                    />
                  )}
                </span>
              </div>
            );
          })}
          {canEditLabels && (
          <Space.Compact style={{ width: "100%", marginTop: 8 }}>
            <Input
              size="small"
              placeholder={t("新类别名，回车添加")}
              value={newLabel}
              maxLength={40}
              onChange={(e) => setNewLabel(e.target.value)}
              onPressEnter={addLabel}
            />
            <Button size="small" icon={<PlusOutlined />} onClick={addLabel} />
          </Space.Compact>
          )}
        </section>

        {isPose && template && (
          <section>
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}>
              <h3 style={{ margin: 0 }}>
                {t("关键点模板")} <span className="dim">{t("{n} 点", { n: nk })}</span>
              </h3>
              {!remote && (
                <Button size="small" type="link" onClick={openTplEdit} style={{ padding: 0 }}>
                  {t("编辑")}
                </Button>
              )}
            </div>
            <div className="dim" style={{ fontSize: 12 }}>{template.name}</div>
          </section>
        )}

        {multi ? (
          <section>
            <h3>
              {t("本图类别")} <span className="dim">{tags.length}</span>
            </h3>
            {tags.length ? (
              tags.map((x) => (
                <div key={x.labelId} className="obj on">
                  <span className="sw" style={{ background: colorOf(x.labelId) }} />
                  <span className="n">{nameOf(x.labelId)}</span>
                  {x.source !== "manual" && (
                    <Tag style={{ marginInlineEnd: 0 }}>
                      {SOURCE_TEXT[x.source] ?? x.source}
                      {x.score != null && ` ${x.score.toFixed(2)}`}
                    </Tag>
                  )}
                  <DeleteOutlined className="del" onClick={() => change({ tags: tagsRef.current.filter((y) => y.labelId !== x.labelId) })} />
                </div>
              ))
            ) : (
              <div className="dim">{t("还没有类别，点上面的类别或按 1–9，可以选好几个")}</div>
            )}
          </section>
        ) : isCls ? (
          <section>
            <h3>{t("本图类别")}</h3>
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
              <div className="dim">{t("还没有类别，点上面的类别或按 1–9")}</div>
            )}
          </section>
        ) : (
          <section>
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}>
              <h3 style={{ margin: 0 }}>
                {t("本图对象")} <span className="dim">{objects.length}</span>
              </h3>
              <span className="row" style={{ gap: 6, fontSize: 12 }}>
                <span className="dim">{t("类别名")}</span>
                <Switch size="small" checked={showNames} onChange={setShowNames} />
              </span>
            </div>
            <Select
              size="small"
              style={{ width: "100%", marginBottom: 6 }}
              allowClear
              showSearch
              optionFilterProp="label"
              placeholder={t("画面上显示全部类别")}
              value={onlyLabel ?? undefined}
              onChange={(v) => setOnlyLabel(v ?? null)}
              options={labels.map((l) => ({ value: l.id, label: t("只显示「{name}」", { name: l.name }) }))}
            />
            {objects.length === 0 && (
              <div className="dim">
                {isSeg ? t("还没有轮廓，按 R 在目标边缘逐点点击，或按 S 智能选取") : isPose ? t("还没有目标，按 R 画框后逐个点关键点") : t("还没有框，按 R 在图上拖动画框")}
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
                {t("关键点")} <span className="dim">{kptCount(selPose)}/{nk}</span>
              </h3>
              <Button size="small" type="link" style={{ padding: 0 }} onClick={() => setPlacing({ key: selPose.key, idx: 0 })}>
                {t("重放关键点")}
              </Button>
            </div>
            <div className="dim" style={{ fontSize: 12, marginBottom: 4 }}>{t("点名字重新放这个点；点状态切换「可见 / 被遮挡」")}</div>
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
                      title={v === 2 ? t("可见（点击改成被遮挡）") : v === 1 ? t("被遮挡（点击改成可见）") : t("还没标")}
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
          <h3>{review ? t("审核") : t("完成")}</h3>
          <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
            <Button size="small" icon={<CheckOutlined />} onClick={finish}>
              {!review ? t("完成并下一张") : isAnnotator ? t("提交并下一张") : t("通过并下一张")}
            </Button>
            {collab && canReview && meta && (meta.status === "labeled" || meta.status === "reviewed") && (
              <Button size="small" danger icon={<RollbackOutlined />} onClick={() => setRejecting("")}>
                {t("打回")}
              </Button>
            )}
            <Button size="small" icon={<StepForwardOutlined />} onClick={() => setStatus("skipped", true)}>
              {t("跳过")}
            </Button>
            {meta && (meta.status === "skipped" || (meta.status === "reviewed" && (review || (objects.length === 0 && !tag)))) && (
              <Button size="small" onClick={() => setStatus(objects.length || tag ? "labeled" : "todo", false)}>
                {t("撤回")}
              </Button>
            )}
          </div>
          {teamInfo && (teamInfo.assignee || teamInfo.labeledBy || teamInfo.reviewedBy) && (
            <div className="dim" style={{ fontSize: 12, marginTop: 6 }}>
              {[
                teamInfo.assignee && t("分配给 {name}", { name: teamInfo.assignee }),
                teamInfo.labeledBy && t("{name} 标的", { name: teamInfo.labeledBy }),
                teamInfo.reviewedBy && t("{name} 审过", { name: teamInfo.reviewedBy }),
              ]
                .filter(Boolean)
                .join(" · ")}
            </div>
          )}
          <div className="dim" style={{ fontSize: 12, marginTop: 6 }}>
            {isAnnotator
              ? t("提交后等审核员审核；被打回的图会出现在「被打回」里，原因显示在图上方。")
              : review
                ? t("「跳过」的图不参与训练，可用于模糊、无关的图片。")
                : isCls
                  ? t("选了类别的图自动算已标注，不用审核；模糊、无关的图点「跳过」，不参与训练。")
                  : t("标了目标的图自动算已标注，不用审核。确实没有目标的图点「完成」，会作为负样本参与训练；模糊、无关的图点「跳过」，不参与训练。")}
          </div>
        </section>
      </aside>

      <div
        className="annot-strip-grip"
        title={t("拖动调整缩略图大小，双击恢复默认")}
        onMouseDown={startStripResize}
        onDoubleClick={() => {
          setStripH(STRIP_H_DEFAULT);
          prefs.set("stripH", String(STRIP_H_DEFAULT));
        }}
      />
      <div className="annot-strip">
        {strip.map((id, i) => {
          const dot = dotClass(id);
          return (
            <div key={id} className={id === imageId ? "t on" : "t"} style={{ height: stripH - 26 }} onClick={() => go(stripStart + i)}>
              <img src={thumbUrl(project.path, id)} alt="" draggable={false} />
              {dot && <span className={dot} />}
            </div>
          );
        })}
      </div>

      <Modal
        title={filter === "queue" && queue ? t("「{name}」这一批标完了", { name: queue.name }) : t("这一批标完了")}
        open={!!endInfo}
        onCancel={() => setEndInfo(null)}
        width={420}
        footer={
          endInfo && (
            <Space>
              {!remote && endInfo.done > 0 && (
                <Button
                  type={endInfo.todo > 0 ? "default" : "primary"}
                  onClick={() => {
                    setEndInfo(null);
                    setPage("train");
                  }}
                >
                  {t("去训练")}
                </Button>
              )}
              {/* 一张都没标（比如全跳过了）：去训练也没用，去数据页看看 */}
              {!remote && endInfo.done === 0 && endInfo.todo === 0 && (
                <Button
                  type="primary"
                  onClick={() => {
                    setEndInfo(null);
                    setPage("data");
                  }}
                >
                  {t("去数据页")}
                </Button>
              )}
              {endInfo.todo > 0 && (
                <Button
                  type="primary"
                  onClick={() => {
                    const first = endInfo.firstTodo;
                    setEndInfo(null);
                    // 指定的一批图里还有没标的：跳回去接着标；否则切到「未标注」
                    if (filter === "queue" && first >= 0) setIdx(first);
                    else {
                      setFilter("todo");
                      setIdsVersion((n) => n + 1);
                    }
                  }}
                >
                  {t("继续标未标注的")}
                </Button>
              )}
              {remote && endInfo.todo === 0 && (
                <Button type="primary" onClick={() => setEndInfo(null)}>
                  {t("知道了")}
                </Button>
              )}
            </Space>
          )
        }
      >
        {endInfo &&
          (endInfo.todo > 0
            ? t("已标注 {done} 张，还有 {todo} 张未标注。", endInfo)
            : t("已标注 {done} 张，没有未标注的图了。", endInfo))}
      </Modal>

      <AiRulesDrawer
        open={rulesOpen}
        onClose={() => setRulesOpen(false)}
        imageId={imageId ?? null}
        canEdit={canEditLabels}
        onApplyPreview={(p) => applyPreds(p, "ai")}
        onSaved={() => loadLabels().catch(() => {})}
      />

      <Modal
        title={t("打回这张图")}
        open={rejecting !== null}
        onCancel={() => setRejecting(null)}
        onOk={reject}
        okText={t("打回")}
        okButtonProps={{ danger: true }}
        cancelText={t("取消")}
        width={440}
      >
        <div className="dim" style={{ fontSize: 12, margin: "8px 0" }}>
          {t("写清楚哪里要改，标注员会在图上方看到。")}
        </div>
        <Space wrap style={{ marginBottom: 8 }}>
          {[t("漏标"), t("框不准"), t("类别标错"), t("多标了"), t("关键点不对")].map((r) => (
            <Tag.CheckableTag
              key={r}
              checked={(rejecting ?? "").split("；").includes(r)}
              onChange={(on) =>
                setRejecting((s) => {
                  const parts = (s ?? "").split("；").filter((x) => x && x !== r);
                  return (on ? [...parts, r] : parts).join("；");
                })
              }
            >
              {r}
            </Tag.CheckableTag>
          ))}
        </Space>
        <Input.TextArea
          autoSize={{ minRows: 2, maxRows: 5 }}
          value={rejecting ?? ""}
          placeholder={t("比如：左下角还有两个安全帽没标")}
          onChange={(e) => setRejecting(e.target.value)}
        />
      </Modal>

      <Modal
        title={t("编辑类别")}
        open={!!editing}
        onCancel={() => setEditing(null)}
        onOk={saveLabelEdit}
        okText={t("保存")}
        cancelText={t("取消")}
        width={420}
      >
        {editing && (
          <>
            <Space style={{ marginTop: 12 }}>
              <ColorPicker
                value={editing.color}
                disabledAlpha
                presets={[{ label: t("常用"), colors: PALETTE }]}
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
                  {t("英文提示词（「文本提示」自动标注用；可写多个，用逗号隔开，如 helmet, hard hat）")}
                </div>
                <Input
                  value={editing.prompt ?? ""}
                  maxLength={120}
                  placeholder={t("例如 helmet")}
                  onChange={(e) => setEditing({ ...editing, prompt: e.target.value })}
                  onPressEnter={saveLabelEdit}
                />
              </div>
            )}
            {!isCls && xuntongEnabled() && (
              <div style={{ marginTop: 12 }}>
                <div className="dim" style={{ fontSize: 12, marginBottom: 4 }}>
                  {t("识别说明（巡瞳 AI「按项目类别识别」用：这个类别具体指什么、什么情况算 / 不算，可不填）")}
                </div>
                <Input.TextArea
                  autoSize={{ minRows: 2, maxRows: 5 }}
                  maxLength={500}
                  value={editing.description ?? ""}
                  placeholder={t("比如：头上没有安全帽的人；戴了帽子但没系带也算")}
                  onChange={(e) => setEditing({ ...editing, description: e.target.value })}
                />
              </div>
            )}
          </>
        )}
      </Modal>

      <Modal
        title={t("关键点模板")}
        open={!!tplEdit}
        onCancel={() => setTplEdit(null)}
        onOk={saveTpl}
        okText={t("保存")}
        cancelText={t("取消")}
        width={520}
      >
        {tplEdit && (
          <>
            <div className="dim" style={{ fontSize: 12, margin: "8px 0" }}>
              {t("点的顺序就是标注时点击的顺序。已经有关键点标注后只能改名字和连线，不能增减点数。")}
            </div>
            <Input
              value={tplEdit.name}
              onChange={(e) => setTplEdit({ ...tplEdit, name: e.target.value })}
              placeholder={t("模板名")}
              style={{ marginBottom: 8 }}
            />
            <div className="row" style={{ gap: 8, alignItems: "flex-start" }}>
              <div style={{ flex: 1 }}>
                <div className="dim" style={{ fontSize: 12, marginBottom: 4 }}>{t("点名（每行一个）")}</div>
                <Input.TextArea rows={12} value={tplEdit.points} onChange={(e) => setTplEdit({ ...tplEdit, points: e.target.value })} />
              </div>
              <div style={{ width: 150 }}>
                <div className="dim" style={{ fontSize: 12, marginBottom: 4 }}>{t("连线（每行一对序号，如 1-2）")}</div>
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
              {t("使用 COCO 人体 17 点")}
            </Button>
          </>
        )}
      </Modal>
    </div>
  );
}
