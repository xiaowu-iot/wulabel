// 审核页（只在团队协作时有）：按成员、状态、文件名筛选；宫格里看着框批量通过 / 打回，可以撤销；
// 逐张细看；抽检一个成员的一部分图，按合格率处理其余的。要动手改就带着队列去标注页，改完点「返回审核」回来。
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type MouseEvent as RMouseEvent, type ReactNode } from "react";
import { App, Button, Checkbox, Empty, Input, Modal, Pagination, Segmented, Select, Slider, Space, Spin, Tag, Tooltip } from "antd";
import { CheckOutlined, CloseOutlined, EditOutlined, LeftOutlined, ReloadOutlined, RightOutlined } from "@ant-design/icons";
import {
  api,
  errText,
  imageUrl,
  thumbUrl,
  type Annotation,
  type BatchResult,
  type KeypointData,
  type Label,
  type ProjectInfo,
  type ReviewItem,
  type ReviewMember,
  type ReviewQuery,
  type ShareEvent,
} from "../api";
import { listen } from "../transport";
import { t } from "../i18n";
import { useApp } from "../store";

type View = "grid" | "single";

interface Saved {
  q: ReviewQuery;
  page: number;
  pageSize: number;
  view: View;
  queue: number[];
  qi: number;
  thumb: number;
}

interface Sample {
  member: number;
  name: string;
  ids: number[];
  ok: number;
  bad: number;
}

const emptyQuery = (): ReviewQuery => ({
  member: null,
  statuses: ["labeled"],
  name: "",
  labelId: null,
  empty: false,
  rejectedBefore: false,
  sort: "time",
});

// 切到别的页（比如去标注页改图）再回来时，筛选、页码、逐张的位置都还在
let saved: Saved | null = null;
const defaults = (): Saved => ({ q: emptyQuery(), page: 1, pageSize: 48, view: "grid", queue: [], qi: 0, thumb: 180 });

/** 协作页「看他的图」：打开审核页时只看这个人的待审核图 */
export function reviewMemberNext(memberId: number) {
  saved = { ...(saved ?? defaults()), q: { ...emptyQuery(), member: memberId }, page: 1, view: "grid" };
}

const statusOpts = () => [
  { key: "labeled", label: t("待审核") },
  { key: "todo", label: t("未标注") },
  { key: "reviewed", label: t("已通过") },
  { key: "rejected", label: t("被打回") },
  { key: "skipped", label: t("已跳过") },
];

const rejectPresets = () => [t("漏标"), t("框不准"), t("类别标错"), t("多标了"), t("关键点不对")];

function statusView(it: ReviewItem): { text: string; color?: string } {
  if (it.rejectReason) return { text: t("被打回"), color: "error" };
  switch (it.status) {
    case "labeled":
      return { text: t("待审核"), color: "processing" };
    case "reviewed":
      return { text: t("已通过"), color: "success" };
    case "skipped":
      return { text: t("已跳过") };
    default:
      return { text: t("未标注") };
  }
}

function timeText(sec: number) {
  if (!sec) return "";
  const d = new Date(sec * 1000);
  const now = new Date();
  const hm = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  if (d.toDateString() === now.toDateString()) return hm;
  return `${d.getMonth() + 1}-${d.getDate()} ${hm}`;
}

/** 图片 + 标注画在上面（框、轮廓、关键点；分类显示类别名） */
function Shot({ src, item, colors, names, big }: { src: string; item: ReviewItem; colors: Map<number, string>; names: Map<number, string>; big?: boolean }) {
  const w = item.width || 1;
  const h = item.height || 1;
  const fs = Math.max(w, h) / 55;
  const shapes: ReactNode[] = [];
  const tags: string[] = [];
  item.annotations.forEach((a: Annotation, i) => {
    const c = (a.labelId != null && colors.get(a.labelId)) || "#3d8bfd";
    const name = (a.labelId != null && names.get(a.labelId)) || "";
    const dash = a.source !== "manual" ? "6 4" : undefined;
    let box: number[] | null = null;
    if (a.type === "tag") {
      if (name) tags.push(name);
      return;
    }
    if (a.type === "bbox") {
      const [x, y, bw, bh] = a.data as number[];
      box = [x, y, bw, bh];
      shapes.push(<rect key={i} x={x} y={y} width={bw} height={bh} fill="none" stroke={c} strokeWidth={2} strokeDasharray={dash} vectorEffect="non-scaling-stroke" />);
    } else if (a.type === "polygon") {
      const d = a.data as number[];
      const pts: string[] = [];
      for (let k = 0; k + 1 < d.length; k += 2) pts.push(`${d[k]},${d[k + 1]}`);
      if (pts.length) {
        let x0 = Infinity, y0 = Infinity;
        for (let k = 0; k + 1 < d.length; k += 2) {
          x0 = Math.min(x0, d[k]);
          y0 = Math.min(y0, d[k + 1]);
        }
        box = [x0, y0, 0, 0];
        shapes.push(<polygon key={i} points={pts.join(" ")} fill={c} fillOpacity={0.18} stroke={c} strokeWidth={2} strokeDasharray={dash} vectorEffect="non-scaling-stroke" />);
      }
    } else if (a.type === "keypoints") {
      const k = a.data as KeypointData;
      if (k?.box?.length === 4) {
        const [x, y, bw, bh] = k.box;
        box = [x, y, bw, bh];
        shapes.push(<rect key={i} x={x} y={y} width={bw} height={bh} fill="none" stroke={c} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />);
      }
      const kp = k?.kpts ?? [];
      for (let j = 0; j + 2 < kp.length; j += 3) {
        if (!(kp[j + 2] > 0)) continue;
        shapes.push(<circle key={`${i}-${j}`} cx={kp[j]} cy={kp[j + 1]} r={fs * 0.35} fill={c} />);
      }
    }
    if (big && box && name) {
      shapes.push(
        <text key={`${i}-t`} x={box[0]} y={Math.max(fs, box[1] - fs * 0.3)} fontSize={fs} fill={c} stroke="rgba(0,0,0,.6)" strokeWidth={fs * 0.12} paintOrder="stroke">
          {name}
        </text>,
      );
    }
  });
  return (
    <div className={big ? "rv-shot big" : "rv-shot"} style={big ? { aspectRatio: `${w} / ${h}` } : undefined}>
      <img src={src} alt={item.name} loading="lazy" draggable={false} />
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet">
        {shapes}
      </svg>
      {tags.length > 0 && <span className="rv-tags">{tags.join("、")}</span>}
    </div>
  );
}

/** 每个类别几个，比如「安全帽 2 · 反光衣 1」；没有标注时说「没有目标」 */
function ClassSummary({ item, colors, names, max }: { item: ReviewItem; colors: Map<number, string>; names: Map<number, string>; max?: number }) {
  const m = new Map<number, number>();
  for (const a of item.annotations) if (a.labelId != null) m.set(a.labelId, (m.get(a.labelId) ?? 0) + 1);
  if (!item.annotations.length) return <span className="dim">{t("没有目标")}</span>;
  const list = [...m.entries()].sort((a, b) => b[1] - a[1]);
  const shown = max ? list.slice(0, max) : list;
  return (
    <>
      {shown.map(([id, n]) => (
        <span key={id} className="rv-cls">
          <i style={{ background: colors.get(id) ?? "#3d8bfd" }} />
          {names.get(id) ?? "?"} {n}
        </span>
      ))}
      {max && list.length > max && <span className="dim">+{list.length - max}</span>}
    </>
  );
}

export default function Review({ project }: { project: ProjectInfo }) {
  const { message } = App.useApp();
  const annotateQueue = useApp((s) => s.annotateQueue);
  const refreshStats = useApp((s) => s.refreshStats);
  const remote = useApp((s) => s.team.remote);

  const s0 = useMemo(() => saved ?? defaults(), []);
  const [q, setQ] = useState<ReviewQuery>(s0.q);
  const [page, setPage] = useState(s0.page);
  const [pageSize, setPageSize] = useState(s0.pageSize);
  const [view, setView] = useState<View>(s0.view);
  const [queue, setQueue] = useState<number[]>(s0.queue);
  const [qi, setQi] = useState(s0.qi);
  const [thumb, setThumb] = useState(s0.thumb);
  const [nameInput, setNameInput] = useState(s0.q.name);
  const [more, setMore] = useState(!!(s0.q.labelId || s0.q.empty || s0.q.rejectedBefore));

  const [items, setItems] = useState<ReviewItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [members, setMembers] = useState<ReviewMember[]>([]);
  const [labels, setLabels] = useState<Label[]>([]);
  const [sel, setSel] = useState<Set<number>>(new Set());
  const [cur, setCur] = useState(0);
  const lastClick = useRef<number | null>(null);
  const [busy, setBusy] = useState(false);

  const [rejectIds, setRejectIds] = useState<number[] | null>(null);
  const [reason, setReason] = useState("");
  const [quick, setQuick] = useState<number | null>(null);
  const [one, setOne] = useState<ReviewItem | null>(null);
  const [sample, setSample] = useState<Sample | null>(null);
  const [sampleAsk, setSampleAsk] = useState<{ member: number | null; ratio: number } | null>(null);
  const [sampleDone, setSampleDone] = useState<{ rest: number[] } | null>(null);
  const gridRef = useRef<HTMLDivElement>(null);

  const colors = useMemo(() => new Map(labels.map((l) => [l.id, l.color])), [labels]);
  const names = useMemo(() => new Map(labels.map((l) => [l.id, l.name])), [labels]);
  const pending = useMemo(() => members.reduce((n, m) => n + m.pending, 0), [members]);

  useEffect(() => {
    saved = { q, page, pageSize, view, queue, qi, thumb };
  }, [q, page, pageSize, view, queue, qi, thumb]);

  useEffect(() => {
    api.listLabels().then(setLabels).catch(() => {});
  }, []);

  // 文件名输入停一下再搜
  useEffect(() => {
    if (nameInput === q.name) return;
    const h = setTimeout(() => {
      setQ((x) => ({ ...x, name: nameInput }));
      setPage(1);
    }, 300);
    return () => clearTimeout(h);
  }, [nameInput, q.name]);

  const seq = useRef(0);
  const load = useCallback(async () => {
    const my = ++seq.current;
    setLoading(true);
    try {
      const [pg, ms] = await Promise.all([api.reviewQuery(q, (page - 1) * pageSize, pageSize), api.reviewMembers(q)]);
      if (my !== seq.current) return;
      // 这页被审完了（比如最后一页全部通过）：退回上一页
      if (!pg.items.length && pg.total > 0 && page > 1) {
        setPage(Math.max(1, Math.ceil(pg.total / pageSize)));
        return;
      }
      setItems(pg.items);
      setTotal(pg.total);
      setMembers(ms);
      setCur((c) => Math.min(c, Math.max(0, pg.items.length - 1)));
    } catch (e) {
      if (my === seq.current) message.error(errText(e));
    } finally {
      if (my === seq.current) setLoading(false);
    }
  }, [q, page, pageSize, message]);

  useEffect(() => {
    load();
  }, [load]);

  // 逐张：取当前这张的详情
  const curId = view === "single" ? queue[qi] : undefined;
  const loadOne = useCallback(async () => {
    if (curId == null) return setOne(null);
    try {
      const r = await api.reviewQuery({ ...emptyQuery(), statuses: [], ids: [curId] }, 0, 1);
      setOne(r.items[0] ?? null);
    } catch (e) {
      message.error(errText(e));
    }
  }, [curId, message]);
  useEffect(() => {
    loadOne();
  }, [loadOne]);

  // 别人提交、改图、审核了：稍等一下刷新（批量操作只来一条）
  const loadRef = useRef(load);
  loadRef.current = load;
  const loadOneRef = useRef(loadOne);
  loadOneRef.current = loadOne;
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    const un = listen<ShareEvent>("share-event", (e) => {
      const k = e.payload.kind;
      if (k !== "ann" && k !== "status" && k !== "status-batch" && k !== "assign") return;
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        loadRef.current();
        loadOneRef.current();
      }, 600);
    });
    return () => {
      if (timer) clearTimeout(timer);
      un.then((f) => f());
    };
  }, []);

  const setQuery = (patch: Partial<ReviewQuery>) => {
    setQ((x) => ({ ...x, ...patch }));
    setPage(1);
    setSel(new Set());
    setCur(0);
    lastClick.current = null;
  };

  // ---------- 批量通过 / 打回 ----------

  const undo = async (token: string, key: string) => {
    message.destroy(key);
    try {
      const ids = await api.reviewUndo(token);
      message.success(t("已撤销，{n} 张恢复成原来的状态", { n: ids.length }));
      load();
      loadOne();
      if (!remote) refreshStats();
    } catch (e) {
      message.error(errText(e));
    }
  };

  const report = (r: BatchResult, pass: boolean) => {
    const skipped: string[] = [];
    if (r.skippedLocked) skipped.push(t("{n} 张有人正在改", { n: r.skippedLocked }));
    if (r.skippedNotPending) skipped.push(t("{n} 张不是待审核", { n: r.skippedNotPending }));
    const head = pass ? t("已通过 {n} 张", { n: r.done }) : t("已打回 {n} 张", { n: r.done });
    const text = skipped.length ? t("{head}，跳过 {skipped}", { head, skipped: skipped.join(" · ") }) : head;
    if (!r.token) {
      message.warning(text);
      return;
    }
    const key = `rv-undo-${r.token}`;
    message.open({
      key,
      type: "success",
      duration: 10,
      content: (
        <span>
          {text}
          <a style={{ marginLeft: 10 }} onClick={() => undo(r.token, key)}>
            {t("撤销")}
          </a>
        </span>
      ),
    });
  };

  const batch = async (ids: number[], pass: boolean, why: string) => {
    if (!ids.length) return;
    setBusy(true);
    try {
      const r = await api.reviewMany(ids, pass, why);
      setSel(new Set());
      report(r, pass);
      await load();
      if (!remote) refreshStats();
      return r;
    } catch (e) {
      message.error(errText(e));
    } finally {
      setBusy(false);
    }
  };

  const openReject = (ids: number[]) => {
    if (!ids.length) return;
    setReason("");
    setRejectIds(ids);
  };

  // ---------- 逐张 ----------

  const advance = () => {
    if (qi < queue.length - 1) {
      setQi(qi + 1);
      return;
    }
    if (sample) {
      finishSample();
      return;
    }
    message.success(t("这一批看完了"));
    setView("grid");
    load();
  };

  /** 逐张、放大看时审一张（走单张审核接口，和标注页一样） */
  const judge = async (id: number, pass: boolean, why: string) => {
    setBusy(true);
    try {
      await api.reviewImage(id, pass, why);
      if (sample && view === "single") setSample({ ...sample, ok: sample.ok + (pass ? 1 : 0), bad: sample.bad + (pass ? 0 : 1) });
      if (!remote) refreshStats();
      if (view === "single") advance();
      else {
        message.success(pass ? t("已通过") : t("已打回，图回到标注员的任务里"));
        load();
      }
    } catch (e) {
      message.error(errText(e));
    } finally {
      setBusy(false);
    }
  };

  const doReject = async () => {
    const ids = rejectIds ?? [];
    const why = reason.trim() || t("需要修改");
    setRejectIds(null);
    if (ids.length === 1 && (view === "single" || quick != null)) {
      setQuick(null);
      await judge(ids[0], false, why);
    } else await batch(ids, false, why);
  };

  const toSingle = async (startId?: number) => {
    try {
      const all = await api.reviewIds(q);
      if (!all.length) {
        message.info(t("没有符合条件的图"));
        return;
      }
      setSample(null);
      setQueue(all);
      setQi(Math.max(0, startId != null ? all.indexOf(startId) : 0));
      setView("single");
    } catch (e) {
      message.error(errText(e));
    }
  };

  const toGrid = () => {
    setView("grid");
    setSample(null);
    load();
  };

  /** 去标注页自己动手改：带着这批图，改完点「返回审核」回来 */
  const edit = async (id: number) => {
    let ids = view === "single" ? queue : items.map((x) => x.id);
    if (view === "grid") {
      try {
        ids = await api.reviewIds(q);
      } catch {
        /* 用这一页 */
      }
    }
    annotateQueue(ids, t("审核中的图"), { from: "review", focus: id });
  };

  // ---------- 抽检 ----------

  const startSample = async () => {
    const ask = sampleAsk;
    if (!ask || ask.member == null) return;
    try {
      const pool = await api.reviewIds({ ...emptyQuery(), member: ask.member, statuses: ["labeled"] });
      if (!pool.length) {
        message.info(t("他现在没有待审核的图"));
        return;
      }
      const k = Math.max(1, Math.round(pool.length * ask.ratio));
      const shuffled = [...pool];
      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }
      const picked = shuffled.slice(0, k).sort((a, b) => a - b);
      const name = members.find((m) => m.memberId === ask.member)?.name ?? "";
      setSampleAsk(null);
      setSample({ member: ask.member, name, ids: picked, ok: 0, bad: 0 });
      setQueue(picked);
      setQi(0);
      setView("single");
    } catch (e) {
      message.error(errText(e));
    }
  };

  const finishSample = async () => {
    if (!sample) return;
    try {
      const pool = await api.reviewIds({ ...emptyQuery(), member: sample.member, statuses: ["labeled"] });
      const inSample = new Set(sample.ids);
      setSampleDone({ rest: pool.filter((id) => !inSample.has(id)) });
    } catch (e) {
      message.error(errText(e));
    }
  };

  const closeSample = async (action: "pass" | "reject" | null) => {
    const rest = sampleDone?.rest ?? [];
    setSampleDone(null);
    setSample(null);
    setView("grid");
    if (action && rest.length) await batch(rest, action === "pass", t("抽检不合格，请自查后重新提交"));
    else load();
  };

  // ---------- 选择、键盘 ----------

  const toggle = (id: number) =>
    setSel((s) => {
      const n = new Set(s);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });

  const clickCard = (e: RMouseEvent, i: number) => {
    const id = items[i].id;
    if (e.shiftKey && lastClick.current != null) {
      const [a, b] = [Math.min(lastClick.current, i), Math.max(lastClick.current, i)];
      setSel((s) => {
        const n = new Set(s);
        for (let k = a; k <= b; k++) if (items[k]) n.add(items[k].id);
        return n;
      });
    } else toggle(id);
    lastClick.current = i;
    setCur(i);
  };

  const selectAll = async () => {
    try {
      const all = await api.reviewIds(q);
      setSel(new Set(all));
      message.info(t("选中了筛选结果里的全部 {n} 张（包括其他页）", { n: all.length }));
    } catch (e) {
      message.error(errText(e));
    }
  };

  const targets = () => (sel.size ? [...sel] : items[cur] ? [items[cur].id] : []);
  const modalOpen = rejectIds != null || sampleAsk != null || sampleDone != null;

  const keyRef = useRef<(e: KeyboardEvent) => void>(() => {});
  keyRef.current = (e: KeyboardEvent) => {
    const el = e.target as HTMLElement | null;
    if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable || el.closest(".ant-select"))) return;
    if (modalOpen || busy) return;
    const k = e.key.toLowerCase();
    if (quick != null) {
      const it = items[quick];
      if (k === "escape" || k === " ") {
        e.preventDefault();
        setQuick(null);
      } else if (k === "arrowright" || k === "arrowleft") {
        e.preventDefault();
        const n = Math.min(items.length - 1, Math.max(0, quick + (k === "arrowright" ? 1 : -1)));
        setQuick(n);
        setCur(n);
      } else if (k === "a" && it?.status === "labeled") {
        setQuick(null);
        judge(it.id, true, "");
      } else if (k === "r" && it) openReject([it.id]);
      return;
    }
    if (view === "single") {
      if (k === "arrowright" || k === "arrowleft") {
        e.preventDefault();
        setQi((i) => Math.min(queue.length - 1, Math.max(0, i + (k === "arrowright" ? 1 : -1))));
      } else if (k === "a" && one?.status === "labeled") judge(one.id, true, "");
      else if (k === "r" && one) openReject([one.id]);
      else if (k === "e" && one) edit(one.id);
      else if (k === "escape") toGrid();
      return;
    }
    if (!items.length) return;
    const cols = gridRef.current ? getComputedStyle(gridRef.current).gridTemplateColumns.split(" ").length : 1;
    let n = cur;
    if (k === "arrowright") n = cur + 1;
    else if (k === "arrowleft") n = cur - 1;
    else if (k === "arrowdown") n = cur + cols;
    else if (k === "arrowup") n = cur - cols;
    if (n !== cur) {
      e.preventDefault();
      n = Math.min(items.length - 1, Math.max(0, n));
      setCur(n);
      gridRef.current?.children[n]?.scrollIntoView({ block: "nearest" });
      return;
    }
    if (k === " ") {
      e.preventDefault();
      setQuick(cur);
    } else if (k === "x") toggle(items[cur].id);
    else if (k === "a" && !e.ctrlKey && !e.metaKey) batch(targets(), true, "");
    else if (k === "a") {
      e.preventDefault();
      setSel((s) => new Set([...s, ...items.map((x) => x.id)]));
    } else if (k === "r") openReject(targets());
    else if (k === "e") edit(items[cur].id);
    else if (k === "escape") setSel(new Set());
  };
  useEffect(() => {
    const h = (e: KeyboardEvent) => keyRef.current(e);
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  // ---------- 界面 ----------

  const rate = (m: ReviewMember) => (m.passed + m.rejected ? Math.round((m.passed / (m.passed + m.rejected)) * 100) : null);
  const allCount = members.reduce((n, m) => n + m.count, 0);

  const head = (
    <div className="rv-head">
      <h2>{t("审核")}</h2>
      <span className="dim">{t("待审核 {n} 张", { n: pending })}</span>
      <div className="spacer" />
      {view === "grid" && (
        <Button size="small" onClick={() => setSampleAsk({ member: q.member ?? members.find((m) => m.pending > 0)?.memberId ?? null, ratio: 0.2 })}>
          {t("抽检…")}
        </Button>
      )}
      <Tooltip title={t("刷新")}>
        <Button size="small" icon={<ReloadOutlined />} onClick={() => (view === "grid" ? load() : loadOne())} />
      </Tooltip>
      <Segmented
        size="small"
        value={view}
        options={[
          { label: t("宫格"), value: "grid" },
          { label: t("逐张"), value: "single" },
        ]}
        onChange={(v) => (v === "grid" ? toGrid() : toSingle(items[cur]?.id))}
      />
    </div>
  );

  const memberBar = (
    <div className="rv-members">
      <span className={q.member == null ? "rv-mchip on" : "rv-mchip"} onClick={() => setQuery({ member: null })}>
        {t("全部成员")} <b>{allCount}</b>
      </span>
      {members.map((m) => {
        const r = rate(m);
        return (
          <Tooltip key={m.memberId} title={t("待审核 {p} 张 · 被通过 {a} 次、被打回 {b} 次", { p: m.pending, a: m.passed, b: m.rejected })}>
            <span className={q.member === m.memberId ? "rv-mchip on" : "rv-mchip"} onClick={() => setQuery({ member: m.memberId })}>
              <span className="rv-av">{m.name.slice(-1)}</span>
              {m.name} <b>{m.count}</b>
              {r != null && <span className={r < 80 ? "rv-rate low" : "rv-rate"}>{t("通过率 {r}%", { r })}</span>}
            </span>
          </Tooltip>
        );
      })}
    </div>
  );

  const filters = (
    <div className="rv-filters">
      <div className="rv-row">
        <span className="dim">{t("状态")}</span>
        {statusOpts().map((s) => (
          <Tag.CheckableTag
            key={s.key}
            checked={q.statuses.includes(s.key)}
            onChange={(on) => setQuery({ statuses: on ? [...q.statuses, s.key] : q.statuses.filter((x) => x !== s.key) })}
          >
            {s.label}
          </Tag.CheckableTag>
        ))}
        <Input
          size="small"
          allowClear
          style={{ width: 200 }}
          placeholder={t("搜文件名")}
          value={nameInput}
          onChange={(e) => setNameInput(e.target.value)}
        />
        <Select
          size="small"
          style={{ width: 170 }}
          value={q.sort}
          onChange={(v) => setQuery({ sort: v })}
          options={[
            { value: "time", label: t("按提交时间（新的在前）") },
            { value: "name", label: t("按文件名") },
          ]}
        />
        <Button size="small" type="link" onClick={() => setMore(!more)}>
          {more ? t("收起") : t("更多筛选")}
          {!more && (q.labelId != null || q.empty || q.rejectedBefore) ? ` · ${t("已用")}` : ""}
        </Button>
      </div>
      {more && (
        <div className="rv-row">
          <span className="dim">{t("含类别")}</span>
          <Select
            size="small"
            style={{ width: 160 }}
            allowClear
            placeholder={t("不限")}
            value={q.labelId ?? undefined}
            onChange={(v) => setQuery({ labelId: v ?? null })}
            options={labels.map((l) => ({ value: l.id, label: l.name }))}
          />
          <Checkbox checked={q.empty} onChange={(e) => setQuery({ empty: e.target.checked })}>
            {t("只看没有目标的")}
          </Checkbox>
          <Checkbox checked={q.rejectedBefore} onChange={(e) => setQuery({ rejectedBefore: e.target.checked })}>
            {t("只看打回过的")}
          </Checkbox>
          <span className="dim">{t("图大小")}</span>
          <Slider style={{ width: 120, margin: "0 8px" }} min={130} max={320} step={10} value={thumb} onChange={setThumb} />
          <Button
            size="small"
            onClick={() => {
              setNameInput("");
              setQuery(emptyQuery());
            }}
          >
            {t("清空筛选")}
          </Button>
        </div>
      )}
    </div>
  );

  let body: ReactNode;
  if (view === "single") {
    const it = one;
    body = (
      <>
        {sample && (
          <div className="rv-sample">
            {t("抽检「{name}」：共 {n} 张，已审 {d} 张，合格 {ok}、不合格 {bad}", {
              name: sample.name,
              n: sample.ids.length,
              d: sample.ok + sample.bad,
              ok: sample.ok,
              bad: sample.bad,
            })}
            <div className="spacer" />
            <Button size="small" onClick={finishSample}>
              {t("结束抽检")}
            </Button>
          </div>
        )}
        <div className="rv-progress">
          <i style={{ width: `${((qi + 1) / Math.max(1, queue.length)) * 100}%` }} />
        </div>
        {!it ? (
          <Spin style={{ margin: 40 }} />
        ) : (
          <div className="rv-single">
            <div className="rv-stage">
              <Shot src={imageUrl(project.path, it.id)} item={it} colors={colors} names={names} big />
              <Button className="rv-nav l" shape="circle" icon={<LeftOutlined />} disabled={qi === 0} onClick={() => setQi(qi - 1)} />
              <Button className="rv-nav r" shape="circle" icon={<RightOutlined />} disabled={qi >= queue.length - 1} onClick={() => setQi(qi + 1)} />
            </div>
            <div className="rv-panel">
              <div className="rv-panel-head">
                <b className="mono">{it.name}</b>
                <Tag color={statusView(it).color}>{statusView(it).text}</Tag>
              </div>
              <div className="dim">{t("第 {i} / {n} 张", { i: qi + 1, n: queue.length })}</div>
              <div className="rv-kv">
                <span>{t("标注人")}</span>
                <span>{it.member ?? "—"}</span>
                <span>{t("提交")}</span>
                <span>{timeText(it.submittedAt) || "—"}</span>
                <span>{t("打回记录")}</span>
                <span>{it.rejectedBefore ? t("打回过") : t("没有")}</span>
              </div>
              {it.rejectReason && <div className="rv-reason">{t("被打回：{reason}", { reason: it.rejectReason })}</div>}
              <div>
                <div style={{ marginBottom: 4 }}>{t("标注对象 {n} 个", { n: it.annotations.length })}</div>
                <div className="rv-objs">
                  <ClassSummary item={it} colors={colors} names={names} />
                </div>
              </div>
              <div className="rv-big">
                <Button danger icon={<CloseOutlined />} disabled={busy} onClick={() => openReject([it.id])}>
                  {t("打回")} <kbd>R</kbd>
                </Button>
                <Button type="primary" icon={<CheckOutlined />} disabled={busy || it.status !== "labeled"} onClick={() => judge(it.id, true, "")}>
                  {t("通过")} <kbd>A</kbd>
                </Button>
              </div>
              {it.status !== "labeled" && <div className="dim">{t("这张不是待审核，不用再通过；有问题仍可以打回。")}</div>}
              <Button icon={<EditOutlined />} onClick={() => edit(it.id)}>
                {t("自己动手改")} <kbd>E</kbd>
              </Button>
              <div className="rv-keys">
                <span>
                  <kbd>←</kbd> <kbd>→</kbd> {t("翻图")}
                </span>
                <span>
                  <kbd>Esc</kbd> {t("回到宫格")}
                </span>
              </div>
            </div>
          </div>
        )}
      </>
    );
  } else {
    body = (
      <>
        {memberBar}
        {filters}
        <div className="rv-act">
          {sel.size ? (
            <>
              <b>{t("已选 {n} 张", { n: sel.size })}</b>
              <Button size="small" type="link" onClick={() => setSel(new Set())}>
                {t("取消选择")}
              </Button>
            </>
          ) : (
            <span className="dim">{t("点图勾选，Shift 连选；双击逐张细看")}</span>
          )}
          <Button size="small" onClick={() => setSel((s) => new Set([...s, ...items.map((x) => x.id)]))}>
            {t("全选本页")}
          </Button>
          <Button size="small" onClick={selectAll} disabled={!total}>
            {t("全选筛选结果（{n}）", { n: total })}
          </Button>
          <div className="spacer" />
          <Button size="small" danger disabled={!sel.size || busy} onClick={() => openReject([...sel])}>
            {t("打回…")}
          </Button>
          <Button size="small" type="primary" disabled={!sel.size || busy} loading={busy} onClick={() => batch([...sel], true, "")}>
            {sel.size ? t("通过 {n} 张", { n: sel.size }) : t("通过")}
          </Button>
        </div>
        <Spin spinning={loading && !items.length}>
          {items.length ? (
            <div className="rv-grid" ref={gridRef} style={{ "--rv-w": `${thumb}px` } as CSSProperties}>
              {items.map((it, i) => {
                const st = statusView(it);
                const picked = sel.has(it.id);
                return (
                  <div
                    key={it.id}
                    className={`rv-card${picked ? " picked" : ""}${i === cur ? " cur" : ""}`}
                    onClick={(e) => clickCard(e, i)}
                    onDoubleClick={() => toSingle(it.id)}
                  >
                    <span className="rv-ck">{picked ? <CheckOutlined /> : null}</span>
                    <span className="rv-st">
                      {it.rejectedBefore && !it.rejectReason && <Tag color="warning">{t("打回过")}</Tag>}
                      <Tag color={st.color}>{st.text}</Tag>
                    </span>
                    <Shot src={thumbUrl(project.path, it.id)} item={it} colors={colors} names={names} />
                    <div className="rv-meta">
                      <span className="mono" title={it.name}>
                        {it.name}
                      </span>
                      <span>{it.member ?? ""}</span>
                    </div>
                    <div className="rv-meta2">
                      <ClassSummary item={it} colors={colors} names={names} max={3} />
                      <span className="spacer" />
                      <span>{timeText(it.submittedAt)}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <Empty
              style={{ padding: 48 }}
              description={q.statuses.length === 1 && q.statuses[0] === "labeled" ? t("没有待审核的图，这个范围都审完了") : t("没有符合条件的图")}
            />
          )}
        </Spin>
        <div className="rv-foot">
          <div className="rv-keys">
            <span>
              <kbd>←↑→↓</kbd> {t("移动")}
            </span>
            <span>
              <kbd>X</kbd> {t("勾选")}
            </span>
            <span>
              <kbd>{t("空格")}</kbd> {t("放大看")}
            </span>
            <span>
              <kbd>A</kbd> {t("通过")}
            </span>
            <span>
              <kbd>R</kbd> {t("打回")}
            </span>
            <span>
              <kbd>E</kbd> {t("去标注页修改")}
            </span>
          </div>
          <div className="spacer" />
          <Pagination
            size="small"
            current={page}
            pageSize={pageSize}
            total={total}
            showSizeChanger
            pageSizeOptions={[24, 48, 96]}
            onChange={(p, ps) => {
              setPage(ps !== pageSize ? 1 : p);
              setPageSize(ps);
              setCur(0);
              lastClick.current = null;
            }}
          />
        </div>
      </>
    );
  }

  const qItem = quick != null ? items[quick] : null;
  const sampleMembers = members.filter((m) => m.pending > 0);
  const doneTotal = sample ? sample.ok + sample.bad : 0;
  const okRate = sample && doneTotal ? Math.round((sample.ok / doneTotal) * 100) : 0;

  return (
    <div className="rv">
      {head}
      {body}

      <Modal
        open={!!qItem}
        width={960}
        footer={null}
        onCancel={() => setQuick(null)}
        title={qItem ? `${qItem.name} · ${qItem.member ?? ""}` : ""}
        destroyOnHidden
      >
        {qItem && (
          <>
            <Shot src={imageUrl(project.path, qItem.id)} item={qItem} colors={colors} names={names} big />
            <div className="rv-quick-foot">
              <span className="rv-keys">
                <span>
                  <kbd>{t("空格")}</kbd> / <kbd>Esc</kbd> {t("关闭")}
                </span>
                <span>
                  <kbd>←</kbd> <kbd>→</kbd> {t("上一张 / 下一张")}
                </span>
              </span>
              <div className="spacer" />
              <Space>
                <Button danger onClick={() => openReject([qItem.id])}>
                  {t("打回")}
                </Button>
                <Button
                  type="primary"
                  disabled={qItem.status !== "labeled"}
                  onClick={() => {
                    setQuick(null);
                    judge(qItem.id, true, "");
                  }}
                >
                  {t("通过")}
                </Button>
              </Space>
            </div>
          </>
        )}
      </Modal>

      <Modal
        title={rejectIds && rejectIds.length > 1 ? t("打回 {n} 张图", { n: rejectIds.length }) : t("打回这张图")}
        open={rejectIds != null}
        onCancel={() => setRejectIds(null)}
        onOk={doReject}
        okText={t("打回")}
        okButtonProps={{ danger: true }}
        cancelText={t("取消")}
        width={440}
        destroyOnHidden
      >
        <div className="dim" style={{ fontSize: 12, margin: "8px 0" }}>
          {t("写清楚哪里要改，标注员会在图上方看到。")}
          {rejectIds && rejectIds.length > 1 ? ` ${t("这几张用同一个原因，各自回到原来标注员的任务里。")}` : ""}
        </div>
        <Space wrap style={{ marginBottom: 8 }}>
          {rejectPresets().map((r) => (
            <Tag.CheckableTag
              key={r}
              checked={reason.split("；").includes(r)}
              onChange={(on) =>
                setReason((s) => {
                  const parts = s.split("；").filter((x) => x && x !== r);
                  return (on ? [...parts, r] : parts).join("；");
                })
              }
            >
              {r}
            </Tag.CheckableTag>
          ))}
        </Space>
        <Input.TextArea
          autoFocus
          autoSize={{ minRows: 2, maxRows: 5 }}
          value={reason}
          placeholder={t("比如：左下角还有两个安全帽没标")}
          onChange={(e) => setReason(e.target.value)}
        />
      </Modal>

      <Modal
        title={t("抽检")}
        open={sampleAsk != null}
        onCancel={() => setSampleAsk(null)}
        onOk={startSample}
        okText={t("开始抽检")}
        okButtonProps={{ disabled: sampleAsk?.member == null }}
        cancelText={t("取消")}
        width={440}
      >
        <div className="dim" style={{ margin: "8px 0 12px" }}>
          {t("成员的图很多时不用每张都看：随机抽一部分逐张审，看合格率再决定其余的图全部通过，还是打回让他自查。")}
        </div>
        {sampleMembers.length ? (
          <Space direction="vertical" style={{ width: "100%" }}>
            <Select
              style={{ width: "100%" }}
              value={sampleAsk?.member ?? undefined}
              placeholder={t("选成员")}
              onChange={(v) => setSampleAsk((s) => (s ? { ...s, member: v } : s))}
              options={sampleMembers.map((m) => ({ value: m.memberId, label: t("{name}（待审核 {n} 张）", { name: m.name, n: m.pending }) }))}
            />
            <Segmented
              value={sampleAsk?.ratio ?? 0.2}
              onChange={(v) => setSampleAsk((s) => (s ? { ...s, ratio: Number(v) } : s))}
              options={[
                { label: t("抽 10%"), value: 0.1 },
                { label: t("抽 20%"), value: 0.2 },
                { label: t("抽 50%"), value: 0.5 },
              ]}
            />
          </Space>
        ) : (
          <div>{t("现在没有待审核的图。")}</div>
        )}
      </Modal>

      <Modal
        title={sample ? t("「{name}」的抽检结果", { name: sample.name }) : t("抽检结果")}
        open={sampleDone != null}
        onCancel={() => closeSample(null)}
        footer={
          <Space wrap>
            <Button onClick={() => closeSample(null)}>{t("先不处理")}</Button>
            <Button danger disabled={!sampleDone?.rest.length} onClick={() => closeSample("reject")}>
              {t("其余 {n} 张全部打回", { n: sampleDone?.rest.length ?? 0 })}
            </Button>
            <Button type="primary" disabled={!sampleDone?.rest.length} onClick={() => closeSample("pass")}>
              {t("其余 {n} 张全部通过", { n: sampleDone?.rest.length ?? 0 })}
            </Button>
          </Space>
        }
      >
        <div className="rv-stat">
          <div>
            <b>{doneTotal}</b>
            <span>{t("已抽检")}</span>
          </div>
          <div>
            <b className={okRate >= 90 ? "ok" : "bad"}>{doneTotal ? `${okRate}%` : "—"}</b>
            <span>{t("合格率")}</span>
          </div>
          <div>
            <b>{sampleDone?.rest.length ?? 0}</b>
            <span>{t("其余待审核")}</span>
          </div>
        </div>
        <div className="dim" style={{ marginTop: 12 }}>
          {!doneTotal
            ? t("还没审任何一张。")
            : okRate >= 90
              ? t("合格率不错，可以把其余的图一起通过。")
              : t("合格率偏低，建议把其余的图打回，让他自查后重新提交。")}
        </div>
      </Modal>
    </div>
  );
}
