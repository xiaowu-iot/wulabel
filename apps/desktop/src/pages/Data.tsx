import { useCallback, useEffect, useRef, useState } from "react";
import { Alert, App, Button, Checkbox, Dropdown, Empty, Input, InputNumber, Modal, Pagination, Progress, Segmented, Select, Tag, Tooltip } from "antd";
import {
  AimOutlined,
  CheckSquareOutlined,
  EyeInvisibleOutlined,
  ClearOutlined,
  CloudDownloadOutlined,
  HistoryOutlined,
  DatabaseOutlined,
  DownOutlined,
  ExportOutlined,
  FileImageOutlined,
  FolderAddOutlined,
  LinkOutlined,
  RollbackOutlined,
  SwapOutlined,
  TeamOutlined,
  VideoCameraOutlined,
} from "@ant-design/icons";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import {
  api,
  errText,
  ImageItem,
  ImageStatus,
  ImportProgress,
  ImportResult,
  isClsTask,
  Label,
  LastImport,
  ProjectInfo,
  StreamFrames,
  StreamInfo,
  thumbUrl,
  VideoProgress,
} from "../api";
import { prefs, useApp, useReview } from "../store";
import { statusTag } from "../status";
import { t } from "../i18n";
import { xuntongEnabled } from "../features";
import ImageViewer from "../components/ImageViewer";
import ExportModal from "../components/ExportModal";
import VersionsModal from "../components/VersionsModal";
import { PickModal, QualityModal } from "../components/DataTools";
import CollabModal from "../components/CollabModal";
import ConvertModal from "../components/ConvertModal";
import { XuntongPullModal } from "../components/Xuntong";

const PAGE_SIZE = 120;
const IMAGE_EXTS = ["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"];
const VIDEO_EXTS = ["mp4", "mov", "mkv", "avi", "webm", "flv", "wmv", "m4v", "ts"];

const FILTERS = [
  { label: t("全部"), value: "all" },
  { label: t("未标注"), value: "todo" },
  // 只在团队协作（审核流程）时用：标注员提交的待审核，审核员通过的已通过
  { label: t("待审核"), value: "labeled" },
  { label: t("已通过"), value: "reviewed" },
  { label: t("已跳过"), value: "skipped" },
];

/** 一个人标注（不用审核）时：已审核算进已标注 */
const SOLO_FILTERS = [
  { label: t("全部"), value: "all" },
  { label: t("未标注"), value: "todo" },
  { label: t("已标注"), value: "done" },
  { label: t("已跳过"), value: "skipped" },
];

type SplitTo = "train" | "val" | "test";
const SPLIT_NAMES: Record<SplitTo, string> = { train: t("训练集"), val: t("验证集"), test: t("测试集") };

/** 按扩展名认视频（拖进窗口时分流：视频去抽帧，图片和文件夹走普通导入） */
const isVideo = (p: string) => {
  const name = p.split(/[\\/]/).pop() ?? "";
  const i = name.lastIndexOf(".");
  return i > 0 && VIDEO_EXTS.includes(name.slice(i + 1).toLowerCase());
};

export default function Data({ project }: { project: ProjectInfo }) {
  const review = useReview();
  const { message, modal } = App.useApp();
  const refreshStats = useApp((s) => s.refreshStats);
  const annotateImage = useApp((s) => s.annotateImage);
  const annotateQueue = useApp((s) => s.annotateQueue);
  const setAppPage = useApp((s) => s.setPage);
  const [filter, setFilter] = useState("all");
  // 按类别筛：类别 id；-1 一个标注都没有；-2 有预标注还没确认
  const [labelFilter, setLabelFilter] = useState<number | null>(null);
  const [labelList, setLabelList] = useState<Label[]>([]);
  const [page, setPage] = useState(1);
  const [items, setItems] = useState<ImageItem[]>([]);
  const [total, setTotal] = useState(0);
  const [viewIndex, setViewIndex] = useState<number | null>(null);
  const [copyMode, setCopyMode] = useState(prefs.get("copyMode", "copy"));
  const [progress, setProgress] = useState<ImportProgress | null>(null);
  // 导入图片时可以停止：off 不能停 / on 能停 / stopping 已点停止
  const [importStop, setImportStop] = useState<"off" | "on" | "stopping">("off");
  const [dragging, setDragging] = useState(false);
  const importing = useRef(false);
  // 这次导入是否复制进项目（进度文字用）
  const importCopy = useRef(true);
  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  // Shift 连选的起点
  const anchor = useRef<number | null>(null);
  const [batchBusy, setBatchBusy] = useState(false);
  const [last, setLast] = useState<LastImport | null>(null);
  const [showExport, setShowExport] = useState(false);
  const [showVersions, setShowVersions] = useState(false);
  const [showQuality, setShowQuality] = useState(false);
  const [showPick, setShowPick] = useState(false);
  const [showCollab, setShowCollab] = useState(false);
  const [showConvert, setShowConvert] = useState(false);
  const [videoPaths, setVideoPaths] = useState<string[]>([]);
  const [videoEvery, setVideoEvery] = useState(Number(prefs.get("videoEvery", "1")));
  const [videoMax, setVideoMax] = useState(Number(prefs.get("videoMax", "0")));
  const [videoProgress, setVideoProgress] = useState<VideoProgress | null>(null);
  const [streamOpen, setStreamOpen] = useState(false);
  const [xtOpen, setXtOpen] = useState(false);
  const [streamUrl, setStreamUrl] = useState(prefs.get("streamUrl", ""));
  const [streamInfo, setStreamInfo] = useState<StreamInfo | null>(null);
  const [probing, setProbing] = useState(false);
  const [streamEvery, setStreamEvery] = useState(Number(prefs.get("streamEvery", "1")));
  const [streamMax, setStreamMax] = useState(Number(prefs.get("streamMax", "200")));
  const [streamMinutes, setStreamMinutes] = useState(Number(prefs.get("streamMinutes", "5")));
  const [streamDedup, setStreamDedup] = useState(prefs.get("streamDedup", "0") === "1");
  const [streamRunning, setStreamRunning] = useState(false);
  const [streamMin, setStreamMin] = useState(false);
  const [streamIds, setStreamIds] = useState<number[]>([]); // 最近入库的帧（最新的在后）
  const [streamSaved, setStreamSaved] = useState(0);
  const [streamDir, setStreamDir] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const r = await api.listImages(filter, (page - 1) * PAGE_SIZE, PAGE_SIZE, labelFilter);
      // 删除后当前页可能空了，退回上一页
      if (r.items.length === 0 && page > 1) {
        setPage(page - 1);
        return;
      }
      setItems(r.items);
      setTotal(r.total);
      setLast(await api.lastImport());
      setLabelList(await api.listLabels());
    } catch (e) {
      message.error(errText(e));
    }
  }, [filter, page, labelFilter, message]);
  // 采集过程中的事件回调里总是调用最新的 load（翻页、筛选变了也对）
  const loadRef = useRef(load);
  loadRef.current = load;

  const afterDelete = async (n: number) => {
    message.success(t("已删除 {n} 张图片", { n }));
    setSelected(new Set());
    setSelecting(false);
    await load();
    await refreshStats();
  };

  const confirmDelete = (title: string, scope: "ids" | "last" | "all", ids?: number[]) => {
    const dlg = modal.confirm({
      title,
      content: t("图片的标注会一起删除，无法恢复。复制进项目的图片副本会删掉，你电脑上的原图不受影响。"),
      okText: t("删除"),
      okButtonProps: { danger: true },
      cancelText: t("取消"),
      onOk: async () => {
        // 删几万张时文件要删一会儿：确认框里显示进度
        const un = await listen<ImportProgress>("delete-progress", (e) => {
          const { done, total } = e.payload;
          if (!total) return;
          dlg.update({
            content: (
              <>
                <Progress size="small" percent={Math.floor((done / total) * 100)} />
                <div className="muted">{t("正在删除图片文件 {done} / {total} 张", { done, total })}</div>
              </>
            ),
          });
        });
        try {
          await afterDelete(await api.deleteImages(scope, ids));
        } catch (e) {
          message.error(errText(e));
        } finally {
          un();
        }
      },
    });
  };

  /** 点一下选 / 取消；按住 Shift 点，把上次点的到这张之间的都选上 */
  const toggle = (id: number, shift: boolean) => {
    const next = new Set(selected);
    const from = anchor.current == null ? -1 : items.findIndex((x) => x.id === anchor.current);
    const to = items.findIndex((x) => x.id === id);
    if (shift && from >= 0 && to >= 0) {
      for (let k = Math.min(from, to); k <= Math.max(from, to); k++) next.add(items[k].id);
    } else if (next.has(id)) next.delete(id);
    else next.add(id);
    anchor.current = id;
    setSelected(next);
  };

  // 换了筛选就清空选择，免得删掉看不见的图；翻页保留（「选中全部 N 张」后可以翻页确认），
  // 删除等操作的确认框里会写明一共多少张
  useEffect(() => {
    setSelected(new Set());
  }, [filter, labelFilter]);
  // Shift 连选只在同一页里
  useEffect(() => {
    anchor.current = null;
  }, [filter, page]);

  /** 选中当前筛选下的全部图（包括其他页） */
  const selectAll = async () => {
    try {
      setSelected(new Set(await api.listImageIds(filter === "all" ? null : filter, labelFilter)));
    } catch (e) {
      message.error(errText(e));
    }
  };

  /** 批量操作做完：清空选择、刷新 */
  const afterBatch = async () => {
    setSelected(new Set());
    anchor.current = null;
    await load();
    await refreshStats();
  };

  const batchStatus = async (status: ImageStatus) => {
    setBatchBusy(true);
    try {
      const n = await api.setStatusMany([...selected], status);
      message.success(status === "skipped" ? t("已把 {n} 张设为跳过", { n }) : t("已恢复 {n} 张（有标注的算已标注）", { n }));
      await afterBatch();
    } catch (e) {
      message.error(errText(e));
    } finally {
      setBatchBusy(false);
    }
  };

  const batchSplit = async (split: SplitTo) => {
    setBatchBusy(true);
    try {
      await api.setSplit([...selected], split);
      message.success(t("已移到{name}，没标注的图不参与划分", { name: SPLIT_NAMES[split] }));
      await afterBatch();
    } catch (e) {
      message.error(errText(e));
    } finally {
      setBatchBusy(false);
    }
  };

  const switchFilter = (v: string) => {
    setFilter(v);
    setPage(1);
  };

  /** 选中的图的原图复制到一个文件夹（比如把没有标注的图拷出来发给别人标） */
  const exportSelected = async () => {
    const dir = await open({ directory: true, title: t("选择把图片导出到哪个文件夹") });
    if (typeof dir !== "string") return;
    setBatchBusy(true);
    try {
      const [n, failed] = await api.exportImages([...selected].sort((a, b) => a - b), dir);
      if (failed.length) {
        modal.warning({
          title: t("导出了 {n} 张，{f} 张复制失败", { n, f: failed.length }),
          width: 640,
          content: <div style={{ maxHeight: 300, overflow: "auto", fontSize: 12 }}>{failed.map((f) => <div key={f}>{f}</div>)}</div>,
        });
      } else {
        message.success({
          content: (
            <span>
              {t("已导出 {n} 张图片", { n })} <a onClick={() => api.revealPath(dir).catch(() => {})}>{t("打开文件夹")}</a>
            </span>
          ),
          duration: 6,
        });
      }
    } catch (e) {
      message.error(errText(e));
    } finally {
      setBatchBusy(false);
    }
  };

  /** 按类别筛出来的这批图送去标注页，一类一类核对 */
  const annotateFiltered = async () => {
    try {
      const ids = await api.listImageIds(filter === "all" ? null : filter, labelFilter);
      const name =
        labelFilter === -1
          ? t("没有标注的图")
          : labelFilter === -2
            ? t("有预标注还没确认的图")
            : labelList.find((l) => l.id === labelFilter)?.name ?? "";
      annotateQueue(ids, name);
    } catch (e) {
      message.error(errText(e));
    }
  };

  useEffect(() => {
    load();
  }, [load]);

  const showResult = useCallback(
    (r: ImportResult) => {
      const base0 = r.duplicated
        ? t("新增 {added} 张，跳过重复 {dup} 张", { added: r.added, dup: r.duplicated })
        : t("新增 {added} 张", { added: r.added });
      // 批量识别的截图：用的是里面的原图，识别结果成了预标注
      const base = r.prelabeled ? base0 + t("（{n} 张批量识别截图已换成原图，识别结果作为预标注）", { n: r.prelabeled }) : base0;
      const text = r.stopped ? t("已停止，{text}，还有 {n} 张没导入", { text: base, n: r.stopped }) : base;
      if (r.failed.length === 0) {
        // 导入了新图，顺手给个去标注的入口
        message.success({
          duration: r.added > 0 ? 6 : 3,
          content: (
            <span>
              {text}
              {r.added > 0 && (
                <>
                  {" "}
                  <a onClick={() => setAppPage("annotate")}>{t("开始标注")}</a>
                </>
              )}
            </span>
          ),
        });
        return;
      }
      modal.warning({
        title: t("{text}，{n} 张读取失败", { text, n: r.failed.length }),
        width: 640,
        content: (
          <div style={{ maxHeight: 300, overflow: "auto", fontSize: 12 }}>
            {r.failed.map((f) => (
              <div key={f}>{f}</div>
            ))}
          </div>
        ),
      });
    },
    [message, modal, setAppPage],
  );

  const runImport = useCallback(
    async (paths: string[]) => {
      if (paths.length === 0 || importing.current) return;
      importing.current = true;
      importCopy.current = prefs.get("copyMode", "copy") === "copy";
      setProgress({ done: 0, total: 0 });
      setImportStop("on");
      const unlisten = await listen<ImportProgress>("import-progress", (e) => setProgress(e.payload));
      try {
        const r = await api.importImages(paths, importCopy.current);
        showResult(r);
        await load();
        await refreshStats();
      } catch (e) {
        message.error(errText(e));
      } finally {
        unlisten();
        setProgress(null);
        setImportStop("off");
        importing.current = false;
      }
    },
    [load, refreshStats, showResult, message],
  );

  const stopImport = () => {
    setImportStop("stopping");
    api.stopImport().catch((e) => message.error(errText(e)));
  };

  // 导入进度分两段（total 是 2 × 张数）：先算指纹、生成缩略图，再复制并写入项目
  const importText = (p: ImportProgress | null) => {
    if (!p || !p.total) return t("正在查找图片…");
    const half = p.total / 2;
    if (p.done <= half) return t("已处理 {done} / {total} 张（计算指纹、生成缩略图）", { done: p.done, total: half });
    const n = p.done - half;
    return importCopy.current
      ? t("复制并写入 {done} / {total} 张", { done: n, total: half })
      : t("写入项目 {done} / {total} 张", { done: n, total: half });
  };

  // 拖进来的东西分流：视频去抽帧设置，图片和文件夹走普通导入；混在一起时先导图片再弹抽帧
  const handleDrop = useCallback(
    async (paths: string[]) => {
      if (importing.current) return;
      const videos = paths.filter(isVideo);
      const rest = paths.filter((p) => !isVideo(p));
      if (rest.length) await runImport(rest);
      if (videos.length) setVideoPaths(videos);
    },
    [runImport],
  );

  // 从资源管理器拖文件或文件夹进来导入
  useEffect(() => {
    const p = getCurrentWebview().onDragDropEvent((e) => {
      const ty = e.payload.type;
      if (ty === "enter" || ty === "over") setDragging(true);
      else if (ty === "leave") setDragging(false);
      else if (ty === "drop") {
        setDragging(false);
        handleDrop(e.payload.paths);
      }
    });
    return () => {
      p.then((un) => un());
    };
  }, [handleDrop]);

  const pickDataset = async () => {
    const dir = await open({
      directory: true,
      title: isClsTask(project.task) ? t("选择分类数据集文件夹（每个类别一个子文件夹）") : t("选择数据集文件夹（YOLO / COCO / VOC）"),
    });
    if (typeof dir !== "string" || importing.current) return;
    importing.current = true;
    importCopy.current = true; // 导入数据集总是复制
    setProgress({ done: 0, total: 0 });
    const unlisten = await listen<ImportProgress>("import-progress", (e) => setProgress(e.payload));
    try {
      const r = await api.importDataset(dir);
      await load();
      await refreshStats();
      modal.success({
        title: t("已导入 {format} 数据集", { format: r.format }),
        width: 520,
        content: (
          <div>
            <div>
              {t("{images} 张图片：新增 {added}，重复 {dup}；导入 {boxes} 个标注（{format}）", {
                images: r.images,
                added: r.added,
                dup: r.duplicated,
                boxes: r.boxes,
                format: r.format,
              })}
            </div>
            {r.labelsAdded.length > 0 && <div className="muted">{t("新增类别：{list}", { list: r.labelsAdded.join(t("、")) })}</div>}
            {r.keptExisting > 0 && <div className="muted">{t("{n} 张重复图片已有标注，保留原标注", { n: r.keptExisting })}</div>}
            {r.failed.length > 0 && <div style={{ color: "var(--warn)" }}>{t("{n} 张读取失败", { n: r.failed.length })}</div>}
          </div>
        ),
      });
    } catch (e) {
      message.error(errText(e));
    } finally {
      unlisten();
      setProgress(null);
      importing.current = false;
    }
  };

  const pickFiles = async () => {
    const r = await open({
      multiple: true,
      title: t("选择图片"),
      filters: [{ name: t("图片"), extensions: IMAGE_EXTS }],
    });
    if (r) runImport(Array.isArray(r) ? r : [r]);
  };

  const pickFolder = async () => {
    const r = await open({ directory: true, multiple: true, title: t("选择图片文件夹（含子文件夹）") });
    if (r) runImport(Array.isArray(r) ? r : [r]);
  };

  const pickVideos = async () => {
    const r = await open({
      multiple: true,
      title: t("选择视频（会抽成图片导入）"),
      filters: [{ name: t("视频"), extensions: VIDEO_EXTS }],
    });
    if (!r) return;
    setVideoPaths(Array.isArray(r) ? r : [r]);
  };

  const startExtract = async () => {
    if (videoPaths.length === 0 || importing.current) return;
    const paths = videoPaths;
    const every = videoEvery;
    const max = videoMax;
    setVideoPaths([]);
    prefs.set("videoEvery", String(every));
    prefs.set("videoMax", String(max));
    importing.current = true;
    setVideoProgress({ stage: "extract", done: 0, total: 0, speed: 0, detail: t("准备中…") });
    const unlisten = await listen<VideoProgress>("video-progress", (e) => setVideoProgress(e.payload));
    try {
      const r = await api.extractVideo(paths, every, max);
      await load();
      await refreshStats();
      const text = r.stopped
        ? t("已停止，抽了 {frames} 帧：新增 {added} 张", { frames: r.frames, added: r.added })
        : r.duplicated
        ? t("从 {videos} 个视频抽出 {frames} 帧：新增 {added} 张，跳过重复 {dup} 张", {
            videos: r.videos,
            frames: r.frames,
            added: r.added,
            dup: r.duplicated,
          })
        : t("从 {videos} 个视频抽出 {frames} 帧：新增 {added} 张", { videos: r.videos, frames: r.frames, added: r.added });
      if (r.failed.length === 0) {
        message.success(text);
      } else {
        modal.warning({
          title: t("{text}，{n} 个处理失败", { text, n: r.failed.length }),
          width: 640,
          content: (
            <div style={{ maxHeight: 300, overflow: "auto", fontSize: 12 }}>
              {r.failed.map((f) => (
                <div key={f}>{f}</div>
              ))}
            </div>
          ),
        });
      }
    } catch (e) {
      message.error(errText(e));
    } finally {
      unlisten();
      setVideoProgress(null);
      importing.current = false;
    }
  };

  const fmtSecs = (d: number | null) =>
    d == null ? "—" : d < 60 ? t("{n} 秒", { n: Math.round(d) }) : t("{m} 分 {s} 秒", { m: Math.floor(d / 60), s: Math.round(d % 60) });

  const testStream = async () => {
    const url = streamUrl.trim();
    if (!url) return;
    setProbing(true);
    setStreamInfo(null);
    try {
      const info = await api.probeStream(url);
      setStreamInfo(info);
      if (info.isLive && streamMax === 0) setStreamMax(200);
    } catch (e) {
      message.error(errText(e));
    } finally {
      setProbing(false);
    }
  };

  const startStream = async () => {
    const url = streamUrl.trim();
    if (!url || importing.current) return;
    const every = streamEvery;
    const max = streamMax;
    const minutes = streamInfo && !streamInfo.isLive ? 0 : streamMinutes; // 回放不需要时长上限
    if (streamInfo?.isLive && max === 0 && minutes === 0) {
      message.warning(t("直播是无限流，请至少设一个上限：最多帧数，或最长采集时长"));
      return;
    }
    const dedup = streamDedup;
    prefs.set("streamUrl", url);
    prefs.set("streamEvery", String(every));
    prefs.set("streamMax", String(max));
    prefs.set("streamMinutes", String(minutes));
    prefs.set("streamDedup", dedup ? "1" : "0");
    setStreamOpen(false);
    setStreamMin(false);
    setStreamIds([]);
    setStreamSaved(0);
    setStreamDir(null);
    importing.current = true;
    setStreamRunning(true);
    setVideoProgress({ stage: "extract", done: 0, total: max, speed: 0, detail: t("连接中…") });
    const unlisten = await listen<VideoProgress>("video-progress", (e) => setVideoProgress(e.payload));
    // 边采边入库：弹框里显示最新的帧，后面的图片列表最多每 2 秒刷新一次
    let lastRefresh = 0;
    const unlistenFrames = await listen<StreamFrames>("stream-frames", (e) => {
      const { ids, dir } = e.payload;
      setStreamIds((prev) => [...prev, ...ids].slice(-12));
      setStreamSaved((n) => n + ids.length);
      setStreamDir(dir);
      const now = Date.now();
      if (now - lastRefresh > 2000) {
        lastRefresh = now;
        loadRef.current();
        refreshStats();
      }
    });
    try {
      const r = await api.extractStream(url, every, max, minutes, dedup);
      const text = r.duplicated
        ? t("采集完成：新增 {added} 张，跳过重复 {dup} 张", { added: r.added, dup: r.duplicated })
        : t("采集完成：新增 {added} 张", { added: r.added });
      const dir = r.dir;
      if (dir) {
        message.success({
          duration: 8,
          content: (
            <span>
              {t("{text}，保存在 {dir}", { text, dir: shortDir(dir) })} <a onClick={() => openDir(dir)}>{t("打开文件夹")}</a>
            </span>
          ),
        });
      } else {
        message.success(text);
      }
    } catch (e) {
      message.error(errText(e));
    } finally {
      unlisten();
      unlistenFrames();
      setVideoProgress(null);
      setStreamRunning(false);
      setStreamMin(false);
      importing.current = false;
      // 出错前可能已经入库了一部分，照样刷新
      await loadRef.current();
      await refreshStats();
    }
  };

  /** 绝对路径只显示项目里的那段：images/stream_xxx */
  const shortDir = (dir: string) => dir.split(/[\\/]/).slice(-2).join("/");

  const openDir = (dir: string) => {
    api.revealPath(dir).catch((e) => message.error(errText(e)));
  };

  const stopStream = async () => {
    try {
      await api.stopExtract();
    } catch (e) {
      message.error(errText(e));
    }
  };

  const s = project.stats;
  const pct = s.total ? Math.round(((s.labeled + s.reviewed) / s.total) * 100) : 0;

  return (
    <div className="page">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 14 }}>
        <div>
          <h1>{t("数据")}</h1>
          <p className="desc" style={{ margin: 0 }}>
            {review
              ? t("{total} 张图片 · 可训练 {n}（待审核 {labeled} + 已通过 {reviewed}）", { total: s.total, n: s.labeled + s.reviewed, labeled: s.labeled, reviewed: s.reviewed })
              : t("{total} 张图片 · 已标注 {n}", { total: s.total, n: s.labeled + s.reviewed })}
          </p>
        </div>
        <div className="row">
          <Tooltip title={t("复制：图片拷进项目文件夹，项目可整体拷走；引用：只记录原路径，不占额外空间")}>
            <Segmented
              value={copyMode}
              onChange={(v) => {
                setCopyMode(v as string);
                prefs.set("copyMode", v as string);
              }}
              options={[
                { label: t("复制到项目"), value: "copy" },
                { label: t("只引用原路径"), value: "ref" },
              ]}
            />
          </Tooltip>
          <Button type="primary" icon={<FolderAddOutlined />} onClick={pickFolder}>
            {t("导入文件夹")}
          </Button>
          <Dropdown
            trigger={["click"]}
            menu={{
              items: [
                { key: "files", icon: <FileImageOutlined />, label: t("导入图片文件") },
                { key: "video", icon: <VideoCameraOutlined />, label: t("导入视频（抽帧）") },
                { key: "stream", icon: <LinkOutlined />, label: t("从视频流抽帧（RTSP / 直播 / 回放）") },
                {
                  key: "dataset",
                  icon: <DatabaseOutlined />,
                  label:
                    isClsTask(project.task)
                      ? t("导入分类数据集（每个类别一个文件夹）")
                      : project.task === "pose"
                        ? t("导入带标注的数据集（YOLO-pose / COCO 关键点）")
                        : t("导入带标注的数据集（YOLO / COCO / VOC）"),
                },
                ...(isClsTask(project.task) || !xuntongEnabled() ? [] : [{ key: "xuntong", icon: <CloudDownloadOutlined />, label: t("从巡瞳 AI 拉取告警图片") }]),
              ],
              onClick: ({ key }) => {
                if (key === "files") pickFiles();
                else if (key === "video") pickVideos();
                else if (key === "stream") setStreamOpen(true);
                else if (key === "xuntong") setXtOpen(true);
                else pickDataset();
              },
            }}
          >
            <Button>
              {t("更多导入")} <DownOutlined />
            </Button>
          </Dropdown>
          <Tooltip title={s.labeled + s.reviewed === 0 ? t("还没有已标注的图片") : undefined}>
            <Button icon={<ExportOutlined />} disabled={s.labeled + s.reviewed === 0} onClick={() => setShowExport(true)}>
              {t("导出数据集")}
            </Button>
          </Tooltip>
          <Dropdown
            trigger={["click"]}
            disabled={s.total === 0}
            menu={{
              items: [
                {
                  key: "last",
                  icon: <RollbackOutlined />,
                  label: last ? t("撤销上次导入（{n} 张）", { n: last.count }) : t("撤销上次导入"),
                  disabled: !last,
                },
                { key: "versions", icon: <HistoryOutlined />, label: t("数据集版本（保存快照、对比、恢复）") },
                { key: "quality", icon: <EyeInvisibleOutlined />, label: t("图片质量筛查（模糊、过暗、过曝）") },
                { key: "pick", icon: <AimOutlined />, label: t("挑选最值得标的图（用模型）") },
                { key: "collab", icon: <TeamOutlined />, label: t("多人协作（分包、合并回来）") },
                // 检测、分割、关键点项目：按每张图标过的类别生成多标签分类项目
                ...(isClsTask(project.task) ? [] : [{ key: "convert", icon: <SwapOutlined />, label: t("转成多标签分类项目") }]),
                { key: "select", icon: <CheckSquareOutlined />, label: t("批量选择") },
                { type: "divider" },
                { key: "all", icon: <ClearOutlined />, label: t("清空全部图片（{n} 张）", { n: s.total }), danger: true },
              ],
              onClick: ({ key }) => {
                if (key === "last" && last) confirmDelete(t("撤销上次导入，删除这 {n} 张图片？", { n: last.count }), "last");
                if (key === "versions") setShowVersions(true);
                if (key === "quality") setShowQuality(true);
                if (key === "pick") setShowPick(true);
                if (key === "collab") setShowCollab(true);
                if (key === "convert") setShowConvert(true);
                if (key === "select") {
                  setSelecting(true);
                  setSelected(new Set());
                  anchor.current = null;
                }
                if (key === "all") confirmDelete(t("清空项目里全部 {n} 张图片？", { n: s.total }), "all");
              },
            }}
          >
            <Button>
              {t("管理")} <DownOutlined />
            </Button>
          </Dropdown>
        </div>
      </div>

      {s.total > 0 && s.labeled + s.reviewed === 0 && (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 12 }}
          message={t("下一步：去标注")}
          action={
            <Button size="small" type="primary" onClick={() => setAppPage("annotate")}>
              {t("开始标注")}
            </Button>
          }
        />
      )}

      <div className="cards" style={{ gridTemplateColumns: "repeat(4, 1fr)", marginBottom: 14 }}>
        <div className="card">
          <div className="muted">{t("标注进度")}</div>
          <div className="kpi">{pct}%</div>
          <Progress percent={pct} showInfo={false} size="small" />
        </div>
        {/* 点统计卡片切换到对应的筛选，再点一下回到全部 */}
        {[
          { label: t("未标注"), n: s.todo, f: "todo" },
          { label: review ? t("待审核") : t("已标注"), n: review ? s.labeled : s.labeled + s.reviewed, f: review ? "labeled" : "done" },
          review ? { label: t("已通过"), n: s.reviewed, f: "reviewed" } : { label: t("已跳过"), n: s.skipped, f: "skipped" },
        ].map((c) => (
          <div
            key={c.f}
            className="card click"
            style={{ borderColor: filter === c.f ? "var(--accent)" : undefined }}
            onClick={() => switchFilter(filter === c.f ? "all" : c.f)}
          >
            <div className="muted">{c.label}</div>
            <div className="kpi">{c.n}</div>
          </div>
        ))}
      </div>

      <div className="row" style={{ marginBottom: 12, justifyContent: "space-between" }}>
        <div className="row" style={{ gap: 8 }}>
          <Segmented value={filter} options={review ? FILTERS : SOLO_FILTERS} onChange={(v) => switchFilter(v as string)} />
          <Select
            style={{ minWidth: 160 }}
            allowClear
            showSearch
            optionFilterProp="label"
            placeholder={t("全部类别")}
            value={labelFilter ?? undefined}
            onChange={(v) => {
              setLabelFilter(v ?? null);
              setPage(1);
            }}
            options={[
              ...labelList.map((l) => ({ value: l.id, label: l.name })),
              { value: -2, label: t("有预标注还没确认的") },
              { value: -1, label: t("没有任何标注的") },
            ]}
          />
          {labelFilter != null && total > 0 && !selecting && (
            <Button size="small" type="primary" onClick={annotateFiltered}>
              {t("去标注这批（{n} 张）", { n: total })}
            </Button>
          )}
        </div>
        {selecting ? (
          <div className="row" style={{ gap: 8 }}>
            <span>{t("已选 {n} 张", { n: selected.size })}</span>
            <Button size="small" onClick={() => setSelected(new Set([...selected, ...items.map((i) => i.id)]))}>
              {t("全选本页")}
            </Button>
            {total > items.length && (
              <Button size="small" onClick={selectAll}>
                {t("选中全部 {n} 张", { n: total })}
              </Button>
            )}
            <Button size="small" onClick={() => setSelected(new Set())}>
              {t("清除选择")}
            </Button>
            <Button size="small" disabled={selected.size === 0 || batchBusy} onClick={() => batchStatus("skipped")}>
              {t("设为跳过")}
            </Button>
            <Tooltip title={t("已有标注的图会恢复成已标注")}>
              <Button size="small" disabled={selected.size === 0 || batchBusy} onClick={() => batchStatus("todo")}>
                {t("恢复未标注")}
              </Button>
            </Tooltip>
            <Button
              size="small"
              disabled={selected.size === 0}
              onClick={() => annotateQueue([...selected].sort((a, b) => a - b), t("选中的图"))}
            >
              {t("去标注")}
            </Button>
            <Tooltip title={t("把选中的图的原图复制到一个文件夹（不带标注）")}>
              <Button size="small" disabled={selected.size === 0 || batchBusy} onClick={exportSelected}>
                {t("导出图片")}
              </Button>
            </Tooltip>
            <Dropdown
              trigger={["click"]}
              disabled={selected.size === 0 || batchBusy}
              menu={{
                items: (Object.keys(SPLIT_NAMES) as SplitTo[]).map((k) => ({
                  key: k,
                  label: { train: t("移到训练集"), val: t("移到验证集"), test: t("移到测试集") }[k],
                })),
                onClick: ({ key }) => batchSplit(key as SplitTo),
              }}
            >
              <Button size="small">
                {t("划分")} <DownOutlined />
              </Button>
            </Dropdown>
            <Button
              size="small"
              danger
              type="primary"
              disabled={selected.size === 0}
              onClick={() => confirmDelete(t("删除选中的 {n} 张图片？", { n: selected.size }), "ids", [...selected])}
            >
              {t("删除")}
            </Button>
            <Button size="small" onClick={() => setSelecting(false)}>
              {t("完成")}
            </Button>
          </div>
        ) : (
          <span className="dim">{t("也可以直接把图片或文件夹拖进窗口")}</span>
        )}
      </div>
      {selecting && (
        <div className="dim" style={{ fontSize: 12, marginTop: -6, marginBottom: 10 }}>
          {t("按住 Shift 点可以连着选一段")}
        </div>
      )}

      {items.length === 0 ? (
        <div className="card" style={{ padding: 48 }}>
          <Empty description={s.total === 0 ? t("还没有图片，导入一些开始吧") : t("没有符合条件的图片")}>
            {s.total === 0 && (
              <Button type="primary" onClick={pickFolder}>
                {t("导入文件夹")}
              </Button>
            )}
          </Empty>
        </div>
      ) : (
        <>
          <div className="thumbs">
            {items.map((it, i) => (
              <div
                key={it.id}
                className={selecting && selected.has(it.id) ? "thumb picked" : "thumb"}
                onClick={(e) => (selecting ? toggle(it.id, e.shiftKey) : setViewIndex(i))}
                title={`${it.name}  ${it.width}×${it.height}`}
              >
                <img src={thumbUrl(project.path, it.id)} loading="lazy" alt={it.name} draggable={false} />
                {selecting && <Checkbox className="pick" checked={selected.has(it.id)} />}
                <Tag className="st" color={statusTag(it.status, review).color}>
                  {statusTag(it.status, review).text}
                </Tag>
                <span className="nm">{it.name}</span>
              </div>
            ))}
          </div>
          {total > PAGE_SIZE && (
            <div className="row" style={{ justifyContent: "center", marginTop: 16 }}>
              <Pagination current={page} pageSize={PAGE_SIZE} total={total} showSizeChanger={false} onChange={setPage} />
            </div>
          )}
        </>
      )}

      <ImageViewer
        projectPath={project.path}
        items={items}
        index={viewIndex}
        onIndex={setViewIndex}
        onAnnotate={(id) => {
          setViewIndex(null);
          annotateImage(id);
        }}
      />

      <ExportModal project={project} open={showExport} onClose={() => setShowExport(false)} />
      <XuntongPullModal
        open={xtOpen}
        onClose={() => setXtOpen(false)}
        onDone={() => {
          load();
          refreshStats();
        }}
      />
      <QualityModal project={project} open={showQuality} onClose={() => setShowQuality(false)} onChanged={load} />
      <PickModal project={project} open={showPick} onClose={() => setShowPick(false)} />
      <ConvertModal project={project} open={showConvert} onClose={() => setShowConvert(false)} />
      <CollabModal
        project={project}
        open={showCollab}
        onClose={() => {
          setShowCollab(false);
          load();
        }}
      />
      <VersionsModal
        project={project}
        open={showVersions}
        onClose={() => {
          setShowVersions(false);
          load();
        }}
      />

      <Modal
        open={!!progress}
        closable={false}
        maskClosable={false}
        title={t("正在导入")}
        footer={
          importStop === "off" ? null : (
            <Button danger loading={importStop === "stopping"} onClick={stopImport}>
              {t("停止")}
            </Button>
          )
        }
      >
        <Progress percent={progress && progress.total ? Math.floor((progress.done / progress.total) * 100) : 0} />
        <div className="muted">{importText(progress)}</div>
      </Modal>

      <Modal
        open={videoPaths.length > 0}
        title={t("视频抽帧")}
        okText={t("开始抽帧")}
        cancelText={t("取消")}
        onOk={startExtract}
        onCancel={() => setVideoPaths([])}
      >
        <p className="muted" style={{ marginTop: 0 }}>
          {t("选了 {n} 个视频，按下面的设置抽成图片导入项目。", { n: videoPaths.length })}
        </p>
        <div className="row" style={{ gap: 8, marginBottom: 10 }}>
          <span>{t("每隔")}</span>
          <InputNumber min={0.1} max={600} step={0.5} value={videoEvery} onChange={(v) => setVideoEvery(v ?? 1)} />
          <span>{t("秒抽 1 帧")}</span>
        </div>
        <div className="row" style={{ gap: 8 }}>
          <span>{t("每个视频最多")}</span>
          <InputNumber min={0} max={100000} step={50} value={videoMax} onChange={(v) => setVideoMax(v ?? 0)} />
          <span>{t("帧（0 = 不限）")}</span>
        </div>
        <p className="dim" style={{ fontSize: 12, marginTop: 12, marginBottom: 0 }}>
          {t("首次使用会自动下载 ffmpeg（约 90MB），之后不再下载。抽出的帧会复制进项目，按内容自动去重。")}
        </p>
      </Modal>

      <Modal
        open={streamOpen}
        title={t("从视频流抽帧")}
        okText={t("开始采集")}
        cancelText={t("取消")}
        okButtonProps={{ disabled: !streamUrl.trim() }}
        onOk={startStream}
        onCancel={() => setStreamOpen(false)}
      >
        <p className="muted" style={{ marginTop: 0 }}>
          {t("支持 RTSP 摄像头 / 直播流,以及 http(s) 的回放、点播地址(mp4、m3u8、flv)。")}
        </p>
        <div className="row" style={{ gap: 8, marginBottom: 8 }}>
          <Input
            placeholder={t("rtsp://…  或  https://…")}
            value={streamUrl}
            onChange={(e) => {
              setStreamUrl(e.target.value);
              setStreamInfo(null);
            }}
            onPressEnter={testStream}
          />
          <Button loading={probing} onClick={testStream}>
            {t("测试连接")}
          </Button>
        </div>
        {streamInfo && (
          <div style={{ fontSize: 12, marginBottom: 10, color: "var(--good, #2fb67c)" }}>
            ✓ {streamInfo.width}×{streamInfo.height} · {streamInfo.codec || t("视频")}
            {streamInfo.fps ? ` · ${streamInfo.fps.toFixed(0)} fps` : ""} ·{" "}
            {streamInfo.isLive ? t("直播流（无限，需设上限）") : t("回放流 · 时长 {d}", { d: fmtSecs(streamInfo.durationSecs) })}
          </div>
        )}
        <div className="row" style={{ gap: 8, marginBottom: 10 }}>
          <span>{t("每隔")}</span>
          <InputNumber min={0.1} max={600} step={0.5} value={streamEvery} onChange={(v) => setStreamEvery(v ?? 1)} />
          <span>{t("秒抽 1 帧")}</span>
        </div>
        <div className="row" style={{ gap: 8, marginBottom: 10 }}>
          <span>{t("最多抽")}</span>
          <InputNumber min={0} max={100000} step={50} value={streamMax} onChange={(v) => setStreamMax(v ?? 0)} />
          <span>{t("帧（0 = 不限）")}</span>
        </div>
        {(!streamInfo || streamInfo.isLive) && (
          <div className="row" style={{ gap: 8, marginBottom: 10 }}>
            <span>{t("最长采集")}</span>
            <InputNumber min={0} max={1440} step={1} value={streamMinutes} onChange={(v) => setStreamMinutes(v ?? 0)} />
            <span>{t("分钟（0 = 不限，仅直播）")}</span>
          </div>
        )}
        <Checkbox checked={streamDedup} onChange={(e) => setStreamDedup(e.target.checked)}>
          {t("跳过几乎相同的帧（画面静止时少存重复图）")}
        </Checkbox>
        <p className="dim" style={{ fontSize: 12, marginTop: 12, marginBottom: 0 }}>
          {t("图片边采边入库，保存在项目的 images/stream_地址_时间/ 文件夹里，采集时就能看到。直播是无限流，务必设「最多帧」或「最长时长」；中途停止也会保留已采的图。首次使用会自动下载 ffmpeg。")}
        </p>
      </Modal>

      <Modal
        open={!!videoProgress && !streamMin}
        closable={false}
        maskClosable={false}
        title={streamRunning ? t("正在采集视频流") : t("正在处理视频")}
        footer={
          streamRunning ? (
            <div className="row" style={{ justifyContent: "flex-end", gap: 8 }}>
              <Button onClick={() => setStreamMin(true)}>{t("最小化")}</Button>
              <Button danger onClick={stopStream}>
                {t("停止采集")}
              </Button>
            </div>
          ) : videoProgress?.stage === "extract" ? (
            // 本地视频抽帧：停止后已抽的帧照常导入
            <Button danger onClick={stopStream}>
              {t("停止")}
            </Button>
          ) : null
        }
      >
        {videoProgress &&
          (() => {
            const vp = videoProgress;
            const pct = vp.total ? Math.floor((vp.done / vp.total) * 100) : 0;
            const text =
              vp.stage === "download"
                ? `${vp.detail} · ${(vp.done / 2 ** 20).toFixed(0)} / ${(vp.total / 2 ** 20).toFixed(0)} MB · ${(vp.speed / 2 ** 20).toFixed(1)} MB/s`
                : vp.stage === "extract"
                  ? streamRunning
                    ? vp.total
                      ? t("已采集 {done} 帧 / 上限 {total} · 已入库 {saved} 张", { done: vp.done, total: vp.total, saved: streamSaved })
                      : t("已采集 {done} 帧 · 已入库 {saved} 张", { done: vp.done, saved: streamSaved })
                    : vp.total
                      ? t("正在抽帧：{detail} · 已抽 {done} 帧 / 约 {total}", { detail: vp.detail, done: vp.done, total: vp.total })
                      : t("正在抽帧：{detail} · 已抽 {done} 帧", { detail: vp.detail, done: vp.done })
                  : // 导入进度的 total 是 2 × 帧数（指纹缩略图、复制写库各一半）
                    t("正在导入 {done} / {total} 帧（生成缩略图）", { done: Math.ceil(vp.done / 2), total: Math.ceil(vp.total / 2) });
            return (
              <>
                <Progress percent={pct} status="active" showInfo={vp.total > 0} />
                {streamRunning && vp.stage === "extract" && (
                  <div className="dim" style={{ fontSize: 12, wordBreak: "break-all" }}>
                    {vp.detail}
                  </div>
                )}
                <div className="muted">{text}</div>
                {streamRunning && vp.stage !== "download" && (
                  <>
                    <div className="stream-strip">
                      {streamIds.length === 0 ? (
                        <div className="dim" style={{ fontSize: 12 }}>
                          {t("等待第一帧…")}
                        </div>
                      ) : (
                        [...streamIds].reverse().map((id, i) => (
                          <img
                            key={id}
                            src={thumbUrl(project.path, id)}
                            alt=""
                            draggable={false}
                            className={i === 0 ? "newest" : undefined}
                          />
                        ))
                      )}
                    </div>
                    {streamDir && (
                      <div className="dim" style={{ fontSize: 12 }}>
                        {t("保存到项目 {dir}", { dir: shortDir(streamDir) })} <a onClick={() => openDir(streamDir)}>{t("打开文件夹")}</a>
                      </div>
                    )}
                  </>
                )}
              </>
            );
          })()}
      </Modal>

      {streamRunning && streamMin && videoProgress && (
        <div
          style={{
            position: "fixed",
            right: 20,
            bottom: 20,
            zIndex: 1000,
            background: "var(--panel, #20242b)",
            border: "1px solid var(--line)",
            borderRadius: 8,
            padding: "10px 14px",
            boxShadow: "0 4px 16px rgba(0,0,0,.35)",
            minWidth: 220,
          }}
        >
          <div className="row" style={{ gap: 10, flexWrap: "nowrap" }}>
            {streamIds.length > 0 && (
              <img
                key={streamIds[streamIds.length - 1]}
                src={thumbUrl(project.path, streamIds[streamIds.length - 1])}
                alt=""
                draggable={false}
                style={{ width: 64, height: 48, objectFit: "cover", borderRadius: 4, flex: "none" }}
              />
            )}
            <div style={{ fontSize: 13 }}>
              {videoProgress.stage === "download"
                ? t("正在下载 ffmpeg…")
                : t("采集中 · 已抽 {n} 帧", { n: videoProgress.done })}
              <div className="dim" style={{ fontSize: 12 }}>
                {t("已入库 {n} 张", { n: streamSaved })}
              </div>
            </div>
          </div>
          <div className="row" style={{ gap: 8, marginTop: 8, justifyContent: "flex-end" }}>
            <Button size="small" onClick={() => setStreamMin(false)}>
              {t("展开")}
            </Button>
            <Button size="small" danger onClick={stopStream}>
              {t("停止")}
            </Button>
          </div>
        </div>
      )}

      {dragging && (
        <div className="drop-mask">
          <div>
            {t("松开鼠标导入图片或视频")}
            <div className="muted" style={{ fontSize: 13, marginTop: 6 }}>
              {t("支持文件夹，会自动查找子文件夹里的图片")}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
