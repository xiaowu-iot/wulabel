import { useCallback, useEffect, useRef, useState } from "react";
import { App, Button, Checkbox, Dropdown, Empty, Modal, Pagination, Progress, Segmented, Tag, Tooltip } from "antd";
import {
  CheckSquareOutlined,
  ClearOutlined,
  DatabaseOutlined,
  DownOutlined,
  ExportOutlined,
  FileImageOutlined,
  FolderAddOutlined,
  RollbackOutlined,
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
  LastImport,
  ProjectInfo,
  thumbUrl,
} from "../api";
import { prefs, useApp } from "../store";
import ImageViewer from "../components/ImageViewer";
import ExportModal from "../components/ExportModal";

const PAGE_SIZE = 120;
const IMAGE_EXTS = ["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"];

const STATUS: Record<ImageStatus, { text: string; color?: string }> = {
  todo: { text: "未标注" },
  labeled: { text: "已标注", color: "processing" },
  reviewed: { text: "已审核", color: "success" },
  skipped: { text: "已跳过", color: "default" },
};

const FILTERS = [
  { label: "全部", value: "all" },
  { label: "未标注", value: "todo" },
  { label: "已标注", value: "labeled" },
  { label: "已审核", value: "reviewed" },
];

export default function Data({ project }: { project: ProjectInfo }) {
  const { message, modal } = App.useApp();
  const refreshStats = useApp((s) => s.refreshStats);
  const annotateImage = useApp((s) => s.annotateImage);
  const [filter, setFilter] = useState("all");
  const [page, setPage] = useState(1);
  const [items, setItems] = useState<ImageItem[]>([]);
  const [total, setTotal] = useState(0);
  const [viewIndex, setViewIndex] = useState<number | null>(null);
  const [copyMode, setCopyMode] = useState(prefs.get("copyMode", "copy"));
  const [progress, setProgress] = useState<ImportProgress | null>(null);
  const [dragging, setDragging] = useState(false);
  const importing = useRef(false);
  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [last, setLast] = useState<LastImport | null>(null);
  const [showExport, setShowExport] = useState(false);

  const load = useCallback(async () => {
    try {
      const r = await api.listImages(filter, (page - 1) * PAGE_SIZE, PAGE_SIZE);
      // 删除后当前页可能空了，退回上一页
      if (r.items.length === 0 && page > 1) {
        setPage(page - 1);
        return;
      }
      setItems(r.items);
      setTotal(r.total);
      setLast(await api.lastImport());
    } catch (e) {
      message.error(errText(e));
    }
  }, [filter, page, message]);

  const afterDelete = async (n: number) => {
    message.success(`已删除 ${n} 张图片`);
    setSelected(new Set());
    setSelecting(false);
    await load();
    await refreshStats();
  };

  const confirmDelete = (title: string, scope: "ids" | "last" | "all", ids?: number[]) => {
    modal.confirm({
      title,
      content: "图片的标注会一起删除，无法恢复。复制进项目的图片副本会删掉，你电脑上的原图不受影响。",
      okText: "删除",
      okButtonProps: { danger: true },
      cancelText: "取消",
      onOk: async () => {
        try {
          await afterDelete(await api.deleteImages(scope, ids));
        } catch (e) {
          message.error(errText(e));
        }
      },
    });
  };

  const toggle = (id: number) => {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelected(next);
  };

  useEffect(() => {
    load();
  }, [load]);

  const showResult = useCallback(
    (r: ImportResult) => {
      const text = `新增 ${r.added} 张` + (r.duplicated ? `，跳过重复 ${r.duplicated} 张` : "");
      if (r.failed.length === 0) {
        message.success(text);
        return;
      }
      modal.warning({
        title: `${text}，${r.failed.length} 张读取失败`,
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
    [message, modal],
  );

  const runImport = useCallback(
    async (paths: string[]) => {
      if (paths.length === 0 || importing.current) return;
      importing.current = true;
      setProgress({ done: 0, total: 0 });
      const unlisten = await listen<ImportProgress>("import-progress", (e) => setProgress(e.payload));
      try {
        const r = await api.importImages(paths, prefs.get("copyMode", "copy") === "copy");
        showResult(r);
        await load();
        await refreshStats();
      } catch (e) {
        message.error(errText(e));
      } finally {
        unlisten();
        setProgress(null);
        importing.current = false;
      }
    },
    [load, refreshStats, showResult, message],
  );

  // 从资源管理器拖文件或文件夹进来导入
  useEffect(() => {
    const p = getCurrentWebview().onDragDropEvent((e) => {
      const t = e.payload.type;
      if (t === "enter" || t === "over") setDragging(true);
      else if (t === "leave") setDragging(false);
      else if (t === "drop") {
        setDragging(false);
        runImport(e.payload.paths);
      }
    });
    return () => {
      p.then((un) => un());
    };
  }, [runImport]);

  const pickDataset = async () => {
    const dir = await open({ directory: true, title: "选择数据集文件夹（YOLO / COCO / VOC）" });
    if (typeof dir !== "string" || importing.current) return;
    importing.current = true;
    setProgress({ done: 0, total: 0 });
    const unlisten = await listen<ImportProgress>("import-progress", (e) => setProgress(e.payload));
    try {
      const r = await api.importDataset(dir);
      await load();
      await refreshStats();
      modal.success({
        title: `已导入 ${r.format} 数据集`,
        width: 520,
        content: (
          <div>
            <div>
              {r.images} 张图片：新增 {r.added}，重复 {r.duplicated}；导入 {r.boxes} 个框
            </div>
            {r.labelsAdded.length > 0 && <div className="muted">新增类别：{r.labelsAdded.join("、")}</div>}
            {r.keptExisting > 0 && <div className="muted">{r.keptExisting} 张重复图片已有标注，保留原标注</div>}
            {r.failed.length > 0 && <div style={{ color: "var(--warn)" }}>{r.failed.length} 张读取失败</div>}
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
      title: "选择图片",
      filters: [{ name: "图片", extensions: IMAGE_EXTS }],
    });
    if (r) runImport(Array.isArray(r) ? r : [r]);
  };

  const pickFolder = async () => {
    const r = await open({ directory: true, multiple: true, title: "选择图片文件夹（含子文件夹）" });
    if (r) runImport(Array.isArray(r) ? r : [r]);
  };

  const s = project.stats;
  const pct = s.total ? Math.round(((s.labeled + s.reviewed) / s.total) * 100) : 0;

  return (
    <div className="page">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 14 }}>
        <div>
          <h1>数据</h1>
          <p className="desc" style={{ margin: 0 }}>
            {s.total} 张图片 · 已标注 {s.labeled + s.reviewed} · 已审核 {s.reviewed}
          </p>
        </div>
        <div className="row">
          <Tooltip title="复制：图片拷进项目文件夹，项目可整体拷走；引用：只记录原路径，不占额外空间">
            <Segmented
              value={copyMode}
              onChange={(v) => {
                setCopyMode(v as string);
                prefs.set("copyMode", v as string);
              }}
              options={[
                { label: "复制到项目", value: "copy" },
                { label: "只引用原路径", value: "ref" },
              ]}
            />
          </Tooltip>
          <Button type="primary" icon={<FolderAddOutlined />} onClick={pickFolder}>
            导入文件夹
          </Button>
          <Dropdown
            trigger={["click"]}
            menu={{
              items: [
                { key: "files", icon: <FileImageOutlined />, label: "导入图片文件" },
                { key: "dataset", icon: <DatabaseOutlined />, label: "导入带标注的数据集（YOLO / COCO / VOC）" },
              ],
              onClick: ({ key }) => (key === "files" ? pickFiles() : pickDataset()),
            }}
          >
            <Button>
              更多导入 <DownOutlined />
            </Button>
          </Dropdown>
          <Button icon={<ExportOutlined />} disabled={s.labeled + s.reviewed === 0} onClick={() => setShowExport(true)}>
            导出数据集
          </Button>
          <Dropdown
            trigger={["click"]}
            disabled={s.total === 0}
            menu={{
              items: [
                {
                  key: "last",
                  icon: <RollbackOutlined />,
                  label: last ? `撤销上次导入（${last.count} 张）` : "撤销上次导入",
                  disabled: !last,
                },
                { key: "select", icon: <CheckSquareOutlined />, label: "选择图片删除" },
                { type: "divider" },
                { key: "all", icon: <ClearOutlined />, label: `清空全部图片（${s.total} 张）`, danger: true },
              ],
              onClick: ({ key }) => {
                if (key === "last" && last) confirmDelete(`撤销上次导入，删除这 ${last.count} 张图片？`, "last");
                if (key === "select") {
                  setSelecting(true);
                  setSelected(new Set());
                }
                if (key === "all") confirmDelete(`清空项目里全部 ${s.total} 张图片？`, "all");
              },
            }}
          >
            <Button>
              管理 <DownOutlined />
            </Button>
          </Dropdown>
        </div>
      </div>

      <div className="cards" style={{ gridTemplateColumns: "repeat(4, 1fr)", marginBottom: 14 }}>
        <div className="card">
          <div className="muted">标注进度</div>
          <div className="kpi">{pct}%</div>
          <Progress percent={pct} showInfo={false} size="small" />
        </div>
        <div className="card">
          <div className="muted">未标注</div>
          <div className="kpi">{s.todo}</div>
        </div>
        <div className="card">
          <div className="muted">已标注</div>
          <div className="kpi">{s.labeled}</div>
        </div>
        <div className="card">
          <div className="muted">已审核</div>
          <div className="kpi">{s.reviewed}</div>
        </div>
      </div>

      <div className="row" style={{ marginBottom: 12, justifyContent: "space-between" }}>
        <Segmented
          value={filter}
          options={FILTERS}
          onChange={(v) => {
            setFilter(v as string);
            setPage(1);
          }}
        />
        {selecting ? (
          <div className="row" style={{ gap: 8 }}>
            <span>已选 {selected.size} 张</span>
            <Button size="small" onClick={() => setSelected(new Set([...selected, ...items.map((i) => i.id)]))}>
              全选本页
            </Button>
            <Button size="small" onClick={() => setSelected(new Set())}>
              清除选择
            </Button>
            <Button
              size="small"
              danger
              type="primary"
              disabled={selected.size === 0}
              onClick={() => confirmDelete(`删除选中的 ${selected.size} 张图片？`, "ids", [...selected])}
            >
              删除
            </Button>
            <Button size="small" onClick={() => setSelecting(false)}>
              完成
            </Button>
          </div>
        ) : (
          <span className="dim">也可以直接把图片或文件夹拖进窗口</span>
        )}
      </div>

      {items.length === 0 ? (
        <div className="card" style={{ padding: 48 }}>
          <Empty description={s.total === 0 ? "还没有图片，导入一些开始吧" : "没有符合条件的图片"}>
            {s.total === 0 && (
              <Button type="primary" onClick={pickFolder}>
                导入文件夹
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
                onClick={() => (selecting ? toggle(it.id) : setViewIndex(i))}
                title={`${it.name}  ${it.width}×${it.height}`}
              >
                <img src={thumbUrl(project.path, it.id)} loading="lazy" alt={it.name} draggable={false} />
                {selecting && <Checkbox className="pick" checked={selected.has(it.id)} />}
                <Tag className="st" color={STATUS[it.status]?.color}>
                  {STATUS[it.status]?.text ?? it.status}
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

      <Modal open={!!progress} closable={false} footer={null} maskClosable={false} title="正在导入">
        <Progress percent={progress && progress.total ? Math.floor((progress.done / progress.total) * 100) : 0} />
        <div className="muted">
          {progress && progress.total
            ? `已处理 ${progress.done} / ${progress.total} 张（计算指纹、生成缩略图）`
            : "正在查找图片…"}
        </div>
      </Modal>

      {dragging && (
        <div className="drop-mask">
          <div>
            松开鼠标导入图片
            <div className="muted" style={{ fontSize: 13, marginTop: 6 }}>
              支持文件夹，会自动查找子文件夹里的图片
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
