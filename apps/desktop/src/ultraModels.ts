import type { Task } from "./api";
import { t } from "./i18n";

/** Ultralytics 系模型（AGPL-3.0）：选中时训练引擎用 ultralytics，模型名就是官方权重文件名，第一次用自动下载。
 *  不在列表里的也能训：手动输入模型名（Ultralytics 认得的都行），或者选本地 .pt / .yaml */
interface Family {
  key: string;
  name: string;
  sizes: string[];
  /** 这个系列有哪些任务的官方权重；segment 只有部分档位时写在 segSizes */
  tasks: Task[];
  segSizes?: string[];
  file: (size: string, suffix: string) => string;
}

const SUFFIX: Partial<Record<Task, string>> = { segment: "-seg", pose: "-pose", classify: "-cls" };

const FAMILIES: Family[] = [
  { key: "yolo11", name: "YOLO11", sizes: ["n", "s", "m", "l", "x"], tasks: ["detect", "segment", "pose", "classify"], file: (s, x) => `yolo11${s}${x}.pt` },
  { key: "yolo26", name: "YOLO26", sizes: ["n", "s", "m", "l", "x"], tasks: ["detect", "segment", "pose", "classify"], file: (s, x) => `yolo26${s}${x}.pt` },
  { key: "yolov8", name: "YOLOv8", sizes: ["n", "s", "m", "l", "x"], tasks: ["detect", "segment", "pose", "classify"], file: (s, x) => `yolov8${s}${x}.pt` },
  { key: "yolo12", name: "YOLO12", sizes: ["n", "s", "m", "l", "x"], tasks: ["detect"], file: (s) => `yolo12${s}.pt` },
  { key: "yolov10", name: "YOLOv10", sizes: ["n", "s", "m", "b", "l", "x"], tasks: ["detect"], file: (s) => `yolov10${s}.pt` },
  { key: "yolov9", name: "YOLOv9", sizes: ["t", "s", "m", "c", "e"], tasks: ["detect", "segment"], segSizes: ["c", "e"], file: (s, x) => `yolov9${s}${x}.pt` },
  { key: "yolov5u", name: "YOLOv5u", sizes: ["n", "s", "m", "l", "x"], tasks: ["detect"], file: (s) => `yolov5${s}u.pt` },
  { key: "rtdetr", name: "RT-DETR", sizes: ["l", "x"], tasks: ["detect"], file: (s) => `rtdetr-${s}.pt` },
];

const SIZE_DESC: Record<string, () => string> = {
  n: () => t("最轻最快，没有独立显卡也能训"),
  t: () => t("最轻最快，没有独立显卡也能训"),
  s: () => t("速度和精度均衡，大多数项目先用它"),
  m: () => t("更准，建议 6GB 以上显存"),
  b: () => t("更准，建议 6GB 以上显存"),
  c: () => t("更准，建议 6GB 以上显存"),
  l: () => t("很准、较慢，建议 8GB 以上显存"),
  x: () => t("最准、最慢，建议 12GB 以上显存"),
  e: () => t("最准、最慢，建议 12GB 以上显存"),
};

export interface UltraModel {
  /** 官方权重文件名，如 yolo11s-seg.pt */
  file: string;
  /** 显示名，如 YOLO11-S 分割 */
  name: string;
  family: string;
  desc: string;
}

const taskWord = (task: Task) =>
  task === "segment" ? t("分割") : task === "pose" ? t("关键点") : task === "classify" || task === "multilabel" ? t("分类") : "";

/** 某个任务能用的 Ultralytics 模型，按系列分组（多标签分类不支持） */
export function ultraGroups(task: Task): { family: string; models: UltraModel[] }[] {
  if (task === "multilabel") return [];
  const suffix = SUFFIX[task] ?? "";
  return FAMILIES.filter((f) => f.tasks.includes(task)).map((f) => ({
    family: f.name,
    models: (task === "segment" && f.segSizes ? f.segSizes : f.sizes).map((s) => ({
      file: f.file(s, suffix),
      name: `${f.name}-${s.toUpperCase()}${taskWord(task) ? " " + taskWord(task) : ""}`,
      family: f.name,
      desc: SIZE_DESC[s]?.() ?? "",
    })),
  }));
}

/** 模型名（官方权重名或本地文件路径）对应的显示名：列表里的用中文名，其他显示文件名 */
export function ultraName(model: string): string {
  const base = model.split(/[\\/]/).pop() ?? model;
  for (const task of ["detect", "segment", "pose", "classify"] as Task[]) {
    for (const g of ultraGroups(task)) {
      const m = g.models.find((x) => x.file === base);
      if (m) return m.name;
    }
  }
  return base;
}

/** 按模型名猜任务（-seg 分割、-pose 关键点、-cls 分类，其他检测）；本地文件猜不出时返回 null，交给训练时检查 */
export function ultraTaskOf(model: string): Task | null {
  const base = (model.split(/[\\/]/).pop() ?? model).toLowerCase();
  const local = /[\\/]/.test(model);
  if (base.includes("-seg")) return "segment";
  if (base.includes("-pose")) return "pose";
  if (base.includes("-cls")) return "classify";
  return local ? null : "detect";
}

/** Ultralytics 训练参数说明（「更多参数」旁边的链接） */
export const ULTRA_ARGS_DOC = "https://docs.ultralytics.com/modes/train/#train-settings";
