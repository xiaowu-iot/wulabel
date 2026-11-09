import { invoke } from "@tauri-apps/api/core";

export type Task = "detect" | "segment" | "classify" | "pose";

export const TASKS: { value: Task; name: string; desc: string }[] = [
  { value: "detect", name: "目标检测", desc: "用矩形框标出目标的位置和类别" },
  { value: "segment", name: "实例分割", desc: "用多边形或画笔描出目标轮廓" },
  { value: "classify", name: "图像分类", desc: "给整张图片打类别标签" },
  { value: "pose", name: "关键点", desc: "标出人体或物体的关键点和骨架" },
];

export const taskName = (t: string) => TASKS.find((x) => x.value === t)?.name ?? t;

export interface Stats {
  total: number;
  todo: number;
  labeled: number;
  reviewed: number;
  skipped: number;
}

export interface ProjectInfo {
  path: string;
  name: string;
  task: Task;
  stats: Stats;
}

export interface RecentProject {
  path: string;
  name: string;
  task: Task;
  openedAt: number;
  exists: boolean;
}

export type ImageStatus = "todo" | "labeled" | "reviewed" | "skipped";

export interface ImageItem {
  id: number;
  name: string;
  width: number;
  height: number;
  status: ImageStatus;
  split: string;
}

export interface ImagePage {
  total: number;
  items: ImageItem[];
}

export interface ImportResult {
  added: number;
  duplicated: number;
  failed: string[];
}

export interface ImportProgress {
  done: number;
  total: number;
}

export interface ExtractResult {
  videos: number;
  frames: number;
  added: number;
  duplicated: number;
  failed: string[];
  /** 视频流采集的帧保存在哪个文件夹（绝对路径）；本地视频抽帧为 null */
  dir: string | null;
}

/** 视频流边采边入库：这一批新入库的图片 id 和保存文件夹 */
export interface StreamFrames {
  ids: number[];
  dir: string;
}

export interface VideoProgress {
  stage: "download" | "extract" | "import";
  done: number;
  total: number;
  speed: number;
  detail: string;
}

export interface StreamInfo {
  width: number;
  height: number;
  fps: number;
  durationSecs: number | null;
  isLive: boolean;
  codec: string;
}

export interface LastImport {
  count: number;
  createdAt: number;
}

export interface Label {
  id: number;
  name: string;
  color: string;
  count: number;
  /** 英文提示词（文本提示自动标注用），可为空 */
  prompt?: string;
}

/** SAM 分割结果：多边形 [x1, y1, ...] 和外接框 [x, y, w, h]，原图像素坐标 */
export interface SamResult {
  points: number[];
  box: number[];
  score: number;
}

/** 巡瞳 AI 平台上可用的识别模型；code = "llm" 是大模型通用识别 */
export interface AiModel {
  code: string;
  name: string;
  description: string;
}

/** 关键点标注数据：目标框 [x, y, w, h] + 关键点 [x, y, v, ...]（v：0 没标、1 被遮挡、2 可见） */
export interface KeypointData {
  box: number[];
  kpts: number[];
}

/**
 * bbox：data = [x, y, w, h]；polygon：data = [x1, y1, x2, y2, ...]；
 * tag：图像分类整图标签，data = []；keypoints：data = KeypointData
 */
export interface Annotation {
  labelId: number | null;
  type: "bbox" | "polygon" | "tag" | "keypoints";
  data: number[] | KeypointData;
  score?: number | null;
  source: string;
}

/** 关键点模板：点名（点击顺序）+ 骨架连线（点序号从 0 开始） */
export interface KeypointTemplate {
  name: string;
  points: string[];
  skeleton: [number, number][];
}

export interface ImageAnnotations {
  id: number;
  name: string;
  width: number;
  height: number;
  status: ImageStatus;
  annotations: Annotation[];
}

export interface SplitSummary {
  train: number;
  val: number;
  test: number;
  unset: number;
}

export type DatasetFormat = "yolo" | "coco" | "voc" | "cls";

export interface ExportResult {
  outDir: string;
  images: number;
  boxes: number;
  polys: number;
  tags: number;
  kpts: number;
  train: number;
  val: number;
  test: number;
  classes: string[];
}

export interface DatasetImportResult {
  format: string;
  images: number;
  added: number;
  duplicated: number;
  boxes: number;
  labelsAdded: string[];
  keptExisting: number;
  failed: string[];
}

export interface Settings {
  python: string | null;
  uv: string | null;
  mirror: "china" | "official";
  envRoot: string | null;
  ffmpeg: string | null;
  /** 巡瞳 AI 平台地址，空 = 默认 https://xuntongai.com */
  aiUrl: string | null;
  /** 巡瞳 AI 的 API Key，只保存在本机 */
  aiKey: string | null;
}

export interface EnvStatus {
  python: string | null;
  managed: boolean;
  ready: boolean;
  info: Record<string, unknown> | null;
  gpu: { name: string; vramMb: number; driver: string; cuda: string } | null;
  variant: string;
  uv: string | null;
  installing: boolean;
  error: string | null;
  settings: Settings;
  envDir: string;
}

export interface SpeedResult {
  china: number | null;
  official: number | null;
  recommend: "china" | "official";
}

export interface TrainConfig {
  model: string;
  epochs: number;
  imgSize: number;
  batch: number;
  lr: number;
  device: "auto" | "cpu" | "cuda";
  onlyReviewed: boolean;
  ratios: [number, number, number];
}

export type RunStatus = "running" | "done" | "stopped" | "failed" | "interrupted";

export interface RunInfo {
  id: number;
  name: string;
  dir: string;
  engine: string;
  status: RunStatus;
  config: TrainConfig & {
    classes?: string[];
    /** anns：全部标注数（框、多边形、关键点目标、分类标签合计）；老记录只有 boxes */
    images?: { train: number; val: number; test: number; boxes: number; anns?: number };
  };
  bestMap50: number | null;
  createdAt: number;
  hasWeights: boolean;
  hasOnnx: boolean;
}

export interface Metric {
  epoch: number;
  loss: number | null;
  map50: number;
  map: number;
  precision: number;
  recall: number;
  lr: number;
  time: number;
}

export interface ClassEval {
  index: number;
  name: string;
  gt: number;
  precision: number;
  recall: number;
  ap50: number | null;
  ap: number | null;
}

export interface SplitEval {
  images: number;
  map50: number;
  map: number;
  precision: number;
  recall: number;
  perClass: ClassEval[];
  confusion: number[][];
  /** 分类模型的错例另有 gt（标的类别序号）、pred（判成的类别序号）、score */
  errors: { imageId: number; missed: number; extra: number; wrong: number; gt?: number; pred?: number; score?: number | null }[];
}

export interface RunDetail {
  run: RunInfo;
  metrics: Metric[];
  eval: { classes: string[]; bestEpoch: number; splits: Record<string, SplitEval> } | null;
  log: string[];
  error: string | null;
}

export interface PredBox {
  labelId: number;
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
}

export interface PredPoly {
  labelId: number;
  points: number[];
  score: number;
}

export interface PredTag {
  labelId: number;
  score: number;
}

export interface PredPose {
  labelId: number;
  box: number[];
  kpts: number[];
  score: number;
}

/** 检测模型给 boxes，分割模型给 polys，关键点模型给 poses，分类模型给 tags（分数从高到低） */
export interface PredResult {
  boxes: PredBox[];
  polys: PredPoly[];
  tags: PredTag[];
  poses: PredPose[];
}

export type IssueKind = "missing" | "wrong_class" | "loose_box" | "extra_box" | "tiny_box" | "dup_box" | "no_label";

export interface Issue {
  kind: IssueKind;
  imageId: number;
  labelId: number | null;
  predLabelId: number | null;
  bbox: [number, number, number, number] | null;
  score: number | null;
  iou: number | null;
  detail: string;
  key: string;
}

export interface HealthReport {
  createdAt: number;
  imagesChecked: number;
  boxesChecked: number;
  runId: number | null;
  runName: string | null;
  issues: Issue[];
  classes: { labelId: number; name: string; color: string; boxes: number; images: number }[];
  sizes: [string, number][];
  dupGroups: number[][];
  suggestions: string[];
  dims: Record<string, [number, number]>;
}

export interface FormatResult {
  format: string;
  path: string;
  ok: boolean;
  note: string;
}

export interface FeedbackInput {
  message: string;
  contact?: string;
  screenshot?: string | null;
}

export const api = {
  submitFeedback: (input: FeedbackInput) => invoke<void>("submit_feedback", { input }),
  liveStart: (runId: number, source: string, conf: number) => invoke<void>("live_start", { runId, source, conf }),
  liveStop: () => invoke<void>("live_stop"),
  liveConf: (conf: number) => invoke<void>("live_conf", { conf }),
  liveSnapshot: () => invoke<void>("live_snapshot"),
  exportFormats: (runId: number, formats: string[], size: number, outDir: string) =>
    invoke<FormatResult[]>("export_formats", { runId, formats, size, outDir }),
  buildService: (runId: number, config: unknown, outDir: string) => invoke<string>("build_service", { runId, config, outDir }),
  healthLoad: () => invoke<HealthReport | null>("health_load"),
  healthRun: (runId: number | null) => invoke<HealthReport>("health_run", { runId }),
  healthIgnore: (keys: string[]) => invoke<void>("health_ignore", { keys }),
  appendAnnotation: (imageId: number, annotation: Annotation) => invoke<ImageStatus>("append_annotation", { imageId, annotation }),
  envStatus: (refresh: boolean) => invoke<EnvStatus>("env_status", { refresh }),
  saveSettings: (settings: Settings) => invoke<void>("save_settings", { settings }),
  envInstall: (variant: string) => invoke<void>("env_install", { variant }),
  speedTest: (variant: string) => invoke<SpeedResult>("speed_test", { variant }),
  listRuns: () => invoke<RunInfo[]>("list_runs"),
  runDetail: (id: number) => invoke<RunDetail>("run_detail", { id }),
  runningRun: () => invoke<number | null>("running_run"),
  startTraining: (config: TrainConfig) => invoke<number>("start_training", { config }),
  stopTraining: (id: number) => invoke<void>("stop_training", { id }),
  deleteRun: (id: number) => invoke<void>("delete_run", { id }),
  exportOnnx: (id: number) => invoke<string>("export_onnx", { id }),
  predictImage: (runId: number, imageId: number, conf: number) => invoke<PredResult>("predict_image", { runId, imageId, conf }),
  prelabelAll: (runId: number, conf: number) => invoke<{ images: number; boxes: number }>("prelabel_all", { runId, conf }),
  revealPath: (path: string) => invoke<void>("reveal_path", { path }),
  splitSummary: () => invoke<SplitSummary>("split_summary"),
  assignSplits: (ratios: [number, number, number], all: boolean) => invoke<SplitSummary>("assign_splits", { ratios, all }),
  exportDataset: (options: { format: DatasetFormat; outDir: string; onlyReviewed: boolean; ratios: [number, number, number] }) =>
    invoke<ExportResult>("export_dataset", { options }),
  importDataset: (dir: string) => invoke<DatasetImportResult>("import_dataset", { dir }),
  lastImport: () => invoke<LastImport | null>("last_import"),
  deleteImages: (scope: "ids" | "last" | "all", ids?: number[]) => invoke<number>("delete_images", { scope, ids }),
  listImageIds: (status: string | null) => invoke<number[]>("list_image_ids", { status }),
  loadAnnotations: (imageId: number) => invoke<ImageAnnotations>("load_annotations", { imageId }),
  saveAnnotations: (imageId: number, annotations: Annotation[]) =>
    invoke<ImageStatus>("save_annotations", { imageId, annotations }),
  setImageStatus: (imageId: number, status: ImageStatus) => invoke<void>("set_image_status", { imageId, status }),
  listLabels: () => invoke<Label[]>("list_labels"),
  addLabel: (name: string, color: string) => invoke<Label>("add_label", { name, color }),
  updateLabel: (id: number, name: string, color: string, prompt?: string) =>
    invoke<void>("update_label", { id, name, color, prompt: prompt ?? null }),
  samSegment: (imageId: number, points: number[][], box: number[] | null) =>
    invoke<SamResult>("sam_segment", { imageId, points, bbox: box }),
  textDetect: (imageId: number, conf: number) => invoke<PredResult>("text_detect", { imageId, conf }),
  textPrelabelAll: (conf: number) => invoke<{ images: number; boxes: number }>("text_prelabel_all", { conf }),
  aiModels: () => invoke<AiModel[]>("ai_models"),
  aiDetect: (imageId: number, model: string, conf: number) => invoke<PredResult>("ai_detect", { imageId, model, conf }),
  aiPrelabelAll: (model: string, conf: number) => invoke<{ images: number; boxes: number }>("ai_prelabel_all", { model, conf }),
  deleteLabel: (id: number) => invoke<number>("delete_label", { id }),

  createProject: (parent: string, name: string, task: Task) =>
    invoke<ProjectInfo>("create_project", { parent, name, task }),
  openProject: (path: string) => invoke<ProjectInfo>("open_project", { path }),
  closeProject: () => invoke<void>("close_project"),
  currentProject: () => invoke<ProjectInfo | null>("current_project"),
  recentProjects: () => invoke<RecentProject[]>("recent_projects"),
  removeRecent: (path: string) => invoke<void>("remove_recent", { path }),
  importImages: (paths: string[], copy: boolean) => invoke<ImportResult>("import_images", { paths, copy }),
  extractVideo: (paths: string[], every: number, maxFrames: number) =>
    invoke<ExtractResult>("extract_video", { paths, every, maxFrames }),
  probeStream: (url: string) => invoke<StreamInfo>("probe_stream", { url }),
  extractStream: (url: string, every: number, maxFrames: number, maxMinutes: number, dedup: boolean) =>
    invoke<ExtractResult>("extract_stream", { url, every, maxFrames, maxMinutes, dedup }),
  stopExtract: () => invoke<void>("stop_extract"),
  keypointTemplate: () => invoke<KeypointTemplate>("keypoint_template"),
  saveKeypointTemplate: (template: KeypointTemplate) => invoke<void>("save_keypoint_template", { template }),
  listImages: (status: string | null, offset: number, limit: number) =>
    invoke<ImagePage>("list_images", { status, offset, limit }),
  projectStats: () => invoke<Stats>("project_stats"),
};

// 图片走自定义协议 xl://，Windows 上的实际地址是 http://xl.localhost
const BASE = navigator.userAgent.includes("Windows") ? "http://xl.localhost" : "xl://localhost";

// 不同项目的图片 id 会重复，带上项目标识，避免浏览器缓存串图
function projectKey(path: string) {
  let h = 0;
  for (let i = 0; i < path.length; i++) h = (h * 31 + path.charCodeAt(i)) | 0;
  return (h >>> 0).toString(36);
}

export const thumbUrl = (projectPath: string, id: number) => `${BASE}/thumb/${id}?p=${projectKey(projectPath)}`;
export const imageUrl = (projectPath: string, id: number) => `${BASE}/image/${id}?p=${projectKey(projectPath)}`;

export const errText = (e: unknown) => (typeof e === "string" ? e : e instanceof Error ? e.message : String(e));
