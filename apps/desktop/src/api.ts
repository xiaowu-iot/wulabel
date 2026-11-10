import { getToken, invoke, isRemote, postJson } from "./transport";

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

export interface InterpResult {
  fromId: number;
  frames: number;
  objects: number;
  unmatched: number;
}

export interface TrackResult {
  frames: number;
  objects: number;
  lost: number;
  hitKeyframe: boolean;
  lastId: number | null;
}

/** 打码：这些类别的标注区域 + 自动找的人脸、车牌 */
export interface AnonymizeConfig {
  labels: number[];
  faces: boolean;
  plates: boolean;
  mode: "mosaic" | "blur" | "black";
}

export interface Label {
  id: number;
  name: string;
  color: string;
  count: number;
  /** 英文提示词（文本提示自动标注用），可为空 */
  prompt?: string;
  /** 识别说明（巡瞳 AI 按项目类别识别时用），可为空 */
  description?: string;
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
  /** 用哪个数据集版本训练；空 = 当前数据 */
  version?: number | null;
  /** 接着哪次训练的模型继续训；空 = 从预训练权重开始 */
  baseRun?: number | null;
  /** 数据增强；空 = 默认 */
  aug?: AugConfig | null;
}

/** 数据增强参数（和 worker 的 augment.py 一致） */
export interface AugConfig {
  hflip: boolean;
  vflip: boolean;
  brightness: number;
  contrast: number;
  saturation: number;
  hue: number;
  blur: number;
  noise: number;
  /** 多尺度训练幅度（检测类） */
  scale: number;
  /** 随机旋转角度（分类） */
  rotate: number;
  /** 随机裁剪最少保留的面积比例（分类） */
  crop: number;
}

export interface VersionInfo {
  id: number;
  name: string;
  note: string;
  createdAt: number;
  images: number;
  annotations: number;
  train: number;
  val: number;
  test: number;
  classes: { name: string; count: number }[];
}

/** 版本和当前数据的差别：added 当前新增的图、removed 版本里有当前没有、changed 标注改过的图 */
export interface VersionDiff {
  added: number[];
  removed: number[];
  changed: number[];
  classes: { name: string; before: number; after: number }[];
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
    /** 训练用的数据集版本名（用当前数据训练时没有） */
    versionName?: string;
    /** 接着哪次训练继续训的（那次的名字） */
    baseRunName?: string;
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

/** 某个置信度阈值下的结果 */
export interface ThresholdPoint {
  conf: number;
  precision: number;
  recall: number;
  f1: number;
  tp?: number;
  fp?: number;
  fn?: number;
}

export interface Thresholds {
  curve: ThresholdPoint[];
  /** F1 最高的阈值 */
  best: ThresholdPoint;
  /** 每个类别 F1 最高的阈值（分类没有） */
  perClass: { index: number; name?: string; conf: number | null; precision: number | null; recall: number | null; f1: number | null }[];
}

/** 外部 ONNX 模型的识别结果 */
export interface OnnxInfo {
  kind: string;
  classes: string[];
  numClasses: number;
  imgSize: number;
  note: string;
  inputs: { name: string; shape: string[] }[];
  outputs: { name: string; shape: string[] }[];
}

export interface ExternalModel {
  name: string;
  kind: string;
  classes: string[];
  imgSize: number;
  /** 只用这些类别（模型类别序号），null 表示全用 */
  useClasses: number[] | null;
}

export interface FpConfig {
  runId: number;
  source: string;
  conf: number;
  fps: number;
  maxSeconds: number;
  consecutive: number;
  cooldown: number;
}

/** 误报测试的统计 */
export interface FpSummary {
  seconds: number;
  frames: number;
  hitFrames: number;
  alarms: number;
  perHour: number;
  perClass: Record<string, number>;
  stopped: boolean;
  dir: string;
}

/** 多人协作分出去的标注包 */
export interface PackageInfo {
  id: number;
  name: string;
  assignee: string;
  dir: string;
  images: number;
  createdAt: number;
  mergedAt: number | null;
}

export interface MergeResult {
  assignee: string;
  updated: number;
  unchanged: number;
  /** 原项目在分包后也改过、没有覆盖的图 */
  conflicts: number[];
  added: number;
  missing: number;
  labelsAdded: string[];
}

/** 图片质量筛查结果。issues：blur 模糊 / dark 过暗 / bright 过曝 / flat 对比度低 */
export interface QualityReport {
  checked: number;
  failed: number;
  items: { imageId: number; issues: string[]; sharpness: number; brightness: number; overexposed: number; contrast: number }[];
}

export interface SplitEval {
  thresholds?: Thresholds;
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
  /** reevaluated：用「重新评估」按当时的数据算的（evaluatedAt 是时间） */
  eval: { classes: string[]; bestEpoch: number | null; splits: Record<string, SplitEval>; reevaluated?: boolean; evaluatedAt?: number } | null;
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
  exportDataset: (
    options: { format: DatasetFormat; outDir: string; onlyReviewed: boolean; ratios: [number, number, number] },
    anonymize: AnonymizeConfig | null = null,
  ) => invoke<ExportResult>("export_dataset", { options, anonymize }),
  importDataset: (dir: string) => invoke<DatasetImportResult>("import_dataset", { dir }),
  lastImport: () => invoke<LastImport | null>("last_import"),
  deleteImages: (scope: "ids" | "last" | "all", ids?: number[]) => invoke<number>("delete_images", { scope, ids }),
  listImageIds: (status: string | null) => invoke<number[]>("list_image_ids", { status }),
  loadAnnotations: (imageId: number) => invoke<ImageAnnotations>("load_annotations", { imageId }),
  saveAnnotations: (imageId: number, annotations: Annotation[]) =>
    invoke<ImageStatus>("save_annotations", { imageId, annotations }),
  setImageStatus: (imageId: number, status: ImageStatus) => invoke<void>("set_image_status", { imageId, status }),
  listLabels: () => invoke<Label[]>("list_labels"),
  interpolateFrames: (imageId: number) => invoke<InterpResult>("interpolate_frames", { imageId }),
  trackFrames: (imageId: number, count: number) => invoke<TrackResult>("track_frames", { imageId, count }),
  addLabel: (name: string, color: string) => invoke<Label>("add_label", { name, color }),
  updateLabel: (id: number, name: string, color: string, prompt?: string, description?: string) =>
    invoke<void>("update_label", { id, name, color, prompt: prompt ?? null, description: description ?? null }),
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
  inspectOnnx: (path: string) => invoke<OnnxInfo>("inspect_onnx", { path }),
  importOnnx: (path: string, model: ExternalModel) => invoke<number>("import_onnx", { path, model }),
  updateExternal: (runId: number, model: ExternalModel) => invoke<void>("update_external", { runId, model }),
  evaluateRun: (runId: number) => invoke<number>("evaluate_run", { runId }),
  fpTest: (config: FpConfig) => invoke<FpSummary>("fp_test", { config }),
  stopFpTest: () => invoke<void>("stop_fp_test"),
  importNegatives: (paths: string[]) => invoke<number>("import_negatives", { paths }),
  augPreview: (aug: AugConfig, imgSize: number) => invoke<string[]>("aug_preview", { aug, imgSize }),
  exportReport: (runId: number, path: string) => invoke<void>("export_report", { runId, path }),
  splitPackages: (options: {
    outDir: string;
    assignees: string[];
    scope: "todo" | "unreviewed" | "all";
    withAnnotations: boolean;
    anonymize?: AnonymizeConfig | null;
  }) =>
    invoke<PackageInfo[]>("split_packages", { options }),
  listPackages: () => invoke<PackageInfo[]>("list_packages"),
  mergePackage: (options: { dir: string; overwrite: boolean; importNew: boolean }) => invoke<MergeResult>("merge_package", { options }),
  qualityScan: (onlyTodo: boolean) => invoke<QualityReport>("quality_scan", { onlyTodo }),
  uncertainImages: (runId: number, limit: number) =>
    invoke<{ imageId: number; score: number; detail: string }[]>("uncertain_images", { runId, limit }),
  listVersions: () => invoke<VersionInfo[]>("list_versions"),
  createVersion: (name: string, note: string, ratios: [number, number, number]) =>
    invoke<VersionInfo>("create_version", { name, note, ratios }),
  renameVersion: (id: number, name: string, note: string) => invoke<void>("rename_version", { id, name, note }),
  deleteVersion: (id: number) => invoke<void>("delete_version", { id }),
  diffVersion: (id: number) => invoke<VersionDiff>("diff_version", { id }),
  restoreVersion: (id: number) => invoke<{ images: number; missing: number; backup: number | null }>("restore_version", { id }),
  exportVersion: (
    id: number,
    options: { format: DatasetFormat; outDir: string; onlyReviewed: boolean; ratios: [number, number, number] },
    anonymize: AnonymizeConfig | null = null,
  ) => invoke<ExportResult>("export_version", { id, options, anonymize }),
  keypointTemplate: () => invoke<KeypointTemplate>("keypoint_template"),
  saveKeypointTemplate: (template: KeypointTemplate) => invoke<void>("save_keypoint_template", { template }),
  listImages: (status: string | null, offset: number, limit: number) =>
    invoke<ImagePage>("list_images", { status, offset, limit }),
  projectStats: () => invoke<Stats>("project_stats"),

  // ---------- 实时协作 ----------
  shareStatus: () => invoke<ShareStatus>("share_status"),
  shareStart: (options: ShareOptions) => invoke<ShareStatus>("share_start", { options }),
  shareStop: () => invoke<void>("share_stop"),
  shareSaveOptions: (options: ShareOptions) => invoke<void>("share_save_options", { options }),
  /** 打开一张图时加锁：别人正在改返回他的名字（只读打开） */
  lockImage: (imageId: number) => invoke<LockResult>(isRemote ? "lock_image" : "share_lock", { imageId }),
  unlockImage: () => invoke<void>(isRemote ? "unlock_image" : "share_unlock"),
  reviewImage: (imageId: number, pass: boolean, reason: string) =>
    invoke<ImageStatus>(isRemote ? "review_image" : "share_review", { imageId, pass, reason }),
  teamMembers: () => invoke<Member[]>("team_members"),
  teamAddMember: (name: string, role: Role, password: string) => invoke<Member>("team_add_member", { name, role, password }),
  teamUpdateMember: (id: number, patch: MemberPatch) => invoke<Member>("team_update_member", { id, patch }),
  teamDeleteMember: (id: number) => invoke<void>("team_delete_member", { id }),
  teamAssign: (options: { memberIds: number[]; scope: "unassigned" | "todo" | "unreviewed"; perPerson: number | null }) =>
    invoke<AssignResult[]>("team_assign", { options }),
  teamUnassign: (memberId: number, onlyUnfinished: boolean) => invoke<number>("team_unassign", { memberId, onlyUnfinished }),
  teamStats: (from: number, to: number) => invoke<MemberStats[]>("team_stats", { from, to }),
  teamExportCsv: (from: number, to: number, path: string) => invoke<number>("team_export_csv", { from, to, path }),
  teamActivity: (limit: number, memberId: number | null, withSaves: boolean) =>
    invoke<ActivityRow[]>("team_activity", { limit, memberId, withSaves }),
  /** 成员端：我是谁、项目、协作选项 */
  whoami: () => invoke<Whoami>("whoami"),
  myStats: () => invoke<{ member: Member | null; all: MemberStats | null; today: MemberStats | null }>("my_stats"),
  login: (name: string, password: string) => postJson<Whoami & { token: string }>("api/login", { name, password }),
  shareInfo: () => postJson<{ project: string; task: Task; hostName: string }>("api/info"),

  // ---------- 巡瞳 AI 识别规则 ----------
  aiRules: () => invoke<AiRules>("ai_rules"),
  /** 返回非空时是提醒（比如同步到巡瞳平台失败） */
  aiSaveRules: (instructions: string | null, classes: ClassRule[]) => invoke<string | null>("ai_save_rules", { instructions, classes }),
  aiSyncRules: () => invoke<string>("ai_sync_rules"),
  aiUnbindRules: () => invoke<void>("ai_unbind_rules"),
  /** 平台上绑定规则的当前状态（启用、适用场景），没绑定返回 null */
  aiRuleSetInfo: () => invoke<{ code: string; enabled: boolean; scenes: string[] | null; name: string | null } | null>("ai_rule_set_info"),
  aiRulesPlatformEnabled: (enabled: boolean) => invoke<void>("ai_rules_platform_enabled", { enabled }),
  aiChatRules: (messages: ChatMsg[]) => invoke<ChatReply>("ai_chat_rules", { messages }),

  // ---------- 巡瞳 AI 平台：上传模型、拉告警图片 ----------
  xtUploadModel: (runId: number, name: string) => invoke<XtUploadResult>("xt_upload_model", { runId, name }),
  xtModels: () => invoke<XtModel[]>("xt_models"),
  xtDeleteModel: (id: string) => invoke<void>("xt_delete_model", { id }),
  /** 登录巡瞳账号：申请登录码并在浏览器打开确认页 */
  xtLoginStart: () =>
    invoke<{ deviceCode: string; userCode: string; verifyUrl: string; expiresIn: number; interval: number }>("xt_login_start"),
  xtLoginPoll: (deviceCode: string) =>
    invoke<{ status: "pending" | "slow_down" | "expired" | "denied" | "ok"; account?: { name?: string; plan?: string } | null; keyName?: string | null }>(
      "xt_login_poll",
      { deviceCode },
    ),
  /** 账号和额度；平台还不支持时是 null */
  xtAccount: () => invoke<XtAccount | null>("xt_account"),
  xtLogout: () => invoke<void>("xt_logout"),
  openUrl: (url: string) => invoke<void>("open_url", { url }),
  xtPullAlarms: (options: { days: number; model: string | null; max: number; withBoxes: boolean }) =>
    invoke<XtPullResult>("xt_pull_alarms", { options }),
};

export interface XtAccount {
  name?: string;
  plan?: string;
  keyName?: string;
  quota?: { name: string; used?: number | null; total?: number | null; unit?: string; expiresAt?: number | null }[];
  expiresAt?: number | null;
}

export interface XtModel {
  id: string | number;
  code: string;
  name: string;
  /** pending 校验中 / ready / failed */
  status: string;
  enabled: boolean;
  message: string;
  createdAt: number | null;
}

export interface XtUploadResult {
  id: string | number | null;
  code: string;
  status: string;
  enabled: boolean;
  message: string;
  consoleUrl: string;
}

export interface XtPullResult {
  fetched: number;
  added: number;
  duplicated: number;
  failed: number;
  noImage: number;
  boxes: number;
  folder: string;
}

// ---------- 实时协作类型 ----------

export type Role = "annotator" | "reviewer" | "admin";

export const ROLE_NAMES: Record<Role, string> = { annotator: "标注员", reviewer: "审核员", admin: "管理员" };

export interface Member {
  id: number;
  name: string;
  role: Role;
  disabled: boolean;
  createdAt: number;
  assigned: number;
  done: number;
  rejected: number;
}

export interface MemberPatch {
  name?: string;
  role?: Role;
  password?: string;
  disabled?: boolean;
}

export type ShareMode = "lan" | "relay" | "xuntong";

export interface ShareOptions {
  mode: ShareMode;
  port: number;
  relayUrl: string;
  relaySecret: string;
  xuntongRelayUrl: string;
  hostName: string;
  allowAi: boolean;
  watermark: boolean;
  membersEditLabels: boolean;
  shareId: string;
}

export interface Presence {
  memberId: number;
  name: string;
  role: Role;
  imageId: number | null;
  since: number;
}

export interface RelayStatus {
  state: "off" | "connecting" | "online" | "error";
  url: string;
  error: string;
}

export interface ShareStatus {
  running: boolean;
  options: ShareOptions;
  lanUrls: string[];
  relay: RelayStatus;
  online: Presence[];
}

export interface ImageTeamInfo {
  assignee: string | null;
  labeledBy: string | null;
  reviewedBy: string | null;
  rejectReason: string;
}

export interface LockResult {
  holder: string | null;
  info: ImageTeamInfo;
}

export interface AssignResult {
  memberId: number;
  name: string;
  count: number;
}

export interface MemberStats {
  memberId: number;
  name: string;
  role: Role;
  assigned: number;
  submitted: number;
  objects: number;
  passed: number;
  rejected: number;
  reviews: number;
  activeSecs: number;
}

export interface ActivityRow {
  id: number;
  time: number;
  member: string;
  imageId: number | null;
  action: string;
  objects: number;
  detail: string;
}

export interface Whoami {
  member: Member;
  project: { name: string; task: Task };
  hostName: string;
  options: { allowAi: boolean; watermark: boolean; membersEditLabels: boolean };
}

/** 实时事件（事件名 share-event） */
export interface ShareEvent {
  kind: "presence" | "ann" | "status" | "lock" | "labels" | "assign" | "kick" | "options" | "closed";
  imageId?: number;
  memberId?: number;
  by?: string;
  status?: ImageStatus;
  action?: string;
  holder?: string | null;
  online?: Presence[];
  allowAi?: boolean;
  watermark?: boolean;
  membersEditLabels?: boolean;
}

// ---------- 巡瞳 AI 识别规则 ----------

export interface ClassRule {
  id?: number;
  name: string;
  prompt?: string;
  description: string;
}

export interface AiRules {
  instructions: string;
  classes: ClassRule[];
  /** 绑定的巡瞳平台规则编号，没同步过是空 */
  ruleSet?: string;
}

export interface ChatMsg {
  role: "user" | "assistant";
  content: string;
  imageId?: number | null;
}

export interface ChatReply {
  reply: string;
  rules: { instructions?: string | null; classes: ClassRule[] } | null;
  preview: PredResult | null;
  previewImageId: number | null;
}

// 图片走自定义协议 xl://，Windows 上的实际地址是 http://xl.localhost
const BASE = navigator.userAgent.includes("Windows") ? "http://xl.localhost" : "xl://localhost";

// 不同项目的图片 id 会重复，带上项目标识，避免浏览器缓存串图
function projectKey(path: string) {
  let h = 0;
  for (let i = 0; i < path.length; i++) h = (h * 31 + path.charCodeAt(i)) | 0;
  return (h >>> 0).toString(36);
}

// 协作成员在浏览器里：图片从主机的接口取，带上登录令牌
export const thumbUrl = (projectPath: string, id: number) =>
  isRemote ? `api/thumb/${id}?t=${encodeURIComponent(getToken())}` : `${BASE}/thumb/${id}?p=${projectKey(projectPath)}`;
export const imageUrl = (projectPath: string, id: number) =>
  isRemote ? `api/image/${id}?t=${encodeURIComponent(getToken())}` : `${BASE}/image/${id}?p=${projectKey(projectPath)}`;

export const errText = (e: unknown) => (typeof e === "string" ? e : e instanceof Error ? e.message : String(e));
