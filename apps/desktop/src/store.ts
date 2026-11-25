import { create } from "zustand";
import { api, Member, ProjectInfo } from "./api";
import { isRemote } from "./transport";

/** 实时协作的状态：成员在浏览器里（remote），或者发起人正在共享（shareRunning） */
export interface TeamState {
  remote: boolean;
  me: Member | null;
  options: { allowAi: boolean; watermark: boolean; membersEditLabels: boolean };
  shareRunning: boolean;
  /** 项目里加过协作成员 */
  hasMembers: boolean;
}

export interface Highlight {
  imageId: number;
  bbox: [number, number, number, number];
  text: string;
}

export type PageKey = "home" | "data" | "annotate" | "review" | "health" | "train" | "eval" | "deploy" | "team" | "settings";

interface AppStore {
  project: ProjectInfo | null;
  page: PageKey;
  /** 进入标注页时要打开的图片 */
  focusImageId: number | null;
  /** 体检跳过来时要高亮的框（原图像素坐标） */
  highlight: Highlight | null;
  /** 指定顺序的一批图去标注（比如模型挑出来最值得标的），标注页会多一个筛选项 */
  /** 指定的一批图去标注；from 为 review 时标注页显示「返回审核」 */
  queue: { ids: number[]; name: string; from?: "review" } | null;
  /** 训练、评估、部署页打开时要选中的训练记录（比如训练页点「看评估」） */
  focusRunId: number | null;
  setProject: (p: ProjectInfo | null) => void;
  setPage: (p: PageKey) => void;
  annotateImage: (id: number, highlight?: Highlight) => void;
  annotateQueue: (ids: number[], name: string, opts?: { from?: "review"; focus?: number }) => void;
  clearQueue: () => void;
  takeFocusImage: () => number | null;
  /** 带着训练记录去训练 / 评估 / 部署页 */
  openRun: (id: number, page: PageKey) => void;
  takeFocusRun: () => number | null;
  refreshStats: () => Promise<void>;
  team: TeamState;
  setTeam: (t: Partial<TeamState>) => void;
}

export const useApp = create<AppStore>((set, get) => ({
  project: null,
  page: "home",
  focusImageId: null,
  highlight: null,
  queue: null,
  focusRunId: null,
  // 要看的图不在之前那批（比如上次批量「去标注」留下的）里时，退出那批，免得打开的是别的图
  annotateImage: (id, highlight) =>
    set({
      focusImageId: id,
      page: "annotate",
      highlight: highlight ?? null,
      ...(get().queue && !get().queue!.ids.includes(id) ? { queue: null } : {}),
    }),
  annotateQueue: (ids, name, opts) =>
    set({ queue: { ids, name, from: opts?.from }, focusImageId: opts?.focus ?? null, highlight: null, page: "annotate" }),
  clearQueue: () => set({ queue: null }),
  takeFocusImage: () => {
    const id = get().focusImageId;
    set({ focusImageId: null });
    return id;
  },
  openRun: (id, page) => set({ focusRunId: id, page }),
  takeFocusRun: () => {
    const id = get().focusRunId;
    set({ focusRunId: null });
    return id;
  },
  setProject: (project) => set({ project, page: project ? "data" : "home", queue: null, focusRunId: null }),
  setPage: (page) => set({ page }),
  team: {
    remote: isRemote,
    me: null,
    options: { allowAi: true, watermark: false, membersEditLabels: true },
    shareRunning: false,
    hasMembers: false,
  },
  setTeam: (t) => set({ team: { ...get().team, ...t } }),
  refreshStats: async () => {
    const p = get().project;
    if (!p) return;
    const stats = await api.projectStats();
    set({ project: { ...p, stats } });
  },
}));

/** 用不用审核流程：只有协作（有成员、正在共享、或是浏览器里的成员）时才审核；
 *  一个人标注时标完就算数，界面上不出现「审核」，已审核的图也显示成已标注 */
export const useReview = () => useApp((s) => s.team.remote || s.team.shareRunning || s.team.hasMembers);

// 本机偏好，读写失败不影响使用
export const prefs = {
  get(key: string, fallback: string) {
    try {
      return localStorage.getItem(`xlabel.${key}`) ?? fallback;
    } catch {
      return fallback;
    }
  },
  set(key: string, value: string) {
    try {
      localStorage.setItem(`xlabel.${key}`, value);
    } catch {
      /* 忽略 */
    }
  },
};
