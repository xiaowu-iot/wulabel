import { create } from "zustand";
import { api, ProjectInfo } from "./api";

export interface Highlight {
  imageId: number;
  bbox: [number, number, number, number];
  text: string;
}

export type PageKey = "home" | "data" | "annotate" | "health" | "train" | "eval" | "deploy" | "settings";

interface AppStore {
  project: ProjectInfo | null;
  page: PageKey;
  /** 进入标注页时要打开的图片 */
  focusImageId: number | null;
  /** 体检跳过来时要高亮的框（原图像素坐标） */
  highlight: Highlight | null;
  /** 指定顺序的一批图去标注（比如模型挑出来最值得标的），标注页会多一个筛选项 */
  queue: { ids: number[]; name: string } | null;
  setProject: (p: ProjectInfo | null) => void;
  setPage: (p: PageKey) => void;
  annotateImage: (id: number, highlight?: Highlight) => void;
  annotateQueue: (ids: number[], name: string) => void;
  clearQueue: () => void;
  takeFocusImage: () => number | null;
  refreshStats: () => Promise<void>;
}

export const useApp = create<AppStore>((set, get) => ({
  project: null,
  page: "home",
  focusImageId: null,
  highlight: null,
  queue: null,
  annotateImage: (id, highlight) => set({ focusImageId: id, page: "annotate", highlight: highlight ?? null }),
  annotateQueue: (ids, name) => set({ queue: { ids, name }, focusImageId: null, highlight: null, page: "annotate" }),
  clearQueue: () => set({ queue: null }),
  takeFocusImage: () => {
    const id = get().focusImageId;
    set({ focusImageId: null });
    return id;
  },
  setProject: (project) => set({ project, page: project ? "data" : "home", queue: null }),
  setPage: (page) => set({ page }),
  refreshStats: async () => {
    const p = get().project;
    if (!p) return;
    const stats = await api.projectStats();
    set({ project: { ...p, stats } });
  },
}));

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
