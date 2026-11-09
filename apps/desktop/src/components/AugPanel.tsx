import { useState } from "react";
import { App, Button, Checkbox, Collapse, Image, Modal, Segmented, Slider, Spin } from "antd";
import { api, AugConfig, errText, Task } from "../api";

const BASE: AugConfig = { hflip: true, vflip: false, brightness: 0, contrast: 0, saturation: 0, hue: 0, blur: 0, noise: 0, scale: 0, rotate: 0, crop: 1 };

/** 预设。「默认」和以前的训练行为一致 */
export const AUG_PRESETS: Record<string, { name: string; det: AugConfig; cls: AugConfig }> = {
  off: { name: "关闭", det: { ...BASE, hflip: false }, cls: { ...BASE, hflip: false, crop: 1 } },
  default: { name: "默认", det: { ...BASE }, cls: { ...BASE, brightness: 0.2, contrast: 0.2, crop: 0.5 } },
  light: {
    name: "轻度",
    det: { ...BASE, brightness: 0.15, contrast: 0.15, saturation: 0.15, scale: 0.1 },
    cls: { ...BASE, brightness: 0.2, contrast: 0.2, saturation: 0.2, crop: 0.5 },
  },
  standard: {
    name: "标准",
    det: { ...BASE, brightness: 0.3, contrast: 0.3, saturation: 0.3, hue: 0.02, blur: 0.1, scale: 0.2 },
    cls: { ...BASE, brightness: 0.3, contrast: 0.3, saturation: 0.3, hue: 0.03, blur: 0.1, rotate: 10, crop: 0.35 },
  },
  strong: {
    name: "强（数据少时）",
    det: { ...BASE, brightness: 0.4, contrast: 0.4, saturation: 0.4, hue: 0.05, blur: 0.2, noise: 0.03, scale: 0.3 },
    cls: { ...BASE, brightness: 0.4, contrast: 0.4, saturation: 0.4, hue: 0.05, blur: 0.2, noise: 0.03, rotate: 20, crop: 0.25 },
  },
};

const pct = (v: number) => `±${Math.round(v * 100)}%`;

/** 训练页的数据增强设置：预设 + 逐项调整 + 效果预览 */
export default function AugPanel({
  value,
  task,
  imgSize,
  fixedSize,
  onChange,
}: {
  value: AugConfig | null | undefined;
  task: Task;
  imgSize: number;
  fixedSize: boolean;
  onChange: (v: AugConfig) => void;
}) {
  const { message } = App.useApp();
  const cls = task === "classify";
  const pick = (k: string) => (cls ? AUG_PRESETS[k].cls : AUG_PRESETS[k].det);
  const a: AugConfig = { ...pick("default"), ...(value ?? {}) };
  const preset = Object.keys(AUG_PRESETS).find((k) => JSON.stringify(pick(k)) === JSON.stringify(a)) ?? "custom";
  const [shots, setShots] = useState<string[] | null>(null);
  const [loading, setLoading] = useState(false);
  const set = (patch: Partial<AugConfig>) => onChange({ ...a, ...patch });

  const preview = async () => {
    setLoading(true);
    setShots([]);
    try {
      setShots(await api.augPreview(a, imgSize));
    } catch (e) {
      message.error(errText(e));
      setShots(null);
    } finally {
      setLoading(false);
    }
  };

  const slider = (label: string, key: keyof AugConfig, max: number, step: number, fmt: (v: number) => string, tip?: string) => (
    <div className="row" style={{ gap: 8, flexWrap: "nowrap", marginBottom: 2 }} title={tip}>
      <span style={{ width: 70, fontSize: 12 }}>{label}</span>
      <Slider style={{ flex: 1, margin: "6px 4px" }} min={0} max={max} step={step} value={a[key] as number} onChange={(v) => set({ [key]: v } as Partial<AugConfig>)} />
      <span className="dim" style={{ width: 44, fontSize: 12 }}>{(a[key] as number) ? fmt(a[key] as number) : "关"}</span>
    </div>
  );

  return (
    <>
      <Collapse
        size="small"
        style={{ marginBottom: 12 }}
        items={[
          {
            key: "aug",
            label: (
              <span>
                数据增强 <span className="dim" style={{ fontSize: 12 }}>· {preset === "custom" ? "自定义" : AUG_PRESETS[preset].name}</span>
              </span>
            ),
            children: (
              <div>
                <Segmented
                  size="small"
                  block
                  value={preset}
                  options={[...Object.entries(AUG_PRESETS).map(([k, v]) => ({ value: k, label: v.name })), ...(preset === "custom" ? [{ value: "custom", label: "自定义" }] : [])]}
                  onChange={(k) => k !== "custom" && onChange(pick(k as string))}
                  style={{ marginBottom: 8 }}
                />
                <div className="dim" style={{ fontSize: 12, marginBottom: 6 }}>
                  训练时随机改变图片，让模型见过更多样子，数据少时效果明显。改完点「预览效果」看看会不会太夸张。
                </div>
                <div className="row" style={{ gap: 14, marginBottom: 4 }}>
                  <Checkbox checked={a.hflip} onChange={(e) => set({ hflip: e.target.checked })}>
                    左右翻转
                  </Checkbox>
                  {task !== "pose" && (
                    <Checkbox checked={a.vflip} onChange={(e) => set({ vflip: e.target.checked })}>
                      上下翻转
                    </Checkbox>
                  )}
                </div>
                {slider("亮度", "brightness", 0.5, 0.05, pct)}
                {slider("对比度", "contrast", 0.5, 0.05, pct)}
                {slider("饱和度", "saturation", 0.5, 0.05, pct)}
                {slider("色调", "hue", 0.1, 0.01, (v) => `±${v.toFixed(2)}`)}
                {slider("模糊", "blur", 0.5, 0.05, (v) => `${Math.round(v * 100)}%`, "随机模糊的概率，模拟对焦不准、运动模糊")}
                {slider("噪点", "noise", 0.1, 0.01, (v) => v.toFixed(2), "模拟夜间、低照度的噪点")}
                {!cls && !fixedSize && slider("多尺度", "scale", 0.5, 0.05, pct, "训练时输入尺寸在这个范围内随机，对远近大小变化大的目标有帮助")}
                {cls && slider("旋转", "rotate", 45, 5, (v) => `±${v}°`)}
                {cls && (
                  <div className="row" style={{ gap: 8, flexWrap: "nowrap" }}>
                    <span style={{ width: 70, fontSize: 12 }}>随机裁剪</span>
                    <Slider style={{ flex: 1, margin: "6px 4px" }} min={0.1} max={1} step={0.05} value={a.crop} onChange={(v) => set({ crop: v })} />
                    <span className="dim" style={{ width: 44, fontSize: 12 }}>{a.crop >= 1 ? "关" : `≥${Math.round(a.crop * 100)}%`}</span>
                  </div>
                )}
                <Button size="small" style={{ marginTop: 6 }} onClick={preview} loading={loading}>
                  预览效果
                </Button>
              </div>
            ),
          },
        ]}
      />
      <Modal title="数据增强效果预览（每张图随机变两次）" open={shots != null} onCancel={() => setShots(null)} footer={null} width={720}>
        {loading ? (
          <div style={{ padding: 40, textAlign: "center" }}>
            <Spin />
          </div>
        ) : (
          <Image.PreviewGroup>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
              {(shots ?? []).map((s, i) => (
                <Image key={i} src={s} style={{ width: "100%", borderRadius: 4 }} />
              ))}
            </div>
          </Image.PreviewGroup>
        )}
        <div className="dim" style={{ fontSize: 12, marginTop: 8 }}>红框是跟着变换后的标注位置。预览不包含多尺度（只影响训练时的缩放尺寸）。</div>
      </Modal>
    </>
  );
}
