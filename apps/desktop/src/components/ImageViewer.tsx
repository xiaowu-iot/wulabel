import { useCallback, useEffect, useRef, useState } from "react";
import { Button, Modal, Spin } from "antd";
import { FormOutlined, LeftOutlined, RightOutlined } from "@ant-design/icons";
import { ImageItem, imageUrl } from "../api";

interface Props {
  projectPath: string;
  items: ImageItem[];
  index: number | null;
  onIndex: (i: number | null) => void;
  onAnnotate?: (imageId: number) => void;
}

/**
 * 大图预览，左右方向键翻页。
 * 图片按预览区算好像素尺寸再显示，并且每张图重新创建 img 元素：
 * WebView2 里用 CSS 缩放大图、外层又有动画时，会出现图片已加载但不重绘的黑屏。
 */
export default function ImageViewer({ projectPath, items, index, onIndex, onAnnotate }: Props) {
  const item = index === null ? null : items[index];
  const src = item ? imageUrl(projectPath, item.id) : null;
  const [shown, setShown] = useState<{ src: string; w: number; h: number; el: HTMLImageElement | null } | null>(null);
  const [canvasEl, setCanvasEl] = useState<HTMLCanvasElement | null>(null);
  const [loading, setLoading] = useState(false);
  const [boxSize, setBoxSize] = useState({ w: 0, h: 0 });
  const observer = useRef<ResizeObserver | null>(null);

  // 弹窗内容是延迟挂载的，用回调 ref 在元素真正出现时再量尺寸
  const box = useCallback((el: HTMLDivElement | null) => {
    observer.current?.disconnect();
    observer.current = null;
    if (!el) return;
    const measure = () => setBoxSize({ w: el.clientWidth, h: el.clientHeight });
    measure();
    observer.current = new ResizeObserver(measure);
    observer.current.observe(el);
  }, []);

  useEffect(() => {
    if (!src || index === null) {
      setShown(null);
      return;
    }
    let alive = true;
    setLoading(true);
    const img = new Image();
    const done = (ok: boolean) => {
      if (!alive) return;
      setShown({ src, w: img.naturalWidth || item!.width, h: img.naturalHeight || item!.height, el: ok ? img : null });
      setLoading(false);
    };
    img.onload = () => (img.decode ? img.decode().catch(() => {}).then(() => done(true)) : done(true));
    img.onerror = () => done(false);
    img.src = src;
    // 预加载前后两张，翻页时直接显示
    for (const i of [index + 1, index - 1]) {
      const it = items[i];
      if (it) new Image().src = imageUrl(projectPath, it.id);
    }
    return () => {
      alive = false;
    };
  }, [src, index, items, projectPath]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (index === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft" && index > 0) onIndex(index - 1);
      if (e.key === "ArrowRight" && index < items.length - 1) onIndex(index + 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [index, items.length, onIndex]);

  let w = 0;
  let h = 0;
  if (shown && boxSize.w && boxSize.h) {
    const s = Math.min(boxSize.w / shown.w, boxSize.h / shown.h);
    w = Math.floor(shown.w * s);
    h = Math.floor(shown.h * s);
  }

  // 大图画到 canvas 上：WebView2 里用 <img> 显示大图时，偶尔会出现图片已加载但画面不刷新的黑屏
  // 弹窗内容晚一拍挂载，用 state 记住 canvas 元素，元素出现时会重新触发绘制
  useEffect(() => {
    const cv = canvasEl;
    if (!cv || !shown?.el || !w || !h) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(shown.el, 0, 0, cv.width, cv.height);
  }, [canvasEl, shown, w, h]);

  return (
    <Modal
      open={!!item}
      onCancel={() => onIndex(null)}
      footer={null}
      width="86vw"
      centered
      transitionName=""
      maskTransitionName=""
      title={
        item && (
          <span className="row" style={{ gap: 8 }}>
            <span>
              {item.name}{" "}
              <span className="dim" style={{ fontWeight: 400 }}>
                · {item.width}×{item.height} · 本页第 {index! + 1} / {items.length} 张 · ← → 翻页
              </span>
            </span>
            {onAnnotate && (
              <Button size="small" type="primary" icon={<FormOutlined />} onClick={() => onAnnotate(item.id)}>
                标注这张
              </Button>
            )}
          </span>
        )
      }
    >
      {item && (
        <div style={{ position: "relative" }}>
          <div className="viewer" ref={box}>
            {shown &&
              (w > 0 && shown.el ? (
                <canvas ref={setCanvasEl} style={{ width: w, height: h, display: "block" }} />
              ) : (
                // 量不到尺寸时退回按容器自动缩放，保证一定能看到图
                <img
                  key={shown.src}
                  src={shown.src}
                  alt={item.name}
                  draggable={false}
                  style={{ maxWidth: "100%", maxHeight: "100%" }}
                />
              ))}
            {loading && (
              <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", pointerEvents: "none" }}>
                <Spin />
              </div>
            )}
          </div>
          <Button
            shape="circle"
            icon={<LeftOutlined />}
            disabled={index === 0}
            onClick={() => onIndex(index! - 1)}
            style={{ position: "absolute", left: 12, top: "50%" }}
          />
          <Button
            shape="circle"
            icon={<RightOutlined />}
            disabled={index === items.length - 1}
            onClick={() => onIndex(index! + 1)}
            style={{ position: "absolute", right: 12, top: "50%" }}
          />
        </div>
      )}
    </Modal>
  );
}
