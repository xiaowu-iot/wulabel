import { useCallback, useEffect, useRef, useState } from "react";
import type Konva from "konva";
import { Image as KImage, Label as KLabel, Layer, Line, Rect, Stage, Tag, Text, Transformer } from "react-konva";

export interface Box {
  key: string;
  labelId: number | null;
  x: number;
  y: number;
  w: number;
  h: number;
  source: string;
  score?: number | null;
}

export type Tool = "select" | "rect";
export type RectXYWH = { x: number; y: number; w: number; h: number };

interface Props {
  src: string;
  imgW: number;
  imgH: number;
  boxes: Box[];
  colorOf: (labelId: number | null) => string;
  nameOf: (labelId: number | null) => string;
  selected: string | null;
  tool: Tool;
  drawColor: string;
  showNames: boolean;
  /** 数字变化时重新适应窗口 */
  fitSignal: number;
  onSelect: (key: string | null) => void;
  onCreate: (r: RectXYWH) => void;
  onUpdate: (key: string, r: RectXYWH) => void;
  onScale?: (scale: number) => void;
  /** 体检跳过来时高亮的问题框 */
  highlight?: { bbox: [number, number, number, number]; text: string } | null;
}

const MIN_SIZE = 2; // 原图像素，小于这个尺寸的框视为误点

export function rgba(hex: string, a: number) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

const round = (v: number) => Math.round(v * 100) / 100;

/**
 * 标注画布。所有坐标都是原图像素坐标，缩放平移只改舞台的变换。
 * 画框过程中的草稿框和十字线直接改 Konva 节点，不走 React 渲染，保证流畅。
 */
export default function AnnotCanvas(p: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const trRef = useRef<Konva.Transformer>(null);
  const draftRef = useRef<Konva.Rect>(null);
  const hLine = useRef<Konva.Line>(null);
  const vLine = useRef<Konva.Line>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [view, setView] = useState({ x: 0, y: 0, s: 1 });
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [spaceHeld, setSpaceHeld] = useState(false);
  const [panning, setPanning] = useState(false);
  const drawStart = useRef<{ x: number; y: number } | null>(null);
  const pan = useRef<{ cx: number; cy: number; vx: number; vy: number } | null>(null);

  // 容器尺寸
  useEffect(() => {
    const el = wrap.current!;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // 加载图片；新图加载完之前不显示旧图，避免尺寸错位
  useEffect(() => {
    let alive = true;
    const im = new window.Image();
    im.onload = () => alive && setImg(im);
    im.src = p.src;
    return () => {
      alive = false;
    };
  }, [p.src]);

  const fit = useCallback(() => {
    if (!size.w || !size.h || !p.imgW || !p.imgH) return;
    const s = Math.min(size.w / p.imgW, size.h / p.imgH) * 0.96;
    setView({ s, x: (size.w - p.imgW * s) / 2, y: (size.h - p.imgH * s) / 2 });
  }, [size.w, size.h, p.imgW, p.imgH]);

  useEffect(fit, [fit, p.src, p.fitSignal]);
  useEffect(() => p.onScale?.(view.s), [view.s, p.onScale]);

  // 按住空格拖动画面
  useEffect(() => {
    const isTyping = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      return t?.tagName === "INPUT" || t?.tagName === "TEXTAREA" || !!t?.closest?.(".ant-modal");
    };
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space" && !isTyping(e)) {
        e.preventDefault();
        setSpaceHeld(true);
      }
    };
    const up = (e: KeyboardEvent) => e.code === "Space" && setSpaceHeld(false);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  const clampPt = (pt: { x: number; y: number }) => ({
    x: Math.min(Math.max(pt.x, 0), p.imgW),
    y: Math.min(Math.max(pt.y, 0), p.imgH),
  });

  const pointer = () => stageRef.current?.getRelativePointerPosition() ?? null;

  const finishDraw = useCallback(() => {
    const d = draftRef.current;
    if (!drawStart.current || !d) return;
    drawStart.current = null;
    const r = { x: d.x(), y: d.y(), w: d.width(), h: d.height() };
    d.visible(false);
    d.getLayer()?.batchDraw();
    if (r.w >= MIN_SIZE && r.h >= MIN_SIZE) p.onCreate({ x: round(r.x), y: round(r.y), w: round(r.w), h: round(r.h) });
  }, [p]);

  // 鼠标在画布外松开也要结束画框和拖动
  useEffect(() => {
    const up = () => {
      if (pan.current) {
        pan.current = null;
        setPanning(false);
      }
      finishDraw();
    };
    window.addEventListener("mouseup", up);
    return () => window.removeEventListener("mouseup", up);
  }, [finishDraw]);

  const onMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    const evt = e.evt;
    if (evt.button === 1 || evt.button === 2 || spaceHeld) {
      evt.preventDefault();
      pan.current = { cx: evt.clientX, cy: evt.clientY, vx: view.x, vy: view.y };
      setPanning(true);
      return;
    }
    if (evt.button !== 0) return;
    if (p.tool === "rect") {
      const pt = pointer();
      const d = draftRef.current;
      if (!pt || !d) return;
      const c = clampPt(pt);
      drawStart.current = c;
      d.setAttrs({ x: c.x, y: c.y, width: 0, height: 0, visible: true });
      return;
    }
    // 选择工具：点空白处取消选中
    if (e.target === e.target.getStage() || e.target.name() === "bg") p.onSelect(null);
  };

  const onMouseMove = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (pan.current) {
      const { cx, cy, vx, vy } = pan.current;
      setView((v) => ({ ...v, x: vx + e.evt.clientX - cx, y: vy + e.evt.clientY - cy }));
      return;
    }
    const pt = pointer();
    if (!pt) return;
    if (p.tool === "rect" && hLine.current && vLine.current) {
      const c = clampPt(pt);
      hLine.current.setAttrs({ points: [0, c.y, p.imgW, c.y], visible: true });
      vLine.current.setAttrs({ points: [c.x, 0, c.x, p.imgH], visible: true });
    }
    if (drawStart.current && draftRef.current) {
      const c = clampPt(pt);
      const s = drawStart.current;
      draftRef.current.setAttrs({
        x: Math.min(s.x, c.x),
        y: Math.min(s.y, c.y),
        width: Math.abs(c.x - s.x),
        height: Math.abs(c.y - s.y),
      });
    }
    hLine.current?.getLayer()?.batchDraw();
  };

  const onMouseLeave = () => {
    hLine.current?.visible(false);
    vLine.current?.visible(false);
    hLine.current?.getLayer()?.batchDraw();
  };

  const onWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const pos = stageRef.current?.getPointerPosition();
    if (!pos) return;
    const old = view.s;
    const s = Math.min(Math.max(old * (e.evt.deltaY > 0 ? 1 / 1.15 : 1.15), 0.02), 60);
    const mx = (pos.x - view.x) / old;
    const my = (pos.y - view.y) / old;
    setView({ s, x: pos.x - mx * s, y: pos.y - my * s });
  };

  // 选中的框挂上调整手柄
  useEffect(() => {
    const tr = trRef.current;
    if (!tr) return;
    const node = p.selected && p.tool === "select" ? stageRef.current?.findOne(`#${p.selected}`) : null;
    tr.nodes(node ? [node] : []);
    tr.getLayer()?.batchDraw();
  }, [p.selected, p.tool, p.boxes]);

  useEffect(() => {
    if (p.tool !== "rect") onMouseLeave();
  }, [p.tool]);

  const s = view.s;
  const fontSize = 12 / s;
  // 大框先画，小框在上面，方便点中
  const ordered = [...p.boxes].sort((a, b) => b.w * b.h - a.w * a.h);
  const cursor = panning ? "grabbing" : spaceHeld ? "grab" : p.tool === "rect" ? "crosshair" : "default";
  const imgReady = img && img.src === p.src;

  return (
    <div ref={wrap} style={{ position: "absolute", inset: 0, cursor }} onContextMenu={(e) => e.preventDefault()}>
      {size.w > 0 && (
        <Stage
          ref={stageRef}
          width={size.w}
          height={size.h}
          x={view.x}
          y={view.y}
          scaleX={s}
          scaleY={s}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseLeave={onMouseLeave}
          onWheel={onWheel}
        >
          <Layer>
            {imgReady ? (
              <KImage name="bg" image={img!} width={p.imgW} height={p.imgH} />
            ) : (
              <Rect name="bg" width={p.imgW} height={p.imgH} fill="#0e1013" />
            )}
            {ordered.map((b) => {
              const c = p.colorOf(b.labelId);
              const sel = b.key === p.selected;
              const editable = p.tool === "select" && !spaceHeld;
              return (
                <Rect
                  key={b.key}
                  id={b.key}
                  x={b.x}
                  y={b.y}
                  width={b.w}
                  height={b.h}
                  stroke={c}
                  strokeWidth={sel ? 2.5 : 1.5}
                  strokeScaleEnabled={false}
                  fill={rgba(c, sel ? 0.22 : 0.08)}
                  dash={b.source === "manual" ? undefined : [6, 3]}
                  listening={editable}
                  draggable={editable}
                  onMouseDown={(e) => {
                    if (e.evt.button !== 0) return;
                    e.cancelBubble = true;
                    p.onSelect(b.key);
                  }}
                  onDragMove={(e) => {
                    const n = e.target;
                    n.x(Math.min(Math.max(n.x(), 0), p.imgW - b.w));
                    n.y(Math.min(Math.max(n.y(), 0), p.imgH - b.h));
                  }}
                  onDragEnd={(e) => p.onUpdate(b.key, { x: round(e.target.x()), y: round(e.target.y()), w: b.w, h: b.h })}
                  onTransformEnd={(e) => {
                    const n = e.target;
                    let x = Math.max(n.x(), 0);
                    let y = Math.max(n.y(), 0);
                    let w = n.width() * n.scaleX();
                    let h = n.height() * n.scaleY();
                    n.scaleX(1);
                    n.scaleY(1);
                    w = Math.min(w, p.imgW - x);
                    h = Math.min(h, p.imgH - y);
                    x = round(x);
                    y = round(y);
                    p.onUpdate(b.key, { x, y, w: round(Math.max(w, MIN_SIZE)), h: round(Math.max(h, MIN_SIZE)) });
                  }}
                />
              );
            })}
            {p.showNames &&
              p.boxes.map((b) => {
                const c = p.colorOf(b.labelId);
                const inside = b.y < fontSize * 1.6;
                const text = p.nameOf(b.labelId) + (b.score != null ? ` ${b.score.toFixed(2)}` : "");
                return (
                  <KLabel key={`t${b.key}`} x={b.x} y={inside ? b.y : b.y - fontSize * 1.5} listening={false}>
                    <Tag fill={c} opacity={0.9} />
                    <Text text={text} fontSize={fontSize} padding={fontSize * 0.2} fill="#fff" />
                  </KLabel>
                );
              })}
            <Transformer
              ref={trRef}
              rotateEnabled={false}
              flipEnabled={false}
              keepRatio={false}
              ignoreStroke
              anchorSize={8}
              anchorStroke="#3d8bfd"
              anchorFill="#fff"
              borderStroke="#ffffff"
              borderDash={[4, 3]}
              boundBoxFunc={(oldB, newB) => {
                // 手柄坐标是屏幕坐标，换算回原图范围检查
                const x0 = view.x - 0.5;
                const y0 = view.y - 0.5;
                const x1 = view.x + p.imgW * s + 0.5;
                const y1 = view.y + p.imgH * s + 0.5;
                if (newB.width < MIN_SIZE * s || newB.height < MIN_SIZE * s) return oldB;
                if (newB.x < x0 || newB.y < y0 || newB.x + newB.width > x1 || newB.y + newB.height > y1) return oldB;
                return newB;
              }}
            />
          </Layer>
          <Layer listening={false}>
            {p.highlight && (
              <>
                <Rect
                  x={p.highlight.bbox[0] - 3 / s}
                  y={p.highlight.bbox[1] - 3 / s}
                  width={p.highlight.bbox[2] + 6 / s}
                  height={p.highlight.bbox[3] + 6 / s}
                  stroke="#ffd43b"
                  strokeWidth={2.5}
                  strokeScaleEnabled={false}
                  dash={[8, 5]}
                />
                <KLabel x={p.highlight.bbox[0] - 3 / s} y={p.highlight.bbox[1] + p.highlight.bbox[3] + 5 / s}>
                  <Tag fill="#ffd43b" />
                  <Text text={`体检：${p.highlight.text}`} fontSize={fontSize} padding={fontSize * 0.25} fill="#1b1e23" />
                </KLabel>
              </>
            )}
            <Rect
              ref={draftRef}
              visible={false}
              stroke={p.drawColor}
              strokeWidth={2}
              strokeScaleEnabled={false}
              fill={rgba(p.drawColor, 0.15)}
            />
            <Line ref={hLine} visible={false} stroke="rgba(255,255,255,.55)" strokeWidth={1} strokeScaleEnabled={false} dash={[4, 4]} />
            <Line ref={vLine} visible={false} stroke="rgba(255,255,255,.55)" strokeWidth={1} strokeScaleEnabled={false} dash={[4, 4]} />
          </Layer>
        </Stage>
      )}
    </div>
  );
}
