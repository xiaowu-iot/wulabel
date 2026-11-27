import { useEffect, useState } from "react";
import { Alert, App, Button, Checkbox, Form, Input, InputNumber, Modal, Select, Spin } from "antd";
import { FolderOpenOutlined } from "@ant-design/icons";
import { open } from "@tauri-apps/plugin-dialog";
import { api, errText, Label, OnnxInfo, taskName } from "../api";
import { t } from "../i18n";

const KINDS = [
  { value: "yolov8", label: t("YOLOv8 / v11（Ultralytics）") },
  { value: "yolov5", label: "YOLOv5 / v7" },
  { value: "yolo_e2e", label: t("YOLOv10 等端到端（模型里已做 NMS）") },
  { value: "yolox", label: t("YOLOX（官方导出）") },
  { value: "rtdetr", label: t("RT-DETR（Ultralytics）") },
  { value: "detr", label: "DETR / RF-DETR" },
  { value: "wulabel", label: t("Wulabel 导出的模型") },
];

/** 导入别处训练好的 ONNX 检测模型，用来预标注、体检、评估、误报测试 */
export default function ImportModelModal({ open: visible, onClose, onDone }: { open: boolean; onClose: () => void; onDone: (runId: number) => void }) {
  const { message } = App.useApp();
  const [path, setPath] = useState("");
  const [info, setInfo] = useState<OnnxInfo | null>(null);
  const [probing, setProbing] = useState(false);
  const [name, setName] = useState("");
  const [kind, setKind] = useState("yolov8");
  const [classText, setClassText] = useState("");
  const [imgSize, setImgSize] = useState(640);
  const [labels, setLabels] = useState<Label[]>([]);
  const [use, setUse] = useState<number[]>([]);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!visible) return;
    setPath("");
    setInfo(null);
    api.listLabels().then(setLabels).catch(() => {});
  }, [visible]);

  const classes = classText.split("\n").map((s) => s.trim()).filter(Boolean);
  const known = new Set(labels.map((l) => l.name));

  // 默认只用和项目类别同名的；一个都对不上时（比如项目还没建类别）全用
  const defaultUse = (cs: string[]) => {
    const hit = cs.map((c, i) => (known.has(c) ? i : -1)).filter((i) => i >= 0);
    return hit.length ? hit : cs.map((_, i) => i);
  };

  const pick = async () => {
    const f = await open({ title: t("选择 ONNX 模型"), filters: [{ name: t("ONNX 模型"), extensions: ["onnx"] }] });
    if (typeof f !== "string") return;
    setPath(f);
    setProbing(true);
    setInfo(null);
    try {
      const i = await api.inspectOnnx(f);
      setInfo(i);
      setKind(i.kind);
      setImgSize(i.imgSize || 640);
      setClassText(i.classes.join("\n"));
      setUse(defaultUse(i.classes));
      setName(f.split(/[\\/]/).pop()?.replace(/\.onnx$/i, "") ?? t("外部模型"));
    } catch (e) {
      message.error(errText(e));
    } finally {
      setProbing(false);
    }
  };

  const save = async () => {
    if (!path || classes.length === 0) return;
    setSaving(true);
    try {
      const all = use.length === classes.length;
      const id = await api.importOnnx(path, {
        name: name.trim() || t("外部模型"),
        kind,
        classes,
        imgSize,
        task: info?.task ?? "detect",
        kptShape: info?.kptShape ?? null,
        useClasses: all ? null : use.filter((i) => i < classes.length),
      });
      message.success(t("已导入，可以用它预标注、评估，或到部署页使用"));
      onDone(id);
      onClose();
    } catch (e) {
      message.error(errText(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal
      title={t("导入外部模型（ONNX）")}
      open={visible}
      onCancel={onClose}
      onOk={save}
      okText={t("导入")}
      cancelText={t("取消")}
      okButtonProps={{ disabled: !info || classes.length === 0, loading: saving }}
      width={640}
      destroyOnHidden
    >
      <p className="muted" style={{ marginTop: 0 }}>
        {t("在别处（比如 Ultralytics YOLO）训好的模型导出成 ONNX 后导进来，就能像训练出的模型一样预标注、评估、实时验证和部署。检测、分割、关键点、分类都支持，任务要和项目一致。")}
      </p>
      <Button icon={<FolderOpenOutlined />} onClick={pick} loading={probing}>
        {path ? t("换一个模型") : t("选择 .onnx 文件")}
      </Button>
      {path && <div className="dim" style={{ fontSize: 12, marginTop: 6, wordBreak: "break-all" }}>{path}</div>}
      {probing && (
        <div style={{ padding: 24, textAlign: "center" }}>
          <Spin />
        </div>
      )}
      {info && (
        <Form layout="vertical" size="small" style={{ marginTop: 12 }}>
          {info.note && <Alert type="info" showIcon style={{ marginBottom: 10 }} message={info.note} />}
          <div className="row" style={{ gap: 12, alignItems: "flex-start" }}>
            <Form.Item label={t("名称")} style={{ flex: 1 }}>
              <Input value={name} maxLength={60} onChange={(e) => setName(e.target.value)} />
            </Form.Item>
            <Form.Item label={t("格式")} tooltip={t("已按模型输出自动识别，不对的话手动改")} style={{ width: 230 }}>
              <Select value={kind} onChange={setKind} options={KINDS} />
            </Form.Item>
            <Form.Item label={t("输入尺寸")} style={{ width: 100 }}>
              <InputNumber min={32} max={4096} step={32} value={imgSize} onChange={(v) => setImgSize(Number(v ?? 640))} />
            </Form.Item>
          </div>
          <div className="dim" style={{ fontSize: 12, marginBottom: 8 }}>
            {taskName(info.task)} · {t("输出：{list}", { list: info.outputs.map((o) => `${o.name} [${o.shape.join(", ")}]`).join(t("；")) })}
          </div>
          <Form.Item
            label={
              info.numClasses
                ? t("类别名（每行一个，顺序要和模型一致；共 {n} 个，模型输出 {k} 类）", { n: classes.length, k: info.numClasses })
                : t("类别名（每行一个，顺序要和模型一致；共 {n} 个）", { n: classes.length })
            }
            extra={t("改成项目里的类别名就能对应上，比如把 person 改成「人」")}
          >
            <Input.TextArea
              rows={6}
              value={classText}
              onChange={(e) => {
                setClassText(e.target.value);
                const cs = e.target.value.split("\n").map((s) => s.trim()).filter(Boolean);
                setUse(defaultUse(cs));
              }}
            />
          </Form.Item>
          <Form.Item label={t("要用的类别")} extra={t("没勾的类别预标注时不会加进来；和项目同名的类别默认勾上")}>
            <div style={{ maxHeight: 140, overflow: "auto" }}>
              <Checkbox.Group
                value={use}
                onChange={(v) => setUse(v as number[])}
                options={classes.map((c, i) => ({ value: i, label: known.has(c) ? `${c} ✓` : c }))}
              />
            </div>
          </Form.Item>
        </Form>
      )}
    </Modal>
  );
}
