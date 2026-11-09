import { useEffect, useState } from "react";
import { App, Button, Checkbox, Form, Input, InputNumber, Modal, Progress, Radio, Space } from "antd";
import { FolderOpenOutlined } from "@ant-design/icons";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import { api, DatasetFormat, errText, ImportProgress, ProjectInfo, SplitSummary } from "../api";
import { prefs } from "../store";

const FORMATS: { value: DatasetFormat; name: string; desc: string }[] = [
  { value: "yolo", name: "YOLO", desc: "images + labels + data.yaml，Ultralytics 等直接可用" },
  { value: "coco", name: "COCO", desc: "annotations/instances_*.json，通用格式" },
  { value: "voc", name: "VOC", desc: "Annotations/*.xml + JPEGImages" },
];

const stamp = () => {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}`;
};

export default function ExportModal({ project, open: visible, onClose }: { project: ProjectInfo; open: boolean; onClose: () => void }) {
  const { message, modal } = App.useApp();
  const [format, setFormat] = useState<DatasetFormat>((prefs.get("exportFormat", "yolo") as DatasetFormat) || "yolo");
  const [onlyReviewed, setOnlyReviewed] = useState(false);
  const [ratios, setRatios] = useState<[number, number, number]>([80, 20, 0]);
  const [parent, setParent] = useState(prefs.get("exportParent", ""));
  const [name, setName] = useState("");
  const [summary, setSummary] = useState<SplitSummary | null>(null);
  const [progress, setProgress] = useState<ImportProgress | null>(null);

  useEffect(() => {
    if (!visible) return;
    setName(`${project.name}_${format}_${stamp()}`);
    api.splitSummary().then(setSummary).catch(() => {});
  }, [visible, project.name, format]);

  const total = summary ? summary.train + summary.val + summary.test + summary.unset : 0;
  const sep = parent.includes("\\") ? "\\" : "/";
  const outDir = parent ? `${parent.replace(/[\\/]+$/, "")}${sep}${name}` : "";
  const r = (): [number, number, number] => [ratios[0] / 100, ratios[1] / 100, ratios[2] / 100];

  const resplit = () => {
    modal.confirm({
      title: "按当前比例重新划分全部图片？",
      content: "已有的训练/验证/测试划分会被打乱重排。之前训练的结果将和新划分不可直接比较。",
      okText: "重新划分",
      cancelText: "取消",
      onOk: async () => {
        try {
          setSummary(await api.assignSplits(r(), true));
          message.success("已重新划分");
        } catch (e) {
          message.error(errText(e));
        }
      },
    });
  };

  const run = async () => {
    if (!parent) {
      message.warning("请选择导出位置");
      return;
    }
    if (!name.trim()) {
      message.warning("请填写文件夹名");
      return;
    }
    setProgress({ done: 0, total: 0 });
    const un = await listen<ImportProgress>("export-progress", (e) => setProgress(e.payload));
    try {
      const res = await api.exportDataset({ format, outDir, onlyReviewed, ratios: r() });
      prefs.set("exportFormat", format);
      prefs.set("exportParent", parent);
      onClose();
      modal.success({
        title: "导出完成",
        content: (
          <div>
            <div>
              {res.images} 张图片、{res.boxes} 个框、{res.classes.length} 个类别
            </div>
            <div className="muted">
              训练 {res.train} · 验证 {res.val} · 测试 {res.test}
            </div>
            <div className="dim" style={{ wordBreak: "break-all", marginTop: 6 }}>
              {res.outDir}
            </div>
          </div>
        ),
      });
    } catch (e) {
      message.error(errText(e));
    } finally {
      un();
      setProgress(null);
    }
  };

  return (
    <Modal
      title="导出数据集"
      open={visible}
      onCancel={onClose}
      onOk={run}
      okText="导出"
      cancelText="取消"
      confirmLoading={!!progress}
      width={600}
      destroyOnHidden
    >
      <Form layout="vertical" style={{ marginTop: 12 }}>
        <Form.Item label="格式">
          <Radio.Group value={format} onChange={(e) => setFormat(e.target.value)}>
            <Space direction="vertical">
              {FORMATS.map((f) => (
                <Radio key={f.value} value={f.value}>
                  <b>{f.name}</b> <span className="muted">· {f.desc}</span>
                </Radio>
              ))}
            </Space>
          </Radio.Group>
        </Form.Item>
        <Form.Item label="范围" extra="未标注和已跳过的图片不导出；已审核但没有框的图片会作为负样本导出。">
          <Checkbox checked={onlyReviewed} onChange={(e) => setOnlyReviewed(e.target.checked)}>
            只导出已审核的图片
          </Checkbox>
        </Form.Item>
        <Form.Item
          label="训练 / 验证 / 测试 比例（%）"
          extra={
            summary && (
              <span>
                当前可导出 {total} 张：训练 {summary.train} · 验证 {summary.val} · 测试 {summary.test}
                {summary.unset > 0 && ` · 未划分 ${summary.unset}（导出时按上面的比例分配）`}
                {total > 0 && (
                  <Button type="link" size="small" onClick={resplit}>
                    全部重新划分
                  </Button>
                )}
              </span>
            )
          }
        >
          <Space>
            {[0, 1, 2].map((i) => (
              <InputNumber
                key={i}
                min={i === 0 ? 1 : 0}
                max={100}
                value={ratios[i]}
                addonBefore={["训练", "验证", "测试"][i]}
                style={{ width: 150 }}
                onChange={(v) => {
                  const next = [...ratios] as [number, number, number];
                  next[i] = Number(v ?? 0);
                  setRatios(next);
                }}
              />
            ))}
          </Space>
        </Form.Item>
        <Form.Item label="导出到" extra={outDir && <span style={{ wordBreak: "break-all" }}>{outDir}</span>}>
          <Space.Compact style={{ width: "100%" }}>
            <Input value={parent} placeholder="选择存放位置" onChange={(e) => setParent(e.target.value)} />
            <Button
              icon={<FolderOpenOutlined />}
              onClick={async () => {
                const d = await open({ directory: true, title: "选择导出位置" });
                if (typeof d === "string") setParent(d);
              }}
            >
              浏览
            </Button>
          </Space.Compact>
          <Input style={{ marginTop: 8 }} value={name} onChange={(e) => setName(e.target.value)} addonBefore="文件夹名" />
        </Form.Item>
      </Form>
      {progress && (
        <Progress percent={progress.total ? Math.floor((progress.done / progress.total) * 100) : 0} size="small" />
      )}
    </Modal>
  );
}
