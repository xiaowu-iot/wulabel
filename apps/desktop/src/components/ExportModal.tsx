import { useEffect, useRef, useState } from "react";
import { App, Button, Checkbox, Form, Input, InputNumber, Modal, Progress, Radio, Space } from "antd";
import { FolderOpenOutlined } from "@ant-design/icons";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import { AnonymizeConfig, api, DatasetFormat, errText, ExportResult, isClsTask, ImportProgress, ProjectInfo, SplitSummary, Task, VersionInfo } from "../api";
import { prefs, useReview } from "../store";
import { t } from "../i18n";
import AnonymizeOptions from "./AnonymizeOptions";

/** 各任务能导出的格式 */
const FORMATS: Record<Task, { value: DatasetFormat; name: string; desc: string }[]> = {
  detect: [
    { value: "yolo", name: "YOLO", desc: t("images + labels + data.yaml，Ultralytics 等直接可用") },
    { value: "coco", name: "COCO", desc: t("annotations/instances_*.json，通用格式") },
    { value: "voc", name: "VOC", desc: "Annotations/*.xml + JPEGImages" },
  ],
  segment: [
    { value: "yolo", name: "YOLO-seg", desc: t("每行一个多边形（归一化），Ultralytics 分割直接可用") },
    { value: "coco", name: "COCO", desc: t("带 segmentation 多边形，通用格式") },
    { value: "voc", name: "VOC", desc: t("只有外接框（VOC 不支持多边形）") },
  ],
  pose: [
    { value: "yolo", name: "YOLO-pose", desc: t("框 + 关键点，data.yaml 带 kpt_shape 和 flip_idx") },
    { value: "coco", name: t("COCO 关键点"), desc: t("keypoints + skeleton，通用格式") },
    { value: "voc", name: "VOC", desc: t("只有目标框（VOC 不支持关键点）") },
  ],
  classify: [{ value: "cls", name: t("分类文件夹"), desc: t("train/类别/图片.jpg，ImageNet、YOLO-cls 通用") }],
  multilabel: [{ value: "cls", name: t("多标签清单"), desc: t("train/images/图片.jpg + annotations/train.json（每张图的 labels 列表）") }],
};

const summaryText = (task: Task, r: ExportResult) =>
  isClsTask(task)
    ? t("{images} 张图片、{classes} 个类别", { images: r.images, classes: r.classes.length })
    : task === "segment"
      ? r.boxes
        ? t("{images} 张图片、{polys} 个轮廓、{boxes} 个框、{classes} 个类别", { images: r.images, polys: r.polys, boxes: r.boxes, classes: r.classes.length })
        : t("{images} 张图片、{polys} 个轮廓、{classes} 个类别", { images: r.images, polys: r.polys, classes: r.classes.length })
      : task === "pose"
        ? t("{images} 张图片、{kpts} 个关键点目标、{classes} 个类别", { images: r.images, kpts: r.kpts, classes: r.classes.length })
        : t("{images} 张图片、{boxes} 个框、{classes} 个类别", { images: r.images, boxes: r.boxes, classes: r.classes.length });

const stamp = () => {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}`;
};

/** version 不为空时导出这个数据集版本（划分固定，不能改比例） */
export default function ExportModal({
  project,
  open: visible,
  onClose,
  version = null,
}: {
  project: ProjectInfo;
  open: boolean;
  onClose: () => void;
  version?: VersionInfo | null;
}) {
  const { message, modal } = App.useApp();
  const formats = FORMATS[project.task];
  const [format, setFormat] = useState<DatasetFormat>(() => {
    const saved = prefs.get("exportFormat", "yolo") as DatasetFormat;
    return formats.some((f) => f.value === saved) ? saved : formats[0].value;
  });
  const [onlyReviewed, setOnlyReviewed] = useState(false);
  const review = useReview();
  const [ratios, setRatios] = useState<[number, number, number]>([80, 20, 0]);
  const [parent, setParent] = useState(prefs.get("exportParent", ""));
  const [name, setName] = useState("");
  const [summary, setSummary] = useState<SplitSummary | null>(null);
  const [progress, setProgress] = useState<ImportProgress | null>(null);
  const [anonymize, setAnonymize] = useState<AnonymizeConfig | null>(null);
  const [stage, setStage] = useState("");
  // 用户自己改过文件夹名，换格式时就不再覆盖
  const nameEdited = useRef(false);

  useEffect(() => {
    if (visible) nameEdited.current = false;
  }, [visible]);

  useEffect(() => {
    if (!visible) return;
    if (!nameEdited.current) {
      const tag = version ? `${version.name}_` : "";
      setName(`${project.name}_${tag}${format}_${stamp()}`.replace(/[\\/:*?"<>|]/g, "_"));
    }
    api.splitSummary().then(setSummary).catch(() => {});
  }, [visible, project.name, format, version]);

  const total = summary ? summary.train + summary.val + summary.test + summary.unset : 0;
  const sep = parent.includes("\\") ? "\\" : "/";
  const outDir = parent ? `${parent.replace(/[\\/]+$/, "")}${sep}${name}` : "";
  const r = (): [number, number, number] => [ratios[0] / 100, ratios[1] / 100, ratios[2] / 100];

  const resplit = () => {
    modal.confirm({
      title: t("按当前比例重新划分全部图片？"),
      content: t("已有的训练/验证/测试划分会被打乱重排。之前训练的结果将和新划分不可直接比较。"),
      okText: t("重新划分"),
      cancelText: t("取消"),
      onOk: async () => {
        try {
          setSummary(await api.assignSplits(r(), true));
          message.success(t("已重新划分"));
        } catch (e) {
          message.error(errText(e));
        }
      },
    });
  };

  const run = async () => {
    if (!parent) {
      message.warning(t("请选择导出位置"));
      return;
    }
    if (!name.trim()) {
      message.warning(t("请填写文件夹名"));
      return;
    }
    if (anonymize && !anonymize.labels.length && !anonymize.faces && !anonymize.plates) {
      message.warning(t("打码：请选择要打码的类别，或勾选自动找人脸 / 车牌"));
      return;
    }
    setProgress({ done: 0, total: 0 });
    setStage("");
    const un = await listen<ImportProgress>("export-progress", (e) => setProgress(e.payload));
    const unLog = await listen<string>("export-log", (e) => setStage(e.payload));
    try {
      const options = { format, outDir, onlyReviewed: review && onlyReviewed, ratios: r() };
      const res = version ? await api.exportVersion(version.id, options, anonymize) : await api.exportDataset(options, anonymize);
      prefs.set("exportFormat", format);
      prefs.set("exportParent", parent);
      onClose();
      modal.success({
        title: t("导出完成"),
        content: (
          <div>
            <div>{summaryText(project.task, res)}</div>
            <div className="muted">
              {t("训练 {train} · 验证 {val} · 测试 {test}", { train: res.train, val: res.val, test: res.test })}
            </div>
            <div className="dim" style={{ wordBreak: "break-all", marginTop: 6 }}>
              {res.outDir}
            </div>
            <Button size="small" icon={<FolderOpenOutlined />} style={{ marginTop: 8 }} onClick={() => api.revealPath(res.outDir).catch((e) => message.error(errText(e)))}>
              {t("打开文件夹")}
            </Button>
          </div>
        ),
      });
    } catch (e) {
      message.error(errText(e));
    } finally {
      un();
      unLog();
      setProgress(null);
      setStage("");
    }
  };

  return (
    <Modal
      title={version ? t("导出数据集版本「{name}」", { name: version.name }) : t("导出数据集")}
      open={visible}
      onCancel={onClose}
      onOk={run}
      okText={t("导出")}
      cancelText={t("取消")}
      confirmLoading={!!progress}
      width={600}
      destroyOnHidden
    >
      <Form layout="vertical" style={{ marginTop: 12 }}>
        <Form.Item label={t("格式")}>
          <Radio.Group value={format} onChange={(e) => setFormat(e.target.value)}>
            <Space direction="vertical">
              {formats.map((f) => (
                <Radio key={f.value} value={f.value}>
                  <b>{f.name}</b> <span className="muted">· {f.desc}</span>
                </Radio>
              ))}
            </Space>
          </Radio.Group>
        </Form.Item>
        <Form.Item
          label={t("范围")}
          extra={
            isClsTask(project.task)
              ? t("未标注和已跳过的图片不导出；没打类别的图片不导出。")
              : review
                ? t("未标注和已跳过的图片不导出；已审核但没有标注的图片会作为负样本导出。")
                : t("未标注和已跳过的图片不导出；标为「没有目标」的图片会作为负样本导出。")
          }
        >
          {review ? (
            <Checkbox checked={onlyReviewed} onChange={(e) => setOnlyReviewed(e.target.checked)}>
              {t("只导出审核通过的图片")}
            </Checkbox>
          ) : (
            <span className="muted">{t("已标注的图片")}</span>
          )}
        </Form.Item>
        {version ? (
          <Form.Item label={t("划分")}>
            <span className="muted">
              {t("版本里固定的划分：训练 {train} · 验证 {val} · 测试 {test}（共 {n} 张）", {
                train: version.train,
                val: version.val,
                test: version.test,
                n: version.images,
              })}
            </span>
          </Form.Item>
        ) : (
        <Form.Item
          label={t("训练 / 验证 / 测试 比例（%）")}
          extra={
            summary && (
              <span>
                {t("当前可导出 {total} 张：训练 {train} · 验证 {val} · 测试 {test}", { total, train: summary.train, val: summary.val, test: summary.test })}
                {summary.unset > 0 && t(" · 未划分 {n}（导出时按上面的比例分配）", { n: summary.unset })}
                {total > 0 && (
                  <Button type="link" size="small" onClick={resplit}>
                    {t("全部重新划分")}
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
                addonBefore={[t("训练"), t("验证"), t("测试")][i]}
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
        )}
        <Form.Item label={t("导出到")} extra={outDir && <span style={{ wordBreak: "break-all" }}>{outDir}</span>}>
          <Space.Compact style={{ width: "100%" }}>
            <Input value={parent} placeholder={t("选择存放位置")} onChange={(e) => setParent(e.target.value)} />
            <Button
              icon={<FolderOpenOutlined />}
              onClick={async () => {
                const d = await open({ directory: true, title: t("选择导出位置") });
                if (typeof d === "string") setParent(d);
              }}
            >
              {t("浏览")}
            </Button>
          </Space.Compact>
          <Input
            style={{ marginTop: 8 }}
            value={name}
            onChange={(e) => {
              nameEdited.current = true;
              setName(e.target.value);
            }}
            addonBefore={t("文件夹名")}
          />
        </Form.Item>
        <Form.Item label={t("打码")}>
          <AnonymizeOptions task={project.task} value={anonymize} onChange={setAnonymize} />
        </Form.Item>
      </Form>
      {progress && (
        <>
          <Progress percent={progress.total ? Math.floor((progress.done / progress.total) * 100) : 0} size="small" />
          {stage && <div className="dim" style={{ fontSize: 12 }}>{stage}</div>}
        </>
      )}
    </Modal>
  );
}
