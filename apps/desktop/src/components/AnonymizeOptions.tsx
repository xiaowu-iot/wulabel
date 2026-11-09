import { useEffect, useState } from "react";
import { Checkbox, Radio, Select, Space } from "antd";
import { AnonymizeConfig, api, Label, Task } from "../api";

/** 名字像人脸、车牌的类别默认勾上 */
const SENSITIVE = /脸|face|车牌|牌照|plate/i;

/**
 * 打码选项（导出数据集、分包时用）。value 为 null 表示不打码。
 * 打码只改导出 / 分包出去的副本，项目里的原图不动。
 */
export default function AnonymizeOptions({
  task,
  value,
  onChange,
}: {
  task: Task;
  value: AnonymizeConfig | null;
  onChange: (v: AnonymizeConfig | null) => void;
}) {
  const [labels, setLabels] = useState<Label[]>([]);
  // 关掉再打开时保留上次的设置
  const [last, setLast] = useState<AnonymizeConfig>({ labels: [], faces: true, plates: true, mode: "mosaic" });
  const hasBoxes = task !== "classify";

  useEffect(() => {
    if (!hasBoxes) return;
    api
      .listLabels()
      .then((ls) => {
        setLabels(ls);
        setLast((c) => (c.labels.length ? c : { ...c, labels: ls.filter((l) => SENSITIVE.test(l.name)).map((l) => l.id) }));
      })
      .catch(() => {});
  }, [hasBoxes]);

  const set = (patch: Partial<AnonymizeConfig>) => {
    const next = { ...(value ?? last), ...patch };
    setLast(next);
    onChange(next);
  };
  const cfg = value ?? last;
  const empty = value && value.labels.length === 0 && !value.faces && !value.plates;

  return (
    <Space direction="vertical" style={{ width: "100%" }}>
      <Checkbox checked={!!value} onChange={(e) => onChange(e.target.checked ? last : null)}>
        给图片打码（人脸、车牌等隐私区域；只改导出的副本，项目里的原图不动）
      </Checkbox>
      {value && (
        <div style={{ paddingLeft: 24 }}>
          <Space direction="vertical" style={{ width: "100%" }}>
            {hasBoxes && (
              <Select
                mode="multiple"
                allowClear
                style={{ width: "100%" }}
                placeholder="这些类别的标注区域打码（可不选）"
                value={cfg.labels}
                onChange={(ids) => set({ labels: ids })}
                options={labels.map((l) => ({ value: l.id, label: l.name }))}
              />
            )}
            <Space wrap>
              <Checkbox checked={cfg.faces} onChange={(e) => set({ faces: e.target.checked })}>
                自动找人脸
              </Checkbox>
              <Checkbox checked={cfg.plates} onChange={(e) => set({ plates: e.target.checked })}>
                自动找车牌
              </Checkbox>
            </Space>
            <Radio.Group value={cfg.mode} onChange={(e) => set({ mode: e.target.value })}>
              <Radio value="mosaic">马赛克</Radio>
              <Radio value="blur">模糊</Radio>
              <Radio value="black">涂黑</Radio>
            </Radio.Group>
            <div className="dim" style={{ fontSize: 12 }}>
              {empty
                ? "还没选要打码的内容"
                : "自动查找用开放词汇检测模型，第一次用要下载模型；图多时比较慢，有 GPU 会快很多。小的、侧脸、远处的车牌可能漏掉，重要数据请抽查。"}
            </div>
          </Space>
        </div>
      )}
    </Space>
  );
}
