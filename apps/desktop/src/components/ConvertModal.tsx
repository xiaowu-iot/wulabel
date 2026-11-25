import { useEffect, useState } from "react";
import { Alert, App, Button, Checkbox, Form, Input, Modal, Progress, Space } from "antd";
import { FolderOpenOutlined } from "@ant-design/icons";
import { open } from "@tauri-apps/plugin-dialog";
import { dirname } from "@tauri-apps/api/path";
import { listen } from "@tauri-apps/api/event";
import { api, errText, ImportProgress, ProjectInfo } from "../api";
import { useApp } from "../store";
import { t } from "../i18n";

interface Values {
  name: string;
  parent: string;
  keepEmpty: boolean;
  emptyLabel: string;
}

/** 检测、分割、关键点项目转成多标签分类项目：每张图标过的类别就是它的标签，新建项目、原项目不动 */
export default function ConvertModal({ project, open: show, onClose }: { project: ProjectInfo; open: boolean; onClose: () => void }) {
  const [form] = Form.useForm<Values>();
  const { message, modal } = App.useApp();
  const setProject = useApp((s) => s.setProject);
  const [busy, setBusy] = useState(false);
  const [prog, setProg] = useState<ImportProgress | null>(null);
  const keepEmpty = Form.useWatch("keepEmpty", form);

  // 默认放在当前项目旁边
  useEffect(() => {
    if (!show) return;
    form.setFieldsValue({ name: t("{name}-多标签", { name: project.name }), keepEmpty: true, emptyLabel: t("正常") });
    dirname(project.path)
      .then((d) => form.setFieldValue("parent", d))
      .catch(() => {});
  }, [show, project, form]);

  const pickParent = async () => {
    const dir = await open({ directory: true, title: t("选择项目存放位置") });
    if (typeof dir === "string") form.setFieldValue("parent", dir);
  };

  const submit = async () => {
    const v = await form.validateFields();
    setBusy(true);
    setProg(null);
    const un = await listen<ImportProgress>("import-progress", (e) => setProg(e.payload));
    try {
      const r = await api.convertToMultilabel(v.parent, v.name.trim(), v.keepEmpty ? v.emptyLabel.trim() : null);
      onClose();
      modal.confirm({
        title: t("已生成多标签分类项目"),
        content:
          r.negatives > 0
            ? t("{images} 张图片、{labels} 个类别，其中 {neg} 张没有目标的图打上了「{empty}」。", { images: r.images, labels: r.labels, neg: r.negatives, empty: v.emptyLabel.trim() })
            : t("{images} 张图片、{labels} 个类别。", { images: r.images, labels: r.labels }),
        okText: t("打开新项目"),
        cancelText: t("留在这里"),
        onOk: async () => {
          try {
            setProject(await api.openProject(r.path));
          } catch (e) {
            message.error(errText(e));
          }
        },
      });
    } catch (e) {
      message.error(errText(e));
    } finally {
      un();
      setBusy(false);
    }
  };

  return (
    <Modal
      title={t("转成多标签分类项目")}
      open={show}
      onCancel={busy ? undefined : onClose}
      onOk={submit}
      okText={t("开始转换")}
      confirmLoading={busy}
      destroyOnHidden
      width={560}
    >
      <Alert
        type="info"
        showIcon
        style={{ margin: "12px 0" }}
        message={t("新建一个多标签分类项目：每张已标注的图，上面标过哪些类别，就打上哪些类别标签。图片复制过去，这个项目不会改动。")}
      />
      <Form form={form} layout="vertical" requiredMark={false} disabled={busy}>
        <Form.Item name="name" label={t("新项目名")} rules={[{ required: true, message: t("请输入项目名") }]}>
          <Input maxLength={60} />
        </Form.Item>
        <Form.Item label={t("存放位置")} required>
          <Space.Compact style={{ width: "100%" }}>
            <Form.Item name="parent" noStyle rules={[{ required: true, message: t("请选择存放位置") }]}>
              <Input placeholder={t("项目会建在这个文件夹下")} />
            </Form.Item>
            <Button icon={<FolderOpenOutlined />} onClick={pickParent}>
              {t("浏览")}
            </Button>
          </Space.Compact>
        </Form.Item>
        <Form.Item name="keepEmpty" valuePropName="checked" style={{ marginBottom: 8 }}>
          <Checkbox>{t("标为「没有目标」的图也要，打上这个类别：")}</Checkbox>
        </Form.Item>
        <Form.Item
          name="emptyLabel"
          rules={[{ required: !!keepEmpty, whitespace: true, message: t("请输入类别名") }]}
          extra={t("多标签分类每张图至少要有一个类别，没问题的图用「正常」「全新」这样的类别。")}
        >
          <Input disabled={!keepEmpty || busy} maxLength={40} style={{ width: 200 }} />
        </Form.Item>
      </Form>
      {busy && prog && prog.total > 0 && <Progress size="small" percent={Math.floor((prog.done / prog.total) * 100)} status="active" />}
    </Modal>
  );
}
