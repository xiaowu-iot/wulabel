import { useState } from "react";
import { App, Button, Form, Input, Modal, Radio, Space } from "antd";
import { FolderOpenOutlined } from "@ant-design/icons";
import { open } from "@tauri-apps/plugin-dialog";
import { api, errText, Task, TASKS } from "../api";
import { prefs, useApp } from "../store";

interface Values {
  name: string;
  parent: string;
  task: Task;
}

export default function NewProjectModal({ openState, onClose }: { openState: boolean; onClose: () => void }) {
  const [form] = Form.useForm<Values>();
  const [busy, setBusy] = useState(false);
  const { message } = App.useApp();
  const setProject = useApp((s) => s.setProject);

  const pickParent = async () => {
    const dir = await open({ directory: true, title: "选择项目存放位置" });
    if (typeof dir === "string") form.setFieldValue("parent", dir);
  };

  const submit = async () => {
    const v = await form.validateFields();
    setBusy(true);
    try {
      const p = await api.createProject(v.parent, v.name, v.task);
      prefs.set("lastParent", v.parent);
      message.success(`项目「${p.name}」已创建`);
      onClose();
      form.resetFields();
      setProject(p);
    } catch (e) {
      message.error(errText(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      title="新建项目"
      open={openState}
      onCancel={onClose}
      onOk={submit}
      okText="创建"
      confirmLoading={busy}
      destroyOnHidden
      width={560}
    >
      <Form
        form={form}
        layout="vertical"
        requiredMark={false}
        initialValues={{ task: "detect", parent: prefs.get("lastParent", "") }}
        style={{ marginTop: 16 }}
      >
        <Form.Item name="name" label="项目名" rules={[{ required: true, message: "请输入项目名" }]}>
          <Input placeholder="例如：工地安全帽检测" autoFocus maxLength={60} />
        </Form.Item>
        <Form.Item label="存放位置" required>
          <Space.Compact style={{ width: "100%" }}>
            <Form.Item name="parent" noStyle rules={[{ required: true, message: "请选择存放位置" }]}>
              <Input placeholder="项目会建在这个文件夹下" />
            </Form.Item>
            <Button icon={<FolderOpenOutlined />} onClick={pickParent}>
              浏览
            </Button>
          </Space.Compact>
        </Form.Item>
        <Form.Item name="task" label="任务类型" extra="创建后不能修改。">
          <Radio.Group style={{ width: "100%" }}>
            <Space direction="vertical" style={{ width: "100%" }}>
              {TASKS.map((t) => (
                <Radio key={t.value} value={t.value}>
                  <b>{t.name}</b> <span className="muted">· {t.desc}</span>
                </Radio>
              ))}
            </Space>
          </Radio.Group>
        </Form.Item>
      </Form>
    </Modal>
  );
}
