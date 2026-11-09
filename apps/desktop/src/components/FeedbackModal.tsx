import { useState } from "react";
import { App, Button, Checkbox, Image, Input, Modal, Space } from "antd";
import { api, errText } from "../api";

export default function FeedbackModal({
  open,
  shot,
  onClose,
}: {
  open: boolean;
  shot: string | null;
  onClose: () => void;
}) {
  const { message } = App.useApp();
  const [text, setText] = useState("");
  const [contact, setContact] = useState("");
  const [withShot, setWithShot] = useState(true);
  const [sending, setSending] = useState(false);

  const close = () => {
    if (!sending) onClose();
  };

  const submit = async () => {
    if (!text.trim()) {
      message.warning("请先填写问题描述");
      return;
    }
    setSending(true);
    try {
      await api.submitFeedback({
        message: text.trim(),
        contact: contact.trim() || undefined,
        screenshot: withShot ? shot : null,
      });
      message.success("反馈已提交，谢谢！");
      setText("");
      setContact("");
      onClose();
    } catch (e) {
      message.error(errText(e));
    } finally {
      setSending(false);
    }
  };

  return (
    <Modal
      open={open}
      title="反馈问题"
      onCancel={close}
      maskClosable={!sending}
      footer={[
        <Button key="cancel" onClick={close} disabled={sending}>
          取消
        </Button>,
        <Button key="send" type="primary" loading={sending} onClick={submit}>
          发送
        </Button>,
      ]}
    >
      <Space direction="vertical" style={{ width: "100%" }} size={12}>
        <div className="dim" style={{ fontSize: 13 }}>
          遇到问题或有建议都可以告诉我。描述、截图（可选）以及软件版本、操作系统会一起发给开发者用于排查，不会上传你的项目图片本身。
        </div>
        <Input.TextArea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={5}
          maxLength={2000}
          showCount
          placeholder="说说遇到了什么问题，或者你希望增加什么…"
          autoFocus
        />
        <Input
          value={contact}
          onChange={(e) => setContact(e.target.value)}
          maxLength={120}
          placeholder="联系方式（选填，方便回复你，如邮箱 / 微信）"
        />
        {shot ? (
          <div style={{ border: "1px solid #eee", borderRadius: 6, padding: 8 }}>
            <Checkbox checked={withShot} onChange={(e) => setWithShot(e.target.checked)} style={{ marginBottom: withShot ? 8 : 0 }}>
              附带当前界面截图
            </Checkbox>
            {withShot && (
              <div style={{ textAlign: "center" }}>
                <Image src={shot} alt="界面截图预览" style={{ maxHeight: 180, objectFit: "contain" }} />
              </div>
            )}
          </div>
        ) : (
          <div className="dim" style={{ fontSize: 12 }}>（这次没能自动截到界面，直接文字描述即可）</div>
        )}
      </Space>
    </Modal>
  );
}
