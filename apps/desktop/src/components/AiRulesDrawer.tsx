// 识别规则：告诉巡瞳 AI 每个类别具体指什么、整体要注意什么；可以和 AI 聊，让它帮着整理规则、在当前图上试识别
import { useEffect, useRef, useState } from "react";
import { Alert, App, Button, Checkbox, Drawer, Empty, Input, Space, Tabs, Tag } from "antd";
import { SendOutlined } from "@ant-design/icons";
import { AiRules, api, ChatMsg, ChatReply, ClassRule, errText, PredResult } from "../api";

interface Props {
  open: boolean;
  onClose: () => void;
  /** 当前标注页上的图（聊天时可以附上） */
  imageId: number | null;
  /** 能不能改规则（协作成员可能没有权限） */
  canEdit: boolean;
  /** 把 AI 试识别的结果加到当前图 */
  onApplyPreview: (preds: PredResult) => void;
  /** 规则保存后（类别的识别说明变了）刷新类别列表 */
  onSaved: () => void;
}

interface Turn extends ChatMsg {
  suggestion?: ChatReply["rules"];
  preview?: PredResult | null;
  previewImageId?: number | null;
}

export default function AiRulesDrawer({ open, onClose, imageId, canEdit, onApplyPreview, onSaved }: Props) {
  const { message } = App.useApp();
  const [rules, setRules] = useState<AiRules | null>(null);
  const [draft, setDraft] = useState<AiRules | null>(null);
  const [saving, setSaving] = useState(false);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [text, setText] = useState("");
  const [withImage, setWithImage] = useState(true);
  const [sending, setSending] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  const load = () =>
    api
      .aiRules()
      .then((r) => {
        setRules(r);
        setDraft(r);
      })
      .catch((e) => message.error(errText(e)));

  useEffect(() => {
    if (open) load();
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [turns, sending]);

  const dirty = !!rules && !!draft && JSON.stringify(rules) !== JSON.stringify(draft);

  const saveDraft = async () => {
    if (!draft) return;
    setSaving(true);
    try {
      await api.aiSaveRules(draft.instructions, draft.classes);
      message.success("识别规则已保存");
      await load();
      onSaved();
    } catch (e) {
      message.error(errText(e));
    } finally {
      setSaving(false);
    }
  };

  const setDesc = (name: string, description: string) =>
    setDraft((d) => (d ? { ...d, classes: d.classes.map((c) => (c.name === name ? { ...c, description } : c)) } : d));

  const send = async () => {
    const content = text.trim();
    if (!content || sending) return;
    const msg: Turn = { role: "user", content, imageId: withImage && imageId != null ? imageId : null };
    const history = [...turns, msg];
    setTurns(history);
    setText("");
    setSending(true);
    try {
      const r = await api.aiChatRules(history.map(({ role, content, imageId }) => ({ role, content, imageId })));
      setTurns([
        ...history,
        { role: "assistant", content: r.reply || "（AI 没有回答）", suggestion: r.rules, preview: r.preview, previewImageId: r.previewImageId },
      ]);
    } catch (e) {
      message.error(errText(e));
      setTurns(history.slice(0, -1));
      setText(content);
    } finally {
      setSending(false);
    }
  };

  /** 采用 AI 建议的规则：合并到草稿并保存 */
  const adopt = async (s: NonNullable<ChatReply["rules"]>) => {
    if (!rules) return;
    const next: AiRules = {
      instructions: s.instructions ?? rules.instructions,
      classes: rules.classes.map((c) => {
        const u = s.classes.find((x) => x.name === c.name);
        return u ? { ...c, description: u.description } : c;
      }),
    };
    try {
      await api.aiSaveRules(next.instructions, next.classes);
      message.success("已采用 AI 建议的规则");
      await load();
      onSaved();
    } catch (e) {
      message.error(errText(e));
    }
  };

  const classDesc = (c: ClassRule) => (
    <div key={c.name} style={{ marginBottom: 10 }}>
      <div style={{ marginBottom: 4 }}>
        <b>{c.name}</b>
        {c.prompt && <span className="dim" style={{ fontSize: 12 }}> · {c.prompt}</span>}
      </div>
      <Input.TextArea
        autoSize={{ minRows: 1, maxRows: 4 }}
        placeholder="这个类别具体指什么、什么情况算 / 不算（可不填）"
        value={c.description}
        disabled={!canEdit}
        maxLength={500}
        onChange={(e) => setDesc(c.name, e.target.value)}
      />
    </div>
  );

  return (
    <Drawer title="识别规则 · 和 AI 说怎么识别" open={open} onClose={onClose} width={520} destroyOnHidden>
      <Tabs
        items={[
          {
            key: "chat",
            label: "和 AI 聊",
            children: (
              <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 170px)" }}>
                <Alert
                  type="info"
                  showIcon
                  style={{ marginBottom: 10 }}
                  message="用大白话告诉 AI 怎么识别，比如「戴了帽子没系带也算未戴安全帽」「反光衣不算」「只看画面下半部分的通道」。AI 会整理成规则，你点「采用」后，之后用巡瞳 AI「按项目类别识别」就按这些规则来，只找项目里的类别。"
                />
                <div ref={listRef} style={{ flex: 1, overflow: "auto", paddingRight: 4 }}>
                  {turns.length === 0 && <Empty description="还没有对话" image={Empty.PRESENTED_IMAGE_SIMPLE} />}
                  {turns.map((t, i) => (
                    <div key={i} style={{ display: "flex", justifyContent: t.role === "user" ? "flex-end" : "flex-start", marginBottom: 10 }}>
                      <div
                        style={{
                          maxWidth: "88%",
                          background: t.role === "user" ? "rgba(61,139,253,.18)" : "var(--panel)",
                          border: "1px solid var(--line)",
                          borderRadius: 8,
                          padding: "8px 10px",
                          whiteSpace: "pre-wrap",
                        }}
                      >
                        {t.content}
                        {t.role === "user" && t.imageId != null && (
                          <div className="dim" style={{ fontSize: 11, marginTop: 4 }}>
                            附图 #{t.imageId}
                          </div>
                        )}
                        {t.suggestion && (
                          <div style={{ marginTop: 8, borderTop: "1px dashed var(--line)", paddingTop: 8 }}>
                            <div style={{ marginBottom: 4 }}>
                              <Tag color="blue">AI 建议更新规则</Tag>
                            </div>
                            {t.suggestion.instructions && <div style={{ fontSize: 12 }}>整体：{t.suggestion.instructions}</div>}
                            {t.suggestion.classes.map((c) => (
                              <div key={c.name} style={{ fontSize: 12 }}>
                                <b>{c.name}</b>：{c.description}
                              </div>
                            ))}
                            {canEdit ? (
                              <Button size="small" type="primary" style={{ marginTop: 6 }} onClick={() => adopt(t.suggestion!)}>
                                采用这些规则
                              </Button>
                            ) : (
                              <div className="dim" style={{ fontSize: 12, marginTop: 4 }}>
                                你没有修改规则的权限，请发给发起人采用
                              </div>
                            )}
                          </div>
                        )}
                        {t.preview && t.previewImageId != null && (
                          <div style={{ marginTop: 8 }}>
                            <Button
                              size="small"
                              disabled={t.previewImageId !== imageId}
                              onClick={() => onApplyPreview(t.preview!)}
                            >
                              {t.previewImageId === imageId ? `把试识别的 ${t.preview.boxes.length} 个框加到这张图` : `试识别的是图 #${t.previewImageId}`}
                            </Button>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                  {sending && <div className="dim">AI 正在思考…</div>}
                </div>
                <div style={{ marginTop: 8 }}>
                  <Input.TextArea
                    autoSize={{ minRows: 2, maxRows: 5 }}
                    placeholder="说说要怎么识别，或者问「这张图左边那个人为什么没认出来」（Ctrl+Enter 发送）"
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                        e.preventDefault();
                        send();
                      }
                      e.stopPropagation();
                    }}
                  />
                  <Space style={{ marginTop: 6, width: "100%", justifyContent: "space-between" }}>
                    <Checkbox checked={withImage} disabled={imageId == null} onChange={(e) => setWithImage(e.target.checked)}>
                      附上当前这张图（AI 会按规则试识别）
                    </Checkbox>
                    <Button type="primary" icon={<SendOutlined />} loading={sending} onClick={send} disabled={!text.trim()}>
                      发送
                    </Button>
                  </Space>
                  <div className="dim" style={{ fontSize: 11, marginTop: 4 }}>
                    每次对话会用巡瞳 AI 的额度（协作成员用的是发起人的额度）。
                  </div>
                </div>
              </div>
            ),
          },
          {
            key: "rules",
            label: "当前规则",
            children: draft ? (
              <div onKeyDown={(e) => e.stopPropagation()}>
                <div style={{ marginBottom: 6 }}>
                  <b>整体规则</b> <span className="dim" style={{ fontSize: 12 }}>（对所有类别都适用）</span>
                </div>
                <Input.TextArea
                  autoSize={{ minRows: 3, maxRows: 8 }}
                  placeholder="比如：只关注画面里的人和车，忽略海报、屏幕里的内容；夜间画面也要识别"
                  value={draft.instructions}
                  disabled={!canEdit}
                  maxLength={2000}
                  onChange={(e) => setDraft({ ...draft, instructions: e.target.value })}
                  style={{ marginBottom: 16 }}
                />
                <div style={{ marginBottom: 8 }}>
                  <b>每个类别的识别说明</b>
                </div>
                {draft.classes.length === 0 ? <Empty description="项目里还没有类别" /> : draft.classes.map(classDesc)}
                {canEdit ? (
                  <Button type="primary" disabled={!dirty} loading={saving} onClick={saveDraft}>
                    保存规则
                  </Button>
                ) : (
                  <div className="dim">发起人没有开放修改类别和规则</div>
                )}
              </div>
            ) : null,
          },
        ]}
      />
    </Drawer>
  );
}
