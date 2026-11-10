// 识别规则：告诉巡瞳 AI 每个类别具体指什么、整体要注意什么；可以和 AI 聊，让它帮着整理规则、在当前图上试识别
import { useEffect, useRef, useState } from "react";
import { Alert, App, Button, Checkbox, Drawer, Empty, Input, Space, Switch, Tabs, Tag } from "antd";
import { SendOutlined } from "@ant-design/icons";
import { AiRules, api, ChatMsg, ChatReply, ClassRule, errText, PredResult } from "../api";
import { useApp } from "../store";

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
  const { message, modal } = App.useApp();
  const [rules, setRules] = useState<AiRules | null>(null);
  const [draft, setDraft] = useState<AiRules | null>(null);
  const [saving, setSaving] = useState(false);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [text, setText] = useState("");
  const [withImage, setWithImage] = useState(true);
  const [sending, setSending] = useState(false);
  const [syncing, setSyncing] = useState(false);
  /** 平台上这套规则是否启用（平台识别也用它）；null 表示还没读到 */
  const [platformOn, setPlatformOn] = useState<boolean | null>(null);
  const remote = useApp((s) => s.team.remote);
  const listRef = useRef<HTMLDivElement>(null);

  const load = () =>
    api
      .aiRules()
      .then((r) => {
        setRules(r);
        setDraft(r);
        // 绑定了平台规则：读一下平台上的启用状态（可能在巡瞳控制台里改过）
        setPlatformOn(null);
        if (r.ruleSet && !remote)
          api
            .aiRuleSetInfo()
            .then((i) => setPlatformOn(i ? i.enabled : null))
            .catch(() => {});
      })
      .catch((e) => message.error(errText(e)));

  const setPlatform = (on: boolean) => {
    const apply = async () => {
      try {
        await api.aiRulesPlatformEnabled(on);
        setPlatformOn(on);
        message.success(on ? "巡瞳平台的识别也会按这套规则了" : "已停用，巡瞳平台恢复按原来的方式识别");
      } catch (e) {
        message.error(errText(e));
      }
    };
    if (!on) {
      apply();
      return;
    }
    modal.confirm({
      title: "让巡瞳平台的识别也用这套规则？",
      content:
        "打开后，你巡瞳账号里的识别（抓拍助手、告警推送、大模型识别接口）都会只按这个项目的类别识别，不再报其他隐患。适用的场景默认是全部，可以在巡瞳控制台的「识别规则」里调。账号里有多套启用的规则时，最早建的那套会接管所有场景。",
      okText: "打开",
      cancelText: "取消",
      onOk: apply,
    });
  };

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
      const warn = await api.aiSaveRules(draft.instructions, draft.classes);
      if (warn) message.warning(warn);
      else message.success(rules?.ruleSet ? "识别规则已保存，并同步到巡瞳平台" : "识别规则已保存");
      await load();
      onSaved();
    } catch (e) {
      message.error(errText(e));
    } finally {
      setSaving(false);
    }
  };

  const sync = async () => {
    if (dirty) {
      message.warning("先保存规则再同步");
      return;
    }
    setSyncing(true);
    try {
      const code = await api.aiSyncRules();
      message.success(`已同步到巡瞳平台（规则 ${code}）`);
      await load();
    } catch (e) {
      message.error(errText(e));
    } finally {
      setSyncing(false);
    }
  };

  const unbind = async () => {
    try {
      await api.aiUnbindRules();
      message.success("已解除绑定，平台上的规则没有删除");
      await load();
    } catch (e) {
      message.error(errText(e));
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
      const warn = await api.aiSaveRules(next.instructions, next.classes);
      if (warn) message.warning(warn);
      else message.success("已采用 AI 建议的规则");
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
                  maxLength={1000}
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
                {!remote && (
                  <div style={{ marginTop: 20, borderTop: "1px solid var(--line)", paddingTop: 12 }}>
                    <div style={{ marginBottom: 6 }}>
                      <b>同步到巡瞳平台</b>
                    </div>
                    <div className="dim" style={{ fontSize: 12, marginBottom: 8 }}>
                      把类别和规则存到巡瞳平台上（默认不启用，不影响平台现在的识别）。绑定后每次保存规则会自动同步内容；想让巡瞳自己的抓拍、告警也只报你要的类别，再打开「平台识别也用这套规则」。
                    </div>
                    {rules?.ruleSet ? (
                      <Space wrap>
                        <Tag color="green">已绑定平台规则 {rules.ruleSet}</Tag>
                        {platformOn !== null && (
                          <span>
                            <Switch size="small" checked={platformOn} onChange={setPlatform} /> 平台识别也用这套规则
                          </span>
                        )}
                        <Button size="small" loading={syncing} onClick={sync}>
                          立即同步
                        </Button>
                        <Button size="small" type="link" onClick={unbind}>
                          解除绑定
                        </Button>
                      </Space>
                    ) : (
                      <Button size="small" loading={syncing} onClick={sync}>
                        同步到巡瞳平台
                      </Button>
                    )}
                  </div>
                )}
              </div>
            ) : null,
          },
        ]}
      />
    </Drawer>
  );
}
