import { useCallback, useEffect, useState } from "react";
import { App, Button, Empty, Input, Modal, Space, Table, Tag, Tooltip } from "antd";
import { DeleteOutlined, DiffOutlined, EditOutlined, ExportOutlined, RollbackOutlined } from "@ant-design/icons";
import { api, errText, ProjectInfo, VersionDiff, VersionInfo } from "../api";
import { useApp } from "../store";
import ExportModal from "./ExportModal";

const fmtTime = (ts: number) => new Date(ts * 1000).toLocaleString();

/** 数据集版本：把当前已标注的数据存成快照，可以导出、训练、和当前对比、恢复 */
export default function VersionsModal({ project, open, onClose }: { project: ProjectInfo; open: boolean; onClose: () => void }) {
  const { message, modal } = App.useApp();
  const refreshStats = useApp((s) => s.refreshStats);
  const [list, setList] = useState<VersionInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [name, setName] = useState("");
  const [note, setNote] = useState("");
  const [creating, setCreating] = useState(false);
  const [exporting, setExporting] = useState<VersionInfo | null>(null);
  const [diff, setDiff] = useState<{ v: VersionInfo; d: VersionDiff } | null>(null);
  const [editing, setEditing] = useState<{ v: VersionInfo; name: string; note: string } | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setList(await api.listVersions());
    } catch (e) {
      message.error(errText(e));
    } finally {
      setLoading(false);
    }
  }, [message]);

  useEffect(() => {
    if (open) load();
  }, [open, load]);

  const create = async () => {
    setCreating(true);
    try {
      const v = await api.createVersion(name, note, [0.8, 0.2, 0]);
      setName("");
      setNote("");
      message.success(`已保存版本「${v.name}」：${v.images} 张图、${v.annotations} 个标注`);
      await load();
    } catch (e) {
      message.error(errText(e));
    } finally {
      setCreating(false);
    }
  };

  const showDiff = async (v: VersionInfo) => {
    try {
      setDiff({ v, d: await api.diffVersion(v.id) });
    } catch (e) {
      message.error(errText(e));
    }
  };

  const restore = (v: VersionInfo) => {
    modal.confirm({
      title: `恢复到版本「${v.name}」？`,
      width: 520,
      content: (
        <div>
          版本里这 {v.images} 张图的标注、训练/验证/测试划分、审核状态会改回保存时的样子。
          <div className="muted" style={{ marginTop: 6 }}>
            恢复前会自动把当前数据存成「恢复前自动备份」版本，随时可以再恢复回来。版本里没有的图不受影响。
          </div>
        </div>
      ),
      okText: "恢复",
      cancelText: "取消",
      onOk: async () => {
        try {
          const r = await api.restoreVersion(v.id);
          message.success(`已恢复 ${r.images} 张图的标注${r.missing ? `，${r.missing} 张图已从项目删除，没法恢复` : ""}`);
          await load();
          await refreshStats();
        } catch (e) {
          message.error(errText(e));
        }
      },
    });
  };

  const remove = (v: VersionInfo) => {
    modal.confirm({
      title: `删除版本「${v.name}」？`,
      content: "只删除这个快照，不影响当前的图片和标注，也不影响已经训练好的模型。",
      okText: "删除",
      okButtonProps: { danger: true },
      cancelText: "取消",
      onOk: async () => {
        try {
          await api.deleteVersion(v.id);
          await load();
        } catch (e) {
          message.error(errText(e));
        }
      },
    });
  };

  const saveEdit = async () => {
    if (!editing) return;
    try {
      await api.renameVersion(editing.v.id, editing.name, editing.note);
      setEditing(null);
      await load();
    } catch (e) {
      message.error(errText(e));
    }
  };

  return (
    <>
      <Modal title="数据集版本" open={open} onCancel={onClose} footer={null} width={860} destroyOnHidden>
        <p className="muted" style={{ marginTop: 0 }}>
          把当前已标注、已审核的图片和标注存成一个快照。之后可以用某个版本训练（方便对比不同数据的效果）、导出、和当前数据对比，或者恢复回去。
        </p>
        <Space.Compact style={{ width: "100%", marginBottom: 12 }}>
          <Input style={{ width: 200 }} placeholder="版本名，如 v1、补充夜间图" value={name} maxLength={60} onChange={(e) => setName(e.target.value)} />
          <Input placeholder="备注（可选）：这一版改了什么" value={note} maxLength={200} onChange={(e) => setNote(e.target.value)} onPressEnter={create} />
          <Button type="primary" loading={creating} onClick={create}>
            保存当前数据为新版本
          </Button>
        </Space.Compact>
        <Table<VersionInfo>
          size="small"
          rowKey="id"
          loading={loading}
          dataSource={list}
          pagination={false}
          locale={{ emptyText: <Empty description="还没有版本" /> }}
          expandable={{
            expandedRowRender: (v) => (
              <div className="row" style={{ flexWrap: "wrap", gap: 6 }}>
                {v.classes.map((c) => (
                  <Tag key={c.name}>
                    {c.name} {c.count}
                  </Tag>
                ))}
              </div>
            ),
          }}
          columns={[
            {
              title: "版本",
              render: (_, v) => (
                <div>
                  <b>{v.name}</b>
                  {v.note && <div className="dim" style={{ fontSize: 12 }}>{v.note}</div>}
                </div>
              ),
            },
            { title: "时间", dataIndex: "createdAt", render: fmtTime, width: 160 },
            { title: "图片", render: (_, v) => `${v.images}（训 ${v.train} / 验 ${v.val} / 测 ${v.test}）`, width: 190 },
            { title: "标注", dataIndex: "annotations", width: 70 },
            {
              title: "",
              width: 170,
              render: (_, v) => (
                <Space size={2}>
                  <Tooltip title="和当前数据对比">
                    <Button size="small" type="text" icon={<DiffOutlined />} onClick={() => showDiff(v)} />
                  </Tooltip>
                  <Tooltip title="导出这个版本">
                    <Button size="small" type="text" icon={<ExportOutlined />} onClick={() => setExporting(v)} />
                  </Tooltip>
                  <Tooltip title="恢复到这个版本">
                    <Button size="small" type="text" icon={<RollbackOutlined />} onClick={() => restore(v)} />
                  </Tooltip>
                  <Tooltip title="改名、备注">
                    <Button size="small" type="text" icon={<EditOutlined />} onClick={() => setEditing({ v, name: v.name, note: v.note })} />
                  </Tooltip>
                  <Tooltip title="删除这个版本">
                    <Button size="small" type="text" icon={<DeleteOutlined />} onClick={() => remove(v)} />
                  </Tooltip>
                </Space>
              ),
            },
          ]}
        />
        <div className="dim" style={{ fontSize: 12, marginTop: 8 }}>在「训练」页的「训练数据」里可以选用某个版本训练。</div>
      </Modal>

      <Modal title={diff ? `版本「${diff.v.name}」→ 当前数据` : ""} open={!!diff} onCancel={() => setDiff(null)} footer={null} width={560}>
        {diff && (
          <>
            <div className="cards" style={{ gridTemplateColumns: "repeat(3, 1fr)", margin: "8px 0 12px" }}>
              <div className="card"><div className="muted">新增的图</div><div className="kpi">{diff.d.added.length}</div></div>
              <div className="card"><div className="muted">移出的图</div><div className="kpi">{diff.d.removed.length}</div></div>
              <div className="card"><div className="muted">标注改过的图</div><div className="kpi">{diff.d.changed.length}</div></div>
            </div>
            {diff.d.classes.length === 0 ? (
              <div className="muted">各类别的标注数量没有变化。</div>
            ) : (
              <Table
                size="small"
                rowKey="name"
                pagination={false}
                dataSource={diff.d.classes}
                columns={[
                  { title: "类别", dataIndex: "name" },
                  { title: "版本里", dataIndex: "before" },
                  { title: "当前", dataIndex: "after" },
                  {
                    title: "变化",
                    render: (_, c) => {
                      const d = c.after - c.before;
                      return <span style={{ color: d > 0 ? "var(--good, #2fb67c)" : "var(--bad)" }}>{d > 0 ? `+${d}` : d}</span>;
                    },
                  },
                ]}
              />
            )}
            <div className="dim" style={{ fontSize: 12, marginTop: 8 }}>「移出的图」是版本里有、现在变成未标注 / 跳过 / 已删除的图。</div>
          </>
        )}
      </Modal>

      <Modal title="修改版本" open={!!editing} onCancel={() => setEditing(null)} onOk={saveEdit} okText="保存" cancelText="取消" width={420}>
        {editing && (
          <Space direction="vertical" style={{ width: "100%", marginTop: 8 }}>
            <Input value={editing.name} maxLength={60} onChange={(e) => setEditing({ ...editing, name: e.target.value })} />
            <Input.TextArea rows={3} value={editing.note} maxLength={200} placeholder="备注" onChange={(e) => setEditing({ ...editing, note: e.target.value })} />
          </Space>
        )}
      </Modal>

      <ExportModal project={project} open={!!exporting} version={exporting} onClose={() => setExporting(null)} />
    </>
  );
}
