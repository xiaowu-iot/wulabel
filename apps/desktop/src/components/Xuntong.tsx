// 巡瞳 AI 平台对接：把训练好的模型上传到平台；把平台上的告警图片拉回来标注
import { useEffect, useState } from "react";
import { Alert, App, Button, Checkbox, Input, InputNumber, Modal, Progress, Select, Space, Tag } from "antd";
import { CloudUploadOutlined } from "@ant-design/icons";
import { AiModel, api, errText, ImportProgress, ProjectInfo, RunInfo, XtPullResult, XtUploadResult } from "../api";
import { listen } from "../transport";
import { useApp } from "../store";

/** 部署页「导出模型」里：上传到巡瞳 AI 平台 */
export function XuntongUpload({ run, project }: { run: RunInfo; project: ProjectInfo }) {
  const { message } = App.useApp();
  const setPage = useApp((s) => s.setPage);
  const [name, setName] = useState(`${project.name} ${run.name}`);
  const [busy, setBusy] = useState(false);
  const [log, setLog] = useState("");
  const [result, setResult] = useState<XtUploadResult | null>(null);
  const supported = project.task === "detect" || project.task === "classify";

  const upload = async () => {
    setBusy(true);
    setResult(null);
    setLog("");
    const un = await listen<string>("xuntong-log", (e) => setLog(e.payload));
    try {
      setResult(await api.xtUploadModel(run.id, name));
    } catch (e) {
      message.error(errText(e));
    } finally {
      un();
      setBusy(false);
    }
  };

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div style={{ marginBottom: 8 }}>
        <b>上传到巡瞳 AI 平台</b>{" "}
        <span className="muted">· 用你在「设置 → 巡瞳 AI」填的账号，上传后在平台上接摄像头、配告警，不用自己部署服务</span>
      </div>
      {!supported ? (
        <div className="dim">巡瞳平台目前只支持检测和分类模型。</div>
      ) : (
        <>
          <Space.Compact style={{ width: "100%", maxWidth: 560 }}>
            <Input addonBefore="模型名" value={name} onChange={(e) => setName(e.target.value)} />
            <Button type="primary" icon={<CloudUploadOutlined />} loading={busy} onClick={upload}>
              上传
            </Button>
          </Space.Compact>
          {busy && log && (
            <div className="dim" style={{ fontSize: 12, marginTop: 6 }}>
              {log}
            </div>
          )}
          {result && (
            <Alert
              style={{ marginTop: 10 }}
              type={result.status === "failed" ? "error" : "success"}
              showIcon
              message={
                <span>
                  {result.status === "failed" ? "平台没有接受这个模型" : "已上传"} <Tag>{result.status}</Tag>
                  {result.code && <span className="muted">编码 {result.code}</span>}
                </span>
              }
              description={
                <span>
                  {result.message || "到巡瞳 AI 控制台的「自定义模型」里配置、启用后就能用。推荐阈值已经一起传上去了。"}{" "}
                  <a href={result.consoleUrl} target="_blank" rel="noreferrer">
                    打开控制台
                  </a>
                </span>
              }
            />
          )}
          <div className="dim" style={{ fontSize: 12, marginTop: 6 }}>
            还没填巡瞳 API Key？<a onClick={() => setPage("settings")}>去设置</a>
          </div>
        </>
      )}
    </div>
  );
}

/** 数据页「更多导入」里：从巡瞳 AI 拉取告警图片 */
export function XuntongPullModal({ open, onClose, onDone }: { open: boolean; onClose: () => void; onDone: () => void }) {
  const { message, modal } = App.useApp();
  const [days, setDays] = useState(7);
  const [model, setModel] = useState<string>("");
  const [models, setModels] = useState<AiModel[] | null>(null);
  const [max, setMax] = useState(200);
  const [withBoxes, setWithBoxes] = useState(true);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<ImportProgress | null>(null);
  const [log, setLog] = useState("");
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!open || models) return;
    api
      .aiModels()
      .then((ms) => {
        setModels(ms.filter((m) => m.code !== "llm" && m.code !== "classes"));
        setErr(null);
      })
      .catch((e) => setErr(errText(e)));
  }, [open, models]);

  const run = async () => {
    setBusy(true);
    setLog("");
    setProgress({ done: 0, total: 0 });
    const subs = [
      listen<ImportProgress>("xuntong-progress", (e) => setProgress(e.payload)),
      listen<string>("xuntong-log", (e) => setLog(e.payload)),
    ];
    try {
      const r: XtPullResult = await api.xtPullAlarms({ days, model: model || null, max, withBoxes });
      onClose();
      onDone();
      modal.success({
        title: "告警图片已拉回",
        content: (
          <div>
            <div>
              {r.fetched} 条告警，新增 {r.added} 张图{r.duplicated ? `，${r.duplicated} 张已经有了` : ""}
              {r.failed ? `，${r.failed} 张下载失败` : ""}
            </div>
            {withBoxes && <div>平台的识别框加成了 {r.boxes} 个预标注（虚线，来源「巡瞳」），到标注页检查：误报的删掉、漏的补上。</div>}
            <div className="dim" style={{ fontSize: 12, marginTop: 6 }}>
              图片在项目的 {r.folder} 文件夹
            </div>
          </div>
        ),
      });
    } catch (e) {
      message.error(errText(e));
    } finally {
      (await Promise.all(subs)).forEach((un) => un());
      setBusy(false);
      setProgress(null);
    }
  };

  return (
    <Modal
      title="从巡瞳 AI 拉取告警图片"
      open={open}
      onCancel={busy ? undefined : onClose}
      onOk={run}
      okText="开始拉取"
      cancelText="取消"
      confirmLoading={busy}
      okButtonProps={{ disabled: !!err }}
      width={520}
      destroyOnHidden
    >
      <p className="muted" style={{ marginTop: 8 }}>
        把巡瞳平台上的告警截图拉回来标注、再训练。现场的误报、漏报正是最值得补的数据。
      </p>
      {err ? (
        <Alert type="warning" showIcon message={err} />
      ) : (
        <Space direction="vertical" style={{ width: "100%" }}>
          <Space wrap>
            <Select
              value={days}
              onChange={setDays}
              style={{ width: 140 }}
              options={[
                { value: 1, label: "最近 1 天" },
                { value: 3, label: "最近 3 天" },
                { value: 7, label: "最近 7 天" },
                { value: 30, label: "最近 30 天" },
                { value: 90, label: "最近 90 天" },
              ]}
            />
            <Select
              value={model}
              onChange={setModel}
              style={{ width: 220 }}
              loading={!models}
              options={[{ value: "", label: "全部模型" }, ...(models ?? []).map((m) => ({ value: m.code, label: m.name }))]}
            />
            <InputNumber min={1} max={5000} value={max} onChange={(v) => setMax(Number(v ?? 200))} addonBefore="最多" addonAfter="张" style={{ width: 170 }} />
          </Space>
          <Checkbox checked={withBoxes} onChange={(e) => setWithBoxes(e.target.checked)}>
            平台识别出的框作为预标注（按类别名对应项目里的类别，没有的会新建）
          </Checkbox>
        </Space>
      )}
      {progress && (
        <div style={{ marginTop: 12 }}>
          <Progress size="small" percent={progress.total ? Math.floor((progress.done / progress.total) * 100) : 0} />
          {log && <div className="dim" style={{ fontSize: 12 }}>{log}</div>}
        </div>
      )}
    </Modal>
  );
}
