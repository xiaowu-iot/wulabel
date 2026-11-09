import { useCallback, useEffect, useRef, useState } from "react";
import { Alert, App, Button, Descriptions, Input, Radio, Select, Space, Spin, Tag } from "antd";
import { CheckCircleOutlined, FolderOpenOutlined, ReloadOutlined, ToolOutlined } from "@ant-design/icons";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import { api, EnvStatus, errText } from "../api";

const VARIANTS = [
  { value: "cu130", label: "CUDA 13.0（需要较新的驱动）" },
  { value: "cu128", label: "CUDA 12.8（RTX 50 系列等新显卡）" },
  { value: "cu126", label: "CUDA 12.6" },
  { value: "cu118", label: "CUDA 11.8（较老的显卡和驱动）" },
  { value: "cpu", label: "CPU 版（没有 NVIDIA 显卡）" },
];

export default function Settings() {
  const { message, modal } = App.useApp();
  const [st, setSt] = useState<EnvStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [installing, setInstalling] = useState(false);
  const [variant, setVariant] = useState<string | null>(null);
  const [lines, setLines] = useState<string[]>([]);
  const [python, setPython] = useState("");
  const [root, setRoot] = useState("");
  const logBox = useRef<HTMLDivElement>(null);

  const load = useCallback(
    async (refresh: boolean) => {
      setLoading(true);
      try {
        const s = await api.envStatus(refresh);
        setSt(s);
        setPython(s.settings.python ?? "");
        setRoot(s.settings.envRoot ?? "");
        setInstalling(s.installing);
      } catch (e) {
        message.error(errText(e));
      } finally {
        setLoading(false);
      }
    },
    [message],
  );

  useEffect(() => {
    load(false);
  }, [load]);

  useEffect(() => {
    const un = listen<string>("env-log", (e) => {
      setLines((ls) => [...ls.slice(-400), e.payload]);
      requestAnimationFrame(() => logBox.current?.scrollTo(0, logBox.current.scrollHeight));
    });
    return () => {
      un.then((f) => f());
    };
  }, []);

  const install = async () => {
    const v = variant ?? st?.variant ?? "cpu";
    setInstalling(true);
    setLines([]);
    try {
      await api.envInstall(v);
      message.success("训练环境安装完成");
    } catch (e) {
      modal.error({ title: "安装没有完成", content: <pre style={{ whiteSpace: "pre-wrap", fontSize: 12 }}>{errText(e)}</pre>, width: 640 });
    } finally {
      setInstalling(false);
      load(true);
    }
  };

  const saveSettings = async (patch: Partial<EnvStatus["settings"]>) => {
    if (!st) return;
    try {
      await api.saveSettings({ ...st.settings, ...patch });
      await load(true);
    } catch (e) {
      message.error(errText(e));
    }
  };

  const info = st?.info as Record<string, string | number | boolean> | null | undefined;

  return (
    <div className="page" style={{ maxWidth: 1000 }}>
      <h1>设置</h1>
      <p className="desc">训练环境和下载源。标注、导入导出不需要训练环境。</p>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
          <h2 style={{ fontSize: 15, margin: 0 }}>
            训练环境{" "}
            {st && (st.ready ? <Tag color="success" icon={<CheckCircleOutlined />}>就绪</Tag> : <Tag color="warning">未就绪</Tag>)}
          </h2>
          <Button size="small" icon={<ReloadOutlined />} loading={loading} onClick={() => load(true)}>
            重新检查
          </Button>
        </div>
        {!st ? (
          <Spin />
        ) : (
          <>
            <Descriptions column={1} size="small" bordered labelStyle={{ width: 140 }}>
              <Descriptions.Item label="显卡">
                {st.gpu ? `${st.gpu.name} · ${(st.gpu.vramMb / 1024).toFixed(0)} GB · 驱动 ${st.gpu.driver} · 支持 CUDA ${st.gpu.cuda}` : "没有检测到 NVIDIA 显卡，只能用 CPU 训练（会慢很多）"}
              </Descriptions.Item>
              <Descriptions.Item label="Python">{st.python ?? <span className="dim">未安装</span>}</Descriptions.Item>
              {info && (
                <Descriptions.Item label="PyTorch">
                  {info.torch ? `${info.torch} · ${info.cuda ? `显卡可用（${info.device}）` : "CPU"}` : <span style={{ color: "var(--bad)" }}>{String(info.error ?? "未安装")}</span>}
                </Descriptions.Item>
              )}
              {st.error && (
                <Descriptions.Item label="问题">
                  <span style={{ color: "var(--bad)", whiteSpace: "pre-wrap" }}>{st.error}</span>
                </Descriptions.Item>
              )}
            </Descriptions>

            {st.managed && (
              <div style={{ marginTop: 16 }}>
                <div className="row" style={{ marginBottom: 8 }}>
                  <span>PyTorch 版本</span>
                  <Select
                    style={{ width: 300 }}
                    value={variant ?? st.variant}
                    onChange={setVariant}
                    options={VARIANTS.map((v) => ({ ...v, label: v.value === st.variant ? `${v.label}（推荐）` : v.label }))}
                  />
                  <Button type="primary" icon={<ToolOutlined />} loading={installing} onClick={install}>
                    {st.ready ? "重新安装" : "一键安装"}
                  </Button>
                </div>
                <div className="dim" style={{ fontSize: 12, marginBottom: 8 }}>
                  GPU 版约 3GB，只需要一次。会自动下载 uv、Python 3.11、PyTorch。
                </div>
                <div className="row" style={{ gap: 8, flexWrap: "nowrap" }}>
                  <span style={{ whiteSpace: "nowrap" }}>安装位置</span>
                  <Input size="small" value={root} placeholder="默认：系统的本机数据目录（C 盘）" onChange={(e) => setRoot(e.target.value)} />
                  <Button
                    size="small"
                    icon={<FolderOpenOutlined />}
                    onClick={async () => {
                      const d = await open({ directory: true, title: "选择训练环境安装位置（建议空间大的盘）" });
                      if (typeof d === "string") setRoot(d);
                    }}
                  />
                  <Button size="small" disabled={(root.trim() || null) === (st.settings.envRoot || null)} onClick={() => saveSettings({ envRoot: root.trim() || null })}>
                    保存
                  </Button>
                </div>
                <div className="dim" style={{ fontSize: 12, marginTop: 4 }}>当前：{st.envDir}。改位置后需要重新安装。</div>
              </div>
            )}

            {(installing || lines.length > 0) && (
              <div
                ref={logBox}
                style={{ marginTop: 12, height: 220, overflow: "auto", background: "#0e1013", borderRadius: 6, padding: 10, font: "12px/1.6 Consolas, monospace", color: "#b8c0c9", whiteSpace: "pre-wrap", userSelect: "text" }}
              >
                {lines.join("\n")}
              </div>
            )}
          </>
        )}
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <h2 style={{ fontSize: 15, marginTop: 0 }}>下载源</h2>
        <Radio.Group value={st?.settings.mirror ?? "china"} onChange={(e) => saveSettings({ mirror: e.target.value })}>
          <Space direction="vertical">
            <Radio value="china">国内镜像（阿里云、清华、npmmirror，推荐国内使用）</Radio>
            <Radio value="official">官方源（PyPI、download.pytorch.org）</Radio>
          </Space>
        </Radio.Group>
      </div>

      <div className="card">
        <h2 style={{ fontSize: 15, marginTop: 0 }}>使用已有的 Python 环境（高级）</h2>
        <p className="muted" style={{ marginTop: 0 }}>
          已经有装好 PyTorch 的环境可以直接指定，就不用再下载。需要 torch、torchvision、numpy、pillow，导出 ONNX 还需要 onnx。
        </p>
        <Space.Compact style={{ width: "100%" }}>
          <Input value={python} placeholder="例如 D:\envs\torch\Scripts\python.exe（留空则用软件自己安装的环境）" onChange={(e) => setPython(e.target.value)} />
          <Button
            icon={<FolderOpenOutlined />}
            onClick={async () => {
              const f = await open({ title: "选择 python.exe", filters: [{ name: "Python", extensions: ["exe", ""] }] });
              if (typeof f === "string") setPython(f);
            }}
          />
          <Button type="primary" onClick={() => saveSettings({ python: python.trim() || null })}>
            保存
          </Button>
        </Space.Compact>
        {st && !st.managed && (
          <Alert style={{ marginTop: 10 }} type="info" showIcon message="正在使用你指定的 Python 环境。清空后保存可改回软件自己管理的环境。" />
        )}
      </div>
    </div>
  );
}
