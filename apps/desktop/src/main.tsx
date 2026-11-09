import React from "react";
import ReactDOM from "react-dom/client";
import { App as AntApp, ConfigProvider, theme } from "antd";
import zhCN from "antd/locale/zh_CN";
import App from "./App";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <ConfigProvider
      locale={zhCN}
      theme={{
        algorithm: theme.darkAlgorithm,
        token: {
          colorPrimary: "#3d8bfd",
          colorBgBase: "#14161a",
          colorBgContainer: "#1b1e23",
          colorBgElevated: "#22262c",
          colorBorder: "#2e333a",
          colorBorderSecondary: "#2e333a",
          borderRadius: 6,
          fontFamily: '"Microsoft YaHei UI","PingFang SC",system-ui,sans-serif',
        },
      }}
    >
      <AntApp>
        <App />
      </AntApp>
    </ConfigProvider>
  </React.StrictMode>,
);
