# Wulabel 反馈服务

软件内「反馈问题」把描述和截图发到作者邮箱。一个很小的 FastAPI 服务，只监听本机，由反向代理提供 HTTPS。

```
桌面端 ──HTTPS──▶ feedback.<域名>/feedback   （反向代理）
                    └─▶ 本服务 127.0.0.1:<端口>
                           └─ 验签 + 限流防刷 + SMTP 发到收件邮箱
```

- 收件邮箱只在服务端 `.env` 里配置，不进客户端。
- 客户端带签名（blake3 keyed-hash），挡随手刷接口；真正兜底的是这里的限流和全局发信上限。
- 下载安装包不经过本服务，走 GitHub / Gitee 的 Release。

## 一、跑起来

需要 Python 3.9+。

```bash
cd server/feedback
python -m venv .venv
# Linux/macOS:  . .venv/bin/activate
# Windows:      .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env     # 填好 APP_SECRET 和 SMTP 信息（见下）
uvicorn app:app --host 127.0.0.1 --port 8787
```

只监听 `127.0.0.1`，由反向代理转发，不要对公网直接开端口。

**常驻为后台服务：**

- **Linux（systemd）：**

  ```ini
  # /etc/systemd/system/wulabel-feedback.service
  [Unit]
  Description=wulabel feedback
  After=network.target
  [Service]
  WorkingDirectory=/opt/wulabel/server/feedback
  ExecStart=/opt/wulabel/server/feedback/.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8787
  Restart=always
  [Install]
  WantedBy=multi-user.target
  ```

- **Windows：** 用 [WinSW](https://github.com/winsw/winsw) 把 `python -m uvicorn app:app --host 127.0.0.1 --port 8787` 注册成服务（开机自启、失败自重启），或用任意进程守护工具。

## 二、发信邮箱（QQ 邮箱 SMTP）

用一个 QQ 邮箱发送（和示例一致），不依赖任何云邮件服务：

1. QQ 邮箱 → 设置 → 账户 → 开启「POP3/SMTP 服务」，生成**授权码**。
2. `.env` 填：`SMTP_HOST=smtp.qq.com`、`SMTP_PORT=465`、`SMTP_USER` 和 `MAIL_FROM` 填你的 QQ 邮箱、`SMTP_PASS` 填**授权码**（不是登录密码）。
3. `FEEDBACK_TO` 填收件箱。

> QQ 邮箱有每日发信上限，配合这里的 `GLOBAL_DAY` 双重兜底，刷不爆。换成其它 SMTP 服务同理，改 host / 端口即可。

## 三、反向代理（二级域名 + HTTPS）

把 `feedback.<域名>` 解析到这台机器，用反向代理转发到本服务并提供 HTTPS：

- Caddy：见 [`../Caddyfile`](../Caddyfile)（自动签证书，最省事）。
- nginx：见 [`nginx.conf.example`](nginx.conf.example)。

## 四、对接客户端

改 `apps/desktop/src-tauri/src/feedback.rs` 顶部的 `FEEDBACK_ENDPOINT` 为你的 `https://feedback.<域名>/feedback`。

签名密钥**不写进源码**：打包时用环境变量注入，要和这里 `.env` 的 `APP_SECRET` 完全一致。

```powershell
# 在桌面端打包机上（PowerShell）
$env:WULABEL_FEEDBACK_SECRET = "<和服务端 .env 的 APP_SECRET 一样>"
cd apps/desktop
npm run tauri build
```

不注入时客户端用的是占位密钥，只能连你自己放开的测试环境，连不上正式服务——这正是为了不把密钥暴露在公开源码里。

## 防刷说明

| 手段 | 作用 |
|---|---|
| 请求签名（时间戳 + keyed-hash） | 挡掉直接 curl / 脚本打接口；防重放（时间偏差 > 5 分钟拒绝） |
| 单设备限流 | 每小时 5 条、每天 20 条 |
| 单 IP 限流 | 每小时 10 条 |
| 全局日上限 | 每天 500 条，保护邮件额度 |
| 内容大小限制 | 描述 ≤ 2000 字，截图 ≤ 约 2MB |

签名密钥只挡低端滥用；限流和全局上限才是硬兜底。要更强可再加人机验证，但桌面端体验差，一般不需要。
