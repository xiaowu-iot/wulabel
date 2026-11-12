# Wulabel 服务端

这里是**软件内「反馈问题」**用到的服务端。桌面软件本身完全离线，只有用户主动点反馈时，才会把描述和截图发到作者邮箱。

```
桌面端 ──HTTPS──▶ feedback.<你的域名>/feedback   （反向代理，提供 HTTPS）
                     └─▶ 本机反馈后端 127.0.0.1:<端口>
                            └─ 验签 + 限流防刷 + SMTP 发到收件邮箱
```

两样东西，放同一台有公网的服务器即可：

| 内容 | 说明 |
|---|---|
| **官网静态站** | 官网静态文件（单独维护，不在本仓库），反向代理直接托管（`wulabel.<域名>`）。 |
| **反馈后端** | `server/feedback/` 的小服务，监听 `127.0.0.1`，由反向代理转发（`feedback.<域名>`）。 |

下载安装包走 Gitee / GitHub 的 Release，**不经本服务**。

## 部署步骤

1. **反馈后端**：按 [`feedback/README.md`](feedback/README.md) 把服务跑起来，监听 `127.0.0.1:<端口>`（只听本机，不对公网开放）。
2. **DNS**：`wulabel.<域名>`、`feedback.<域名>` 两条 A 记录都指向这台服务器。
3. **反向代理 + HTTPS**：用 Caddy 或 nginx，给两个子域签证书并转发：
   - `wulabel.<域名>` → 静态托管官网目录
   - `feedback.<域名>` → `http://127.0.0.1:<端口>`
   - Caddy 示例见 [`Caddyfile`](Caddyfile)；nginx 示例见 [`feedback/nginx.conf.example`](feedback/nginx.conf.example)。
4. **客户端对接**：见 `feedback/README.md` 的「对接客户端」，把反馈入口和签名密钥配上。

> 整套可以和你服务器上已有的站点共用同一个反向代理，互不影响。
