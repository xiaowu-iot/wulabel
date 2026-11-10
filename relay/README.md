# Wulabel 协作中转服务

团队里有人不在同一个局域网（外包、在家办公）时，发起人的 Wulabel 可以通过中转服务器「在线协作」：

```
成员的浏览器 ──HTTPS──▶ 中转服务器 ◀──WebSocket── 发起人电脑上的 Wulabel（主机）
```

- 主机**主动连**中转，不需要公网 IP、不用改路由器、不用端口映射。
- 图片和标注只在内存里转发，**不落盘**。数据只经过你自己的服务器。
- 主机断线会自动重连；主机不在线时，成员打开地址会看到「发起人不在线」的提示。

## 部署（自建）

需要一台有公网地址的服务器（云服务器即可），Python 3.9 以上。

```bash
pip install aiohttp
python wulabel_relay.py --port 8095 --secret 换成你自己的长密钥
```

然后用 nginx（或 caddy）加 HTTPS，反向代理到 8095，**要支持 WebSocket**：

```nginx
server {
    listen 443 ssl;
    server_name relay.example.com;
    # ssl_certificate ...;

    location / {
        proxy_pass http://127.0.0.1:8095;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        client_max_body_size 50m;
    }
}
```

放在子路径下（比如 `/wulabel-relay/`）也可以，这时加上 `--public-url https://example.com/wulabel-relay/`，让中转知道自己对外的地址。

在 Wulabel 里：**协作 → 共享方式选「在线（自建中转）」**，填中转地址 `https://relay.example.com/` 和同样的密钥，开启后会显示成员访问的地址，形如 `https://relay.example.com/s/xxxxxxxxxxxx/`。

### 用 systemd 常驻

```ini
# /etc/systemd/system/wulabel-relay.service
[Unit]
Description=Wulabel relay
After=network.target

[Service]
ExecStart=/usr/bin/python3 /opt/wulabel-relay/wulabel_relay.py --port 8095 --secret 换成你自己的长密钥
Restart=always
User=nobody

[Install]
WantedBy=multi-user.target
```

```bash
systemctl enable --now wulabel-relay
```

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--host` | 127.0.0.1 | 监听地址，前面有 nginx 时保持默认 |
| `--port` | 8095 | 端口 |
| `--auth` | secret | `secret` 用密钥；`xuntong` 用巡瞳 AI 的 API Key（官方中转用） |
| `--secret` | | 密钥，至少 8 位 |
| `--xuntong-url` | https://xuntongai.com | `--auth xuntong` 时验证 API Key 的平台地址 |
| `--public-url` | 按请求头推断 | 对外地址 |
| `--max-shares-per-key` | 3 | 同一个密钥 / API Key 最多同时开几个共享，0 不限 |
| `--max-body-mb` | 50 | 单个请求最大 MB |
| `--timeout` | 120 | 等主机响应的秒数 |

## 安全

- 主机要带密钥（或有效的巡瞳 API Key）才能连上；同一个共享编号只能被同一个密钥使用，别人顶不掉。
- 成员仍然要用发起人给的**名字和密码**登录，中转本身看不到也不保存密码以外的任何数据；HTTPS 由你的 nginx 提供。
- 标注员只能看到分给自己的图，导出、训练、删除等只能在发起人的电脑上做。
