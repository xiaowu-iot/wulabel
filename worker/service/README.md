# 推理服务包

由 xlabel 生成。拉取摄像头视频流，用训练好的模型检测，发现目标就推送告警；同时提供 HTTP 检测接口。

## 运行

需要 Python 3.9 及以上（不需要 PyTorch，也不需要显卡）。

- Windows：双击 `run.bat`
- Linux：`sh run.sh`

首次运行会自动创建 Python 环境并安装 onnxruntime、opencv、numpy（约 100MB，使用清华镜像）。之后直接启动。

有 NVIDIA 显卡想用 GPU 推理：把 `requirements.txt` 里的 `onnxruntime` 换成 `onnxruntime-gpu`，删除 `venv` 文件夹后重新运行。

## 配置 config.json

```jsonc
{
  "model": "model.onnx",
  "gpu": true,                 // 有 onnxruntime-gpu 时用显卡
  "max_side": 1280,            // 检测前把画面长边缩到这个尺寸，越小越快
  "port": 8090,                // HTTP 接口端口
  "interval": 1.0,             // 每路视频每隔几秒检测一帧
  "streams": [
    {
      "name": "北门",
      "url": "rtsp://admin:密码@192.168.1.64:554/Streaming/Channels/101",
      "region": [[0.1, 0.4], [0.9, 0.4], [0.9, 1.0], [0.1, 1.0]]  // 可选：检测区域，0~1 的比例坐标
    }
  ],
  "alert": {
    "labels": ["未戴安全帽"],  // 只对这些类别告警，空表示全部类别
    "min_score": 0.5,          // 置信度阈值
    "cooldown": 60,            // 同一路视频同一类别，多少秒内只告警一次
    "hours": "07:00-19:00",    // 生效时段，空表示全天，跨天写 "22:00-06:00"
    "save_image": true         // 保存告警截图到 alerts/
  },
  "webhook": { "url": "http://你的平台/api/alarm", "with_image": false, "headers": {} },
  "mqtt": { "host": "", "port": 1883, "topic": "xlabel/alarm", "username": "", "password": "" },
  "reflux": { "enabled": true, "low": 0.3, "high": 0.5, "interval": 60 }
}
```

- 视频源也可以填本地视频文件路径，用于测试（默认循环播放）。
- 检测区域按目标框的**底边中点**判断（人、车站的位置）。

## 告警内容

Webhook（POST JSON）和 MQTT 收到的消息：

```json
{
  "stream": "北门",
  "time": "2026-11-01T08:30:15",
  "labels": ["未戴安全帽"],
  "detections": [{ "label": "未戴安全帽", "score": 0.91, "box": [x, y, w, h] }],
  "width": 1920,
  "height": 1080,
  "image": "alerts/20261101/北门_083015.jpg"
}
```

Webhook 开启 `with_image` 时，会多一个 `image_base64` 字段（带框的 JPEG）。

## HTTP 接口

| 接口 | 说明 |
|---|---|
| `GET /health` | 服务状态、各路视频是否在线 |
| `GET /alerts?limit=50` | 最近的告警 |
| `GET /alerts/20261101/xxx.jpg` | 告警截图 |
| `POST /predict?conf=0.4` | 上传一张图片（请求体直接放图片文件，或 JSON `{"image": "base64"}`），返回检测结果 |

```bash
curl -X POST --data-binary @test.jpg "http://127.0.0.1:8090/predict?conf=0.4"
```

## 现场回流

置信度在 `low`～`high` 之间、模型拿不准的画面，会保存到 `reflux/`（每路视频每 `interval` 秒最多一张）。
定期把 `reflux` 文件夹拷回来，在 xlabel 的「部署」页点「导入现场回流图片」，标注后重新训练，模型会越用越准。

## 文件

```
model.onnx / model.json   模型和类别
service.py                服务程序
config.json               配置
alerts/                   告警截图
reflux/                   回流图片
logs/service.log          运行日志
```
