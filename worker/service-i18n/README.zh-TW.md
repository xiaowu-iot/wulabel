# 推論服務包

由 Wulabel 產生。擷取攝影機影像串流，用訓練好的模型偵測，發現目標就推播警報；同時提供 HTTP 偵測 API。

## 執行

需要 Python 3.9 以上版本（不需要 PyTorch，也不需要顯示卡）。

- Windows：雙擊 `run.bat`
- Linux：`sh run.sh`

第一次執行會自動建立 Python 環境並安裝 onnxruntime、opencv、numpy（約 100MB）。之後直接啟動。

有 NVIDIA 顯示卡想用 GPU 推論：把 `requirements.txt` 裡的 `onnxruntime` 換成 `onnxruntime-gpu`，刪除 `venv` 資料夾後重新執行。

## 設定 config.json

```jsonc
{
  "model": "model.onnx",
  "gpu": true,                 // 有 onnxruntime-gpu 時使用顯示卡
  "max_side": 1280,            // 偵測前把畫面長邊縮到這個尺寸，越小越快
  "port": 8090,                // HTTP API 連接埠
  "interval": 1.0,             // 每路串流每隔幾秒偵測一個影格
  "streams": [
    {
      "name": "北門",
      "url": "rtsp://admin:密碼@192.168.1.64:554/Streaming/Channels/101",
      "region": [[0.1, 0.4], [0.9, 0.4], [0.9, 1.0], [0.1, 1.0]]  // 選填：偵測區域，0~1 的比例座標
    }
  ],
  "alert": {
    "labels": ["未戴安全帽"],  // 只對這些類別發出警報，空白表示全部類別
    "min_score": 0.5,          // 信心值門檻
    "class_scores": {},        // 每個類別單獨的閾值，如 {"刮痕": 0.3}，沒寫的類別用 min_score
    "cooldown": 60,            // 同一路串流同一類別，多少秒內只警報一次
    "hours": "07:00-19:00",    // 生效時段，空白表示全天，跨日寫 "22:00-06:00"
    "save_image": true         // 儲存警報截圖到 alerts/
  },
  "webhook": { "url": "http://你的平台/api/alarm", "with_image": false, "headers": {} },
  "mqtt": { "host": "", "port": 1883, "topic": "xlabel/alarm", "username": "", "password": "" },
  "reflux": { "enabled": true, "low": 0.3, "high": 0.5, "interval": 60 }
}
```

- 影像來源支援 RTSP、HTTP-FLV、HLS（m3u8）、http 上的 MP4，也可以填本機影片檔路徑用於測試（本機檔案和 http 點播預設循環播放）。本機影片要填到檔案本身，例如 `D:\\測試\\1.mp4`（JSON 裡反斜線寫兩個，或者寫成 `D:/測試/1.mp4`），不能只填資料夾；執行記錄一直顯示「連線失敗」時先查這個。一個資料夾的圖片要辨識，用 `POST /predict` 逐張傳。本機影片只想播一遍的話，給這一路加 `"loop": false`。要把一批影片各看一遍、按資料夾出結果，用 Wulabel 部署頁的「批次辨識」更方便。
- 偵測區域依目標框的**底邊中點**判斷（人、車所站的位置）。

## 警報內容

Webhook（POST JSON）和 MQTT 收到的訊息：

```json
{
  "stream": "北門",
  "time": "2026-11-01T08:30:15",
  "labels": ["未戴安全帽"],
  "detections": [{ "label": "未戴安全帽", "score": 0.91, "box": [x, y, w, h] }],
  "width": 1920,
  "height": 1080,
  "image": "alerts/20261101/北門_083015.jpg"
}
```

Webhook 開啟 `with_image` 時，會多一個 `image_base64` 欄位（畫好框的 JPEG）。

## 不同模型的結果

`detections` 裡每一項的欄位依模型而定，`GET /health` 和 `POST /predict` 回傳的 `task` 說明是哪種模型：

| 模型 | task | 每一項的欄位 |
|---|---|---|
| 物件偵測 | `detect` | `label`、`score`、`box` [x, y, w, h] |
| 實例分割 | `segment` | 同上，多一個 `polygon` [x1, y1, x2, y2, …]（目標輪廓，原圖像素） |
| 關鍵點 | `pose` | 同上，多一個 `keypoints` [x, y, v, …]（v 為 2 表示找到了，0 表示沒找到）；警報截圖依 `config.json` 裡的 `skeleton` 畫骨架 |
| 影像分類 | `classify` | `label`、`score`，沒有 `box`：機率達到 `min_score` 的類別各一項（最多 5 個；多標籤分類模型不限個數）。分類依整張畫面判斷，不依 `region` 過濾 |

## HTTP API

| API | 說明 |
|---|---|
| `GET /health` | 服務狀態、各路串流是否在線 |
| `GET /alerts?limit=50` | 最近的警報 |
| `GET /alerts/20261101/xxx.jpg` | 警報截圖 |
| `POST /predict?conf=0.4` | 上傳一張圖片（請求本文直接放圖片檔，或 JSON `{"image": "base64"}`），回傳偵測結果 |

```bash
curl -X POST --data-binary @test.jpg "http://127.0.0.1:8090/predict?conf=0.4"
```

## 現場回流

信心值在 `low`～`high` 之間、模型拿不準的畫面，會儲存到 `reflux/`（每路串流每 `interval` 秒最多一張）。
定期把 `reflux` 資料夾複製回來，在 Wulabel 的「部署」頁點「匯入現場回流圖片」，標註後重新訓練，模型會越用越準。

## 檔案

```
model.onnx / model.json   模型和類別
service.py                服務程式
config.json               設定
alerts/                   警報截圖
reflux/                   回流圖片
logs/service.log          執行記錄
```
