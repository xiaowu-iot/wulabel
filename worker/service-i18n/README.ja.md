# 推論サービスパッケージ

Wulabel で作成されたパッケージです。カメラの映像ストリームを取得し、学習済みモデルで検出して、対象が見つかるとアラートを送信します。HTTP の検出 API も提供します。

## 実行

Python 3.9 以降が必要です（PyTorch も GPU も不要です）。

- Windows：`run.bat` をダブルクリック
- Linux：`sh run.sh`

初回実行時は Python 環境を自動で作成し、onnxruntime、opencv、numpy（約 100 MB）をインストールします。2 回目以降はすぐに起動します。

NVIDIA の GPU で推論する場合：`requirements.txt` の `onnxruntime` を `onnxruntime-gpu` に置き換え、`venv` フォルダーを削除してから再実行してください。

## 設定：config.json

```jsonc
{
  "model": "model.onnx",
  "gpu": true,                 // onnxruntime-gpu があれば GPU を使用
  "max_side": 1280,            // 検出前にフレームの長辺をこのサイズに縮小。小さいほど高速
  "port": 8090,                // HTTP API のポート
  "interval": 1.0,             // 各ストリームで何秒ごとに 1 フレームを検出するか
  "streams": [
    {
      "name": "北門",
      "url": "rtsp://admin:パスワード@192.168.1.64:554/Streaming/Channels/101",
      "region": [[0.1, 0.4], [0.9, 0.4], [0.9, 1.0], [0.1, 1.0]]  // 任意：検出領域。0～1 の比率座標
    }
  ],
  "alert": {
    "labels": ["ヘルメット未着用"],  // これらのクラスのみアラート。空なら全クラス
    "min_score": 0.5,          // 信頼度のしきい値
    "cooldown": 60,            // 同じストリーム・同じクラスのアラートは N 秒に 1 回まで
    "hours": "07:00-19:00",    // 有効な時間帯。空なら終日。日をまたぐ場合は "22:00-06:00"
    "save_image": true         // アラートのスナップショットを alerts/ に保存
  },
  "webhook": { "url": "http://プラットフォーム/api/alarm", "with_image": false, "headers": {} },
  "mqtt": { "host": "", "port": 1883, "topic": "xlabel/alarm", "username": "", "password": "" },
  "reflux": { "enabled": true, "low": 0.3, "high": 0.5, "interval": 60 }
}
```

- 映像ソースは RTSP、HTTP-FLV、HLS（m3u8）、http 上の MP4 に対応しています。ローカルの動画ファイルのパスも指定でき、テストに使えます（ローカルファイルと http のオンデマンド動画はデフォルトでループ再生）。
- 検出領域に入っているかは、検出枠の**下辺の中点**（人や車が立っている位置）で判定します。

## アラートの内容

Webhook（POST JSON）と MQTT に送られるメッセージ：

```json
{
  "stream": "北門",
  "time": "2026-11-01T08:30:15",
  "labels": ["ヘルメット未着用"],
  "detections": [{ "label": "ヘルメット未着用", "score": 0.91, "box": [x, y, w, h] }],
  "width": 1920,
  "height": 1080,
  "image": "alerts/20261101/北門_083015.jpg"
}
```

Webhook で `with_image` を有効にすると、`image_base64` フィールド（枠を描画した JPEG）が追加されます。

## モデルごとの結果

`detections` の各要素のフィールドはモデルによって異なります。`GET /health` と `POST /predict` が返す `task` でモデルの種類がわかります：

| モデル | task | 各要素のフィールド |
|---|---|---|
| 物体検出 | `detect` | `label`、`score`、`box` [x, y, w, h] |
| インスタンスセグメンテーション | `segment` | 同上に加えて `polygon` [x1, y1, x2, y2, …]（対象の輪郭、元画像のピクセル座標） |
| キーポイント | `pose` | 同上に加えて `keypoints` [x, y, v, …]（v が 2 なら検出、0 なら未検出）。アラート画像は `config.json` の `skeleton` で骨格を描画 |
| 画像分類 | `classify` | `label`、`score`（`box` なし）：確率が `min_score` 以上のクラスごとに 1 要素（最大 5 個。マルチラベル分類モデルは上限なし）。分類は画面全体で判定し、`region` では絞り込みません |

## HTTP API

| エンドポイント | 説明 |
|---|---|
| `GET /health` | サービスの状態、各ストリームがオンラインかどうか |
| `GET /alerts?limit=50` | 最近のアラート |
| `GET /alerts/20261101/xxx.jpg` | アラートのスナップショット |
| `POST /predict?conf=0.4` | 画像を 1 枚アップロード（リクエスト本文に画像ファイルをそのまま入れるか、JSON `{"image": "base64"}`）すると、検出結果を返します |

```bash
curl -X POST --data-binary @test.jpg "http://127.0.0.1:8090/predict?conf=0.4"
```

## 現場データの還流

信頼度が `low`～`high` の間で、モデルが判断に迷ったフレームは `reflux/` に保存されます（各ストリームで `interval` 秒ごとに最大 1 枚）。
定期的に `reflux` フォルダーを持ち帰り、Wulabel の「デプロイ」ページで「現場から戻った画像をインポート」をクリックし、アノテーションして再学習すれば、使うほどモデルの精度が上がります。

## ファイル

```
model.onnx / model.json   モデルとクラス
service.py                サービス本体
config.json               設定
alerts/                   アラートのスナップショット
reflux/                   還流画像
logs/service.log          ログ
```
