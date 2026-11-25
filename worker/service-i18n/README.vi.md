# Gói dịch vụ suy luận

Được tạo bởi Wulabel. Gói này lấy luồng video từ camera, dùng mô hình đã huấn luyện để phát hiện và đẩy cảnh báo khi tìm thấy mục tiêu; đồng thời cung cấp API phát hiện qua HTTP.

## Chạy

Cần Python 3.9 trở lên (không cần PyTorch, cũng không cần card đồ họa).

- Windows: nhấp đúp `run.bat`
- Linux: `sh run.sh`

Lần chạy đầu tiên sẽ tự tạo môi trường Python và cài onnxruntime, opencv, numpy (khoảng 100 MB). Từ lần sau sẽ khởi động ngay.

Muốn suy luận bằng GPU NVIDIA: đổi `onnxruntime` thành `onnxruntime-gpu` trong `requirements.txt`, xóa thư mục `venv` rồi chạy lại.

## Cấu hình config.json

```jsonc
{
  "model": "model.onnx",
  "gpu": true,                 // dùng GPU khi đã cài onnxruntime-gpu
  "max_side": 1280,            // thu nhỏ cạnh dài của khung hình về kích thước này trước khi phát hiện, càng nhỏ càng nhanh
  "port": 8090,                // cổng API HTTP
  "interval": 1.0,             // mỗi luồng video cứ bao nhiêu giây thì phát hiện một khung hình
  "streams": [
    {
      "name": "Cổng Bắc",
      "url": "rtsp://admin:matkhau@192.168.1.64:554/Streaming/Channels/101",
      "region": [[0.1, 0.4], [0.9, 0.4], [0.9, 1.0], [0.1, 1.0]]  // tùy chọn: vùng phát hiện, tọa độ theo tỉ lệ 0~1
    }
  ],
  "alert": {
    "labels": ["khong_mu_bao_ho"],  // chỉ cảnh báo các lớp này, để trống là tất cả các lớp
    "min_score": 0.5,          // ngưỡng độ tin cậy
    "cooldown": 60,            // cùng một luồng, cùng một lớp, trong bao nhiêu giây chỉ cảnh báo một lần
    "hours": "07:00-19:00",    // khung giờ áp dụng, để trống là cả ngày, qua nửa đêm thì ghi "22:00-06:00"
    "save_image": true         // lưu ảnh chụp cảnh báo vào alerts/
  },
  "webhook": { "url": "http://nen-tang-cua-ban/api/alarm", "with_image": false, "headers": {} },
  "mqtt": { "host": "", "port": 1883, "topic": "xlabel/alarm", "username": "", "password": "" },
  "reflux": { "enabled": true, "low": 0.3, "high": 0.5, "interval": 60 }
}
```

- Nguồn video hỗ trợ RTSP, HTTP-FLV, HLS (m3u8), MP4 qua http, cũng có thể là đường dẫn tới tệp video cục bộ để thử nghiệm (tệp cục bộ và video theo yêu cầu qua http mặc định phát lặp lại).
- Vùng phát hiện được xét theo **trung điểm cạnh dưới** của khung đối tượng (vị trí người, xe đứng).

## Nội dung cảnh báo

Thông điệp gửi tới Webhook (POST JSON) và MQTT:

```json
{
  "stream": "Cổng Bắc",
  "time": "2026-11-01T08:30:15",
  "labels": ["khong_mu_bao_ho"],
  "detections": [{ "label": "khong_mu_bao_ho", "score": 0.91, "box": [x, y, w, h] }],
  "width": 1920,
  "height": 1080,
  "image": "alerts/20261101/Cổng Bắc_083015.jpg"
}
```

Khi bật `with_image` cho Webhook, sẽ có thêm trường `image_base64` (ảnh JPEG đã vẽ khung).

## Kết quả theo từng loại mô hình

Các trường của mỗi mục trong `detections` tùy theo mô hình. Trường `task` do `GET /health` và `POST /predict` trả về cho biết loại mô hình:

| Mô hình | task | Các trường của mỗi mục |
|---|---|---|
| Phát hiện đối tượng | `detect` | `label`, `score`, `box` [x, y, w, h] |
| Phân đoạn thực thể | `segment` | Như trên, thêm `polygon` [x1, y1, x2, y2, …] (đường viền đối tượng, pixel ảnh gốc) |
| Điểm chính | `pose` | Như trên, thêm `keypoints` [x, y, v, …] (v = 2: tìm thấy, 0: không tìm thấy); ảnh chụp cảnh báo vẽ khung xương theo `skeleton` trong `config.json` |
| Phân loại ảnh | `classify` | `label`, `score`, không có `box`: mỗi lớp có xác suất đạt `min_score` là một mục (tối đa 5; mô hình đa nhãn không giới hạn). Phân loại xét toàn bộ khung hình, không lọc theo `region` |

## API HTTP

| Endpoint | Mô tả |
|---|---|
| `GET /health` | Trạng thái dịch vụ, từng luồng video có trực tuyến hay không |
| `GET /alerts?limit=50` | Các cảnh báo gần đây |
| `GET /alerts/20261101/xxx.jpg` | Ảnh chụp cảnh báo |
| `POST /predict?conf=0.4` | Tải lên một ảnh (đặt thẳng tệp ảnh vào thân yêu cầu, hoặc JSON `{"image": "base64"}`), trả về kết quả phát hiện |

```bash
curl -X POST --data-binary @test.jpg "http://127.0.0.1:8090/predict?conf=0.4"
```

## Dữ liệu hồi lưu từ hiện trường

Những khung hình mà mô hình chưa chắc chắn (độ tin cậy nằm giữa `low` và `high`) sẽ được lưu vào `reflux/` (mỗi luồng tối đa một ảnh mỗi `interval` giây).
Định kỳ chép thư mục `reflux` về, vào trang "Triển khai" của Wulabel và nhấp "Nhập ảnh thu từ hiện trường", gán nhãn rồi huấn luyện lại — mô hình sẽ càng dùng càng chính xác.

## Tệp

```
model.onnx / model.json   mô hình và các lớp
service.py                chương trình dịch vụ
config.json               cấu hình
alerts/                   ảnh chụp cảnh báo
reflux/                   ảnh hồi lưu
logs/service.log          nhật ký chạy
```
