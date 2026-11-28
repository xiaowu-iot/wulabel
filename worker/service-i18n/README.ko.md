# 추론 서비스 패키지

Wulabel에서 생성한 패키지입니다. 카메라 영상 스트림을 받아 학습된 모델로 검출하고, 대상이 발견되면 경보를 푸시합니다. HTTP 검출 API도 제공합니다.

## 실행

Python 3.9 이상이 필요합니다(PyTorch와 그래픽카드는 필요 없습니다).

- Windows: `run.bat` 더블클릭
- Linux: `sh run.sh`

처음 실행하면 Python 환경을 자동으로 만들고 onnxruntime, opencv, numpy(약 100MB)를 설치합니다. 그다음부터는 바로 시작됩니다.

NVIDIA 그래픽카드로 GPU 추론을 하려면: `requirements.txt`의 `onnxruntime`을 `onnxruntime-gpu`로 바꾸고, `venv` 폴더를 삭제한 뒤 다시 실행하세요.

## 설정: config.json

```jsonc
{
  "model": "model.onnx",
  "gpu": true,                 // onnxruntime-gpu가 있으면 GPU 사용
  "max_side": 1280,            // 검출 전에 프레임의 긴 변을 이 크기로 축소, 작을수록 빠름
  "port": 8090,                // HTTP API 포트
  "interval": 1.0,             // 스트림마다 몇 초에 한 프레임씩 검출할지
  "streams": [
    {
      "name": "북문",
      "url": "rtsp://admin:비밀번호@192.168.1.64:554/Streaming/Channels/101",
      "region": [[0.1, 0.4], [0.9, 0.4], [0.9, 1.0], [0.1, 1.0]]  // 선택: 검출 영역, 0~1 비율 좌표
    }
  ],
  "alert": {
    "labels": ["안전모_미착용"],  // 이 클래스만 경보, 비우면 모든 클래스
    "min_score": 0.5,          // 신뢰도 임계값
    "class_scores": {},        // 클래스별 임계값(예: {"긁힘": 0.3}), 없는 클래스는 min_score 사용
    "cooldown": 60,            // 같은 스트림·같은 클래스는 N초에 한 번만 경보
    "hours": "07:00-19:00",    // 적용 시간대, 비우면 하루 종일, 자정을 넘기면 "22:00-06:00"
    "save_image": true         // 경보 스냅샷을 alerts/에 저장
  },
  "webhook": { "url": "http://플랫폼/api/alarm", "with_image": false, "headers": {} },
  "mqtt": { "host": "", "port": 1883, "topic": "xlabel/alarm", "username": "", "password": "" },
  "reflux": { "enabled": true, "low": 0.3, "high": 0.5, "interval": 60 },
  "watch": { "dir": "D:/videos", "out": "watch_results", "every": 1.0, "interval": 10, "settle": 5 }  // 선택: 폴더 모드
}
```

- 영상 소스는 RTSP, HTTP-FLV, HLS(m3u8), http의 MP4를 지원하며, 로컬 동영상 파일 경로를 넣어 테스트할 수도 있습니다(로컬 파일과 http 주문형 영상은 기본적으로 반복 재생). 로컬 영상은 폴더가 아니라 파일 자체를 입력해야 합니다(예: `D:\\test\\1.mp4`, JSON에서는 백슬래시를 두 번 쓰거나 `D:/test/1.mp4`로 씁니다). 로그에 "연결 실패"가 계속 뜨면 먼저 이것을 확인하세요. 폴더 안의 이미지를 인식하려면 `POST /predict`로 한 장씩 보내세요. 로컬 영상을 한 번만 재생하려면 해당 스트림에 `"loop": false`를 추가하세요. 여러 영상을 한 번씩 보고 폴더별로 결과를 내려면 Wulabel 배포 페이지의 "일괄 인식"이 더 편합니다.
- 검출 영역은 박스의 **아래쪽 변 중점**(사람이나 차량이 서 있는 위치)을 기준으로 판단합니다.

## 경보 내용

Webhook(POST JSON)과 MQTT로 전송되는 메시지:

```json
{
  "stream": "북문",
  "time": "2026-11-01T08:30:15",
  "labels": ["안전모_미착용"],
  "detections": [{ "label": "안전모_미착용", "score": 0.91, "box": [x, y, w, h] }],
  "width": 1920,
  "height": 1080,
  "image": "alerts/20261101/북문_083015.jpg"
}
```

Webhook에서 `with_image`를 켜면 `image_base64` 필드(박스를 그린 JPEG)가 추가됩니다.

## 모델별 결과

`detections` 각 항목의 필드는 모델에 따라 다릅니다. `GET /health`와 `POST /predict`가 반환하는 `task`로 모델 종류를 알 수 있습니다:

| 모델 | task | 각 항목의 필드 |
|---|---|---|
| 객체 검출 | `detect` | `label`, `score`, `box` [x, y, w, h] |
| 인스턴스 분할 | `segment` | 위와 같고 `polygon` [x1, y1, x2, y2, …]가 추가됨(대상 윤곽, 원본 이미지 픽셀) |
| 키포인트 | `pose` | 위와 같고 `keypoints` [x, y, v, …]가 추가됨(v가 2면 찾음, 0이면 못 찾음). 경보 캡처는 `config.json`의 `skeleton`으로 골격을 그림 |
| 이미지 분류 | `classify` | `label`, `score`, `box` 없음: 확률이 `min_score` 이상인 클래스마다 한 항목(최대 5개, 다중 라벨 분류 모델은 제한 없음). 분류는 화면 전체로 판단하며 `region`으로 거르지 않음 |

## HTTP API

| 엔드포인트 | 설명 |
|---|---|
| `GET /health` | 서비스 상태, 각 스트림의 온라인 여부 |
| `GET /alerts?limit=50` | 최근 경보 |
| `GET /alerts/20261101/xxx.jpg` | 경보 스냅샷 |
| `POST /predict?conf=0.4` | 이미지 한 장 업로드(요청 본문에 이미지 파일을 그대로 넣거나 JSON `{"image": "base64"}`), 검출 결과 반환 |

```bash
curl -X POST --data-binary @test.jpg "http://127.0.0.1:8090/predict?conf=0.4"
```

## 폴더 모드

설정에 `watch.dir`를 적으면 켜집니다. 서비스가 `interval`초마다 이 폴더(하위 폴더 포함)를 확인해 새로 넣은 이미지와 영상을 자동으로 인식하며, 카메라를 연결하지 않아도 됩니다.

- 파일은 놓인 지 `settle`초가 지나고 크기가 더 이상 바뀌지 않아야 인식합니다. 아직 복사 중인 파일을 절반만 인식하는 일은 없습니다.
- 영상은 `every`초마다 한 프레임을 봅니다. 인식된 클래스마다 박스를 그린 스크린샷을 한 장(점수가 가장 높은 프레임) 저장하며, 파일이 있는 하위 폴더(예: 주문 번호)별로 `out`(기본은 서비스 패키지 안의 `watch_results/`)에 둡니다.
- 파일마다 `watch_results/인식결과.csv`에 한 행을 추가합니다: 시간, 파일, 폴더, 결과, 최고 점수, 스크린샷.
- 경보 규칙(클래스, 신뢰도, 시간대)에 걸린 것은 평소처럼 Webhook / MQTT로 푸시하며, 메시지에 `file`, `folder` 두 항목이 더 들어갑니다.
- 인식한 파일은 `watch_results/.wulabel-watch.json`에 기록되므로 서비스를 다시 시작해도 다시 인식하지 않습니다. 파일이 바뀌면 다시 인식합니다. Wulabel 일괄 인식의 결과 폴더는 자동으로 건너뜁니다.
- `GET /health`의 `watch`에서 몇 개 파일을 인식했는지, 마지막 파일이 무엇인지 볼 수 있습니다.

## 현장 데이터 회수

신뢰도가 `low`~`high` 사이로 모델이 확신하지 못한 프레임은 `reflux/`에 저장됩니다(스트림마다 `interval`초에 최대 한 장).
주기적으로 `reflux` 폴더를 복사해 와서 Wulabel의 '배포' 페이지에서 '현장 회수 이미지 가져오기'를 클릭하고, 라벨링한 뒤 다시 학습하면 쓸수록 모델이 정확해집니다.

## 파일

```
model.onnx / model.json   모델과 클래스
service.py                서비스 프로그램
config.json               설정
alerts/                   경보 스냅샷
reflux/                   회수 이미지
watch_results/            폴더 모드 스크린샷과 표
logs/service.log          실행 로그
```
