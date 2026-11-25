# Paket layanan inferensi

Dibuat oleh Wulabel. Paket ini mengambil stream video kamera, menjalankan model hasil pelatihan untuk mendeteksi, dan mengirim alarm saat target ditemukan. Paket ini juga menyediakan API deteksi HTTP.

## Menjalankan

Membutuhkan Python 3.9 atau lebih baru (tidak perlu PyTorch maupun kartu grafis).

- Windows: klik dua kali `run.bat`
- Linux: `sh run.sh`

Saat pertama kali dijalankan, lingkungan Python dibuat otomatis lalu onnxruntime, opencv, dan numpy diinstal (sekitar 100 MB). Setelah itu langsung berjalan.

Untuk inferensi dengan GPU NVIDIA: ganti `onnxruntime` menjadi `onnxruntime-gpu` di `requirements.txt`, hapus folder `venv`, lalu jalankan lagi.

## Konfigurasi config.json

```jsonc
{
  "model": "model.onnx",
  "gpu": true,                 // pakai GPU jika onnxruntime-gpu terpasang
  "max_side": 1280,            // perkecil sisi panjang frame ke ukuran ini sebelum deteksi; makin kecil makin cepat
  "port": 8090,                // port API HTTP
  "interval": 1.0,             // deteksi satu frame setiap N detik per stream
  "streams": [
    {
      "name": "Gerbang utara",
      "url": "rtsp://admin:katasandi@192.168.1.64:554/Streaming/Channels/101",
      "region": [[0.1, 0.4], [0.9, 0.4], [0.9, 1.0], [0.1, 1.0]]  // opsional: area deteksi, koordinat rasio 0–1
    }
  ],
  "alert": {
    "labels": ["tanpa_helm"],  // hanya beri alarm untuk kelas ini; kosong berarti semua kelas
    "min_score": 0.5,          // ambang keyakinan
    "class_scores": {},        // ambang per kelas, mis. {"goresan": 0.3}; kelas lain memakai min_score
    "cooldown": 60,            // paling banyak satu alarm per N detik untuk stream dan kelas yang sama
    "hours": "07:00-19:00",    // jam aktif; kosong berarti sepanjang hari; melewati tengah malam tulis "22:00-06:00"
    "save_image": true         // simpan tangkapan alarm ke alerts/
  },
  "webhook": { "url": "http://platform-anda/api/alarm", "with_image": false, "headers": {} },
  "mqtt": { "host": "", "port": 1883, "topic": "xlabel/alarm", "username": "", "password": "" },
  "reflux": { "enabled": true, "low": 0.3, "high": 0.5, "interval": 60 }
}
```

- Sumber video mendukung RTSP, HTTP-FLV, HLS (m3u8), dan MP4 lewat http, juga bisa berupa path file video lokal untuk pengujian (file lokal dan video on-demand lewat http diputar berulang secara default).
- Area deteksi diperiksa berdasarkan **titik tengah sisi bawah** setiap kotak (tempat orang atau kendaraan berdiri).

## Isi alarm

Pesan yang dikirim ke Webhook (POST JSON) dan MQTT:

```json
{
  "stream": "Gerbang utara",
  "time": "2026-11-01T08:30:15",
  "labels": ["tanpa_helm"],
  "detections": [{ "label": "tanpa_helm", "score": 0.91, "box": [x, y, w, h] }],
  "width": 1920,
  "height": 1080,
  "image": "alerts/20261101/Gerbang utara_083015.jpg"
}
```

Jika `with_image` diaktifkan pada Webhook, ada tambahan field `image_base64` (JPEG dengan kotak tergambar).

## Hasil menurut jenis model

Field setiap item di `detections` bergantung pada model. Field `task` yang dikembalikan `GET /health` dan `POST /predict` menunjukkan jenisnya:

| Model | task | Field setiap item |
|---|---|---|
| Deteksi objek | `detect` | `label`, `score`, `box` [x, y, w, h] |
| Segmentasi instans | `segment` | Sama, ditambah `polygon` [x1, y1, x2, y2, …] (kontur objek, piksel gambar asli) |
| Keypoint | `pose` | Sama, ditambah `keypoints` [x, y, v, …] (v = 2: ditemukan, 0: tidak ditemukan); gambar alarm menggambar kerangka sesuai `skeleton` di `config.json` |
| Klasifikasi gambar | `classify` | `label`, `score`, tanpa `box`: satu item untuk setiap kelas yang probabilitasnya mencapai `min_score` (maksimal 5; tanpa batas untuk model multi-label). Klasifikasi menilai seluruh frame dan tidak disaring dengan `region` |

## API HTTP

| Endpoint | Keterangan |
|---|---|
| `GET /health` | Status layanan dan apakah setiap stream online |
| `GET /alerts?limit=50` | Alarm terbaru |
| `GET /alerts/20261101/xxx.jpg` | Tangkapan alarm |
| `POST /predict?conf=0.4` | Unggah satu gambar (file gambar langsung sebagai body permintaan, atau JSON `{"image": "base64"}`) dan dapatkan hasil deteksi |

```bash
curl -X POST --data-binary @test.jpg "http://127.0.0.1:8090/predict?conf=0.4"
```

## Data balik dari lapangan

Frame yang membuat model ragu (keyakinan antara `low` dan `high`) disimpan ke `reflux/` (paling banyak satu per stream setiap `interval` detik).
Salin folder `reflux` secara berkala, klik "Impor gambar dari lapangan" di halaman "Deploy" pada Wulabel, anotasi gambarnya, lalu latih ulang — model makin akurat seiring pemakaian.

## File

```
model.onnx / model.json   model dan kelas
service.py                program layanan
config.json               konfigurasi
alerts/                   tangkapan alarm
reflux/                   gambar dari lapangan
logs/service.log          log
```
