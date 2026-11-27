<div align="center">

# Wulabel

**Offline AI annotation and training for the desktop**

Import data → annotate → train → evaluate → pre-label with your model → export and deploy, all in one app

English · [简体中文](README.md)

[Website](https://wulabel.xiaowuiot.com/en/) · [Download](#download) · [User guide](https://wulabel.xiaowuiot.com/en/guide.html) ([Markdown](docs/i18n/en/使用教程-详细版.md)) · [Releases](https://github.com/xiaowu-iot/wulabel/releases)

[![Latest release](https://img.shields.io/badge/release-v0.6.5-brightgreen)](https://github.com/xiaowu-iot/wulabel/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Platforms](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](#download)

</div>

## Highlights

- **Fully offline**: images, annotations and models stay on your own computer
- **Ready to train**: one-click training environment setup that detects your GPU and picks the right PyTorch build
- **Four tasks**: object detection, instance segmentation, image classification, keypoints
- **YOLO-family training**: YOLOX Nano / Tiny / S / M / L, plus Faster R-CNN, RetinaNet, Mask R-CNN and more
- **Gets faster as you go**: SAM outlines objects with one click, text prompts find objects automatically; train on a small batch, then let the model pre-label the rest
- **Team collaboration**: members annotate in their browser, with task assignment, review and per-person statistics
- **Deploy to the field**: ONNX, OpenVINO, TensorRT, RKNN, and a ready-to-run inference service package
- **Commercial-friendly**: Apache-2.0; built-in models only use BSD / Apache licensed code and weights (torchvision, YOLOX)
- **Multilingual UI** (from v0.5.0): follows your system language — English, 简体中文, 繁體中文, 日本語, 한국어, Español, Português, Русский, Deutsch, Français, Tiếng Việt, Bahasa Indonesia

## Download

Get the latest installers from [GitHub Releases](https://github.com/xiaowu-iot/wulabel/releases/latest):

| System | File |
|---|---|
| Windows 10 / 11 (64-bit) | `Wulabel_<version>_x64-setup.exe` |
| macOS 11+ (Apple silicon and Intel) | `Wulabel_<version>_universal.dmg` |
| Linux x86_64 | `Wulabel_<version>_amd64.deb` or `Wulabel_<version>_amd64.AppImage` |
| Linux ARM64 | `Wulabel_<version>_arm64.deb` or `Wulabel_<version>_aarch64.AppImage` |

Files ending in `.sig` or `.tar.gz` are used by the in-app updater; you don't need them for a manual install.

**First install**

- **Windows**: if you see "Windows protected your PC", click "More info" → "Run anyway" (the installer is not code-signed yet).
- **macOS**: drag Wulabel into Applications. If macOS says the developer cannot be verified, right-click Wulabel in Applications → "Open" (on macOS 15+, use System Settings → Privacy & Security → "Open Anyway"). If it says the app is damaged, run `xattr -cr /Applications/Wulabel.app` in Terminal.
- **Linux**: `sudo apt install ./Wulabel_<version>_amd64.deb`, or `chmod +x` the AppImage and run it.
- Then open **Settings** and click the one-click training environment install. You can skip this if you only annotate.

Updates: the app checks for new versions on startup; click "Update now" and it downloads, verifies, installs and restarts by itself.

## Features

- **Data**: import images and folders with de-duplication and auto-rotation; extract frames from videos and from live or playback streams (RTSP, HTTP-FLV, HLS/m3u8, MP4); import YOLO / COCO / VOC / class-folder datasets (format detected automatically); train / val / test split; image quality screening; pick the most valuable images to label first; anonymize faces and license plates on export
- **Annotation**: boxes, polygons, classification, keypoints; switch classes with 1–9; undo / redo, copy from previous image; keyframe interpolation and tracking for video frames
- **AI assistance**: SAM click-to-outline; text-prompt detection; pre-label with your own trained model (single image or batch)
- **Health check**: rule checks (tiny boxes, duplicate boxes and images, class imbalance) and model checks (likely missed labels, wrong classes, loose boxes, extra boxes)
- **Dataset versions**: snapshot annotations, compare, export, train on a version, restore
- **Training**: detection — YOLOX, Faster R-CNN, RetinaNet, FCOS, SSDLite; segmentation — Mask R-CNN; keypoints — Keypoint R-CNN; classification — MobileNetV3, EfficientNet, ResNet50; you can also train Ultralytics YOLO11, YOLO26, YOLOv8, YOLO12, YOLOv10, YOLOv9 and RT-DETR (detection, segmentation, keypoints, classification), enter any model name or continue from a local .pt, with automatic ONNX conversion. Augmentation, resume training, run comparison; batch size is reduced automatically when GPU memory runs out; import external ONNX models (YOLOv5 / v8 / v11, YOLOX, RT-DETR…, any task) for pre-labeling, evaluation and deployment
- **Evaluation**: mAP, precision, recall (accuracy and F1 for classification); per-class metrics, confusion matrix, error browser; threshold suggestions; false-positive test; delivery report export
- **Deployment**: live test on a camera or video stream; export ONNX, OpenVINO, TensorRT / RKNN conversion packages; generate an inference service package (multi-stream, alert rules, Webhook / MQTT). Live test and service packages support detection, segmentation, keypoint and classification models
- **Collaboration**: your computer hosts, members log in from a browser over LAN or online; roles, assignment, per-image locks, review, workload stats, watermark. Or split into offline packages and merge them back

## Development

Requires Node.js 22+ and Rust (stable); on Windows also the Visual Studio C++ build tools and WebView2.

```bash
cd apps/desktop
npm install
npm run tauri dev      # run in development
npm run tauri build    # build installers
```

Tests:

```bash
cd apps/desktop/src-tauri && cargo test            # Rust unit tests
cd worker && python -m unittest discover -s tests  # Python tests (metrics, YOLOX smoke test)
```

### Translations

UI text uses the Chinese source string as the key. Translations live in three places, one JSON file per language:

- `apps/desktop/src/locales/` — UI (`t("中文")` in TypeScript)
- `apps/desktop/src-tauri/locales/` — backend messages (`t!("中文")` in Rust)
- `worker/xlabel_worker/locales/` — training / inference logs (`tr("中文")` in Python)

Missing entries fall back to English, then Chinese. Run `python apps/desktop/scripts/i18n_check.py` to see untranslated text and missing entries per language. Corrections to translations are very welcome — open an issue or a pull request.

## License

[Apache-2.0](LICENSE). Third-party code (YOLOX, Apache-2.0) is listed in [NOTICE](NOTICE).

Author: 小物学长 (177183719@qq.com). Questions and feature requests: [GitHub Issues](https://github.com/xiaowu-iot/wulabel/issues) or email.
