# Wulabel

离线 AI 标注训练桌面软件：**导入数据 → 标注 → 训练 → 评估 → 用模型预标注 → 导出部署**，一个软件走完。

- 完全离线可用，数据不出本机
- 一键安装训练环境，自动识别显卡、选择 PyTorch 版本，支持国内镜像
- 越标越快：标一小批就能训练，再用训练出的模型预标注剩下的图
- 导出 YOLO / COCO / VOC 数据集和 ONNX 模型
- Windows 优先，后续支持 macOS
- Apache-2.0 开源，内置模型只用 BSD / Apache 许可的 torchvision

**官网**：<https://wulabel.xiaowuiot.com>　|　**使用教程**：[docs/使用教程.md](docs/使用教程.md)（在线版 <https://wulabel.xiaowuiot.com/guide.html>）　|　**反馈服务部署**：[server/feedback/](server/feedback/)

> 作者：小物学长。安装包发布在 Gitee / GitHub 的 Release，不经自建服务器。

## 当前功能（v0.1 开发中）

| 模块 | 功能 |
|---|---|
| 项目 | 新建/打开、最近项目；一个项目就是一个文件夹 |
| 数据 | 导入图片/文件夹/拖拽（复制或引用、按内容去重、EXIF 摆正）；导入带标注的 YOLO/COCO/VOC 数据集（自动识别格式）；撤销上次导入、多选删除、清空；导出 YOLO/COCO/VOC；训练/验证/测试划分 |
| 标注 | 矩形框：画框、选中、拖动、调整大小；类别管理（1–9 快捷切换）；撤销重做；复制上一张；自动保存；审核/跳过；模型预标注（单张 P 键、批量） |
| 体检 | 规则检查：极小框、重复框、近似重复图片（感知哈希）、类别不均衡、框大小分布；模型检查：疑似漏标、类别错、框不准、多余框；一键加为标注、忽略、跳到标注页高亮问题框 |
| 训练 | Faster R-CNN（MobileNetV3 / ResNet50），实时 loss 和 mAP 曲线、日志、中途停止、训练记录 |
| 评估 | mAP50、mAP50-95、精确率、召回率、各类别指标、混淆矩阵、错例浏览（点击直接去改标注） |
| 部署 | 实时验证（RTSP 摄像头/本地视频，截图加入数据集）；导出 ONNX（动态/固定尺寸）、OpenVINO（软件内转换并验证）、TensorRT/RKNN 转换包；生成推理服务包（多路拉流、告警规则、Webhook/MQTT 推送、HTTP 检测接口、现场回流）；导入回流图片 |
| 设置 | 训练环境一键安装、安装位置、下载源、使用已有 Python 环境 |

### 后续版本

| 版本 | 主题 |
|---|---|
| v0.2 | SAM 交互分割、实例分割、文本提示标注、AI 平台辅助、视频抽帧 |
| v0.3 | 训练对比、数据集版本 |
| v0.4 | 分类、关键点 |
| v1.0 | 正式版，macOS；单阶段检测模型（更适合 NPU/边缘设备） |

## 开发

需要：Node.js 20+、Rust（stable）、Windows 上还需要 VS C++ 生成工具和 WebView2。

```bash
cd apps/desktop
npm install
npm run tauri dev      # 开发运行
npm run tauri build    # 打安装包
```

测试：

```bash
# Rust 单元测试
cd apps/desktop/src-tauri
cargo test

# 训练全流程集成测试（需要装有 torch 的 Python）
XLABEL_TEST_PYTHON=/path/to/python cargo test end_to_end -- --nocapture

# Python 指标测试
cd worker
python -m unittest discover -s tests
```

## 目录

```
apps/desktop/
├─ src/                     React 前端
│  ├─ pages/                项目、数据、标注、训练、评估、设置
│  └─ components/           标注画布、导出弹窗、大图预览、折线图
└─ src-tauri/src/           Rust 核心
   ├─ project.rs            项目新建/打开、最近项目
   ├─ db.rs                 SQLite 表结构与迁移
   ├─ images.rs             图片导入、缩略图、删除
   ├─ annotations.rs        标注读写、类别、图片状态
   ├─ dataset.rs            数据集划分与导出
   ├─ dataset_import.rs     YOLO/COCO/VOC 导入
   ├─ health.rs             数据体检
   ├─ runtime.rs            训练环境：显卡检测、uv、PyTorch 安装
   ├─ training.rs           训练任务、推理服务（预标注）、ONNX 导出
   ├─ deploy.rs             实时验证、多格式导出、推理服务包
   ├─ protocol.rs           xl:// 图片协议
   └─ commands.rs           前端调用的命令
worker/                     Python 训练与推理进程（随安装包分发）
└─ xlabel_worker/
   ├─ train.py              训练与评估
   ├─ serve.py              常驻推理服务
   ├─ export.py             导出 ONNX / OpenVINO / TensorRT、RKNN 转换包
   ├─ live.py               实时验证（RTSP / 视频文件）
   ├─ metrics.py            mAP、混淆矩阵（只依赖 numpy）
   └─ models.py             torchvision 检测模型
└─ service/                 推理服务包模板（只依赖 onnxruntime + OpenCV，含中文说明）
```

桌面端和 worker 之间通过标准输入输出逐行传 JSON，不开端口。

## 项目文件格式

```
我的项目/
├─ project.json      项目名、任务类型
├─ annotations.db    SQLite：图片、类别、标注、训练记录
├─ images/           复制模式导入的图片
├─ runs/<时间>/      每次训练：dataset/、best.pt、last.pt、metrics.jsonl、eval.json、log.txt、model.onnx
└─ .cache/thumbs/    缩略图（可删除，会自动重建）
```

## 许可证

[Apache-2.0](LICENSE)
