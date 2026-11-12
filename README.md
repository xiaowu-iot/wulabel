# Wulabel

离线 AI 标注训练桌面软件：**导入数据 → 标注 → 训练 → 评估 → 用模型预标注 → 导出部署**，一个软件走完。

- 完全离线可用，数据不出本机
- 一键安装训练环境，自动识别显卡、选择 PyTorch 版本，支持国内镜像
- 四种任务：目标检测、实例分割、图像分类、关键点
- 越标越快：SAM 点一下出轮廓、文本提示自动找目标；标一小批就能训练，再用训练出的模型预标注剩下的图
- 导出 YOLO（含 seg / pose）/ COCO / VOC / 分类文件夹数据集和 ONNX 模型
- Windows 和 macOS（Apple 芯片用 MPS 加速训练）
- Apache-2.0 开源，内置模型只用 BSD / Apache 许可的 torchvision

**官网**：<https://wulabel.xiaowuiot.com>　|　**使用教程**：[docs/使用教程.md](docs/使用教程.md)（在线版 <https://wulabel.xiaowuiot.com/guide.html>）　|　**反馈服务部署**：[server/feedback/](server/feedback/)

> 作者：小物学长。打包方法见 [docs/打包发布.md](docs/打包发布.md)。

## 下载

最新版 **v0.2.1**，点文件名直接下载：

| 系统 | 官网（国内推荐） | GitHub |
|---|---|---|
| Windows 10 / 11（64 位） | [Wulabel_0.2.1_x64-setup.exe](https://wulabel.xiaowuiot.com/download/Wulabel_0.2.1_x64-setup.exe) | [Wulabel_0.2.1_x64-setup.exe](https://github.com/xiaowu-iot/wulabel/releases/download/v0.2.1/Wulabel_0.2.1_x64-setup.exe) |
| macOS 11 及以上（Apple 芯片和 Intel 通用） | [Wulabel_0.2.1_universal.dmg](https://wulabel.xiaowuiot.com/download/Wulabel_0.2.1_universal.dmg) | [Wulabel_0.2.1_universal.dmg](https://github.com/xiaowu-iot/wulabel/releases/download/v0.2.1/Wulabel_0.2.1_universal.dmg) |

- 历史版本和更新说明：[GitHub Releases](https://github.com/xiaowu-iot/wulabel/releases) ｜ [Gitee 发行版](https://gitee.com/xiaowu-iot/wulabel/releases)
- Windows 双击安装；Mac 打开 .dmg，把 Wulabel 拖进「应用程序」。
- Mac 第一次打开会提示「无法验证开发者」（没有苹果开发者签名）：在「应用程序」里右键 Wulabel → 打开；macOS 15 以后到「系统设置 → 隐私与安全性」点「仍要打开」。提示「已损坏」的话，在终端执行 `xattr -cr /Applications/Wulabel.app`。

## 当前功能（v0.2.1）

| 模块 | 功能 |
|---|---|
| 项目 | 新建/打开、最近项目；一个项目就是一个文件夹；任务类型：目标检测 / 实例分割 / 图像分类 / 关键点 |
| 数据 | 导入图片/文件夹/拖拽（复制或引用、按内容去重、EXIF 摆正）；视频抽帧；视频流抽帧（RTSP/直播/回放 URL，测试连接、边采边入库、可停止）；导入带标注的数据集（YOLO 含 seg/pose、COCO 含分割/关键点、VOC、分类文件夹，自动识别）；撤销上次导入、多选删除、清空；按任务导出（可批量给人脸、车牌等区域打码）；训练/验证/测试划分；图片质量筛查（模糊、过暗过曝、低对比）；主动学习挑图（挑模型最没把握的图先标）；从巡瞳 AI 平台拉取告警图片（识别框作为预标注） |
| 数据集版本 | 把已标注数据存成快照；和当前数据对比、导出某个版本、用某个版本训练、恢复（恢复前自动备份） |
| 多人协作 | 实时协作：发起人的电脑当主机，成员用浏览器登录就能标（不用装软件）；局域网或在线（自建中转 / 巡瞳官方中转）；成员和角色（标注员、审核员）、分配任务、按张加锁（别人在改就只读、实时同步）、审核打回（带原因）、在线状态、工作量统计（导出 CSV 计件）、操作记录、水印。离线分包：按人数分成几个标注包（独立项目，可带预标注、可打码），标完合并回来，冲突提示 |
| 标注 | 矩形框；多边形轮廓；分类（数字键一键一张）；关键点（可编辑模板，逐个点、遮挡/跳过、拖点微调）；类别管理（1–9 快捷切换）；撤销重做；复制上一张；视频帧关键帧插值、往后自动跟踪；自动保存；审核/跳过 |
| AI 辅助 | SAM 智能选取（点选/框选出轮廓）；文本提示（按英文类别名开放词汇检测，分割项目再用 SAM 出轮廓）；巡瞳 AI 平台（按项目类别识别：只认项目里的类别、按识别说明来；专用模型、大模型识别）；识别规则（每个类别的识别说明 + 整体规则，可以和 AI 聊天整理规则、在当前图上试识别）；自己训练的模型预标注（单张 P 键、批量） |
| 体检 | 规则检查：极小框、重复框、近似重复图片（感知哈希）、类别不均衡、框大小分布；模型检查：疑似漏标、类别错、框不准、多余框（分类：疑似标错类别）；一键加为标注、忽略、跳到标注页高亮问题框 |
| 训练 | 检测：Faster R-CNN（MobileNetV3 / ResNet50）、单阶段 RetinaNet / FCOS / SSDLite；分割：Mask R-CNN；关键点：Keypoint R-CNN；分类：MobileNetV3 / EfficientNet-B0 / ResNet50。数据增强（预设 + 自定义，效果预览）；接着已有模型继续训练；导入外部 ONNX 模型（YOLOv5/v8、YOLOX、RT-DETR 等，用于预标注、评估）；实时曲线、日志、中途停止、训练记录、训练对比 |
| 评估 | 检测类：mAP50、mAP50-95、精确率、召回率；分类：准确率、宏平均 F1；各类别指标、混淆矩阵、错例浏览（点击直接去改标注）；PR 曲线、阈值推荐；误报测试（拿不含目标的视频/视频流跑，统计每小时误报，误报帧一键加为负样本）；用当前数据重新评估；导出交付报告（单个 HTML） |
| 部署 | 实时验证（RTSP 摄像头/本地视频，截图加入数据集）；导出 ONNX（动态/固定尺寸）、OpenVINO（软件内转换并验证）、TensorRT/RKNN 转换包（检测、分类）；生成推理服务包（检测；多路拉流、告警规则、Webhook/MQTT 推送、HTTP 检测接口、现场回流）；导入回流图片；上传到巡瞳 AI 平台（在平台上接摄像头、配告警） |
| 设置 | 训练环境一键安装、安装位置、下载源、使用已有 Python 环境、巡瞳 AI 平台 |
| 协作中转 | `relay/wulabel_relay.py`：在线协作用的中转服务（Python + aiohttp），可以自建，也可以用巡瞳官方中转，部署见 [relay/README.md](relay/README.md) |

### 后续版本

v0.2.0 已包含下表 v0.2–v1.1 的全部功能。

| 版本 | 主题 | 状态 |
|---|---|---|
| v0.2 | SAM 交互分割、实例分割、文本提示标注、AI 平台辅助、视频抽帧、视频流抽帧 | 已发布（v0.2.0） |
| v0.3 | 训练对比、数据集版本 | 已发布（v0.2.0） |
| v0.4 | 分类、关键点 | 已发布（v0.2.0） |
| v1.0 | 正式版，macOS；单阶段检测模型 | 已发布（v0.2.0，Mac 版无苹果签名） |
| v1.1 | 实时多人协作（局域网 / 在线）、巡瞳 AI 平台深度对接（按类别识别、聊天定规则、上传模型、拉告警图） | 已发布（v0.2.0） |

已知限制：分割、关键点模型暂不支持导出部署（可在软件内预标注）；分割、关键点的评估按目标框计算指标。

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
│  └─ components/           标注画布、导出弹窗、数据集版本、训练对比、大图预览、折线图
└─ src-tauri/src/           Rust 核心
   ├─ project.rs            项目新建/打开、最近项目
   ├─ db.rs                 SQLite 表结构与迁移
   ├─ images.rs             图片导入、缩略图、删除
   ├─ video.rs              视频抽帧、视频流抽帧（ffmpeg）
   ├─ annotations.rs        标注读写、类别、图片状态
   ├─ keypoints.rs          关键点模板
   ├─ dataset.rs            数据集划分与导出
   ├─ dataset_import.rs     YOLO/COCO/VOC/分类文件夹导入
   ├─ versions.rs           数据集版本（快照、对比、恢复）
   ├─ health.rs             数据体检
   ├─ runtime.rs            训练环境：显卡检测、uv、PyTorch 安装
   ├─ training.rs           训练任务、推理服务（预标注）、ONNX 导出
   ├─ assist.rs             SAM 智能选取、文本提示标注
   ├─ ai.rs                 巡瞳 AI 平台辅助标注
   ├─ deploy.rs             实时验证、多格式导出、推理服务包
   ├─ protocol.rs           xl:// 图片协议
   └─ commands.rs           前端调用的命令
worker/                     Python 训练与推理进程（随安装包分发）
└─ xlabel_worker/
   ├─ train.py              训练与评估（检测 / 分割 / 关键点 / 分类）
   ├─ serve.py              常驻推理服务（预标注、SAM、文本提示）
   ├─ assist.py             SAM、开放词汇检测（transformers，首次使用时安装）
   ├─ export.py             导出 ONNX / OpenVINO / TensorRT、RKNN 转换包
   ├─ live.py               实时验证（RTSP / 视频文件）
   ├─ metrics.py            mAP、分类指标、混淆矩阵（只依赖 numpy）
   └─ models.py             torchvision 模型（检测、分割、关键点、分类）
└─ service/                 推理服务包模板（只依赖 onnxruntime + OpenCV，含中文说明）
```

桌面端和 worker 之间通过标准输入输出逐行传 JSON，不开端口。

## 项目文件格式

```
我的项目/
├─ project.json      项目名、任务类型
├─ annotations.db    SQLite：图片、类别、标注、关键点模板、数据集版本、训练记录
├─ images/           复制模式导入的图片；stream_*/ 是视频流采集的帧
├─ runs/<时间>/      每次训练：dataset/、best.pt、last.pt、metrics.jsonl、eval.json、log.txt、model.onnx
└─ .cache/thumbs/    缩略图（可删除，会自动重建）
```

## 许可证

[Apache-2.0](LICENSE)
