"""导出 ONNX，以及部署用的多种格式"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from . import models
from .i18n import tr
from .protocol import emit, log


class _ClsNet(torch.nn.Module):
    """分类模型导出用：输入 RGB 0~1，模型里做 ImageNet 归一化，输出各类别概率（多标签是每个类别各自的 sigmoid 分数）"""

    def __init__(self, m, multi: bool = False):
        super().__init__()
        self.m = m
        self.multi = multi
        self.register_buffer("mean", torch.tensor(models.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(models.IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        y = self.m((x - self.mean) / self.std)
        return torch.sigmoid(y) if self.multi else torch.softmax(y, dim=1)


CLS_OUTPUTS = tr("probs [N, 类别数]，各类别概率（已 softmax），类别顺序见 classes")
ML_OUTPUTS = tr("probs [N, 类别数]，每个类别各自的分数（已 sigmoid，多标签：分数达到阈值的类别都算有），类别顺序见 classes")


def _cls_meta(ck: dict) -> dict:
    """分类模型 model.json 里的输出说明；多标签另加 multiLabel，部署服务按它决定返回几个类别"""
    if ck.get("multi_label"):
        return {"outputs": ML_OUTPUTS, "multiLabel": True}
    return {"outputs": CLS_OUTPUTS}
DET_OUTPUTS = tr("boxes [N,4] xyxy 像素坐标；labels [N] 类别序号从 1 开始（0 是背景）；scores [N]")
SEG_OUTPUTS = tr("boxes [N,4] xyxy 像素坐标；labels [N] 类别序号从 1 开始（0 是背景）；scores [N]；masks [N,1,H,W] 每个目标的掩膜概率（0~1，和输入同尺寸，大于 0.5 的是目标）")
POSE_OUTPUTS = tr("boxes [N,4] xyxy 像素坐标；labels [N] 类别序号从 1 开始（0 是背景）；scores [N]；keypoints [N,K,3] 每个关键点的 x、y（像素）和可见性；keypoints_scores [N,K] 每个点的分数")


def _det_keys(key: str) -> list[str]:
    """检测类模型导出的输出名：检测 3 个，分割多 masks，关键点多 keypoints、keypoints_scores"""
    keys = ["boxes", "labels", "scores"]
    if models.is_seg(key):
        keys.append("masks")
    elif models.is_pose(key):
        keys += ["keypoints", "keypoints_scores"]
    return keys


def _det_outputs(key: str) -> str:
    return SEG_OUTPUTS if models.is_seg(key) else POSE_OUTPUTS if models.is_pose(key) else DET_OUTPUTS
# YOLOX 的固定尺寸导出用官方格式：没有 NMS，适合 TensorRT / RKNN 等边缘设备，巡瞳平台的 yolox 解码也认这个格式
YOLOX_OUTPUTS = tr("output [1, N, 5+类别数]，YOLOX 官方格式（已解码）：每行 cx, cy, w, h（输入尺寸上的像素）、目标分数 obj、各类别分数；最终分数 = obj × 类别分数，过滤后按类别做 NMS（模型里没有 NMS）")


def _yolox_input(size: int) -> str:
    return tr("image: float32 [1, 3, {size}, {size}]，BGR，0~255 不归一化；原图按比例缩放后放在左上角，其余补 114", size=size)


def _export_yolox_native(model, size: int, path: Path) -> None:
    from .yolox_det import NativeExport

    torch.onnx.export(
        NativeExport(model).eval(),
        (torch.rand(1, 3, size, size) * 255,),
        str(path),
        opset_version=17,
        input_names=["image"],
        output_names=["output"],
        dynamo=False,
    )


def _export_cls(model, size: int, path: Path, multi: bool = False) -> None:
    torch.onnx.export(
        _ClsNet(model, multi).eval(),
        (torch.rand(1, 3, size, size),),
        str(path),
        opset_version=17,
        input_names=["image"],
        output_names=["probs"],
        dynamic_axes={"image": {0: "batch"}, "probs": {0: "batch"}},
        dynamo=False,
    )


ONNX_META_KEYS = ("kind", "task", "classes", "imgSize", "kptShape", "multiLabel", "engine", "model", "source")
ULTRA_INPUT = "image: float32 [1, 3, S, S], RGB, 0~1, letterbox (keep ratio, center, pad 114)"


def _copy_onnx(src: Path, dst: Path) -> dict:
    """已经是 ONNX 的模型（Ultralytics 等引擎训练后导出的、导入的）：复制过去，写好 model.json"""
    from . import onnx_decode

    m, info = onnx_decode.load(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if Path(src).resolve() != Path(dst).resolve():
        shutil.copy(src, dst)
    meta = {k: info[k] for k in ONNX_META_KEYS if info.get(k) is not None}
    meta.update(kind=m.kind, task=m.task, classes=m.classes, imgSize=m.size)
    if m.kind != "wulabel":
        meta.setdefault("input", ULTRA_INPUT.replace("S, S", f"{m.size}, {m.size}"))
    Path(dst).with_name("model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def run(weights: str, out: str) -> None:
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if Path(weights).suffix.lower() == ".onnx":
        log(tr("正在导出 ONNX…"))
        _copy_onnx(Path(weights), out_p)
        emit("done", path=str(out_p))
        return
    model, ck = models.load(Path(weights), torch.device("cpu"))
    size = int(ck["img_size"])
    cls = models.is_cls(ck["model"])
    log(tr("正在导出 ONNX…"))
    if cls:
        _export_cls(model, size, out_p, bool(ck.get("multi_label")))
        feed = np.random.rand(1, 3, size, size).astype(np.float32)
        meta = {
            "classes": ck["classes"],
            "input": tr("image: float32 [N, 3, {size}, {size}]，RGB，0~1；原图直接缩放到 {size}×{size}（模型内部已做归一化）", size=size),
            **_cls_meta(ck),
            "model": ck["model"],
        }
    else:
        _export_onnx(model, size, out_p, True, _det_keys(ck["model"]))
        feed = np.random.rand(3, size, size).astype(np.float32)
        meta = {
            "classes": ck["classes"],
            "input": tr("image: float32 [3, H, W]，RGB，0~1，不需要归一化和缩放"),
            "outputs": _det_outputs(ck["model"]),
            "model": ck["model"],
        }
    out_p.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 有 onnxruntime 就顺手验证一下能不能跑
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_p), providers=["CPUExecutionProvider"])
        sess.run(None, {"image": feed})
        log(tr("已用 onnxruntime 验证导出的模型可以运行"))
    except ImportError:
        pass
    emit("done", path=str(out_p))


# ---------- 部署用的多格式导出 ----------
# 转换包 README 按行拼：每行一个词条，换成别的语言时整行替换


def _note(title: str, body: str) -> str:
    return f"{title}\n\n{body}\n"


def _readme_note(fmt: str, two_stage: bool, yolox: bool) -> str:
    if yolox:
        return _note(
            tr("## YOLOX 模型说明"),
            tr("这是 YOLOX 官方格式的模型：模型里没有 NMS，算子都是普通卷积和激活，TensorRT 和 RK 系列 NPU 都能完整跑。")
            + "\n"
            + tr("输出每行是 `cx, cy, w, h, obj, 各类别分数`（输入尺寸上的像素，已解码），后处理：分数 = obj × 类别分数，过滤低分后按类别做 NMS，\n再把框除以缩放比例换回原图。可以参考 YOLOX 官方仓库的 demo（ONNXRuntime / TensorRT / ncnn）。"),
        )
    if fmt == "tensorrt":
        if two_stage:
            return _note(tr("## 注意"), tr("这个模型是两阶段检测模型（R-CNN 系列），含 RoIAlign 等算子。较新的 TensorRT（8.6 及以上）通常能转换，但老版本可能报不支持的算子，速度也不如单阶段模型。转换失败时，可以直接用 onnxruntime-gpu 加载 `model_static.onnx` 推理；或者回到软件里改训 RetinaNet / FCOS / SSDLite 这类单阶段模型。"))
        return _note(tr("## 说明"), tr("这个模型是单阶段模型，没有 RoIAlign 这类两阶段算子，TensorRT 一般能直接转换。"))
    if two_stage:
        return _note(tr("## 重要说明"), tr("这个模型是两阶段检测模型（R-CNN 系列），包含 RoIAlign、NMS 等 NPU 支持有限的算子。转换可能失败，或者部分算子退回 CPU 执行，速度不理想。**RK 系列 NPU 上更推荐单阶段模型**，可以回到软件里改训 SSDLite / RetinaNet / FCOS。本转换包按标准流程生成，能否在你的芯片和 SDK 版本上跑通，需要实测。"))
    return _note(tr("## 说明"), tr("这个模型是单阶段模型，比两阶段的 R-CNN 更适合 NPU。不同芯片和 SDK 版本对个别算子（如 NMS）支持不同，能否全部跑在 NPU 上需要实测；不支持的部分会退回 CPU。"))


def _trt_io(size: int, task: str) -> list[str]:
    """task：det / seg / pose / cls / yolox"""
    if task == "yolox":
        inp = tr("- 输入 `image`：float32 `[1, 3, {size}, {size}]`，BGR，0~255，不归一化。原图按比例缩放后放在左上角，其余补 114，补到 {size}×{size}。", size=size)
    else:
        inp = tr("- 输入 `image`：float32 `[1, 3, {size}, {size}]`，RGB，0~1。原图需要先缩放（保持比例，空白处补 0）到 {size}×{size}。", size=size)
    out = {
        "cls": tr("- 输出 `probs` [N, 类别数]，各类别概率。"),
        "yolox": tr("- 输出 `output` [1, N, 5+类别数]，YOLOX 官方格式，见下面的说明。"),
        "seg": tr("- 输出 `boxes` [N,4]（xyxy，基于输入尺寸）、`labels` [N]（从 1 开始）、`scores` [N]、`masks` [N,1,H,W]（掩膜概率，大于 0.5 的是目标）。"),
        "pose": tr("- 输出 `boxes` [N,4]（xyxy，基于输入尺寸）、`labels` [N]（从 1 开始）、`scores` [N]、`keypoints` [N,K,3]（关键点 x、y、可见性）、`keypoints_scores` [N,K]。"),
    }.get(task) or tr("- 输出 `boxes` [N,4]（xyxy，基于输入尺寸）、`labels` [N]（从 1 开始）、`scores` [N]。")
    return [inp, out, tr("- 类别见 `model.json`。")]


def _rknn_io(size: int, task: str) -> list[str]:
    if task == "yolox":
        inp = tr("- 输入：uint8 `[1, 3, {size}, {size}]`，BGR，0~255，不归一化。原图按比例缩放后放在左上角，其余补 114，补到 {size}×{size}。", size=size)
    else:
        inp = tr("- 输入：uint8 RGB `[1, 3, {size}, {size}]`（转换时已把 /255 归一化写进模型）。原图需要先缩放（保持比例，空白处补 0）到 {size}×{size}。", size=size)
    out = {
        "cls": tr("- 输出 `probs`，各类别概率，类别见 `model.json`。"),
        "yolox": tr("- 输出 `output` [1, N, 5+类别数]，YOLOX 官方格式，见下面的说明。类别见 `model.json`。"),
        "seg": tr("- 输出 `boxes`、`labels`、`scores`、`masks` [N,1,H,W]（掩膜概率，大于 0.5 的是目标），含义同 ONNX，类别见 `model.json`。"),
        "pose": tr("- 输出 `boxes`、`labels`、`scores`、`keypoints` [N,K,3]（关键点 x、y、可见性）、`keypoints_scores` [N,K]，含义同 ONNX，类别见 `model.json`。"),
    }.get(task) or tr("- 输出 `boxes`、`labels`、`scores`，含义同 ONNX，类别见 `model.json`。")
    return [inp, out]


def _readme(fmt: str, size: int, task: str, two_stage: bool) -> str:
    if fmt == "tensorrt":
        lines = [
            tr("# TensorRT 转换包"),
            "",
            tr("TensorRT 引擎（.engine）和显卡型号、TensorRT 版本绑定，必须在**目标设备上**生成（例如 Jetson、部署服务器）。"),
            "",
            tr("## 步骤"),
            "",
            tr("1. 把整个文件夹拷到目标设备，确认已安装 TensorRT（Jetson 刷机自带；服务器需安装 TensorRT 并能运行 `trtexec`）。"),
            tr("2. Windows 运行 `build_engine.bat`，Linux/Jetson 运行 `bash build_engine.sh`。"),
            tr("3. 生成 `model.engine`。"),
            "",
            tr("## 输入输出"),
            "",
            *_trt_io(size, task),
        ]
    else:
        lines = [
            tr("# RKNN 转换包（瑞芯微 RK3588 / RK3568 / RK3566 等）"),
            "",
            tr("瑞芯微官方转换工具 rknn-toolkit2 只支持 **Linux x86_64**（如 Ubuntu 20.04/22.04），不能在 Windows 上运行。"),
            "",
            tr("## 步骤"),
            "",
            tr("1. 在 Ubuntu 电脑上安装 rknn-toolkit2（见瑞芯微官方仓库 airockchip/rknn-toolkit2，按 Python 版本安装对应的 whl）。"),
            tr("2. 把整个文件夹拷过去，运行：`python convert_rknn.py rk3588`（芯片型号按实际填写）。"),
            tr("3. 生成 `model.rknn`，拷到板子上用 rknn-toolkit-lite2 或 RKNN C API 加载。"),
            "",
            tr("## 输入输出"),
            "",
            *_rknn_io(size, task),
        ]
    return "\n".join(lines + ["", _readme_note(fmt, two_stage, task == "yolox")])


def _rknn_script(yolox: bool) -> str:
    # 生成的脚本里的提示用 json.dumps 写成 Python 字符串字面量，译文里有引号也不会出错
    q = lambda s: json.dumps(s, ensure_ascii=False)  # noqa: E731
    if yolox:
        norm = "# " + tr("输入 uint8 BGR 0~255，YOLOX 不需要归一化")
        std = "[[1, 1, 1]]"
    else:
        norm = "# " + tr("输入 uint8 RGB，模型内部要的是 0~1，这里用 std=255 完成归一化")
        std = "[[255, 255, 255]]"
    return "\n".join([
        "# " + tr("在 Linux x86_64 上运行：python convert_rknn.py rk3588"),
        "import sys",
        "from rknn.api import RKNN",
        "",
        'target = sys.argv[1] if len(sys.argv) > 1 else "rk3588"',
        "rknn = RKNN(verbose=True)",
        norm,
        f"rknn.config(mean_values=[[0, 0, 0]], std_values={std}, target_platform=target)",
        f'assert rknn.load_onnx(model="model_static.onnx") == 0, {q(tr("加载 ONNX 失败"))}',
        f"assert rknn.build(do_quantization=False) == 0, {q(tr('转换失败（部分算子可能不被支持，见 README）'))}",
        f'assert rknn.export_rknn("model.rknn") == 0, {q(tr("导出失败"))}',
        f"print({q(tr('已生成 model.rknn'))})",
        "rknn.release()",
        "",
    ])


class _Batched(torch.nn.Module):
    """固定尺寸版本：输入带 batch 维度 [1, 3, H, W]，TensorRT、RKNN 等工具要求这种形状"""

    def __init__(self, m, keys: list[str]):
        super().__init__()
        self.m = m
        self.keys = keys

    def forward(self, x):
        o = self.m([x[0]])[0]
        return tuple(o[k] for k in self.keys)


def _export_onnx(model, size: int, path: Path, dynamic: bool, keys: list[str] | None = None) -> None:
    """检测、分割、关键点模型导出 ONNX。keys：输出名（见 _det_keys）"""
    keys = keys or ["boxes", "labels", "scores"]
    if dynamic:
        net, dummy = model, ([torch.rand(3, size, size)],)
    else:
        net, dummy = _Batched(model, keys).eval(), (torch.rand(1, 3, size, size),)
    axes = {k: {0: "n"} for k in keys}
    if dynamic:
        axes["image"] = {1: "height", 2: "width"}
    torch.onnx.export(
        net,
        dummy,
        str(path),
        opset_version=17,
        input_names=["image"],
        output_names=keys,
        dynamic_axes=axes,
        dynamo=False,
    )


def run_formats(cfg: dict) -> None:
    """cfg: weights, out, formats(["onnx","onnx_static","openvino","tensorrt","rknn"]), size"""
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    if Path(cfg["weights"]).suffix.lower() == ".onnx":
        return _run_formats_onnx(cfg, out)
    model, ck = models.load(Path(cfg["weights"]), torch.device("cpu"))
    size = int(cfg.get("size") or ck["img_size"])
    formats = cfg["formats"]
    cls = models.is_cls(ck["model"])
    two_stage = models.kind_of(ck["model"]) == "rcnn"
    yolox = models.kind_of(ck["model"]) == "yolox"
    if yolox:
        size = max(32, round(size / 32) * 32)
    meta = {
        "classes": ck["classes"],
        "model": ck["model"],
        **(_cls_meta(ck) if cls else {"outputs": _det_outputs(ck["model"])}),
    }
    results = []

    def done(fmt, path, ok=True, note=""):
        results.append({"format": fmt, "path": str(path), "ok": ok, "note": note})
        emit("format", format=fmt, path=str(path), ok=ok, note=note)

    dyn = out / "onnx" / "model.onnx"
    static = out / "onnx_static" / "model_static.onnx"
    need_dyn = any(f in formats for f in ("onnx", "openvino"))
    need_static = any(f in formats for f in ("onnx_static", "tensorrt", "rknn"))
    # 步骤进度：动态 ONNX、固定 ONNX 各一步，后面每个格式一步（名字用格式，界面换成格式名）
    steps = (["onnx"] if need_dyn else []) + (["onnx_static"] if need_static else [])
    steps += [f for f in ("openvino", "tensorrt", "rknn") if f in formats]
    passed = []

    def stage(name):
        emit("progress", step=len(passed), total=len(steps), name=name)
        passed.append(name)

    # 分类模型只有固定尺寸一种（批大小可变），「动态」和「固定」导出的是同一个模型
    cls_input = tr("image: float32 [N, 3, {size}, {size}]，RGB，0~1；原图直接缩放到 {size}×{size}（模型内部已做归一化）", size=size)
    if need_dyn:
        stage("onnx")
        log(tr("导出 ONNX（动态尺寸）…"))
        dyn.parent.mkdir(parents=True, exist_ok=True)
        if cls:
            _export_cls(model, size, dyn, bool(ck.get("multi_label")))
        else:
            _export_onnx(model, size, dyn, True, _det_keys(ck["model"]))
        (dyn.parent / "model.json").write_text(
            json.dumps({**meta, "input": cls_input if cls else tr("image: float32 [3, H, W]，RGB，0~1，任意尺寸")}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if "onnx" in formats:
            done("onnx", dyn)
    if need_static:
        stage("onnx_static")
        log(tr("导出 ONNX（固定尺寸 {size}×{size}）…", size=size))
        static.parent.mkdir(parents=True, exist_ok=True)
        static_meta = dict(meta)
        if cls:
            _export_cls(model, size, static, bool(ck.get("multi_label")))
        elif yolox:
            _export_yolox_native(model, size, static)
            static_meta.update(outputs=YOLOX_OUTPUTS, format="yolox", inputSize=size)
        else:
            _export_onnx(model, size, static, False, _det_keys(ck["model"]))
        if cls:
            static_input = cls_input
        elif yolox:
            static_input = _yolox_input(size)
        else:
            static_input = tr("image: float32 [1, 3, {size}, {size}]，RGB，0~1；原图按比例缩放后补边到该尺寸", size=size)
        (static.parent / "model.json").write_text(
            json.dumps({**static_meta, "input": static_input}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if "onnx_static" in formats:
            done("onnx_static", static)

    if "openvino" in formats:
        stage("openvino")
        try:
            os.environ.setdefault("OPENVINO_TELEMETRY", "0")
            import openvino as ov

            log(tr("转换 OpenVINO…"))
            d = out / "openvino"
            d.mkdir(parents=True, exist_ok=True)
            m = ov.convert_model(str(dyn))
            ov.save_model(m, str(d / "model.xml"))
            shutil.copy(dyn.parent / "model.json", d / "model.json")
            # 用 CPU 跑一遍验证
            import numpy as np

            core = ov.Core()
            compiled = core.compile_model(str(d / "model.xml"), "CPU")
            shape = (1, 3, size, size) if cls else (3, size, size)
            compiled([np.random.rand(*shape).astype(np.float32)])
            done("openvino", d / "model.xml", note=tr("已用 OpenVINO CPU 推理验证"))
        except ImportError:
            done("openvino", "", ok=False, note=tr("没有安装 OpenVINO"))
        except Exception as e:
            done("openvino", "", ok=False, note=tr("转换失败：{e}", e=e))

    key = ck["model"]
    task = "yolox" if yolox else "cls" if cls else "seg" if models.is_seg(key) else "pose" if models.is_pose(key) else "det"
    for fmt, files in (
        ("tensorrt", {
            "build_engine.bat": "trtexec --onnx=model_static.onnx --saveEngine=model.engine --fp16\r\npause\r\n",
            "build_engine.sh": "#!/bin/sh\nset -e\ncd \"$(dirname \"$0\")\"\ntrtexec --onnx=model_static.onnx --saveEngine=model.engine --fp16\n",
        }),
        ("rknn", {"convert_rknn.py": _rknn_script(yolox)}),
    ):
        if fmt not in formats:
            continue
        stage(fmt)
        d = out / fmt
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(static, d / "model_static.onnx")
        shutil.copy(static.parent / "model.json", d / "model.json")
        (d / "README.md").write_text(_readme(fmt, size, task, two_stage), encoding="utf-8")
        for name, text in files.items():
            (d / name).write_text(text, encoding="utf-8", newline="")
        done(fmt, d, note=tr("转换包：需要在目标设备上完成转换，见 README.md"))

    # 只要了转换包、没要 ONNX 时，中间文件不保留
    if "onnx_static" not in formats and static.parent.exists():
        shutil.rmtree(static.parent, ignore_errors=True)
    if "onnx" not in formats and dyn.parent.exists() and "openvino" in formats:
        shutil.rmtree(dyn.parent, ignore_errors=True)
    emit("progress", step=len(steps), total=len(steps), name="")
    emit("done", results=results)


def _run_formats_onnx(cfg: dict, out: Path) -> None:
    """已经是 ONNX 的模型：ONNX 直接复制；OpenVINO 从 ONNX 转；TensorRT / RKNN 转换包基于同一个 ONNX"""
    src = Path(cfg["weights"])
    formats = cfg["formats"]
    steps = [f for f in ("onnx", "onnx_static", "openvino", "tensorrt", "rknn") if f in formats]
    results = []
    passed = []

    def done(fmt, path, ok=True, note=""):
        results.append({"format": fmt, "path": str(path), "ok": ok, "note": note})
        emit("format", format=fmt, path=str(path), ok=ok, note=note)

    def stage(name):
        emit("progress", step=len(passed), total=len(steps), name=name)
        passed.append(name)

    dyn = out / "onnx" / "model.onnx"
    meta = _copy_onnx(src, dyn)
    size = int(meta.get("imgSize") or 640)
    if "onnx" in formats:
        stage("onnx")
        done("onnx", dyn)
    if "onnx_static" in formats:
        stage("onnx_static")
        static = out / "onnx_static" / "model_static.onnx"
        _copy_onnx(src, static)
        done("onnx_static", static)
    if "openvino" in formats:
        stage("openvino")
        try:
            os.environ.setdefault("OPENVINO_TELEMETRY", "0")
            import openvino as ov

            log(tr("转换 OpenVINO…"))
            d = out / "openvino"
            d.mkdir(parents=True, exist_ok=True)
            ov.save_model(ov.convert_model(str(dyn)), str(d / "model.xml"))
            shutil.copy(dyn.parent / "model.json", d / "model.json")
            done("openvino", d / "model.xml")
        except ImportError:
            done("openvino", "", ok=False, note=tr("没有安装 OpenVINO"))
        except Exception as e:
            done("openvino", "", ok=False, note=tr("转换失败：{e}", e=e))
    wulabel = meta.get("kind") == "wulabel"
    for fmt, files in (
        ("tensorrt", {
            "build_engine.bat": "trtexec --onnx=model_static.onnx --saveEngine=model.engine --fp16\r\npause\r\n",
            "build_engine.sh": "#!/bin/sh\nset -e\ncd \"$(dirname \"$0\")\"\ntrtexec --onnx=model_static.onnx --saveEngine=model.engine --fp16\n",
        }),
        ("rknn", {"convert_rknn.py": _rknn_script(False)}),
    ):
        if fmt not in formats:
            continue
        stage(fmt)
        d = out / fmt
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, d / "model_static.onnx")
        shutil.copy(dyn.parent / "model.json", d / "model.json")
        task = {"segment": "seg", "pose": "pose", "classify": "cls"}.get(meta.get("task"), "det")
        text = _readme(fmt, size, task, False) if wulabel else _readme_ultra(fmt, size)
        (d / "README.md").write_text(text, encoding="utf-8")
        for name, body in files.items():
            (d / name).write_text(body, encoding="utf-8", newline="")
        done(fmt, d, note=tr("转换包：需要在目标设备上完成转换，见 README.md"))
    if "onnx" not in formats and dyn.parent.exists():
        shutil.rmtree(dyn.parent, ignore_errors=True)
    emit("progress", step=len(steps), total=len(steps), name="")
    emit("done", results=results)


def _readme_ultra(fmt: str, size: int) -> str:
    """Ultralytics 系模型（YOLOv8 / 11 等）转换包的说明：输入输出是 Ultralytics 原始格式"""
    head = tr("# TensorRT 转换包") if fmt == "tensorrt" else tr("# RKNN 转换包（瑞芯微 RK3588 / RK3568 / RK3566 等）")
    if fmt == "tensorrt":
        steps = [
            tr("TensorRT 引擎（.engine）和显卡型号、TensorRT 版本绑定，必须在**目标设备上**生成（例如 Jetson、部署服务器）。"),
            "",
            tr("## 步骤"),
            "",
            tr("1. 把整个文件夹拷到目标设备，确认已安装 TensorRT（Jetson 刷机自带；服务器需安装 TensorRT 并能运行 `trtexec`）。"),
            tr("2. Windows 运行 `build_engine.bat`，Linux/Jetson 运行 `bash build_engine.sh`。"),
            tr("3. 生成 `model.engine`。"),
        ]
    else:
        steps = [
            tr("瑞芯微官方转换工具 rknn-toolkit2 只支持 **Linux x86_64**（如 Ubuntu 20.04/22.04），不能在 Windows 上运行。"),
            "",
            tr("## 步骤"),
            "",
            tr("1. 在 Ubuntu 电脑上安装 rknn-toolkit2（见瑞芯微官方仓库 airockchip/rknn-toolkit2，按 Python 版本安装对应的 whl）。"),
            tr("2. 把整个文件夹拷过去，运行：`python convert_rknn.py rk3588`（芯片型号按实际填写）。"),
            tr("3. 生成 `model.rknn`，拷到板子上用 rknn-toolkit-lite2 或 RKNN C API 加载。"),
        ]
    io = [
        tr("## 输入输出"),
        "",
        tr("- 输入：`[1, 3, {size}, {size}]`，RGB，0~1。原图按比例缩放后居中，四周补 114，补到 {size}×{size}。", size=size),
        tr("- 输出是 Ultralytics 原始格式（模型里没有 NMS）：检测 `[1, 4+类别数, N]`；分割另有掩膜原型；关键点每个目标后面跟关键点。后处理可以参考 Ultralytics 官方示例，或推理服务包里的 `onnx_decode.py`。"),
        tr("- 类别见 `model.json`。"),
    ]
    return "\n".join([head, "", *steps, "", *io, ""])
