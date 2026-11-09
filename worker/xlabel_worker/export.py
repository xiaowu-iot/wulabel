"""导出 ONNX，以及部署用的多种格式"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch

from . import models
from .protocol import emit, log


def run(weights: str, out: str) -> None:
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    model, ck = models.load(Path(weights), torch.device("cpu"))
    size = int(ck["img_size"])
    dummy = [torch.rand(3, size, size)]
    log("正在导出 ONNX…")
    torch.onnx.export(
        model,
        (dummy,),
        str(out_p),
        opset_version=17,
        input_names=["image"],
        output_names=["boxes", "labels", "scores"],
        dynamic_axes={"image": {1: "height", 2: "width"}, "boxes": {0: "n"}, "labels": {0: "n"}, "scores": {0: "n"}},
        dynamo=False,
    )
    meta = {
        "classes": ck["classes"],
        "input": "image: float32 [3, H, W]，RGB，0~1，不需要归一化和缩放",
        "outputs": "boxes [N,4] xyxy 像素坐标；labels [N] 类别序号从 1 开始（0 是背景）；scores [N]",
        "model": ck["model"],
    }
    out_p.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 有 onnxruntime 就顺手验证一下能不能跑
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_p), providers=["CPUExecutionProvider"])
        sess.run(None, {"image": np.random.rand(3, size, size).astype(np.float32)})
        log("已用 onnxruntime 验证导出的模型可以运行")
    except ImportError:
        pass
    emit("done", path=str(out_p))


# ---------- 部署用的多格式导出 ----------

TRT_README = """# TensorRT 转换包

TensorRT 引擎（.engine）和显卡型号、TensorRT 版本绑定，必须在**目标设备上**生成（例如 Jetson、部署服务器）。

## 步骤

1. 把整个文件夹拷到目标设备，确认已安装 TensorRT（Jetson 刷机自带；服务器需安装 TensorRT 并能运行 `trtexec`）。
2. Windows 运行 `build_engine.bat`，Linux/Jetson 运行 `bash build_engine.sh`。
3. 生成 `model.engine`。

## 输入输出

- 输入 `image`：float32 `[1, 3, {size}, {size}]`，RGB，0~1。原图需要先缩放（保持比例，空白处补 0）到 {size}×{size}。
- 输出 `boxes` [N,4]（xyxy，基于输入尺寸）、`labels` [N]（从 1 开始）、`scores` [N]。
- 类别见 `model.json`。

## 注意

内置的 Faster R-CNN 是两阶段模型，含 RoIAlign 等算子。较新的 TensorRT（8.6 及以上）通常能转换，但老版本可能报不支持的算子，速度也不如 YOLO 这类单阶段模型。转换失败时，可以直接用 onnxruntime-gpu 加载 `model_static.onnx` 推理。
"""

RKNN_README = """# RKNN 转换包（瑞芯微 RK3588 / RK3568 / RK3566 等）

瑞芯微官方转换工具 rknn-toolkit2 只支持 **Linux x86_64**（如 Ubuntu 20.04/22.04），不能在 Windows 上运行。

## 步骤

1. 在 Ubuntu 电脑上安装 rknn-toolkit2（见瑞芯微官方仓库 airockchip/rknn-toolkit2，按 Python 版本安装对应的 whl）。
2. 把整个文件夹拷过去，运行：`python convert_rknn.py rk3588`（芯片型号按实际填写）。
3. 生成 `model.rknn`，拷到板子上用 rknn-toolkit-lite2 或 RKNN C API 加载。

## 输入输出

- 输入：uint8 RGB `[1, 3, {size}, {size}]`（转换时已把 /255 归一化写进模型）。原图需要先缩放（保持比例，空白处补 0）到 {size}×{size}。
- 输出 `boxes`、`labels`、`scores`，含义同 ONNX，类别见 `model.json`。

## 重要说明

内置的 Faster R-CNN 是两阶段检测模型，包含 RoIAlign、NMS 等 NPU 支持有限的算子。转换可能失败，或者部分算子退回 CPU 执行，速度不理想。**RK 系列 NPU 上更推荐 YOLO 等单阶段模型。** 本转换包按标准流程生成，能否在你的芯片和 SDK 版本上跑通，需要实测。
"""

RKNN_SCRIPT = """# 在 Linux x86_64 上运行：python convert_rknn.py rk3588
import sys
from rknn.api import RKNN

target = sys.argv[1] if len(sys.argv) > 1 else "rk3588"
rknn = RKNN(verbose=True)
# 输入 uint8 RGB，模型内部要的是 0~1，这里用 std=255 完成归一化
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=target)
assert rknn.load_onnx(model="model_static.onnx") == 0, "加载 ONNX 失败"
assert rknn.build(do_quantization=False) == 0, "转换失败（Faster R-CNN 的部分算子可能不被支持，见 README）"
assert rknn.export_rknn("model.rknn") == 0, "导出失败"
print("已生成 model.rknn")
rknn.release()
"""


class _Batched(torch.nn.Module):
    """固定尺寸版本：输入带 batch 维度 [1, 3, H, W]，TensorRT、RKNN 等工具要求这种形状"""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        o = self.m([x[0]])[0]
        return o["boxes"], o["labels"], o["scores"]


def _export_onnx(model, size: int, path: Path, dynamic: bool) -> None:
    if dynamic:
        net, dummy = model, ([torch.rand(3, size, size)],)
    else:
        net, dummy = _Batched(model).eval(), (torch.rand(1, 3, size, size),)
    torch.onnx.export(
        net,
        dummy,
        str(path),
        opset_version=17,
        input_names=["image"],
        output_names=["boxes", "labels", "scores"],
        dynamic_axes=(
            {"image": {1: "height", 2: "width"}, "boxes": {0: "n"}, "labels": {0: "n"}, "scores": {0: "n"}}
            if dynamic
            else {"boxes": {0: "n"}, "labels": {0: "n"}, "scores": {0: "n"}}
        ),
        dynamo=False,
    )


def run_formats(cfg: dict) -> None:
    """cfg: weights, out, formats(["onnx","onnx_static","openvino","tensorrt","rknn"]), size"""
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    model, ck = models.load(Path(cfg["weights"]), torch.device("cpu"))
    size = int(cfg.get("size") or ck["img_size"])
    formats = cfg["formats"]
    meta = {
        "classes": ck["classes"],
        "model": ck["model"],
        "outputs": "boxes [N,4] xyxy 像素坐标；labels [N] 类别序号从 1 开始（0 是背景）；scores [N]",
    }
    results = []

    def done(fmt, path, ok=True, note=""):
        results.append({"format": fmt, "path": str(path), "ok": ok, "note": note})
        emit("format", format=fmt, path=str(path), ok=ok, note=note)

    dyn = out / "onnx" / "model.onnx"
    static = out / "onnx_static" / "model_static.onnx"
    need_dyn = any(f in formats for f in ("onnx", "openvino"))
    need_static = any(f in formats for f in ("onnx_static", "tensorrt", "rknn"))
    if need_dyn:
        log("导出 ONNX（动态尺寸）…")
        dyn.parent.mkdir(parents=True, exist_ok=True)
        _export_onnx(model, size, dyn, True)
        (dyn.parent / "model.json").write_text(
            json.dumps({**meta, "input": "image: float32 [3, H, W]，RGB，0~1，任意尺寸"}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if "onnx" in formats:
            done("onnx", dyn)
    if need_static:
        log(f"导出 ONNX（固定尺寸 {size}×{size}）…")
        static.parent.mkdir(parents=True, exist_ok=True)
        _export_onnx(model, size, static, False)
        (static.parent / "model.json").write_text(
            json.dumps({**meta, "input": f"image: float32 [1, 3, {size}, {size}]，RGB，0~1；原图按比例缩放后补边到该尺寸"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if "onnx_static" in formats:
            done("onnx_static", static)

    if "openvino" in formats:
        try:
            os.environ.setdefault("OPENVINO_TELEMETRY", "0")
            import openvino as ov

            log("转换 OpenVINO…")
            d = out / "openvino"
            d.mkdir(parents=True, exist_ok=True)
            m = ov.convert_model(str(dyn))
            ov.save_model(m, str(d / "model.xml"))
            shutil.copy(dyn.parent / "model.json", d / "model.json")
            # 用 CPU 跑一遍验证
            import numpy as np

            core = ov.Core()
            compiled = core.compile_model(str(d / "model.xml"), "CPU")
            compiled([np.random.rand(3, size, size).astype(np.float32)])
            done("openvino", d / "model.xml", note="已用 OpenVINO CPU 推理验证")
        except ImportError:
            done("openvino", "", ok=False, note="没有安装 OpenVINO")
        except Exception as e:
            done("openvino", "", ok=False, note=f"转换失败：{e}")

    for fmt, readme, files in (
        ("tensorrt", TRT_README, {
            "build_engine.bat": "trtexec --onnx=model_static.onnx --saveEngine=model.engine --fp16\r\npause\r\n",
            "build_engine.sh": "#!/bin/sh\nset -e\ncd \"$(dirname \"$0\")\"\ntrtexec --onnx=model_static.onnx --saveEngine=model.engine --fp16\n",
        }),
        ("rknn", RKNN_README, {"convert_rknn.py": RKNN_SCRIPT}),
    ):
        if fmt not in formats:
            continue
        d = out / fmt
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(static, d / "model_static.onnx")
        shutil.copy(static.parent / "model.json", d / "model.json")
        (d / "README.md").write_text(readme.replace("{size}", str(size)), encoding="utf-8")
        for name, text in files.items():
            (d / name).write_text(text, encoding="utf-8", newline="")
        done(fmt, d, note="转换包：需要在目标设备上完成转换，见 README.md")

    # 只要了转换包、没要 ONNX 时，中间文件不保留
    if "onnx_static" not in formats and static.parent.exists():
        shutil.rmtree(static.parent, ignore_errors=True)
    if "onnx" not in formats and dyn.parent.exists() and "openvino" in formats:
        shutil.rmtree(dyn.parent, ignore_errors=True)
    emit("done", results=results)
