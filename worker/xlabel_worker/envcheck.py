import platform
import sys

from .protocol import emit


def run() -> None:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        import torchvision

        info["torch"] = torch.__version__
        info["torchvision"] = torchvision.__version__
        info["cuda"] = torch.cuda.is_available()
        info["cudaVersion"] = torch.version.cuda
        if info["cuda"]:
            info["device"] = torch.cuda.get_device_name(0)
            info["vram"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            # 真跑一次运算，确认显卡可用（新显卡配旧 CUDA 版本会在这里报错）
            x = torch.ones(64, 64, device="cuda")
            info["cudaOk"] = float((x @ x).sum()) == 64 * 64 * 64
        else:
            mps = getattr(torch.backends, "mps", None)
            info["mps"] = bool(mps is not None and mps.is_available())
            info["device"] = "Apple 芯片（MPS）" if info["mps"] else "CPU"
    except Exception as e:
        info["error"] = str(e)
    for m in ("numpy", "PIL", "onnx", "onnxruntime"):
        try:
            __import__(m)
            info[m] = True
        except Exception:
            info[m] = False
    emit("env", **info)
