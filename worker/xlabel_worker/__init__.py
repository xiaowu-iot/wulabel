"""xlabel 训练与推理进程。

由桌面端启动，通过标准输出逐行输出 JSON 事件：
    {"type": "log", "msg": "..."}
    {"type": "progress", ...}
    {"type": "metric", ...}
    {"type": "done", ...}
    {"type": "error", "msg": "..."}
"""

__version__ = "0.1.0"
