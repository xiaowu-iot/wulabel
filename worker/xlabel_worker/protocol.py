import json
import sys
import threading

_lock = threading.Lock()


def emit(kind: str, **fields) -> None:
    """输出一行 JSON 事件给桌面端"""
    line = json.dumps({"type": kind, **fields}, ensure_ascii=False)
    with _lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def log(msg: str) -> None:
    emit("log", msg=msg)
