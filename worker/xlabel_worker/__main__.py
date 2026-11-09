"""命令行入口：python -m xlabel_worker <命令>

    check                 输出环境信息
    train <配置.json>     训练
    serve                 常驻推理服务
    export <权重> <输出>  导出 ONNX
    onnx-info <模型>      识别外部 ONNX 模型的格式、类别
    evaluate <配置.json>  用现在的数据重新评估模型
    fptest <配置.json>    误报测试
"""

import json
import sys
import traceback


def main() -> int:
    # 桌面端按 UTF-8 解析输出
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stdin.reconfigure(encoding="utf-8")
    from .protocol import emit

    args = sys.argv[1:]
    if not args:
        emit("error", msg="缺少命令")
        return 2
    cmd = args[0]
    try:
        if cmd == "check":
            from .envcheck import run

            run()
        elif cmd == "train":
            from . import train

            with open(args[1], encoding="utf-8") as f:
                train.run(json.load(f))
        elif cmd == "serve":
            from . import serve

            serve.run()
        elif cmd == "export":
            from . import export

            export.run(args[1], args[2])
        elif cmd == "export-formats":
            from . import export

            with open(args[1], encoding="utf-8") as f:
                export.run_formats(json.load(f))
        elif cmd == "live":
            from . import live

            with open(args[1], encoding="utf-8") as f:
                live.run(json.load(f))
        elif cmd == "onnx-info":
            from . import onnx_model

            emit("onnx", **onnx_model.inspect(args[1]))
        elif cmd == "evaluate":
            from . import evaluate

            with open(args[1], encoding="utf-8") as f:
                evaluate.run(json.load(f))
        elif cmd == "anonymize":
            from . import anonymize

            with open(args[1], encoding="utf-8") as f:
                anonymize.run(json.load(f))
        elif cmd == "augpreview":
            from . import augment

            with open(args[1], encoding="utf-8") as f:
                augment.preview(json.load(f))
        elif cmd == "fptest":
            from . import fptest

            with open(args[1], encoding="utf-8") as f:
                fptest.run(json.load(f))
        else:
            emit("error", msg=f"未知命令：{cmd}")
            return 2
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        emit("error", msg=str(e), trace=traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
