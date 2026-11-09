@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
  echo 首次运行：创建 Python 环境并安装依赖（需要已安装 Python 3.9 及以上）...
  python -m venv venv || (echo 没有找到 Python，请先安装 Python 3.9+ 并勾选 Add to PATH & pause & exit /b 1)
  venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple || (pause & exit /b 1)
)
venv\Scripts\python.exe service.py config.json
pause
