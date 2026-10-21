@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First install the environment using COMPONENT4_README.md.
  pause
  exit /b 1
)
if not exist "multimodal_fusion\artifacts\fusion_model.joblib" (
  .venv\Scripts\python.exe -m multimodal_fusion train
  if errorlevel 1 (pause & exit /b 1)
)
start "" "http://127.0.0.1:8040"
.venv\Scripts\python.exe -m multimodal_fusion serve
pause
