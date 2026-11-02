@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo ============================================================
echo   Запуск Depth Anything 3 ^& V2 Studio...
echo ============================================================
python app.py
pause
