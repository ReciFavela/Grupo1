@echo off
REM Baixa as fotos do TACO (lixo real com objetos marcados) para testar o sistema
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv_build\Scripts\python.exe" (
    ".venv_build\Scripts\python.exe" ferramentas\baixar_fotos_taco.py
) else (
    python ferramentas\baixar_fotos_taco.py
)
echo.
pause
