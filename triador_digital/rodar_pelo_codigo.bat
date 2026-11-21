@echo off
REM ======================================================================
REM  rodar_pelo_codigo.bat - Abre o Triador Digital sem gerar o .exe
REM  (para quem e da TI e quer testar mudancas no codigo)
REM ======================================================================
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv_build" python -m venv .venv_build
call ".venv_build\Scripts\activate.bat"
python -m pip install -q -r requirements.txt
python triador.py
