@echo off
REM ======================================================================
REM  gerar_executavel.bat - Cria o programa TriadorDigital.exe
REM  Basta dar dois cliques neste arquivo (no Windows).
REM  Precisa do Python instalado (python.org ou Anaconda).
REM  Leva uns 5 minutos na primeira vez (baixa as bibliotecas).
REM ======================================================================
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo  === Triador Digital - gerando o executavel ===
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo  [ERRO] Nao encontrei o Python.
    echo  Instale pelo site python.org ^(marque "Add python.exe to PATH"^)
    echo  ou abra este arquivo pelo "Anaconda Prompt".
    pause
    exit /b 1
)

echo  1/4 Criando um ambiente separado para nao baguncar o seu Python...
if not exist ".venv_build" python -m venv .venv_build
call ".venv_build\Scripts\activate.bat"

echo  2/4 Instalando as bibliotecas ^(pode demorar^)...
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt pyinstaller
if errorlevel 1 (
    echo  [ERRO] Falhou ao instalar as bibliotecas. Veja se a internet esta ligada.
    pause
    exit /b 1
)

echo  3/4 Testando o modelo antes de empacotar...
python triador.py --teste
if errorlevel 1 (
    echo  [ERRO] O teste do modelo falhou. Veja a mensagem acima.
    pause
    exit /b 1
)

echo  4/4 Criando o executavel...
python -m PyInstaller --noconfirm --clean TriadorDigital.spec
if errorlevel 1 (
    echo  [ERRO] O PyInstaller falhou. Veja a mensagem acima.
    pause
    exit /b 1
)

REM fotos da esteira ficam ao lado do .exe para qualquer um poder trocar
xcopy /E /I /Y "imagens_esteira" "dist\TriadorDigital\imagens_esteira" >nul
copy /Y "COMO_USAR.txt" "dist\TriadorDigital\COMO_USAR.txt" >nul

echo  Compactando em TriadorDigital.zip ...
powershell -NoProfile -Command "Compress-Archive -Path 'dist\TriadorDigital' -DestinationPath 'TriadorDigital.zip' -Force"

echo.
echo  ===================================================
echo   PRONTO!
echo   Programa:  dist\TriadorDigital\TriadorDigital.exe
echo   Para enviar a outra pessoa: TriadorDigital.zip
echo  ===================================================
echo.
pause
