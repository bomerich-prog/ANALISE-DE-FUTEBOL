@echo off
chcp 65001 >nul
title Conversor de Video MP4
cd /d "%~dp0"

echo.
echo  ========================================
echo   PitchView - Plataforma de Analise
echo  ========================================
echo.

where node >nul 2>&1
if %errorlevel% neq 0 (
    echo ERRO: Node.js nao encontrado.
    echo Instale em https://nodejs.org e execute novamente.
    echo.
    pause
    exit /b 1
)

if not exist "node_modules\" (
    echo Instalando dependencias...
    call npm install
    if %errorlevel% neq 0 (
        echo Falha ao instalar dependencias.
        pause
        exit /b 1
    )
)

echo Iniciando servidor em http://localhost:3000
echo.
start "" "http://localhost:3000"
node server/index.js
pause
