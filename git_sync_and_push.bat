@echo off
echo Sincronizando con el repositorio remoto...
git pull --rebase origin main
IF %ERRORLEVEL% NEQ 0 (
    echo Error durante git pull. Por favor resolvé los conflictos manualmente si los hay.
    pause
    exit /b
)
echo Subiendo cambios al repositorio remoto...
git push origin main
pause
