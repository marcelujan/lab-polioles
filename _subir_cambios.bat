@echo off
echo --------------------------------------------------
echo Iniciando procedimiento de Git para subir cambios...
echo --------------------------------------------------

REM Cambiar al directorio del proyecto
cd /d "G:\Mi unidad\lab-polioles"

REM Verificar que Git esté disponible
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git no está instalado o no está en el PATH.
    echo Descargalo desde: https://git-scm.com/download/win
    pause
    exit /b
)
echo Git detectado correctamente.

REM Agregar todos los cambios al área de staging
git add .

REM Realizar el commit (modificá el mensaje si lo deseás)
git commit -m "Actualización de la app"

REM Hacer push a la rama principal
git push

echo --------------------------------------------------
echo Proceso completado. Los cambios se han subido a GitHub.
pause
