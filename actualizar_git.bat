@echo off
REM Script para actualizar el repositorio Git

echo.
echo Agregando cambios al repositorio...
git add .

echo.
set /p msg="Escribí el mensaje de commit: "
git commit -m "%msg%"

echo.
echo Subiendo cambios a GitHub...
git push

echo.
pause
