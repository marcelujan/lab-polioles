@echo off
echo =====================================
echo 🚀 Iniciando limpieza de clave Firebase
echo =====================================

cd /d "H:\Mi unidad\lab-polioles"

REM Verificar si Git está disponible
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git no está instalado o no está en el PATH
    echo ➡️ Descargalo desde: https://git-scm.com/download/win
    pause
    exit /b
)

echo ✅ Git detectado

REM Eliminar clave del índice (sin borrar el archivo local)
git rm --cached firebase_key.json

REM Agregar a .gitignore (si no está)
findstr /c:"firebase_key.json" .gitignore >nul 2>&1 || echo firebase_key.json>>.gitignore

REM Hacer commit y push
git add .gitignore
git commit -m "🧹 Remover firebase_key.json del repo público"
git push

echo ✅ Listo. firebase_key.json fue eliminado del repositorio.
pause
