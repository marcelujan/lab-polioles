@echo off
setlocal EnableDelayedExpansion

echo ============================================
echo ⚙️ Subida automática de cambios + ver app online
echo ============================================

cd /d "H:\Mi unidad\lab-polioles"

REM 1. Verificar Git
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git no está instalado o no está en el PATH
    pause
    exit /b
)

REM 2. Configurar identidad (solo la primera vez)
git config --global user.name "marcelujan"
git config --global user.email "marcelo.lujan@mi.unc.edu.ar"

echo ✅ Git configurado correctamente

REM 3. Ver si hay cambios
git status --porcelain > temp_git_status.txt
set /p has_changes=<temp_git_status.txt
del temp_git_status.txt

if not "!has_changes!"=="" (
    echo 📝 Cambios detectados. Haciendo commit y push...

    git add .
    git commit -m "🚀 Subida automática con cambios recientes"
    git push
    echo ✅ Código subido a GitHub

) else (
    echo ⚠️ No se detectaron cambios nuevos para subir.
)

REM 4. Abrir Streamlit App en navegador
start https://lab-polioles.streamlit.app

echo 🌐 App abierta en tu navegador. ¡Todo listo!
pause
