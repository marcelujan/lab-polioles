@echo off
echo ============================================
echo ⚙️ Configurando identidad de Git para Marcelo
echo ============================================

cd /d "H:\Mi unidad\lab-polioles"

REM Verificar si Git está disponible
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git no está instalado o no está en el PATH
    pause
    exit /b
)

REM Configurar nombre y email (global, una sola vez)
git config --global user.name "marcelujan"
git config --global user.email "marcelo.lujan@mi.unc.edu.ar"

echo ✅ Git configurado correctamente con tu identidad
echo --------------------------------------------

REM Agregar gitignore (por si hubo cambios)
git add .gitignore

REM Commit y push
git commit -m "🔧 Configuración de Git y actualización de .gitignore"
git push

echo ✅ ¡Listo! Todo actualizado y tu identidad quedó configurada.
pause
