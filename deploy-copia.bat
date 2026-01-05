@echo off
echo Agregando todos los cambios...
git add .

echo Realizando commit automático...
git commit -m "Actualización automática desde script"

echo Sincronizando con el repositorio remoto (pull --rebase)...
git pull --rebase origin main

echo Subiendo cambios...
git push origin main

pause
