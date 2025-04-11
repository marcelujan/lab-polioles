@echo off
echo.
git add .
set /p msg="Mensaje de commit: "
git commit -m "%msg%"
echo.
git push
pause
