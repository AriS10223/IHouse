@echo off
title i-House — International Student Advisor
echo.
echo  Starting i-House...
echo.
py run_app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  Something went wrong. Make sure Python is installed:
    echo    https://www.python.org/downloads/
    echo.
    pause
)
