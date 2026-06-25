@echo off
REM Install / update all pip dependencies into the glod conda environment.
REM Run this once after any change to requirements.txt.

set PYTHON=C:\Users\param\anaconda3\envs\glod\python.exe

echo.
echo Installing requirements into glod env...
echo.
"%PYTHON%" -m pip install --upgrade -r "%~dp0requirements.txt"
echo.
echo Verifying key imports...
"%PYTHON%" -c "import PyQt6; import matplotlib; import numpy; import pandas; print('Core OK')"
"%PYTHON%" -c "import pyqtgraph; print('pyqtgraph OK')"
"%PYTHON%" -c "import serial; print('pyserial OK')"
"%PYTHON%" -c "import senxor; print('pysenxor-lite OK')"
echo.
echo Done.
pause
