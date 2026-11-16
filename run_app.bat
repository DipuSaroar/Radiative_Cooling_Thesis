@echo off
REM Double-click to start the Radiative Cooling web app on this computer.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating the Python environment, first run only...
  py -3.11 -m venv .venv
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt -r app\requirements.txt
)
".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py
pause
