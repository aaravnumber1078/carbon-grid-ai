
@echo off
cd /d "C:\CARBON GRID AI"
set KMP_DUPLICATE_LIB_OK=TRUE

start "CarbonGrid-AI Backend" cmd /k "python -m uvicorn carbongrid.api.main:app --host 127.0.0.1 --port 8000"

timeout /t 5 /nobreak >nul
start "" http://127.0.0.1:8000/