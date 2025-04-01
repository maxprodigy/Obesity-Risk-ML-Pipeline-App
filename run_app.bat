@echo off
echo Obesity Risk Predictor Application

echo Choose an option:
echo 1. Start full application (backend + frontend)
echo 2. Start backend API only
echo 3. Start frontend only
echo 4. Test model functionality
echo 5. Get model information
echo 6. Exit

choice /C 123456 /M "Enter your choice:"

if errorlevel 6 goto :exit
if errorlevel 5 goto :model_info
if errorlevel 4 goto :test_model
if errorlevel 3 goto :frontend_only
if errorlevel 2 goto :backend_only
if errorlevel 1 goto :full_app

:full_app
echo Starting full application...
start cmd /k "cd src && python api.py"
timeout /t 3
start cmd /k "cd frontend && npm start"
echo Application started!
echo - Backend: http://localhost:8000
echo - Frontend: http://localhost:3000
goto :end

:backend_only
echo Starting backend API only...
start cmd /k "cd src && python api.py"
echo Backend API started at http://localhost:8000
goto :end

:frontend_only
echo Starting frontend only...
start cmd /k "cd frontend && npm start"
echo Frontend started at http://localhost:3000
goto :end

:test_model
echo Testing model functionality...
cd src
python test_model.py
pause
goto :end

:model_info
echo Getting model information...
cd src
python get_model_info.py
pause
goto :end

:exit
echo Exiting...
goto :end

:end 