@echo off
echo ============================================================
echo  WatchDog – Setup (Windows)
echo ============================================================

python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Done!  Run either:
echo   python app.py            (web dashboard at http://localhost:5000)
echo   python run_headless.py   (OpenCV window, no browser)
pause
