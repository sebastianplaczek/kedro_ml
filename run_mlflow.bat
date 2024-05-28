@echo off
cd /d "C:\projects\kedro_ml"
call venv\Scripts\activate
call mlflow server --host 127.0.0.1 --port 8080