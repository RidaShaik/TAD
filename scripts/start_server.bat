@echo off
setlocal

:: start Flask sever
cd /d "%~dp0..\flask-server"
echo Starting Flask server...
start "Flask Server" cmd /c "python server.py"

:: start node server
cd /d "%~dp0..\client"
echo Starting Node.js server...
start "Node.js Server" cmd /c "npm start"

echo Both servers have been started.
exit
