@echo off
echo Stopping Flask and Node.js servers...

:: Kill Flask server 
taskkill /F /IM python.exe /T

:: Kill Node server
taskkill /F /IM node.exe /T

echo Servers stopped.
exit
