# Get the absolute path of the script's directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Start Flask server
Write-Host "Starting Flask server..."
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "$scriptDir\..\flask-server\server.py"

# Start Node.js server
Write-Host "Starting Node.js server..."
Start-Process -NoNewWindow -FilePath "cmd.exe" -ArgumentList "/c cd `"$scriptDir\..\client`" && npm start"

Write-Host "Both servers have been started."
