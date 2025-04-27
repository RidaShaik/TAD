Write-Host "Stopping Flask and Node.js servers..."

# Kill Flask server
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

# Kill Node server
Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force

Write-Host "Servers stopped."
