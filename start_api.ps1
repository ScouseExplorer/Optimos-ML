# start_api.ps1
# PowerShell script to start the LinkedIn Growth AI API

Write-Host "`n🚀 Starting LinkedIn Growth AI API..." -ForegroundColor Green
Write-Host "="*60 -ForegroundColor Green

# Change to the correct directory
Set-Location "C:\Users\akinl\LinkedIn-AI\LinkedIn-growth"

# Start the API server
& "C:\Users\akinl\LinkedIn-AI\.venv\Scripts\python.exe" -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

Write-Host "`n✅ Server stopped." -ForegroundColor Yellow
