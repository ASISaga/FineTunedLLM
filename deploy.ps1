# PowerShell deployment script for FineTunedLLM on Azure
# This script automates the deployment process using Azure Developer CLI

param(
    [Parameter(Mandatory=$false)]
    [string]$Environment = "dev",
    
    [Parameter(Mandatory=$false)]
    [string]$Location = "eastus",
    
    [Parameter(Mandatory=$false)]
    [string]$OpenAIApiKey,
    
    [Parameter(Mandatory=$false)]
    [string]$AnthropicApiKey,
    
    [Parameter(Mandatory=$false)]
    [switch]$PreviewOnly = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipLogin = $false
)

Write-Host "🚀 FineTunedLLM Azure Deployment Script" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Check prerequisites
Write-Host "📋 Checking prerequisites..." -ForegroundColor Yellow

# Check if azd is installed
try {
    $azdVersion = azd version 2>$null
    Write-Host "✅ Azure Developer CLI found: $azdVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Azure Developer CLI not found. Please install it first." -ForegroundColor Red
    Write-Host "   Download from: https://docs.microsoft.com/en-us/azure/developer/azure-developer-cli/install-azd" -ForegroundColor Yellow
    exit 1
}

# Check if az CLI is installed
try {
    $azVersion = az version --query '"azure-cli"' -o tsv 2>$null
    Write-Host "✅ Azure CLI found: $azVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Azure CLI not found. Please install it first." -ForegroundColor Red
    exit 1
}

# Navigate to project directory
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot
Write-Host "📁 Working directory: $projectRoot" -ForegroundColor Blue

# Login to Azure (unless skipped)
if (-not $SkipLogin) {
    Write-Host "🔐 Logging in to Azure..." -ForegroundColor Yellow
    try {
        azd auth login
        Write-Host "✅ Successfully logged in to Azure" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to login to Azure" -ForegroundColor Red
        exit 1
    }
}

# Set environment variables
Write-Host "⚙️ Configuring environment..." -ForegroundColor Yellow

azd env set AZURE_ENV_NAME $Environment
azd env set AZURE_LOCATION $Location

# Set API keys if provided
if ($OpenAIApiKey) {
    Write-Host "🔑 Setting OpenAI API key..." -ForegroundColor Yellow
    azd env set OPENAI_API_KEY $OpenAIApiKey
} else {
    Write-Host "⚠️ OpenAI API key not provided. You'll need to set it manually:" -ForegroundColor Yellow
    Write-Host "   azd env set OPENAI_API_KEY 'your-key-here'" -ForegroundColor Gray
}

if ($AnthropicApiKey) {
    Write-Host "🔑 Setting Anthropic API key..." -ForegroundColor Yellow
    azd env set ANTHROPIC_API_KEY $AnthropicApiKey
} else {
    Write-Host "⚠️ Anthropic API key not provided. You'll need to set it manually:" -ForegroundColor Yellow
    Write-Host "   azd env set ANTHROPIC_API_KEY 'your-key-here'" -ForegroundColor Gray
}

# Preview or deploy
if ($PreviewOnly) {
    Write-Host "👀 Previewing deployment..." -ForegroundColor Yellow
    try {
        azd provision --preview
        Write-Host "✅ Preview completed successfully" -ForegroundColor Green
        Write-Host "🔄 To deploy, run: azd up" -ForegroundColor Blue
    } catch {
        Write-Host "❌ Preview failed" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "🚀 Starting deployment..." -ForegroundColor Yellow
    
    # Check if API keys are set
    $openaiKey = azd env get-value OPENAI_API_KEY 2>$null
    $anthropicKey = azd env get-value ANTHROPIC_API_KEY 2>$null
    
    if (-not $openaiKey -or -not $anthropicKey) {
        Write-Host "❌ API keys are required for deployment" -ForegroundColor Red
        Write-Host "Please set them using:" -ForegroundColor Yellow
        if (-not $openaiKey) {
            Write-Host "   azd env set OPENAI_API_KEY 'your-openai-key'" -ForegroundColor Gray
        }
        if (-not $anthropicKey) {
            Write-Host "   azd env set ANTHROPIC_API_KEY 'your-anthropic-key'" -ForegroundColor Gray
        }
        Write-Host "Then run the script again." -ForegroundColor Yellow
        exit 1
    }
    
    try {
        Write-Host "📦 Deploying infrastructure and applications..." -ForegroundColor Blue
        azd up
        
        Write-Host "✅ Deployment completed successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "🔗 Getting service endpoints..." -ForegroundColor Yellow
        azd env get-values | Where-Object { $_ -match "FUNCTION_URL" }
        
        Write-Host ""
        Write-Host "📋 Next steps:" -ForegroundColor Cyan
        Write-Host "1. Upload test documents to the 'input-documents' container" -ForegroundColor Gray
        Write-Host "2. Monitor processing via Application Insights" -ForegroundColor Gray
        Write-Host "3. Check generated summaries in the 'summaries' container" -ForegroundColor Gray
        Write-Host "4. See DEPLOYMENT.md for detailed usage instructions" -ForegroundColor Gray
        
    } catch {
        Write-Host "❌ Deployment failed" -ForegroundColor Red
        Write-Host "Check the error messages above and try again." -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""
Write-Host "🎉 Script completed!" -ForegroundColor Green
