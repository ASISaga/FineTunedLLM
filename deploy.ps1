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

# Validate project structure
Write-Host "🔍 Validating project structure..." -ForegroundColor Yellow

$requiredPaths = @(
    "KnowledgeModel\DomainContextManager.py",
    "KnowledgeModel\AbstractiveSummarizer.py", 
    "KnowledgeModel\DomainAwareTrainer.py",
    "azure-functions\summarization-pipeline\function_app_domain_aware.py",
    "azure-functions\finetuning-pipeline\function_app.py",
    "infra\main.bicep",
    "azure.yaml"
)

$missingFiles = @()
foreach ($path in $requiredPaths) {
    if (-not (Test-Path $path)) {
        $missingFiles += $path
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "❌ Missing required files:" -ForegroundColor Red
    foreach ($file in $missingFiles) {
        Write-Host "   - $file" -ForegroundColor Gray
    }
    exit 1
}

Write-Host "✅ Project structure validated" -ForegroundColor Green

# Validate Bicep template
Write-Host "🔧 Validating Bicep template..." -ForegroundColor Yellow
try {
    $bicepBuild = az bicep build --file "infra\main.bicep" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Bicep template is valid" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Bicep validation warnings:" -ForegroundColor Yellow
        Write-Host $bicepBuild -ForegroundColor Gray
    }
} catch {
    Write-Host "❌ Bicep template validation failed" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

# Validate domain context files
Write-Host "🎯 Validating domain context system..." -ForegroundColor Yellow
try {
    # Check if DomainContextManager can be imported (basic syntax check)
    $pythonCheck = python -c "
import ast
try:
    with open('KnowledgeModel/DomainContextManager.py', 'r') as f:
        ast.parse(f.read())
    print('DomainContextManager: Valid')
except Exception as e:
    print(f'DomainContextManager: Error - {e}')
    
try:
    with open('KnowledgeModel/AbstractiveSummarizer.py', 'r') as f:
        ast.parse(f.read())
    print('AbstractiveSummarizer: Valid')
except Exception as e:
    print(f'AbstractiveSummarizer: Error - {e}')
" 2>$null

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Domain context files validated" -ForegroundColor Green
        Write-Host $pythonCheck -ForegroundColor Gray
    } else {
        Write-Host "⚠️ Python validation not available (Python not found)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️ Could not validate Python files (Python not available)" -ForegroundColor Yellow
}

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
        Write-Host "📦 Deploying domain-aware FineTunedLLM infrastructure and applications..." -ForegroundColor Blue
        azd up
        
        Write-Host "✅ Deployment completed successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "🔗 Getting service endpoints..." -ForegroundColor Yellow
        $endpoints = azd env get-values | Where-Object { $_ -match "FUNCTION_URL" }
        $endpoints
        
        Write-Host ""
        Write-Host "🎯 Domain-Aware Features Deployed:" -ForegroundColor Cyan
        Write-Host "   • Technical domain processing (APIs, software, systems)" -ForegroundColor Gray
        Write-Host "   • Medical domain processing (clinical, healthcare)" -ForegroundColor Gray
        Write-Host "   • Legal domain processing (contracts, compliance)" -ForegroundColor Gray
        Write-Host "   • Financial domain processing (banking, investment)" -ForegroundColor Gray
        Write-Host "   • Automatic domain detection from filenames" -ForegroundColor Gray
        Write-Host "   • Domain-specific fine-tuning pipelines" -ForegroundColor Gray
        
        Write-Host ""
        Write-Host "📋 Next steps:" -ForegroundColor Cyan
        Write-Host "1. Upload test documents with domain keywords in filenames:" -ForegroundColor Gray
        Write-Host "   - technical_api_documentation.pdf" -ForegroundColor DarkGray
        Write-Host "   - medical_clinical_study.pdf" -ForegroundColor DarkGray
        Write-Host "   - legal_contract_review.pdf" -ForegroundColor DarkGray
        Write-Host "   - financial_market_analysis.pdf" -ForegroundColor DarkGray
        Write-Host "2. Monitor domain-specific processing via Application Insights" -ForegroundColor Gray
        Write-Host "3. Check domain-enhanced summaries in storage containers" -ForegroundColor Gray
        Write-Host "4. Use /api/domains endpoint to list available domains" -ForegroundColor Gray
        Write-Host "5. See DEPLOYMENT.md for detailed usage instructions" -ForegroundColor Gray
        
    } catch {
        Write-Host "❌ Deployment failed" -ForegroundColor Red
        Write-Host "Check the error messages above and try again." -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""
Write-Host "🎉 Script completed!" -ForegroundColor Green
