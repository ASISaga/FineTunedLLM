// Main parameters file for development environment
using './main.bicep'

param environmentName = readEnvironmentVariable('AZURE_ENV_NAME', 'dev')
param location = readEnvironmentVariable('AZURE_LOCATION', 'eastus')
param openaiApiKey = readEnvironmentVariable('OPENAI_API_KEY')
param anthropicApiKey = readEnvironmentVariable('ANTHROPIC_API_KEY')
