// Main Bicep template for FineTunedLLM serverless infrastructure
// This template provisions all Azure resources needed for the adaptive summarization
// and fine-tuning pipelines including Function Apps, Storage, Key Vault, and AI services

@description('The environment name (e.g., dev, staging, prod)')
param environmentName string = 'dev'

@description('The primary location for all resources')
param location string = resourceGroup().location

@description('The OpenAI API key for fine-tuning operations')
@secure()
param openaiApiKey string

@description('The Anthropic API key for Claude summarization')
@secure()
param anthropicApiKey string

@description('The unique suffix for resource naming')
param resourceSuffix string = uniqueString(resourceGroup().id)

// Variables for consistent naming
var abbrs = loadJsonContent('abbreviations.json')
var resourceToken = toLower(uniqueString(subscription().id, environmentName, location))
var tags = {
  'azd-env-name': environmentName
  'project': 'finetuned-llm'
  'component': 'serverless-pipeline'
}

// Storage Account for blob triggers and data storage
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: '${abbrs.storageStorageAccounts}${resourceToken}'
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    supportsHttpsTrafficOnly: true
    networkAcls: {
      defaultAction: 'Allow'
    }
  }
}

// Storage containers for pipeline data
resource inputContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  name: '${storageAccount.name}/default/input-documents'
  properties: {
    publicAccess: 'None'
  }
}

resource summaryContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  name: '${storageAccount.name}/default/summaries'
  properties: {
    publicAccess: 'None'
  }
}

resource trainingDataContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  name: '${storageAccount.name}/default/training-data'
  properties: {
    publicAccess: 'None'
  }
}

resource modelsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  name: '${storageAccount.name}/default/models'
  properties: {
    publicAccess: 'None'
  }
}

// Key Vault for secure secret management
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: '${abbrs.keyVaultVaults}${resourceToken}'
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
}

// Store API keys in Key Vault
resource openaiApiKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: 'openai-api-key'
  parent: keyVault
  properties: {
    value: openaiApiKey
    contentType: 'text/plain'
  }
}

resource anthropicApiKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: 'anthropic-api-key'
  parent: keyVault
  properties: {
    value: anthropicApiKey
    contentType: 'text/plain'
  }
}

// Log Analytics Workspace for monitoring
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${abbrs.operationalInsightsWorkspaces}${resourceToken}'
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// Application Insights for function monitoring
resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${abbrs.insightsComponents}${resourceToken}'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
    IngestionMode: 'LogAnalytics'
  }
}

// App Service Plan for Function Apps (Consumption plan for serverless)
resource appServicePlan 'Microsoft.Web/serverfarms@2023-01-01' = {
  name: '${abbrs.webServerFarms}${resourceToken}'
  location: location
  tags: tags
  sku: {
    name: 'Y1'
    tier: 'Dynamic'
  }
  kind: 'functionapp'
  properties: {
    reserved: true // Linux
  }
}

// Managed Identity for Function Apps
resource functionAppIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${abbrs.managedIdentityUserAssignedIdentities}${resourceToken}'
  location: location
  tags: tags
}

// Grant Key Vault access to Function App identity
resource keyVaultRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, functionAppIdentity.id, 'Key Vault Secrets User')
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6') // Key Vault Secrets User
    principalId: functionAppIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Grant Storage access to Function App identity
resource storageRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storageAccount.id, functionAppIdentity.id, 'Storage Blob Data Contributor')
  scope: storageAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe') // Storage Blob Data Contributor
    principalId: functionAppIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Function App for Summarization Pipeline
resource summarizationFunctionApp 'Microsoft.Web/sites@2023-01-01' = {
  name: '${abbrs.webSitesFunctions}summarization-${resourceToken}'
  location: location
  tags: union(tags, { 'azd-service-name': 'summarization-pipeline' })
  kind: 'functionapp,linux'
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${functionAppIdentity.id}': {}
    }
  }
  properties: {
    serverFarmId: appServicePlan.id
    reserved: true
    siteConfig: {
      linuxFxVersion: 'Python|3.11'
      appSettings: [
        {
          name: 'AzureWebJobsStorage'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
        }
        {
          name: 'WEBSITE_CONTENTAZUREFILECONNECTIONSTRING'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
        }
        {
          name: 'WEBSITE_CONTENTSHARE'
          value: 'summarization-content'
        }
        {
          name: 'FUNCTIONS_EXTENSION_VERSION'
          value: '~4'
        }
        {
          name: 'FUNCTIONS_WORKER_RUNTIME'
          value: 'python'
        }
        {
          name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
          value: applicationInsights.properties.ConnectionString
        }
        {
          name: 'AZURE_CLIENT_ID'
          value: functionAppIdentity.properties.clientId
        }
        {
          name: 'KEY_VAULT_URL'
          value: keyVault.properties.vaultUri
        }
        {
          name: 'STORAGE_ACCOUNT_NAME'
          value: storageAccount.name
        }
        {
          name: 'INPUT_CONTAINER'
          value: 'input-documents'
        }
        {
          name: 'SUMMARY_CONTAINER'
          value: 'summaries'
        }
      ]
      cors: {
        allowedOrigins: ['*']
      }
    }
  }
}

// Function App for Fine-tuning Pipeline
resource finetuningFunctionApp 'Microsoft.Web/sites@2023-01-01' = {
  name: '${abbrs.webSitesFunctions}finetuning-${resourceToken}'
  location: location
  tags: union(tags, { 'azd-service-name': 'finetuning-pipeline' })
  kind: 'functionapp,linux'
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${functionAppIdentity.id}': {}
    }
  }
  properties: {
    serverFarmId: appServicePlan.id
    reserved: true
    siteConfig: {
      linuxFxVersion: 'Python|3.11'
      appSettings: [
        {
          name: 'AzureWebJobsStorage'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
        }
        {
          name: 'WEBSITE_CONTENTAZUREFILECONNECTIONSTRING'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
        }
        {
          name: 'WEBSITE_CONTENTSHARE'
          value: 'finetuning-content'
        }
        {
          name: 'FUNCTIONS_EXTENSION_VERSION'
          value: '~4'
        }
        {
          name: 'FUNCTIONS_WORKER_RUNTIME'
          value: 'python'
        }
        {
          name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
          value: applicationInsights.properties.ConnectionString
        }
        {
          name: 'AZURE_CLIENT_ID'
          value: functionAppIdentity.properties.clientId
        }
        {
          name: 'KEY_VAULT_URL'
          value: keyVault.properties.vaultUri
        }
        {
          name: 'STORAGE_ACCOUNT_NAME'
          value: storageAccount.name
        }
        {
          name: 'TRAINING_DATA_CONTAINER'
          value: 'training-data'
        }
        {
          name: 'MODELS_CONTAINER'
          value: 'models'
        }
        {
          name: 'SUMMARY_CONTAINER'
          value: 'summaries'
        }
      ]
      cors: {
        allowedOrigins: ['*']
      }
    }
  }
}

// Outputs for azd and other tools
output AZURE_LOCATION string = location
output AZURE_TENANT_ID string = subscription().tenantId
output AZURE_SUBSCRIPTION_ID string = subscription().subscriptionId

output STORAGE_ACCOUNT_NAME string = storageAccount.name
output KEY_VAULT_NAME string = keyVault.name
output KEY_VAULT_URL string = keyVault.properties.vaultUri

output SUMMARIZATION_FUNCTION_NAME string = summarizationFunctionApp.name
output SUMMARIZATION_FUNCTION_URL string = 'https://${summarizationFunctionApp.properties.defaultHostName}'

output FINETUNING_FUNCTION_NAME string = finetuningFunctionApp.name  
output FINETUNING_FUNCTION_URL string = 'https://${finetuningFunctionApp.properties.defaultHostName}'

output FUNCTION_APP_IDENTITY_ID string = functionAppIdentity.id
output FUNCTION_APP_IDENTITY_CLIENT_ID string = functionAppIdentity.properties.clientId

output APPLICATION_INSIGHTS_CONNECTION_STRING string = applicationInsights.properties.ConnectionString
output LOG_ANALYTICS_WORKSPACE_ID string = logAnalytics.properties.customerId
