# Deploying Quantum Circuit Optimizer (QCO) to Microsoft Azure

This guide provides step-by-step instructions to deploy the containerized QCO web application to **Azure App Service (Linux Web App for Containers)** or **Azure Container Apps (ACA)**.

---

## Architecture Overview

```
                      +-----------------------------+
                      |       Azure Front Door      |
                      |   (SSL Termination / DNS)   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Azure App Service (Linux) |
                      |    Container: python:3.12   |
                      |      Port: 8501 (HTTP)      |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Streamlit App Engine (QCO) |
                      |  - Circuit Studio           |
                      |  - PyG CircuitGNN Inference |
                      |  - Unitary Equivalence Hub  |
                      |  - Multi-Method Benchmark   |
                      +-----------------------------+
```

---

## Option A: Deploy via Azure App Service (Recommended for Universities / Students)

### Prerequisites
1. [Azure CLI (`az`)](https://learn.microsoft.com/cli/azure/install-azure-cli) installed and authenticated (`az login`).
2. Docker CLI installed locally or access to GitHub Actions.
3. An active Azure Subscription (Student, Free, or Pay-as-you-go).

### Step 1: Create Azure Resource Group & Azure Container Registry (ACR)
```bash
# Define deployment variables
RESOURCE_GROUP="rg-qco-lab"
LOCATION="eastus"
ACR_NAME="acrqcolab$RANDOM"
APP_NAME="qco-learning-platform"
APP_PLAN="asp-qco-linux"

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry (Basic SKU)
az acr create --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic \
  --admin-enabled true
```

### Step 2: Build & Push Image to ACR
```bash
# Log in to your ACR
az acr login --name $ACR_NAME

# Retrieve ACR login server
ACR_SERVER=$(az acr show --name $ACR_NAME --query loginServer --output tsv)

# Build and tag image using Azure Container Registry Tasks (no local docker daemon needed!)
az acr build --registry $ACR_NAME --image qco-app:latest .
```

### Step 3: Create App Service Plan & Web App
```bash
# Create Linux App Service Plan (B1 or higher recommended for PyTorch memory requirements)
az appservice plan create \
  --name $APP_PLAN \
  --resource-group $RESOURCE_GROUP \
  --sku B1 \
  --is-linux

# Create Web App for Containers
az webapp create \
  --resource-group $RESOURCE_GROUP \
  --plan $APP_PLAN \
  --name $APP_NAME \
  --deployment-container-image-name "$ACR_SERVER/qco-app:latest"

# Configure ACR credentials for the Web App
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" --output tsv)
az webapp config container set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --docker-custom-image-name "$ACR_SERVER/qco-app:latest" \
  --docker-registry-server-url "https://$ACR_SERVER" \
  --docker-registry-server-user $ACR_NAME \
  --docker-registry-server-password $ACR_PASSWORD
```

### Step 4: Configure App Settings & Health Probe
```bash
# Set Streamlit port configuration for Azure container proxying
az webapp config appsettings set \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --settings \
    WEBSITES_PORT=8501 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Configure health check probe path
az webapp config set \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --generic-configurations '{"healthCheckPath": "/_stcore/health"}'
```

---

## Option B: Deploy via Azure Container Apps (ACA)

For serverless scale-to-zero when students are inactive:

```bash
# Create Log Analytics workspace
az monitor log-analytics workspace create \
  --resource-group $RESOURCE_GROUP \
  --workspace-name "law-qco"

# Create Container App Environment
az containerapp env create \
  --name "cae-qco" \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

# Deploy Container App
az containerapp create \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment "cae-qco" \
  --image "$ACR_SERVER/qco-app:latest" \
  --target-port 8501 \
  --ingress external \
  --cpu 1.0 \
  --memory 2.0Gi \
  --registry-server $ACR_SERVER \
  --registry-username $ACR_NAME \
  --registry-password $ACR_PASSWORD \
  --min-replicas 1 \
  --max-replicas 3
```

---

## Local Verification with Docker Compose

Before pushing to Azure, verify the container locally:

```bash
# Build and run with docker compose
docker compose up --build

# Open browser at:
http://localhost:8501

# Verify container health endpoint:
curl http://localhost:8501/_stcore/health
```

---

## Monitoring and Maintenance
- **Streaming Logs**: `az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP`
- **CPU / Memory Metrics**: Available in Azure Portal under App Service Metrics.
