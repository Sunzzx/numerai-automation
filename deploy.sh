#!/bin/bash
set -e

FUNC_APP_NAME="${1:-numerai-bot}"
RESOURCE_GROUP="${2:-numerai-rg}"
LOCATION="${3:-eastus}"
STORAGE_ACCOUNT="${4:-numeraistorage$RANDOM}"

echo "=== Numerai AutoML Bot — Full Deploy ==="

# 1. Login check
az account show > /dev/null 2>&1 || az login

# 2. Create resource group if not exists
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none
echo "✓ Resource group: $RESOURCE_GROUP"

# 3. Create storage account (required for timer triggers)
az storage account create \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --sku Standard_LRS \
  --output none
echo "✓ Storage account: $STORAGE_ACCOUNT"

# 4. Create Function App (Python 3.11, consumption plan)
az functionapp create \
  --resource-group "$RESOURCE_GROUP" \
  --consumption-plan-location "$LOCATION" \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --name "$FUNC_APP_NAME" \
  --storage-account "$STORAGE_ACCOUNT" \
  --os-type linux \
  --output none
echo "✓ Function App created: $FUNC_APP_NAME"

# 5. Set App Settings (reads from environment or prompts)
az functionapp config appsettings set \
  --name "$FUNC_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --settings \
    "NUMERAI_PUBLIC_ID=${NUMERAI_PUBLIC_ID}" \
    "NUMERAI_SECRET_KEY=${NUMERAI_SECRET_KEY}" \
    "NUMERAI_MODEL_IDS=${NUMERAI_MODEL_IDS}" \
  --output none
echo "✓ App settings configured"

# 6. Deploy code
func azure functionapp publish "$FUNC_APP_NAME" --python
echo "✓ Code deployed"

# 7. Trigger first run immediately
HTTP_URL=$(az functionapp function show \
  --resource-group "$RESOURCE_GROUP" \
  --name "$FUNC_APP_NAME" \
  --function-name run_now \
  --query "invokeUrlTemplate" -o tsv 2>/dev/null || \
  echo "https://${FUNC_APP_NAME}.azurewebsites.net/api/run-now")

echo ""
echo "=== Deploy Complete ==="
echo "Manual trigger URL: $HTTP_URL"
echo "Timer: runs automatically every 6 hours"
echo "Monitor: https://portal.azure.com → Function Apps → $FUNC_APP_NAME → Monitor"
echo ""
echo "First run starting in ~60 seconds (run_on_startup=True)..."