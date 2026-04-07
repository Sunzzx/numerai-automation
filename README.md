# Numerai AutoML Bot

Fully autonomous Numerai Classic Tournament bot deployed on Azure Functions.
Trains, tunes, ensembles, and submits predictions every round — zero manual intervention.

## What It Does
- Downloads Numerai v5.0 dataset every round
- Monthly AutoML: tunes LightGBM + XGBoost + MLP Neural Net with Optuna
- Selects top-2 models by Spearman CORR, softmax-weighted ensemble
- Neutralizes predictions for maximum TC (True Contribution) score
- Rank-normalizes to uniform  per Numerai spec
- Submits to all your registered models every 6 hours automatically
- Never submits the same round twice (state tracking)

## Quick Deploy (first time)

### Prerequisites
- Azure CLI installed: `brew install azure-cli`
- Azure Functions Core Tools: `npm install -g azure-functions-core-tools@4`
- Active Azure subscription
- Numerai API keys from https://numer.ai/account
- Your Numerai model UUIDs from https://numer.ai/models

### Deploy
```bash
export NUMERAI_PUBLIC_ID="your-public-id"
export NUMERAI_SECRET_KEY="your-secret-key"
export NUMERAI_MODEL_IDS="model-uuid-1,model-uuid-2"

bash deploy.sh numerai-bot numerai-rg eastus
```

### Re-deploy after code changes
```bash
func azure functionapp publish numerai-bot --python
```

### Trigger manually anytime
```bash
curl https://numerai-bot.azurewebsites.net/api/run-now
```

## Monitor
- Azure Portal → Function Apps → numerai-bot → Monitor → Invocations
- Check logs for: `Submitted round X → model Y` and `ensemble_scores`

## Earnings Strategy
1. **Never miss a round** — timer fires every 6h, covers all submission windows
2. **Ensemble + neutralization** — maximizes TC score (unique alpha contribution)
3. **Monthly retraining** — adapts to new data patterns automatically
4. **Stake NMR** — after 20+ consecutive rounds of positive CORR, stake via https://numer.ai

## Required Azure App Settings
| Setting | Description |
|---|---|
| `NUMERAI_PUBLIC_ID` | From numer.ai/account → API Keys |
| `NUMERAI_SECRET_KEY` | From numer.ai/account → API Keys |
| `NUMERAI_MODEL_IDS` | Comma-separated UUIDs from numer.ai/models |
