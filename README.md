# Numerai Automation

Azure Functions-based Numerai automation bot.

## What it does
- Runs every 6 hours on a timer.
- Exposes `/api/run-now` for manual trigger.
- Uses `NUMERAI_PUBLIC_ID` and `NUMERAI_SECRET_KEY` app settings.

## Files
- `function_app.py`: Function app with timer + HTTP trigger.
- `host.json`: Function host settings.
- `requirements.txt`: Python dependencies.
