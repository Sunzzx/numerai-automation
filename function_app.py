import json
import logging
import os

import azure.functions as func
from numerapi import NumerAPI

app = func.FunctionApp()


def run_bot() -> dict:
    try:
        public_id = os.environ["NUMERAI_PUBLIC_ID"]
        secret_key = os.environ["NUMERAI_SECRET_KEY"]
        api = NumerAPI(public_id=public_id, secret_key=secret_key)

        state_file = "/tmp/last_round.json"
        current_round = api.get_current_round()

        if os.path.exists(state_file):
            with open(state_file, "r", encoding="utf-8") as f:
                last = json.load(f)
            if last.get("round") == current_round:
                logging.info("Round %s already submitted. Skipping.", current_round)
                return {"skipped": True, "round": current_round}

        for dataset, path in [
            ("v5.0/train.parquet", "/tmp/train.parquet"),
            ("v5.0/validation.parquet", "/tmp/validation.parquet"),
            ("v5.0/live.parquet", "/tmp/live.parquet"),
        ]:
            if not os.path.exists(path) or dataset == "v5.0/live.parquet":
                api.download_dataset(dataset, dest_path=path)
                logging.info("Downloaded %s", dataset)

        import joblib
        from datetime import datetime

        model_path = f"/tmp/model_{datetime.utcnow().strftime('%Y_%m')}.pkl"

        if not os.path.exists(model_path):
            import numpy as np
            import pandas as pd
            from lightgbm import LGBMRegressor
            from sklearn.model_selection import GroupKFold

            train = pd.read_parquet("/tmp/train.parquet")
            val = pd.read_parquet("/tmp/validation.parquet")
            data = pd.concat([train, val])

            features = [c for c in data.columns if c.startswith("feature_")]
            X = data[features].fillna(0.5)
            y = data["target"]
            groups = data["era"]

            model = LGBMRegressor(
                n_estimators=2000,
                learning_rate=0.01,
                max_depth=5,
                num_leaves=31,
                colsample_bytree=0.1,
                n_jobs=-1,
                verbose=-1,
            )
            _ = GroupKFold
            _ = groups
            model.fit(X, y)
            joblib.dump(model, model_path)
            logging.info("Model trained and saved to %s", model_path)
        else:
            model = joblib.load(model_path)
            logging.info("Loaded cached model from %s", model_path)

        import numpy as np
        import pandas as pd
        from scipy.stats import rankdata

        live = pd.read_parquet("/tmp/live.parquet")
        features = [c for c in live.columns if c.startswith("feature_")]
        X_live = live[features].fillna(0.5)

        raw_preds = model.predict(X_live)

        def neutralize(preds, features_df, proportion=0.5):
            exposures = features_df.values
            exposures = np.hstack([exposures, np.ones((len(exposures), 1))])
            preds = preds - proportion * (exposures @ np.linalg.pinv(exposures) @ preds)
            return preds

        neutralized = neutralize(raw_preds, X_live)
        ranked = (rankdata(neutralized) - 0.5) / len(neutralized)

        submission = pd.DataFrame({"prediction": ranked}, index=X_live.index)
        submission.to_csv("/tmp/submission.csv")
        logging.info("Predictions generated: %d rows", len(submission))

        model_ids = os.environ.get("NUMERAI_MODEL_IDS", "").split(",")
        submitted = []
        for model_id in model_ids:
            model_id = model_id.strip()
            if not model_id:
                continue
            api.upload_predictions("/tmp/submission.csv", model_id=model_id)
            logging.info("Submitted round %s for model_id %s", current_round, model_id)
            submitted.append(model_id)

        with open(state_file, "w", encoding="utf-8") as f:
            json.dump({"round": current_round, "submitted_models": submitted}, f)

        return {"round": current_round, "submitted": submitted}
    except Exception as e:
        logging.error("run_bot FAILED: %s", str(e), exc_info=True)
        return {"error": str(e)}


@app.timer_trigger(schedule="0 0 */6 * * *", arg_name="mytimer", run_on_startup=True, use_monitor=True)
def run_numeraibot(mytimer: func.TimerRequest) -> None:
    run_bot()


@app.route(route="run-now", methods=["GET", "POST"], auth_level=func.AuthLevel.ANONYMOUS)
def run_now(req: func.HttpRequest) -> func.HttpResponse:
    result = run_bot()
    return func.HttpResponse(json.dumps(result), mimetype="application/json")
