import json
import logging
import os
import threading

import azure.functions as func
from numerapi import NumerAPI

app = func.FunctionApp()


def download_data(api):
    for dataset, path in [
        ("v5.0/train.parquet", "/tmp/train.parquet"),
        ("v5.0/validation.parquet", "/tmp/validation.parquet"),
        ("v5.0/live.parquet", "/tmp/live.parquet"),
    ]:
        if not os.path.exists(path) or dataset == "v5.0/live.parquet":
            api.download_dataset(dataset, dest_path=path)
            logging.info("Downloaded %s", dataset)


def get_features_and_target(data):
    features = [c for c in data.columns if c.startswith("feature_")]
    X = data[features].fillna(0.5)
    y = data["target"]
    groups = data["era"]
    return X, y, groups, features


def cross_val_score_era(model, X, y, groups, n_splits=5):
    """Returns mean Spearman correlation across era-based folds."""
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    oof_preds = np.zeros(len(y))
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        corr, _ = spearmanr(preds, y_val)
        scores.append(corr)
        oof_preds[val_idx] = preds
    return float(sum(scores) / len(scores)), oof_preds


def tune_and_train(X, y, groups, features, model_path):
    import joblib
    import numpy as np
    import optuna
    import pandas as pd
    from lightgbm import LGBMRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import QuantileTransformer
    from xgboost import XGBRegressor

    _ = features
    results = {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def lgbm_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "n_jobs": -1,
            "verbose": -1,
        }
        model = LGBMRegressor(**params)
        score, _oof = cross_val_score_era(model, X, y, groups)
        return score

    lgbm_study = optuna.create_study(direction="maximize")
    lgbm_study.optimize(lgbm_objective, n_trials=30, timeout=900)
    best_lgbm_params = lgbm_study.best_params
    best_lgbm_params.update({"n_jobs": -1, "verbose": -1})
    lgbm_final = LGBMRegressor(**best_lgbm_params)
    lgbm_score, lgbm_oof = cross_val_score_era(lgbm_final, X, y, groups)
    lgbm_final.fit(X, y)
    results["lgbm"] = {"model": lgbm_final, "score": lgbm_score, "oof": lgbm_oof}
    logging.info("LightGBM tuned CV score: %.5f", lgbm_score)

    def xgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "tree_method": "hist",
            "n_jobs": -1,
            "verbosity": 0,
        }
        model = XGBRegressor(**params)
        score, _oof = cross_val_score_era(model, X, y, groups)
        return score

    xgb_study = optuna.create_study(direction="maximize")
    xgb_study.optimize(xgb_objective, n_trials=20, timeout=600)
    best_xgb_params = xgb_study.best_params
    best_xgb_params.update({"tree_method": "hist", "n_jobs": -1, "verbosity": 0})
    xgb_final = XGBRegressor(**best_xgb_params)
    xgb_score, xgb_oof = cross_val_score_era(xgb_final, X, y, groups)
    xgb_final.fit(X, y)
    results["xgb"] = {"model": xgb_final, "score": xgb_score, "oof": xgb_oof}
    logging.info("XGBoost tuned CV score: %.5f", xgb_score)

    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    X_scaled = pd.DataFrame(qt.fit_transform(X), columns=X.columns, index=X.index)

    def mlp_objective(trial):
        layer_size = trial.suggest_categorical("layer_size", [64, 128, 256])
        n_layers = trial.suggest_int("n_layers", 2, 4)
        hidden = tuple([layer_size] * n_layers)
        params = {
            "hidden_layer_sizes": hidden,
            "learning_rate_init": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "max_iter": 200,
            "early_stopping": True,
            "random_state": 42,
        }
        model = MLPRegressor(**params)
        score, _oof = cross_val_score_era(model, X_scaled, y, groups)
        return score

    mlp_study = optuna.create_study(direction="maximize")
    mlp_study.optimize(mlp_objective, n_trials=15, timeout=300)
    best_mlp_params = mlp_study.best_params
    layer_size = best_mlp_params.pop("layer_size")
    n_layers = best_mlp_params.pop("n_layers")
    best_mlp_params["hidden_layer_sizes"] = tuple([layer_size] * n_layers)
    best_mlp_params.update({"max_iter": 200, "early_stopping": True, "random_state": 42})
    mlp_final = MLPRegressor(**best_mlp_params)
    mlp_score, mlp_oof = cross_val_score_era(mlp_final, X_scaled, y, groups)
    mlp_final.fit(X_scaled, y)
    results["mlp"] = {"model": mlp_final, "score": mlp_score, "oof": mlp_oof, "scaler": qt}
    logging.info("MLP tuned CV score: %.5f", mlp_score)

    ranked_models = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)
    top2 = ranked_models[:2]
    logging.info(
        "Top models: %s (%.5f), %s (%.5f)",
        top2[0][0],
        top2[0][1]["score"],
        top2[1][0],
        top2[1][1]["score"],
    )

    scores = np.array([m[1]["score"] for m in top2])
    weights = np.exp(scores) / np.exp(scores).sum()

    ensemble = {
        "models": [(name, info["model"], info.get("scaler")) for name, info in top2],
        "weights": weights,
        "scores": {name: info["score"] for name, info in results.items()},
    }

    joblib.dump(ensemble, model_path)
    logging.info(
        "Ensemble saved to %s. Weights: %s=%0.3f, %s=%0.3f",
        model_path,
        top2[0][0],
        float(weights[0]),
        top2[1][0],
        float(weights[1]),
    )
    return ensemble


def predict_and_neutralize(ensemble, X_live, features):
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata

    _ = features
    blended = np.zeros(len(X_live))
    for (name, model, scaler), weight in zip(ensemble["models"], ensemble["weights"]):
        X_input = scaler.transform(X_live) if scaler else X_live.values
        if scaler:
            X_input = pd.DataFrame(X_input, columns=X_live.columns, index=X_live.index)
        preds = model.predict(X_input)
        preds_ranked = (rankdata(preds) - 0.5) / len(preds)
        blended += weight * preds_ranked
        logging.info("Model %s contributing weight %.3f", name, float(weight))

    exposures = X_live.values
    exposures = np.hstack([exposures, np.ones((len(exposures), 1))])
    blended = blended - 0.5 * (exposures @ np.linalg.pinv(exposures) @ blended)

    final = (rankdata(blended) - 0.5) / len(blended)
    return final


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

        download_data(api)

        import joblib
        import pandas as pd
        from datetime import datetime

        model_path = f"/tmp/ensemble_{datetime.utcnow().strftime('%Y_%m')}.pkl"

        if not os.path.exists(model_path):
            logging.info("Training new ensemble for %s...", datetime.utcnow().strftime("%Y-%m"))
            train = pd.read_parquet("/tmp/train.parquet")
            val = pd.read_parquet("/tmp/validation.parquet")
            data = pd.concat([train, val])
            X, y, groups, features = get_features_and_target(data)
            ensemble = tune_and_train(X, y, groups, features, model_path)
        else:
            ensemble = joblib.load(model_path)
            logging.info(
                "Loaded cached ensemble from %s | Scores: %s",
                model_path,
                ensemble.get("scores", {}),
            )

        live = pd.read_parquet("/tmp/live.parquet")
        features = [c for c in live.columns if c.startswith("feature_")]
        X_live = live[features].fillna(0.5)
        ranked = predict_and_neutralize(ensemble, X_live, features)

        submission = pd.DataFrame({"prediction": ranked}, index=X_live.index)
        submission.to_csv("/tmp/submission.csv")
        logging.info(
            "Predictions generated: %d rows, mean=%.4f, std=%.4f",
            len(submission),
            float(submission["prediction"].mean()),
            float(submission["prediction"].std()),
        )

        model_ids = os.environ.get("NUMERAI_MODEL_IDS", "").split(",")
        submitted = []
        for model_id in model_ids:
            model_id = model_id.strip()
            if not model_id:
                continue
            api.upload_predictions("/tmp/submission.csv", model_id=model_id)
            logging.info("Submitted round %s -> model %s", current_round, model_id)
            submitted.append(model_id)

        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "round": current_round,
                    "submitted_models": submitted,
                    "ensemble_scores": ensemble.get("scores", {}),
                },
                f,
            )

        return {
            "round": current_round,
            "submitted": submitted,
            "ensemble_scores": ensemble.get("scores", {}),
        }
    except Exception as e:
        logging.error("run_bot FAILED: %s", str(e), exc_info=True)
        return {"error": str(e)}


@app.timer_trigger(schedule="0 * * * * *", arg_name="mytimer", run_on_startup=True, use_monitor=True)
def run_numeraibot(mytimer: func.TimerRequest) -> None:
    status_file = "/tmp/bot_status.json"

    if os.path.exists(status_file):
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                status = json.load(f)
            if status.get("running"):
                logging.info("Skipping timer run because a bot run is already active.")
                return
        except Exception:
            # If status cannot be parsed, proceed and let run_bot handle errors.
            pass

    with open(status_file, "w", encoding="utf-8") as f:
        json.dump({"running": True, "started_by": "timer"}, f)

    try:
        result = run_bot()
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({"running": False, "last_result": result}, f)
    except Exception as e:
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({"running": False, "error": str(e)}, f)


@app.route(route="run-now", methods=["GET", "POST"], auth_level=func.AuthLevel.ANONYMOUS)
def run_now(req: func.HttpRequest) -> func.HttpResponse:
    """
    Returns 202 immediately and runs the bot in a background thread.
    Check Azure Portal -> Monitor -> Invocations for live progress logs.
    """
    status_file = "/tmp/bot_status.json"

    # If already running, return current status
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            status = json.load(f)
        if status.get("running"):
            return func.HttpResponse(
                json.dumps({"message": "Bot already running", "status": status}),
                mimetype="application/json",
                status_code=202,
            )

    def run_in_background():
        with open(status_file, "w") as f:
            json.dump({"running": True, "started_at": str(__import__("datetime").datetime.utcnow())}, f)
        try:
            result = run_bot()
            with open(status_file, "w") as f:
                json.dump({"running": False, "last_result": result}, f)
        except Exception as e:
            with open(status_file, "w") as f:
                json.dump({"running": False, "error": str(e)}, f)

    thread = threading.Thread(target=run_in_background, daemon=True)
    thread.start()

    return func.HttpResponse(
        json.dumps({
            "message": "Bot started in background. Check Azure Portal logs for progress.",
            "monitor_url": "https://portal.azure.com -> Function Apps -> numerai-bot -> Monitor",
            "status_tip": "Call /api/run-now again to check if still running."
        }),
        mimetype="application/json",
        status_code=202,
    )
