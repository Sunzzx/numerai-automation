import json
import logging
import os

import azure.functions as func
from numerapi import NumerAPI

app = func.FunctionApp()


def run_bot() -> dict:
    public_id = os.environ["NUMERAI_PUBLIC_ID"]
    secret_key = os.environ["NUMERAI_SECRET_KEY"]
    api = NumerAPI(public_id=public_id, secret_key=secret_key)
    account = api.get_account()
    current_round = api.get_current_round()
    logging.info("Numerai auth ok: %s", bool(account))
    logging.info("Numerai current round: %s", current_round)
    return {"auth_ok": bool(account), "current_round": current_round}


@app.timer_trigger(schedule="0 0 */6 * * *", arg_name="mytimer", run_on_startup=True, use_monitor=True)
def run_numeraibot(mytimer: func.TimerRequest) -> None:
    run_bot()


@app.route(route="run-now", methods=["GET", "POST"], auth_level=func.AuthLevel.ANONYMOUS)
def run_now(req: func.HttpRequest) -> func.HttpResponse:
    result = run_bot()
    return func.HttpResponse(json.dumps(result), mimetype="application/json")
