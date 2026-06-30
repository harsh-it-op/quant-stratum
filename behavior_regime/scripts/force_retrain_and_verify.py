from datetime import datetime
import numpy as np
import joblib
import pandas as pd

from behavior_regime.scripts.behavior_pipeline import (
    DATA_PATH,
    MODELS_DIR,
    create_behavior_features,
    retrain_models,
    predict_regimes,
    write_behavior_reports,
)


def main() -> int:
    baseline_path = MODELS_DIR / "behavior_regime_components.pkl"
    print("baseline_exists_before", baseline_path.exists())
    if baseline_path.exists():
        before = joblib.load(baseline_path)
        print("before_version", before.get("version"))
        print("before_last_trained", before.get("last_trained"))

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    features = create_behavior_features(df)

    retrain_ok = retrain_models(features, datetime.now().date())
    print("retrain_ok", retrain_ok)

    if baseline_path.exists():
        after = joblib.load(baseline_path)
        print("after_version", after.get("version"))
        print("after_last_trained", after.get("last_trained"))

        slow_model = after.get("slow_model")
        if slow_model is not None:
            diag = np.diag(slow_model.transmat_)
            print("slow_diag", [round(float(x), 6) for x in diag])
            print("slow_max_diag", round(float(diag.max()), 6))

    predictions = predict_regimes()
    artifacts = write_behavior_reports(predictions)
    print("pred_rows", len(predictions))
    print("artifacts", artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
