def get_advisory_payload():
    import pandas as pd
    from pathlib import Path
    from backend.decision_engine import AdvisoryDecisionEngine

    root_dir = Path(__file__).resolve().parents[2]
    market_timeline = root_dir / "market_regime" / "features" / "regime_timeline_history.csv"
    behavior_predictions = root_dir / "output" / "behavior_regime" / "behavior_regime_predictions.csv"
    
    # 1. Load Data
    try:
        mrkt_df = pd.read_csv(market_timeline)
        behav_df = pd.read_csv(behavior_predictions)
        mrkt = mrkt_df.iloc[-1]
        behav = behav_df.iloc[-1]

        # Calculate time since last flip for fast behavior
        try:
            states = behav_df["behavior_fast_state"].values
            curr_state = states[-1]
            idx = len(states) - 2
            while idx >= 0 and states[idx] == curr_state:
                idx -= 1
            time_since_last_fast_flip = len(states) - 1 - idx
        except Exception:
            time_since_last_fast_flip = 10

    except Exception as e:
        return {"error": "Missing model outputs. Run training pipelines first."}

    market_state = mrkt["adaptive_combined_state"] # e.g. "Durable - Calm"
    market_confidence = float(mrkt.get("combined_confidence", mrkt.get("confidence", 0.5)))
    
    slow_state = behav["behavior_slow_state"]
    slow_conf = float(behav.get("behavior_slow_confidence", 0.0))
    fast_state = behav["behavior_fast_state"]
    fast_conf = float(behav.get("behavior_fast_confidence", 0.0))
    fast_gap = float(behav.get("behavior_fast_prob_gap", 0.0))
    hybrid_action = behav.get("hybrid_action", "")

    # 2. Mocking a stock fetch for now (in product this would take ticker RSI arguments)
    stock_data = {"rsi": 65.0, "momentum_20d": 0.0, "relative_strength": "neutral"}
    
    # 3. Decision Engine Routing
    policy = AdvisoryDecisionEngine.evaluate(
        market_regime=market_state,
        market_confidence=market_confidence,
        behavior_slow=slow_state,
        slow_conf=slow_conf,
        behavior_fast=fast_state,
        fast_conf=fast_conf,
        fast_gap=fast_gap,
        time_since_last_fast_flip=time_since_last_fast_flip,
        stock_data=stock_data
    )

    # 4. Formulate Output
    return {
        "market_regime": market_state,
        "behavior_slow": slow_state,
        "behavior_fast": fast_state,
        "behavior_fast_confidence": fast_conf,
        "behavior_fast_used": policy["tactical_override"],
        "hybrid_behavior": hybrid_action,
        "stock_state": f"RSI {stock_data['rsi']} -> {policy['stock_filter']}",
        "final_action": policy["final_action"],
        "exposure_limit": policy["exposure_limit"],
        "risk_status": policy["risk_status"]
    }

if __name__ == '__main__':
    import json
    print(json.dumps(get_advisory_payload(), indent=2))
