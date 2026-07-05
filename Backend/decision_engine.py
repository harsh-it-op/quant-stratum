import pandas as pd
from typing import Dict, Any, List

class AdvisoryDecisionEngine:
    """
    SaaS Core Decision Engine v2.0
    Converts Market Risk + Structural/Tactical Models into a Unified Composite Score.
    """
    FAST_OVERRIDE_MIN_CONFIDENCE = 0.90
    FAST_OVERRIDE_MIN_PERSIST_BARS = 4
    
    @staticmethod
    def evaluate(market_regime: str, market_confidence: float,
                 behavior_slow: str, slow_conf: float,
                 behavior_fast: str, fast_conf: float, fast_gap: float,
                 time_since_last_fast_flip: int, 
                 stock_data: Dict[str, Any] = None) -> Dict[str, Any]:
                 
        if stock_data is None:
            stock_data = {"rsi": 50.0, "momentum_20d": 0.0, "relative_strength": "neutral"}
            
        score = 0.0
        reasoning = []
        
        # --- PRE-COMPUTE FLAGS ---
        is_behavior_mismatch = (behavior_slow != behavior_fast)
        is_high_market_stress = ("Stress" in market_regime) or ("Fragile" in market_regime and market_confidence > 0.7)
        
        # --- 1. MARKET LAYER SCORING ---
        if "Stress" in market_regime:
            score -= 4.0
            reasoning.append(f"Market is in tail-risk Stress regime (-4.0)")
            exposure_limit = "30-50%"
            risk_status = "CRITICAL - Capital Preservation"
        elif "Fragile" in market_regime:
            score -= 1.5
            reasoning.append(f"Market structure is Fragile (-1.5)")
            exposure_limit = "75%"
            risk_status = "Elevated Risk"
        else:
            score += 2.0
            reasoning.append(f"Market risk is Durable/Calm (+2.0)")
            exposure_limit = "100%"
            risk_status = "Normal"

        # --- 2. STRUCTURAL SLOW BEHAVIOR SCORING ---
        if behavior_slow == "Trending":
            score += 2.5
            reasoning.append(f"Slow structural behavior indicates broad Trend (+2.5, conf: {slow_conf:.2f})")
        elif behavior_slow == "Mean-Reverting":
            score -= 1.0
            reasoning.append(f"Slow structural behavior is Mean-Reverting sideways (-1.0, conf: {slow_conf:.2f})")
        else: # Noisy
            score -= 2.0
            reasoning.append(f"Slow structural behavior is Noisy/Broken (-2.0, conf: {slow_conf:.2f})")

        # --- 3. DISAGREEMENT & TACTICAL OVERRIDE ---
        fast_used = False
        
        if is_behavior_mismatch:
            reasoning.append("Mismatch detected between Fast and Slow behavior regimes.")
            if is_high_market_stress:
                score -= 1.5
                reasoning.append("Mismatch + High Market Stress -> Defensive posture (-1.5)")
                exposure_limit = "50%" # override
            else:
                score -= 0.5
                reasoning.append("Mismatch + Calm Market -> Reduced confidence/size (-0.5)")
                
            # Requires high confidence AND persistence to let fast model influence score.
            if (
                fast_conf >= AdvisoryDecisionEngine.FAST_OVERRIDE_MIN_CONFIDENCE
                and time_since_last_fast_flip >= AdvisoryDecisionEngine.FAST_OVERRIDE_MIN_PERSIST_BARS
            ):
                fast_used = True
                fast_signal = 0.0
                if behavior_fast == "Trending":
                    fast_signal = 1.5
                elif behavior_fast == "Mean-Reverting":
                    fast_signal = -0.5
                elif behavior_fast == "Noisy":
                    fast_signal = -1.0
                score += fast_signal
                reasoning.append(f"Fast tactical {behavior_fast} state persistent and confident -> Impact: {fast_signal:+.2f}")
            else:
                reasoning.append(
                    f"Ignored fast tactical {behavior_fast} "
                    f"(needs persist >= {AdvisoryDecisionEngine.FAST_OVERRIDE_MIN_PERSIST_BARS} "
                    f"and conf >= {AdvisoryDecisionEngine.FAST_OVERRIDE_MIN_CONFIDENCE:.2f}, "
                    f"got persist {time_since_last_fast_flip}, conf {fast_conf:.2f})"
                )
        else:
            reasoning.append("Fast and Slow behavior states are in agreement (High conviction).")
            score += 1.0
            reasoning.append("Agreement Bonus (+1.0)")
            
        # --- 4. STOCK-SPECIFIC SCORING ---
        rsi = stock_data.get("rsi", 50)
        rel_str = stock_data.get("relative_strength", "neutral")
        
        if rsi > 75:
            score -= 1.5
            reasoning.append(f"Stock is heavily overbought (RSI {round(rsi, 1)}) (-1.5)")
            stock_filter = "Distribution / Take Profit"
        elif rsi < 30:
            score += 1.0
            reasoning.append(f"Stock is deeply oversold (RSI {round(rsi, 1)}) (+1.0)")
            stock_filter = "Oversold Bounce Candidate"
        else:
            stock_filter = "Standard Setup"
            
        if rel_str == "outperform":
            score += 1.0
            reasoning.append("Stock is outperforming the Nifty 500 (+1.0)")
        elif rel_str == "underperform":
            score -= 1.0
            reasoning.append("Stock is lagging the Nifty 500 (-1.0)")

        # --- 5. SCORE TO ACTION TRANSLATION ---
        if score >= 4.0:
            action = "Aggressive Buy / Hold Winners"
        elif score >= 2.0:
            action = "Buy on Dips / Accumulate"
        elif score >= 0.0:
            action = "Neutral / Wait for Confirmation"
        elif score >= -2.0:
            action = "Reduce Exposure / Tighten Stops"
        else:
            action = "Cash is a Position / Hedge Longs"
            
        # Absolute Stress Override
        if "Stress" in market_regime:
            action = "Hedge existing longs, avoid breakouts"
            stock_filter = "No Entry (Market Override)"

        return {
            "composite_score": round(score, 2),
            "final_action": action,
            "exposure_limit": exposure_limit,
            "risk_status": risk_status,
            "tactical_override": fast_used,
            "stock_filter": stock_filter,
            "reasoning": reasoning
        }
