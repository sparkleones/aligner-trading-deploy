"""ML Trade Filter — live XGBoost win-probability gate for V14.

Loads a pre-trained XGBoost model (saved by backtesting/ml_trade_classifier.py)
and provides real-time win-probability predictions for trade signals.

The model is trained offline on historical V14 backtest data with walk-forward
validation. In live mode, it takes the same 25 features and returns P(win).

Usage:
    from orchestrator.ml_trade_filter import MLTradeFilter

    ml = MLTradeFilter()
    if ml.is_ready:
        win_prob = ml.predict(features_dict)
        if win_prob < 0.35:
            skip_trade()

Model training:
    Run `python -m backtesting.ml_trade_classifier` to train and save the model.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Model persistence path
MODEL_DIR = Path(__file__).parent.parent / "data" / "models"
MODEL_PATH = MODEL_DIR / "xgb_trade_filter.joblib"

# Feature columns (must match build_feature_matrix in ml_train_enhanced.py)
FEATURE_COLS = [
    # Original 26 features
    "is_put", "confidence", "vix", "entry_minute", "lots",
    "hour_of_entry", "day_of_week", "month_num", "is_expiry",
    "strike_distance_pct", "premium_pct",
    "rsi_at_entry", "bb_width_at_entry", "atr_pct_at_entry",
    "ema9_above_21", "above_ema50", "st_direction",
    "macd_hist_at_entry", "adx_at_entry", "vwap_proximity",
    "squeeze_at_entry", "stoch_k_at_entry",
    "above_sma20", "above_sma50", "price_vs_open_pct",
    "entry_type_encoded",
    # V14 Enhanced features (14 new)
    "cci_at_entry", "williams_r_at_entry", "rsi_divergence",
    "pcr_extreme", "vix_zone", "adx_regime", "rsi_zone",
    "minutes_in_day", "momentum_alignment", "volatility_ratio",
    "stoch_zone", "time_to_close", "is_monday", "is_expiry_eve",
]


class MLTradeFilter:
    """Live ML win-probability filter using pre-trained XGBoost.

    Loads the model from disk on init. If no model exists, operates
    in pass-through mode (is_ready=False, all trades allowed).
    """

    def __init__(self, model_path: Optional[Path] = None):
        self._model = None
        self._model_path = model_path or MODEL_PATH
        self._load_model()

    def _load_model(self) -> None:
        """Load pre-trained model from disk."""
        if not self._model_path.exists():
            logger.info("ML filter: no model at %s — pass-through mode", self._model_path)
            return

        try:
            import joblib
            self._model = joblib.load(self._model_path)
            logger.info("ML filter: loaded model from %s", self._model_path)
        except ImportError:
            # Try pickle fallback
            try:
                import pickle
                with open(self._model_path, "rb") as f:
                    self._model = pickle.load(f)
                logger.info("ML filter: loaded model (pickle) from %s", self._model_path)
            except Exception as e:
                logger.warning("ML filter: failed to load model: %s", e)
        except Exception as e:
            logger.warning("ML filter: failed to load model: %s", e)

    @property
    def is_ready(self) -> bool:
        """True if model is loaded and can make predictions."""
        return self._model is not None

    def predict(self, features: dict) -> float:
        """Predict win probability for a trade.

        Parameters
        ----------
        features : dict
            Feature dictionary with keys matching FEATURE_COLS.
            Missing features default to 0.

        Returns
        -------
        float
            Win probability [0, 1]. Higher = more likely to be profitable.
        """
        if not self.is_ready:
            return 0.5  # Neutral — don't block

        try:
            # Build feature vector in correct column order
            X = np.array([[features.get(col, 0) for col in FEATURE_COLS]])

            # predict_proba returns [[P(loss), P(win)]]
            prob = self._model.predict_proba(X)[0, 1]
            return float(prob)
        except Exception as e:
            logger.debug("ML predict error: %s — returning 0.5", e)
            return 0.5  # Fail open — don't block on errors

    @staticmethod
    def save_model(model, path: Optional[Path] = None) -> None:
        """Save a trained model to disk for live use.

        Call this from ml_trade_classifier.py after training:
            from orchestrator.ml_trade_filter import MLTradeFilter
            MLTradeFilter.save_model(xgb_model)
        """
        save_path = path or MODEL_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import joblib
            joblib.dump(model, save_path)
            logger.info("ML model saved to %s", save_path)
        except ImportError:
            import pickle
            with open(save_path, "wb") as f:
                pickle.dump(model, f)
            logger.info("ML model saved (pickle) to %s", save_path)
