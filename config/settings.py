"""
Application settings loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BrokerSettings:
    """Broker API connection settings."""
    api_key: str = field(default_factory=lambda: os.getenv("BROKER_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("BROKER_API_SECRET", ""))
    totp_secret: str = field(default_factory=lambda: os.getenv("BROKER_TOTP_SECRET", ""))
    user_id: str = field(default_factory=lambda: os.getenv("BROKER_USER_ID", ""))
    password: str = field(default_factory=lambda: os.getenv("BROKER_PASSWORD", ""))
    broker_type: str = field(default_factory=lambda: os.getenv("BROKER_TYPE", "zerodha"))
    base_url: str = field(default_factory=lambda: os.getenv("BROKER_BASE_URL", "https://api.kite.trade"))
    websocket_url: str = field(default_factory=lambda: os.getenv("BROKER_WS_URL", "wss://ws.kite.trade"))


@dataclass
class TradingSettings:
    """Core trading parameters."""
    capital: float = field(default_factory=lambda: float(os.getenv("TRADING_CAPITAL", "1000000")))
    max_daily_loss_pct: float = field(default_factory=lambda: float(os.getenv("MAX_DAILY_LOSS_PCT", "0.02")))
    max_position_size_pct: float = field(default_factory=lambda: float(os.getenv("MAX_POSITION_SIZE_PCT", "0.10")))
    max_open_positions: int = field(default_factory=lambda: int(os.getenv("MAX_OPEN_POSITIONS", "10")))
    default_index: str = field(default_factory=lambda: os.getenv("DEFAULT_INDEX", "NIFTY"))
    enable_straddle: bool = field(default_factory=lambda: os.getenv("ENABLE_STRADDLE", "true").lower() == "true")
    enable_pairs: bool = field(default_factory=lambda: os.getenv("ENABLE_PAIRS", "true").lower() == "true")
    enable_dispersion: bool = field(default_factory=lambda: os.getenv("ENABLE_DISPERSION", "false").lower() == "true")
    paper_trading: bool = field(default_factory=lambda: os.getenv("PAPER_TRADING", "true").lower() == "true")


@dataclass
class MLSettings:
    """Machine learning model configuration."""
    model_path: str = field(default_factory=lambda: os.getenv("MODEL_PATH", "models/ddqn_v3.pt"))
    retrain_interval_days: int = field(default_factory=lambda: int(os.getenv("RETRAIN_INTERVAL_DAYS", "7")))
    lookback_bars: int = field(default_factory=lambda: int(os.getenv("LOOKBACK_BARS", "100")))
    bar_interval_minutes: int = field(default_factory=lambda: int(os.getenv("BAR_INTERVAL_MINUTES", "15")))
    use_gpu: bool = field(default_factory=lambda: os.getenv("USE_GPU", "false").lower() == "true")


@dataclass
class InfraSettings:
    """Infrastructure and deployment settings."""
    grpc_host: str = field(default_factory=lambda: os.getenv("GRPC_HOST", "127.0.0.1"))
    grpc_port: int = field(default_factory=lambda: int(os.getenv("GRPC_PORT", "50051")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_file: str = field(default_factory=lambda: os.getenv("LOG_FILE", "logs/trading.log"))
    static_ip: Optional[str] = field(default_factory=lambda: os.getenv("STATIC_IP"))
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "ap-south-1"))


@dataclass
class DataSettings:
    """Historical data provider settings."""
    provider: str = field(default_factory=lambda: os.getenv("DATA_PROVIDER", "truedata"))
    truedata_user: str = field(default_factory=lambda: os.getenv("TRUEDATA_USER", ""))
    truedata_password: str = field(default_factory=lambda: os.getenv("TRUEDATA_PASSWORD", ""))
    cache_dir: str = field(default_factory=lambda: os.getenv("DATA_CACHE_DIR", "data/cache"))
    history_days: int = field(default_factory=lambda: int(os.getenv("HISTORY_DAYS", "365")))


@dataclass
class AppSettings:
    """Root application settings container."""
    broker: BrokerSettings = field(default_factory=BrokerSettings)
    trading: TradingSettings = field(default_factory=TradingSettings)
    ml: MLSettings = field(default_factory=MLSettings)
    infra: InfraSettings = field(default_factory=InfraSettings)
    data: DataSettings = field(default_factory=DataSettings)


def load_settings() -> AppSettings:
    """Load settings from .env file and environment variables.

    .env file values are loaded first, then overridden by any
    actual environment variables that are set.
    """
    # Load .env file into os.environ
    from pathlib import Path
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    # Only set if not already in environment (env vars take precedence)
                    if key not in os.environ or not os.environ[key]:
                        os.environ[key] = value
    return AppSettings()
