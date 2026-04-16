"""
AI Market Brain — Multi-provider real-time market analysis for trading decisions.

Supports multiple AI providers (cheapest to most expensive):
  1. Groq (Llama 3.3 70B)  — FREE tier, 394 tok/s, 1000 req/day
  2. Gemini Flash-Lite     — FREE tier, 1000 req/day
  3. DeepSeek Chat         — ~Rs 2-6/day with caching
  4. GPT-4o-mini           — ~Rs 6/day, best JSON schema compliance
  5. Claude Haiku           — ~Rs 35/day
  6. Claude Sonnet          — ~Rs 37/day (deep analysis only)

Auto-selects cheapest available provider. Uses condition-adaptive prompts
that change based on VIX level, time of day, position state, and market regime.

Integrates with V14 R5 agent to provide:
  - Real-time market regime analysis with condition-aware prompts
  - Intelligent trade recommendations
  - Dynamic confluence override for high-conviction setups
  - Risk assessment and position commentary
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ai_brain")

# State file for dashboard to read
BRAIN_STATE_FILE = Path(__file__).parent.parent / "data" / "claude_brain.json"

# ── AI Provider Configs ──
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "fast_model": "llama-3.1-8b-instant",
        "env_key": "GROQ_API_KEY",
        "cost_per_1m_input": 0.0,  # Free tier: 1000 req/day
        "max_tokens": 800,
        "type": "openai",
    },
    "gemini": {
        "base_url": None,
        "model": "gemini-2.5-flash-lite",
        "fast_model": "gemini-2.5-flash-lite",
        "env_key": "GEMINI_API_KEY",
        "cost_per_1m_input": 0.0,  # Free tier: 1000 req/day
        "max_tokens": 800,
        "type": "gemini",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "fast_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "cost_per_1m_input": 0.14,
        "max_tokens": 800,
        "type": "openai",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "fast_model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
        "cost_per_1m_input": 0.15,
        "max_tokens": 800,
        "type": "openai",
    },
    "haiku": {
        "base_url": None,
        "model": "claude-sonnet-4-20250514",
        "fast_model": "claude-sonnet-4-20250514",
        "env_key": "ANTHROPIC_API_KEY",
        "cost_per_1m_input": 3.0,
        "max_tokens": 600,
        "type": "anthropic",
    },
    "claude": {
        "base_url": None,
        "model": "claude-sonnet-4-20250514",
        "fast_model": "claude-sonnet-4-20250514",
        "env_key": "ANTHROPIC_API_KEY",
        "cost_per_1m_input": 3.0,
        "max_tokens": 800,
        "type": "anthropic",
    },
}

# Default priority: FREE tiers first, then cheapest paid
PROVIDER_PRIORITY = ["groq", "gemini", "deepseek", "openai", "haiku", "claude"]


class AIMarketBrain:
    """Multi-provider AI brain for live market analysis.

    Auto-selects the cheapest available provider. Falls back through the
    priority chain if a provider fails.

    Cost comparison for 75 bars/day (~150 calls):
      - DeepSeek:  ~$0.02/day  (~Rs 1.7/day)
      - Groq:      ~$0.01/day  (free tier)
      - Haiku:     ~$0.04/day  (~Rs 3.3/day)
      - GPT-4o-mini: ~$0.02/day
      - Sonnet:    ~$0.45/day  (~Rs 37/day)
    """

    def __init__(
        self,
        provider: str = "",
        api_key: str = "",
        analysis_interval: int = 300,  # Deep analysis every 5 minutes
        fast_interval: int = 0,        # Quick assessment disabled by default (0=off)
    ):
        self.analysis_interval = analysis_interval
        self.fast_interval = fast_interval
        self._last_analysis_time = 0.0
        self._last_fast_time = 0.0
        self._last_analysis: dict = {}
        self._analysis_history: list[dict] = []
        self._news_cache: list[dict] = []
        self._last_news_fetch = 0.0
        self._total_calls = 0
        self._total_cost_usd = 0.0
        self._enabled = False
        self._provider_name = ""
        self._provider_config: dict = {}
        self._client = None
        self._client_type = ""

        # Try explicit provider first, then auto-detect cheapest available
        if provider and provider in PROVIDERS:
            self._try_init_provider(provider, api_key)

        if not self._enabled:
            # Auto-detect: try cheapest available provider
            for pname in PROVIDER_PRIORITY:
                if self._try_init_provider(pname):
                    break

        if self._enabled:
            logger.info(
                "AI Market Brain initialized | provider=%s model=%s cost=$%.2f/M tokens",
                self._provider_name,
                self._provider_config.get("model", "?"),
                self._provider_config.get("cost_per_1m_input", 0),
            )
        else:
            logger.info(
                "AI Market Brain disabled — no API keys found. "
                "Set DEEPSEEK_API_KEY (cheapest), GROQ_API_KEY (free), "
                "or ANTHROPIC_API_KEY in .env"
            )

    def _try_init_provider(self, provider_name: str, api_key: str = "") -> bool:
        """Try to initialize a specific provider. Returns True on success."""
        config = PROVIDERS.get(provider_name)
        if not config:
            return False

        key = api_key or os.environ.get(config["env_key"], "")
        if not key:
            return False

        try:
            if config["type"] == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic(api_key=key)
                self._client_type = "anthropic"
            elif config["type"] == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=key)
                self._client = genai
                self._client_type = "gemini"
            else:
                # OpenAI-compatible (DeepSeek, Groq, OpenAI, etc.)
                from openai import OpenAI
                self._client = OpenAI(api_key=key, base_url=config.get("base_url"))
                self._client_type = "openai"

            self._enabled = True
            self._provider_name = provider_name
            self._provider_config = config
            return True
        except ImportError as e:
            logger.debug("Provider %s import failed: %s", provider_name, e)
        except Exception as e:
            logger.debug("Provider %s init failed: %s", provider_name, e)
        return False

    @property
    def enabled(self) -> bool:
        return self._enabled and self._client is not None

    @property
    def provider(self) -> str:
        return self._provider_name

    @property
    def cost_today(self) -> float:
        return self._total_cost_usd

    def should_analyze(self) -> bool:
        """Check if it's time for a new deep analysis."""
        return self.enabled and (time.time() - self._last_analysis_time >= self.analysis_interval)

    def analyze_market(
        self,
        spot_price: float,
        prev_close: float,
        bars: list[dict],
        vix: float,
        pcr: float,
        support: float,
        resistance: float,
        is_expiry_day: bool,
        open_positions: list[dict],
        closed_positions: list[dict],
        realized_pnl: float,
        unrealized_pnl: float,
        capital: float,
        indicators: dict | None = None,
        v14_last_signal: str = "",
        v14_confluence_status: str = "",
    ) -> dict:
        """Run full AI analysis on current market state."""
        if not self.enabled:
            return self._default_analysis()

        now = time.time()
        if now - self._last_analysis_time < self.analysis_interval:
            return self._last_analysis or self._default_analysis()

        self._last_analysis_time = now

        try:
            prompt = self._build_analysis_prompt(
                spot_price=spot_price, prev_close=prev_close, bars=bars,
                vix=vix, pcr=pcr, support=support, resistance=resistance,
                is_expiry_day=is_expiry_day, open_positions=open_positions,
                closed_positions=closed_positions, realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl, capital=capital,
                indicators=indicators, v14_last_signal=v14_last_signal,
                v14_confluence_status=v14_confluence_status,
            )

            result = self._call_llm(prompt, max_tokens=self._provider_config.get("max_tokens", 800))
            self._last_analysis = result
            self._analysis_history.append({
                "time": datetime.now().isoformat(),
                **result,
            })
            if len(self._analysis_history) > 50:
                self._analysis_history = self._analysis_history[-50:]

            self._write_state(result)
            logger.info(
                "AI analysis [%s]: %s | action=%s | conviction=%s | %s",
                self._provider_name,
                result.get("regime", "?"),
                result.get("recommended_action", "?"),
                result.get("conviction", "?"),
                result.get("one_liner", ""),
            )
            return result

        except Exception as e:
            logger.error("AI analysis failed [%s]: %s", self._provider_name, e)
            return self._last_analysis or self._default_analysis()

    def get_quick_assessment(self, spot: float, vix: float, rsi: float,
                             vwap: float, bias: str) -> dict:
        """Fast assessment for per-bar decisions (uses cheapest model)."""
        if not self.enabled:
            return {"override": False, "reason": "AI disabled"}

        prompt = (
            f"NIFTY spot={spot:.0f} VIX={vix:.1f} RSI={rsi:.1f} VWAP={vwap:.0f} "
            f"bias={bias}. Should the trading model override confluence filters for a "
            f"{'PUT' if bias in ('bearish', 'strong_bearish') else 'CALL'} entry? "
            f"Reply ONLY with JSON: {{\"override\": true/false, \"reason\": \"...\"}}"
        )
        try:
            result = self._call_llm(prompt, max_tokens=200, use_fast=True)
            return result
        except Exception as e:
            logger.debug("Quick assessment failed: %s", e)
        return {"override": False, "reason": "assessment_error"}

    def _call_llm(self, prompt: str, max_tokens: int = 800, use_fast: bool = False) -> dict:
        """Call the LLM provider and parse JSON response."""
        model = self._provider_config["fast_model" if use_fast else "model"]

        if self._client_type == "anthropic":
            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            input_tokens = response.usage.input_tokens
        elif self._client_type == "gemini":
            genai_model = self._client.GenerativeModel(model)
            response = genai_model.generate_content(
                prompt,
                generation_config={"max_output_tokens": max_tokens, "temperature": 0.3},
            )
            text = response.text.strip()
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) if hasattr(response, "usage_metadata") else 0
        else:
            # OpenAI-compatible (DeepSeek, Groq, etc.)
            response = self._client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content.strip()
            input_tokens = response.usage.prompt_tokens if response.usage else 0

        # Track cost
        self._total_calls += 1
        cost = (input_tokens / 1_000_000) * self._provider_config.get("cost_per_1m_input", 0)
        self._total_cost_usd += cost

        # Parse JSON
        if "{" in text:
            json_str = text[text.index("{"):text.rindex("}") + 1]
            result = json.loads(json_str)
            result["timestamp"] = datetime.now().isoformat()
            result["model"] = model
            result["provider"] = self._provider_name
            return result

        return self._default_analysis()

    def _build_analysis_prompt(self, **kwargs) -> str:
        """Build condition-adaptive market analysis prompt.

        The prompt changes based on:
        - VIX level (calm/elevated/crisis → different risk focus)
        - Time of day (opening/midday/closing → different strategy focus)
        - Position state (flat/holding/losing → different advice focus)
        - Market regime (trending/ranging → different indicator weight)
        """
        bars = kwargs.get("bars", [])
        spot = kwargs.get("spot_price", 0)
        prev_close = kwargs.get("prev_close", 0)
        vix = kwargs.get("vix", 0)
        capital = kwargs.get("capital", 0)
        realized_pnl = kwargs.get("realized_pnl", 0)
        unrealized_pnl = kwargs.get("unrealized_pnl", 0)
        day_change = spot - prev_close if prev_close > 0 else 0
        day_change_pct = (day_change / prev_close * 100) if prev_close > 0 else 0
        indicators = kwargs.get("indicators") or {}
        open_pos = kwargs.get("open_positions", [])
        closed_pos = kwargs.get("closed_positions", [])

        # ── Condition-adaptive context ──
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        time_mins = hour * 60 + minute

        # Time context
        if time_mins < 570:  # Before 9:30
            time_context = "PRE-MARKET: Focus on gap analysis and opening range prediction."
        elif time_mins < 600:  # 9:15-10:00
            time_context = "OPENING HOUR: High volatility expected. Look for ORB setups. Avoid false breakouts."
        elif time_mins < 720:  # 10:00-12:00
            time_context = "MORNING SESSION: Trend establishing. Focus on trend-following signals."
        elif time_mins < 810:  # 12:00-13:30
            time_context = "LUNCH LULL: Low volume, choppy action. Avoid new entries unless strong conviction."
        elif time_mins < 870:  # 13:30-14:30
            time_context = "AFTERNOON SESSION: Institutional activity picks up. Watch for reversals."
        elif time_mins < 930:  # 14:30-15:30
            time_context = "CLOSING HOUR: Theta decay accelerates. Exit weak positions. Last entry window."
        else:
            time_context = "POST-MARKET: Analyze the day's action for tomorrow's preparation."

        # VIX context
        if vix < 13:
            vix_context = "LOW VIX (<13): Calm market. Premiums are cheap — good for buying options. Breakouts likely."
        elif vix < 16:
            vix_context = "MODERATE VIX (13-16): Sweet spot for option buying. Good risk-reward on directional bets."
        elif vix < 20:
            vix_context = "ELEVATED VIX (16-20): Increased uncertainty. Size down positions. Wider stops needed."
        elif vix < 25:
            vix_context = "HIGH VIX (20-25): Crisis-level fear. Big moves possible. Only high-conviction trades."
        else:
            vix_context = f"EXTREME VIX ({vix:.0f}): Panic mode. Protect capital above all. Consider staying flat."

        # Position context
        total_pnl = realized_pnl + unrealized_pnl
        daily_pnl_pct = (total_pnl / capital * 100) if capital > 0 else 0
        if len(open_pos) == 0:
            pos_context = "FLAT: No open positions. Focus on identifying the next high-probability entry."
        elif daily_pnl_pct < -2:
            pos_context = f"LOSING DAY ({daily_pnl_pct:.1f}%): Prioritize capital preservation. Reduce size or stop trading."
        elif daily_pnl_pct > 2:
            pos_context = f"WINNING DAY ({daily_pnl_pct:.1f}%): Protect gains. Trail stops tighter. Be selective on new entries."
        else:
            pos_context = f"HOLDING ({len(open_pos)} positions, P&L: {daily_pnl_pct:.1f}%): Monitor and manage. Adjust stops if needed."

        # Capital context
        if capital < 15000:
            capital_context = f"SMALL ACCOUNT (Rs {capital:,.0f}): Trade only 1 lot. Strict risk management. No averaging."
        elif capital < 50000:
            capital_context = f"MODERATE ACCOUNT (Rs {capital:,.0f}): Trade 1-3 lots max. Focus on quality over quantity."
        else:
            capital_context = f"ACCOUNT (Rs {capital:,.0f}): Standard position sizing applies."

        # Recent bars summary
        recent_bars = bars[-30:] if bars else []
        bar_summary = ""
        if recent_bars:
            highs = [b.get("high", 0) for b in recent_bars]
            lows = [b.get("low", 0) for b in recent_bars]
            closes = [b.get("close", 0) for b in recent_bars]
            day_high = max(highs) if highs else 0
            day_low = min(lows) if lows else 0
            bar_summary = (
                f"Day High={day_high:.0f} Day Low={day_low:.0f} "
                f"Range={day_high - day_low:.0f}pts "
                f"Last 5 closes: {[f'{c:.0f}' for c in closes[-5:]]}"
            )

        # Positions detail
        pos_summary = "No open positions."
        if open_pos:
            pos_lines = [
                f"  {p.get('symbol','')} {p.get('side','')} x{p.get('qty',0)} "
                f"entry={p.get('entry_price',0):.2f} ltp={p.get('current_price',0):.2f} "
                f"pnl={p.get('pnl',0):.2f}"
                for p in open_pos
            ]
            pos_summary = "Open:\n" + "\n".join(pos_lines)

        # R5 indicators
        r5_block = ""
        if indicators:
            r5_block = (
                f"Connors RSI={indicators.get('connors_rsi', 'N/A')} "
                f"KAMA={'Up' if indicators.get('kama_slope_up') else 'Down'} "
                f"PSAR={'Bull' if indicators.get('psar_bullish') else 'Bear'} "
                f"Donchian={'BrkUp' if indicators.get('donchian_breakout_up') else 'BrkDn' if indicators.get('donchian_breakout_down') else 'Range'} "
                f"HA=G{indicators.get('ha_green_streak', 0)}/R{indicators.get('ha_red_streak', 0)} "
                f"ATR={indicators.get('atr', 0):.1f}"
            )

        return f"""You are a NIFTY options trading analyst. Analyze LIVE data and give actionable intelligence.

CONTEXT:
{time_context}
{vix_context}
{pos_context}
{capital_context}

MARKET: NIFTY {spot:.0f} ({day_change:+.0f} / {day_change_pct:+.1f}%) VIX={vix:.1f} PCR={kwargs.get('pcr', 0):.2f}
S/R: {kwargs.get('support', 0):.0f}/{kwargs.get('resistance', 0):.0f} Expiry={'YES' if kwargs.get('is_expiry_day') else 'No'}
BARS: {bar_summary}

INDICATORS: RSI={indicators.get('rsi', 'N/A')} EMA={'9>21' if indicators.get('ema9_above_ema21') else '9<21'} VWAP={indicators.get('vwap', 'N/A')}
ST={'Bull' if indicators.get('st_direction', 0) > 0 else 'Bear'} MACD_H={indicators.get('macd_hist', 'N/A')} Squeeze={'ON' if indicators.get('squeeze_on') else 'OFF'}
{r5_block}

POSITIONS: {pos_summary}
P&L: Realized={realized_pnl:.0f} Unrealized={unrealized_pnl:.0f} Capital={capital:.0f}
V14 R5: Signal={kwargs.get('v14_last_signal', 'none')} Confluence={kwargs.get('v14_confluence_status', 'unknown')}

Respond ONLY with JSON (no markdown):
{{"regime":"trending_up|trending_down|range_bound|volatile|choppy","regime_strength":"strong|moderate|weak","recommended_action":"BUY_CALL|BUY_PUT|HOLD|EXIT_ALL|REDUCE_SIZE","conviction":"high|medium|low","key_levels":{{"immediate_support":0,"immediate_resistance":0,"breakout_above":0,"breakdown_below":0}},"risk_assessment":"low|moderate|high|extreme","override_confluence":false,"override_reason":"","position_advice":"","sentiment":"bullish|bearish|neutral","one_liner":"1-line summary","detailed_analysis":"2-3 sentences"}}"""

    def _default_analysis(self) -> dict:
        return {
            "regime": "unknown",
            "regime_strength": "unknown",
            "recommended_action": "HOLD",
            "conviction": "low",
            "key_levels": {},
            "risk_assessment": "moderate",
            "override_confluence": False,
            "override_reason": "",
            "position_advice": "",
            "sentiment": "neutral",
            "one_liner": "AI analysis not available",
            "detailed_analysis": "",
            "timestamp": datetime.now().isoformat(),
            "model": "none",
            "provider": "none",
        }

    def _write_state(self, analysis: dict) -> None:
        """Write analysis to JSON for dashboard consumption."""
        try:
            state = {
                "last_updated": datetime.now().isoformat(),
                "enabled": self.enabled,
                "provider": self._provider_name,
                "model": self._provider_config.get("model", ""),
                "total_calls": self._total_calls,
                "cost_usd": round(self._total_cost_usd, 4),
                "analysis": analysis,
                "history": self._analysis_history[-10:],
            }
            BRAIN_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = BRAIN_STATE_FILE.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, default=str)
            tmp.replace(BRAIN_STATE_FILE)
        except Exception as e:
            logger.debug("Failed to write brain state: %s", e)

    def get_latest_analysis(self) -> dict:
        """Get the most recent analysis (from memory or file)."""
        if self._last_analysis:
            return self._last_analysis
        try:
            if BRAIN_STATE_FILE.exists():
                with open(BRAIN_STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("analysis", self._default_analysis())
        except Exception:
            pass
        return self._default_analysis()


# ── Backward-compatible alias ──
# Old code uses ClaudeMarketBrain — redirect to new multi-provider brain
ClaudeMarketBrain = AIMarketBrain
