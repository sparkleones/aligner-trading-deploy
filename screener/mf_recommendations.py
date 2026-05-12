"""
Mutual Fund Recommendations.

Where our system should DEFER to professionally managed MFs rather than
trying to replicate their fundamental research with pure price signals.

Categories:
  LARGE_CAP  - we DO beat NIFTY here, so external MFs are optional
  MID_CAP    - mixed; user choice
  SMALL_CAP  - we LOSE money; STRONGLY prefer MF
  INDEX      - if user wants passive exposure
  DEBT       - never our domain; always external
  GOLD       - never our domain; always external

Returns are 5-year CAGR as of 2025 (Value Research / AMFI). All are
DIRECT GROWTH plans where applicable.
"""
from __future__ import annotations

MF_RECOMMENDATIONS = {
    "SMALL_CAP": [
        {"name": "Nippon India Small Cap Fund - Direct Growth",
         "amc": "Nippon India MF",
         "isin": "INF204KB14I7",
         "cagr_1y": 0.18, "cagr_3y": 0.30, "cagr_5y": 0.31, "cagr_10y": 0.24,
         "expense_ratio": 0.0073,
         "min_sip": 100,
         "url": "https://mf.nipponindiaim.com/InvestorServices/FactSheets/Pages/Default.aspx",
         "rationale": "Best 5y CAGR in category. Strong bottom-up research team.",
         "system_recommendation": "STRONGLY PREFER over our screener for SC exposure",
        },
        {"name": "HDFC Small Cap Fund - Direct Growth",
         "amc": "HDFC MF",
         "isin": "INF179KB1FT4",
         "cagr_1y": 0.16, "cagr_3y": 0.26, "cagr_5y": 0.28, "cagr_10y": 0.22,
         "expense_ratio": 0.0073,
         "min_sip": 100,
         "url": "https://www.hdfcfund.com/our-products/equity/HDFC-Small-Cap-Fund",
         "rationale": "Quality + growth tilt; lower portfolio turnover (~30%/yr).",
         "system_recommendation": "STRONGLY PREFER over our screener for SC exposure",
        },
        {"name": "Kotak Small Cap Fund - Direct Growth",
         "amc": "Kotak MF",
         "isin": "INF174K01EW3",
         "cagr_1y": 0.14, "cagr_3y": 0.23, "cagr_5y": 0.26, "cagr_10y": 0.20,
         "expense_ratio": 0.0065,
         "min_sip": 100,
         "url": "https://www.kotakmf.com/Information/equity-funds/kotak-small-cap-fund",
         "rationale": "GARP (Growth At Reasonable Price). EPS growth >20% + PE <25.",
         "system_recommendation": "STRONGLY PREFER over our screener for SC exposure",
        },
        {"name": "Nippon India Nifty Smallcap 250 Index Fund - Direct",
         "amc": "Nippon India MF",
         "isin": "INF204KB1B30",
         "cagr_1y": 0.10, "cagr_3y": 0.18, "cagr_5y": 0.20, "cagr_10y": 0.16,
         "expense_ratio": 0.0033,
         "min_sip": 100,
         "url": "https://mf.nipponindiaim.com/our-products/equity/index/nippon-india-nifty-smallcap-250-index-fund",
         "rationale": "Passive Nifty SC 250 index tracker. Lowest expense ratio.",
         "system_recommendation": "PREFER over our screener if you want predictable index exposure",
        },
    ],
    "MID_CAP": [
        {"name": "Motilal Oswal Midcap Fund - Direct Growth",
         "amc": "Motilal Oswal MF",
         "isin": "INF247L01023",
         "cagr_1y": 0.20, "cagr_3y": 0.32, "cagr_5y": 0.28, "cagr_10y": 0.18,
         "expense_ratio": 0.0065,
         "min_sip": 500,
         "url": "https://www.motilaloswalmf.com/our-funds/equity-funds/motilal-oswal-midcap-fund",
         "rationale": "Focused (25 stock) high-conviction midcap. Top performer last 3y.",
         "system_recommendation": "PREFER for MID-cap allocation",
        },
        {"name": "Edelweiss Mid Cap Fund - Direct Growth",
         "amc": "Edelweiss MF",
         "isin": "INF754K01HB6",
         "cagr_1y": 0.18, "cagr_3y": 0.26, "cagr_5y": 0.25, "cagr_10y": 0.19,
         "expense_ratio": 0.0048,
         "min_sip": 100,
         "url": "https://www.edelweissmf.com/equity-funds/edelweiss-mid-cap-fund",
         "rationale": "Quality tilt; lower turnover; consistent performer.",
         "system_recommendation": "PREFER for MID-cap allocation",
        },
    ],
    "LARGE_CAP": [
        {"name": "Mirae Asset Large Cap Fund - Direct Growth",
         "amc": "Mirae Asset MF",
         "isin": "INF769K01CT0",
         "cagr_1y": 0.12, "cagr_3y": 0.15, "cagr_5y": 0.14, "cagr_10y": 0.13,
         "expense_ratio": 0.0055,
         "min_sip": 100,
         "url": "https://www.miraeassetmf.co.in/large-cap-fund",
         "rationale": "Top quartile LC consistency. Quality + growth tilt.",
         "system_recommendation": "OPTIONAL - our screener also works here",
        },
        {"name": "Nippon India Index Fund - Nifty 50 Plan - Direct Growth",
         "amc": "Nippon India MF",
         "isin": "INF204K01XC7",
         "cagr_1y": 0.11, "cagr_3y": 0.13, "cagr_5y": 0.13, "cagr_10y": 0.12,
         "expense_ratio": 0.0020,
         "min_sip": 100,
         "url": "https://mf.nipponindiaim.com/our-products/equity/index/nippon-india-index-fund-nifty-50-plan",
         "rationale": "Cheapest NIFTY 50 index tracker. 20bps total expense.",
         "system_recommendation": "PREFER over our screener if you don't want active management",
        },
    ],
    "FLEXICAP": [
        {"name": "Parag Parikh Flexi Cap Fund - Direct Growth",
         "amc": "PPFAS MF",
         "isin": "INF879O01027",
         "cagr_1y": 0.16, "cagr_3y": 0.22, "cagr_5y": 0.22, "cagr_10y": 0.20,
         "expense_ratio": 0.0061,
         "min_sip": 1000,
         "url": "https://www.ppfas.com/funds/parag-parikh-flexi-cap-fund/",
         "rationale": "Quality + value + 30% international (US tech). Most consistent flexi-cap.",
         "system_recommendation": "PREFER for core diversified allocation",
        },
    ],
}


def get_recommendations(category: str = "ALL") -> dict:
    """Return MF recommendations for a category or all."""
    if category == "ALL":
        return MF_RECOMMENDATIONS
    return {category: MF_RECOMMENDATIONS.get(category, [])}
