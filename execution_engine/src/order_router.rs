use std::collections::HashMap;
use std::time::{Duration, Instant};

use rand::Rng;
use tracing::{debug, info, warn, instrument};
use uuid::Uuid;

use crate::rate_limiter::{RateLimiter, RateLimitError};
use crate::trading::{self, OrderRequest, OrderResponse, OrderSide, OrderType};

// ─── NSE Freeze Limits ────────────────────────────────────────────────────────
// Maximum single-order quantity before automatic slicing is required.

lazy_static::lazy_static! {
    static ref FREEZE_LIMITS: HashMap<&'static str, i32> = {
        let mut m = HashMap::new();
        m.insert("NIFTY", 1800);
        m.insert("BANKNIFTY", 600);
        m.insert("FINNIFTY", 1200);
        m.insert("MIDCPNIFTY", 2800);
        m.insert("NIFTYNXT50", 600);
        m
    };
}

/// Randomized micro-delay range between sliced order tranches.
const SLICE_DELAY_MIN_MS: u64 = 50;
const SLICE_DELAY_MAX_MS: u64 = 150;

/// Delta thresholds for moneyness-based order type selection.
const ATM_DELTA_THRESHOLD: f64 = 0.40;
const DEEP_OTM_DELTA_THRESHOLD: f64 = 0.15;

/// Errors from the order router.
#[derive(Debug)]
pub enum OrderRouterError {
    RateLimitExceeded(RateLimitError),
    ValidationFailed(String),
    BrokerError(String),
}

impl std::fmt::Display for OrderRouterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderRouterError::RateLimitExceeded(e) => write!(f, "Rate limit: {e}"),
            OrderRouterError::ValidationFailed(msg) => write!(f, "Validation: {msg}"),
            OrderRouterError::BrokerError(msg) => write!(f, "Broker: {msg}"),
        }
    }
}

impl std::error::Error for OrderRouterError {}

/// A single tranche of a sliced order.
#[derive(Debug, Clone)]
pub struct OrderTranche {
    pub tranche_id: String,
    pub parent_request_id: String,
    pub quantity: i32,
    pub tranche_index: usize,
    pub total_tranches: usize,
}

/// The order router handles validation, slicing, rate limiting, and routing.
pub struct OrderRouter {
    rate_limiter: RateLimiter,
}

impl OrderRouter {
    pub fn new(rate_limiter: RateLimiter) -> Self {
        Self { rate_limiter }
    }

    /// Validate an incoming order request.
    #[instrument(skip(self), fields(
        request_id = %order.request_id,
        symbol = %order.symbol,
        quantity = order.quantity,
    ))]
    pub fn validate_order(&self, order: &OrderRequest) -> Result<(), OrderRouterError> {
        let start = Instant::now();

        if order.symbol.is_empty() {
            return Err(OrderRouterError::ValidationFailed(
                "Symbol is required".to_string(),
            ));
        }

        if order.quantity <= 0 {
            return Err(OrderRouterError::ValidationFailed(
                "Quantity must be positive".to_string(),
            ));
        }

        if order.order_type() == OrderType::Limit && order.price <= 0.0 {
            return Err(OrderRouterError::ValidationFailed(
                "Limit orders require a positive price".to_string(),
            ));
        }

        if (order.order_type() == OrderType::Sl || order.order_type() == OrderType::SlM)
            && order.trigger_price <= 0.0
        {
            return Err(OrderRouterError::ValidationFailed(
                "SL orders require a positive trigger price".to_string(),
            ));
        }

        let latency = start.elapsed();
        debug!(
            latency_us = latency.as_micros() as u64,
            "Order validation passed"
        );
        Ok(())
    }

    /// Get the freeze limit for a given underlying symbol.
    pub fn freeze_limit_for(symbol: &str) -> Option<i32> {
        // Extract underlying from symbol like "NIFTY24MAR22000CE"
        let underlying = extract_underlying(symbol);
        FREEZE_LIMITS.get(underlying.as_str()).copied()
    }

    /// Slice an order into tranches that respect NSE freeze limits.
    #[instrument(skip(self), fields(
        request_id = %order.request_id,
        symbol = %order.symbol,
        quantity = order.quantity,
    ))]
    pub fn slice_order(&self, order: &OrderRequest) -> Vec<OrderTranche> {
        let freeze_limit = Self::freeze_limit_for(&order.symbol).unwrap_or(i32::MAX);
        let quantity = order.quantity;

        if quantity <= freeze_limit {
            debug!(
                quantity = quantity,
                freeze_limit = freeze_limit,
                "Order within freeze limit, no slicing needed"
            );
            return vec![OrderTranche {
                tranche_id: Uuid::new_v4().to_string(),
                parent_request_id: order.request_id.clone(),
                quantity,
                tranche_index: 0,
                total_tranches: 1,
            }];
        }

        let mut tranches = Vec::new();
        let mut remaining = quantity;
        let mut index = 0;

        while remaining > 0 {
            let tranche_qty = remaining.min(freeze_limit);
            remaining -= tranche_qty;
            tranches.push(OrderTranche {
                tranche_id: Uuid::new_v4().to_string(),
                parent_request_id: order.request_id.clone(),
                quantity: tranche_qty,
                tranche_index: index,
                total_tranches: 0, // filled below
            });
            index += 1;
        }

        let total = tranches.len();
        for t in &mut tranches {
            t.total_tranches = total;
        }

        info!(
            original_qty = quantity,
            freeze_limit = freeze_limit,
            tranches = total,
            "Order sliced into tranches"
        );

        tranches
    }

    /// Select order type based on option moneyness (delta).
    ///
    /// - ATM (delta >= 0.40): use Market order for fast fill
    /// - Deep OTM (delta <= 0.15): use Limit order to avoid slippage
    /// - Otherwise: use the order type specified in the request
    pub fn select_order_type_by_delta(
        requested_type: OrderType,
        delta: Option<f64>,
    ) -> OrderType {
        match delta {
            Some(d) if d.abs() >= ATM_DELTA_THRESHOLD => {
                debug!(
                    delta = d,
                    selected = "MARKET",
                    "ATM option — using Market order for fast fill"
                );
                OrderType::Market
            }
            Some(d) if d.abs() <= DEEP_OTM_DELTA_THRESHOLD => {
                debug!(
                    delta = d,
                    selected = "LIMIT",
                    "Deep OTM option — using Limit order to control slippage"
                );
                OrderType::Limit
            }
            Some(d) => {
                debug!(
                    delta = d,
                    selected = ?requested_type,
                    "Using requested order type"
                );
                requested_type
            }
            None => {
                debug!("No delta provided, using requested order type");
                requested_type
            }
        }
    }

    /// Submit an order: validate -> rate-limit -> slice -> route each tranche.
    #[instrument(skip(self), fields(
        request_id = %order.request_id,
        symbol = %order.symbol,
        side = ?OrderSide::try_from(order.side),
        quantity = order.quantity,
    ))]
    pub async fn submit_order(
        &self,
        order: &OrderRequest,
        delta: Option<f64>,
    ) -> Result<OrderResponse, OrderRouterError> {
        let submit_start = Instant::now();

        // 1. Validate
        self.validate_order(order)?;

        // 2. Slice
        let tranches = self.slice_order(order);
        let tranche_count = tranches.len() as u32;

        // 3. Rate-limit: acquire tokens for all tranches
        self.rate_limiter
            .try_acquire(tranche_count)
            .await
            .map_err(OrderRouterError::RateLimitExceeded)?;

        // 4. Determine effective order type based on delta / moneyness
        let effective_order_type =
            Self::select_order_type_by_delta(order.order_type(), delta);

        // 5. Route each tranche
        let mut rng = rand::thread_rng();
        let mut last_order_id = String::new();

        for (i, tranche) in tranches.iter().enumerate() {
            let tranche_start = Instant::now();

            info!(
                tranche_id = %tranche.tranche_id,
                tranche_index = tranche.tranche_index,
                total_tranches = tranche.total_tranches,
                quantity = tranche.quantity,
                order_type = ?effective_order_type,
                "Routing order tranche"
            );

            // Simulate sending to broker (replace with actual broker API call)
            last_order_id = format!("ORD-{}", Uuid::new_v4());

            let tranche_latency = tranche_start.elapsed();
            info!(
                tranche_id = %tranche.tranche_id,
                broker_order_id = %last_order_id,
                latency_us = tranche_latency.as_micros() as u64,
                "Tranche routed successfully"
            );

            // Randomized micro-delay between slices (not after the last one)
            if i < tranches.len() - 1 {
                let delay_ms = rng.gen_range(SLICE_DELAY_MIN_MS..=SLICE_DELAY_MAX_MS);
                debug!(delay_ms = delay_ms, "Inter-tranche delay");
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }
        }

        let total_latency = submit_start.elapsed();
        let now_ms = chrono::Utc::now().timestamp_millis();

        info!(
            request_id = %order.request_id,
            order_id = %last_order_id,
            total_tranches = tranches.len(),
            total_latency_ms = total_latency.as_millis() as f64,
            "Order submission complete"
        );

        Ok(OrderResponse {
            request_id: order.request_id.clone(),
            order_id: last_order_id,
            success: true,
            message: format!(
                "Order placed in {} tranche(s)",
                tranches.len()
            ),
            timestamp_ms: now_ms,
            latency_ms: total_latency.as_millis() as f64,
        })
    }

    /// Cancel an order by its broker order ID.
    #[instrument(skip(self), fields(order_id = %order_id))]
    pub async fn cancel_order(&self, order_id: &str) -> Result<OrderResponse, OrderRouterError> {
        let start = Instant::now();
        info!("Cancelling order");

        // Acquire 1 token for the cancel request
        self.rate_limiter
            .try_acquire(1)
            .await
            .map_err(OrderRouterError::RateLimitExceeded)?;

        // Simulate broker cancel API call
        let latency = start.elapsed();
        let now_ms = chrono::Utc::now().timestamp_millis();

        info!(
            order_id = %order_id,
            latency_ms = latency.as_millis() as f64,
            "Order cancellation sent"
        );

        Ok(OrderResponse {
            request_id: String::new(),
            order_id: order_id.to_string(),
            success: true,
            message: "Cancellation request sent".to_string(),
            timestamp_ms: now_ms,
            latency_ms: latency.as_millis() as f64,
        })
    }

    /// Emergency square-off: cancel all pending orders and close all positions.
    #[instrument(skip(self))]
    pub async fn square_off_all(
        &self,
        reason: &str,
    ) -> Result<trading::SquareOffResponse, OrderRouterError> {
        let start = Instant::now();
        warn!(reason = %reason, "EMERGENCY SQUARE-OFF initiated");

        // In production, this would:
        // 1. Fetch all pending orders from broker and cancel them
        // 2. Fetch all open positions and place opposing orders
        // For now, we log and return a placeholder response.

        let latency = start.elapsed();
        info!(
            latency_ms = latency.as_millis() as f64,
            "Square-off execution complete"
        );

        Ok(trading::SquareOffResponse {
            success: true,
            orders_cancelled: 0,
            positions_closed: 0,
            message: format!("Square-off initiated: {reason}"),
        })
    }
}

/// Extract the underlying index name from an options symbol.
///
/// Examples:
///   "NIFTY24MAR22000CE" -> "NIFTY"
///   "BANKNIFTY24MAR48000PE" -> "BANKNIFTY"
///   "FINNIFTY24MAR22500CE" -> "FINNIFTY"
///   "MIDCPNIFTY24MAR10000CE" -> "MIDCPNIFTY"
///   "NIFTYNXT5024MAR22000CE" -> "NIFTYNXT50"
fn extract_underlying(symbol: &str) -> String {
    // Known underlying names in order of longest first to avoid partial matches
    const UNDERLYINGS: &[&str] = &[
        "MIDCPNIFTY",
        "NIFTYNXT50",
        "BANKNIFTY",
        "FINNIFTY",
        "NIFTY",
    ];

    let upper = symbol.to_uppercase();
    for u in UNDERLYINGS {
        if upper.starts_with(u) {
            return u.to_string();
        }
    }

    // Fallback: take alphabetic prefix
    upper
        .chars()
        .take_while(|c| c.is_alphabetic())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_underlying() {
        assert_eq!(extract_underlying("NIFTY24MAR22000CE"), "NIFTY");
        assert_eq!(extract_underlying("BANKNIFTY24MAR48000PE"), "BANKNIFTY");
        assert_eq!(extract_underlying("FINNIFTY24MAR22500CE"), "FINNIFTY");
        assert_eq!(extract_underlying("MIDCPNIFTY24MAR10000CE"), "MIDCPNIFTY");
        assert_eq!(extract_underlying("NIFTYNXT5024MAR22000CE"), "NIFTYNXT50");
    }

    #[test]
    fn test_freeze_limits() {
        assert_eq!(OrderRouter::freeze_limit_for("NIFTY24MAR22000CE"), Some(1800));
        assert_eq!(OrderRouter::freeze_limit_for("BANKNIFTY24MAR48000PE"), Some(600));
        assert_eq!(OrderRouter::freeze_limit_for("FINNIFTY24MAR22500CE"), Some(1200));
        assert_eq!(OrderRouter::freeze_limit_for("MIDCPNIFTY24MAR10000CE"), Some(2800));
        assert_eq!(OrderRouter::freeze_limit_for("NIFTYNXT5024MAR22000CE"), Some(600));
        assert_eq!(OrderRouter::freeze_limit_for("UNKNOWN"), None);
    }

    #[test]
    fn test_order_slicing() {
        let limiter = RateLimiter::new();
        let router = OrderRouter::new(limiter);

        let order = OrderRequest {
            request_id: "test-1".into(),
            symbol: "NIFTY24MAR22000CE".into(),
            exchange: "NFO".into(),
            side: 0,
            order_type: 0,
            quantity: 5000, // > 1800 freeze limit
            price: 0.0,
            trigger_price: 0.0,
            product: 0,
            tag: String::new(),
            strike_price: 22000.0,
            option_type: "CE".into(),
            expiry: "2024-03-28".into(),
        };

        let tranches = router.slice_order(&order);
        // 5000 / 1800 = 2 full (1800 each) + 1 partial (1400)
        assert_eq!(tranches.len(), 3);
        assert_eq!(tranches[0].quantity, 1800);
        assert_eq!(tranches[1].quantity, 1800);
        assert_eq!(tranches[2].quantity, 1400);

        let total_qty: i32 = tranches.iter().map(|t| t.quantity).sum();
        assert_eq!(total_qty, 5000);
    }

    #[test]
    fn test_moneyness_order_type() {
        // ATM
        assert_eq!(
            OrderRouter::select_order_type_by_delta(OrderType::Limit, Some(0.50)),
            OrderType::Market
        );
        // Deep OTM
        assert_eq!(
            OrderRouter::select_order_type_by_delta(OrderType::Market, Some(0.10)),
            OrderType::Limit
        );
        // In between — use requested
        assert_eq!(
            OrderRouter::select_order_type_by_delta(OrderType::Sl, Some(0.25)),
            OrderType::Sl
        );
    }
}
