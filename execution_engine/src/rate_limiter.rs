use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;
use tracing::{debug, warn, instrument};

/// SEBI mandates a maximum of 10 orders per second per trading member.
/// We cap at 9 to maintain a safety margin.
const MAX_TOKENS: u32 = 9;
const REFILL_INTERVAL: Duration = Duration::from_secs(1);

/// Errors returned by the rate limiter.
#[derive(Debug, Clone)]
pub enum RateLimitError {
    /// The order would exceed the SEBI rate limit.
    ExceedsLimit {
        available: u32,
        requested: u32,
        retry_after_ms: u64,
    },
}

impl std::fmt::Display for RateLimitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RateLimitError::ExceedsLimit {
                available,
                requested,
                retry_after_ms,
            } => write!(
                f,
                "Rate limit exceeded: {requested} tokens requested, {available} available. \
                 Retry after {retry_after_ms}ms"
            ),
        }
    }
}

impl std::error::Error for RateLimitError {}

/// Internal state of the token bucket.
struct TokenBucketInner {
    tokens: u32,
    last_refill: Instant,
    /// Counters for observability.
    total_allowed: u64,
    total_rejected: u64,
}

/// Thread-safe token-bucket rate limiter.
///
/// - Capacity: 9 tokens (orders) per second
/// - Rolling 1-second refill window
/// - Safe for concurrent use via `Arc<Mutex<...>>`
#[derive(Clone)]
pub struct RateLimiter {
    inner: Arc<Mutex<TokenBucketInner>>,
}

impl RateLimiter {
    /// Create a new rate limiter with full token bucket.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(TokenBucketInner {
                tokens: MAX_TOKENS,
                last_refill: Instant::now(),
                total_allowed: 0,
                total_rejected: 0,
            })),
        }
    }

    /// Refill tokens based on elapsed time since last refill.
    fn refill(inner: &mut TokenBucketInner) {
        let now = Instant::now();
        let elapsed = now.duration_since(inner.last_refill);

        if elapsed >= REFILL_INTERVAL {
            // Full refill for each complete second elapsed
            let refills = (elapsed.as_millis() / REFILL_INTERVAL.as_millis()) as u32;
            inner.tokens = MAX_TOKENS.min(inner.tokens + refills * MAX_TOKENS);
            inner.last_refill = now;

            debug!(
                tokens = inner.tokens,
                refills = refills,
                "Token bucket refilled"
            );
        }
    }

    /// Try to acquire `count` tokens for sending orders.
    ///
    /// Returns `Ok(remaining)` with the remaining token count on success,
    /// or `Err(RateLimitError)` if there are insufficient tokens.
    #[instrument(skip(self), fields(requested = count))]
    pub async fn try_acquire(&self, count: u32) -> Result<u32, RateLimitError> {
        let mut inner = self.inner.lock().await;
        Self::refill(&mut inner);

        if inner.tokens >= count {
            inner.tokens -= count;
            inner.total_allowed += 1;

            debug!(
                remaining = inner.tokens,
                total_allowed = inner.total_allowed,
                "Rate limiter: tokens acquired"
            );
            Ok(inner.tokens)
        } else {
            inner.total_rejected += 1;
            let elapsed = Instant::now().duration_since(inner.last_refill);
            let retry_after_ms = REFILL_INTERVAL
                .saturating_sub(elapsed)
                .as_millis() as u64;

            warn!(
                available = inner.tokens,
                requested = count,
                retry_after_ms = retry_after_ms,
                total_rejected = inner.total_rejected,
                "Rate limiter: request rejected — would exceed SEBI 10 OPS threshold"
            );

            Err(RateLimitError::ExceedsLimit {
                available: inner.tokens,
                requested: count,
                retry_after_ms,
            })
        }
    }

    /// Return current available tokens (for monitoring / risk dashboards).
    pub async fn available_tokens(&self) -> u32 {
        let mut inner = self.inner.lock().await;
        Self::refill(&mut inner);
        inner.tokens
    }

    /// Return lifetime stats: (total_allowed, total_rejected).
    pub async fn stats(&self) -> (u64, u64) {
        let inner = self.inner.lock().await;
        (inner.total_allowed, inner.total_rejected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_acquire() {
        let limiter = RateLimiter::new();
        // Should succeed for first 9 tokens
        for i in 0..9 {
            let result = limiter.try_acquire(1).await;
            assert!(result.is_ok(), "Token {i} should succeed");
        }
        // 10th should fail
        let result = limiter.try_acquire(1).await;
        assert!(result.is_err(), "10th token should fail");
    }

    #[tokio::test]
    async fn test_bulk_acquire() {
        let limiter = RateLimiter::new();
        let result = limiter.try_acquire(5).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 4); // 9 - 5 = 4 remaining

        let result = limiter.try_acquire(5).await;
        assert!(result.is_err()); // only 4 left, need 5
    }

    #[tokio::test]
    async fn test_refill_after_window() {
        let limiter = RateLimiter::new();
        // Exhaust all tokens
        let _ = limiter.try_acquire(9).await;
        assert!(limiter.try_acquire(1).await.is_err());

        // Wait for refill
        tokio::time::sleep(Duration::from_secs(1)).await;

        // Should work again
        assert!(limiter.try_acquire(1).await.is_ok());
    }
}
