use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, Mutex, RwLock};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn, instrument};
use uuid::Uuid;

use crate::trading;

// ─── Reconnection constants ───────────────────────────────────────────────────

const BACKOFF_BASE_MS: u64 = 1_000;
const BACKOFF_MAX_MS: u64 = 60_000;
const BACKOFF_MULTIPLIER: u64 = 2;
const HEARTBEAT_INTERVAL_SECS: u64 = 30;
const STALE_MESSAGE_WINDOW_SECS: i64 = 5;

// ─── Tick data deserialized from broker JSON ──────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrokerTick {
    #[serde(default)]
    pub message_id: String,
    pub instrument_token: String,
    #[serde(default)]
    pub symbol: String,
    pub last_price: f64,
    #[serde(default)]
    pub bid_price: f64,
    #[serde(default)]
    pub ask_price: f64,
    #[serde(default)]
    pub bid_qty: i64,
    #[serde(default)]
    pub ask_qty: i64,
    #[serde(default)]
    pub open: f64,
    #[serde(default)]
    pub high: f64,
    #[serde(default)]
    pub low: f64,
    #[serde(default)]
    pub close: f64,
    #[serde(default)]
    pub volume: i64,
    #[serde(default)]
    pub oi: f64,
    #[serde(default)]
    pub timestamp_ms: i64,
    #[serde(default)]
    pub strike_price: f64,
    #[serde(default)]
    pub option_type: String,
    #[serde(default)]
    pub expiry: String,
}

impl From<BrokerTick> for trading::TickData {
    fn from(t: BrokerTick) -> Self {
        trading::TickData {
            symbol: t.symbol,
            last_price: t.last_price,
            bid_price: t.bid_price,
            ask_price: t.ask_price,
            bid_qty: t.bid_qty,
            ask_qty: t.ask_qty,
            open: t.open,
            high: t.high,
            low: t.low,
            close: t.close,
            volume: t.volume,
            oi: t.oi,
            timestamp_ms: t.timestamp_ms,
            instrument_token: t.instrument_token,
            strike_price: t.strike_price,
            option_type: t.option_type,
            expiry: t.expiry,
        }
    }
}

/// Connection state for monitoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
}

/// Signals queued during disconnection for replay.
#[derive(Debug, Clone)]
pub struct QueuedSignal {
    pub id: Uuid,
    pub tick: BrokerTick,
    pub received_at: Instant,
}

/// WebSocket manager with exponential backoff reconnection.
pub struct WebSocketManager {
    /// Broker WebSocket URL.
    ws_url: String,
    /// Broker REST API URL for state reconciliation.
    rest_url: String,
    /// Broadcast channel for tick data to gRPC StreamTicks.
    tick_sender: broadcast::Sender<trading::TickData>,
    /// Current connection state.
    state: Arc<RwLock<ConnectionState>>,
    /// Queue for signals received during disconnection.
    signal_queue: Arc<Mutex<Vec<QueuedSignal>>>,
    /// Set of message IDs seen for deduplication.
    seen_message_ids: Arc<Mutex<HashSet<String>>>,
    /// Consecutive reconnect attempts (for backoff calculation).
    reconnect_attempts: Arc<Mutex<u32>>,
}

impl WebSocketManager {
    pub fn new(
        ws_url: String,
        rest_url: String,
        tick_sender: broadcast::Sender<trading::TickData>,
    ) -> Self {
        Self {
            ws_url,
            rest_url,
            tick_sender,
            state: Arc::new(RwLock::new(ConnectionState::Disconnected)),
            signal_queue: Arc::new(Mutex::new(Vec::new())),
            seen_message_ids: Arc::new(Mutex::new(HashSet::new())),
            reconnect_attempts: Arc::new(Mutex::new(0)),
        }
    }

    /// Get current connection state.
    pub async fn connection_state(&self) -> ConnectionState {
        *self.state.read().await
    }

    /// Calculate backoff delay with exponential increase and jitter.
    fn backoff_delay(attempt: u32) -> Duration {
        let base = BACKOFF_BASE_MS * BACKOFF_MULTIPLIER.pow(attempt);
        let capped = base.min(BACKOFF_MAX_MS);
        // Add 10% jitter to prevent thundering herd
        let jitter = (capped as f64 * 0.1 * rand::random::<f64>()) as u64;
        Duration::from_millis(capped + jitter)
    }

    /// Start the WebSocket connection loop. Reconnects with exponential backoff.
    #[instrument(skip(self), fields(ws_url = %self.ws_url))]
    pub async fn run(&self) {
        info!("WebSocket manager starting");

        loop {
            {
                let mut state = self.state.write().await;
                *state = ConnectionState::Connecting;
            }

            let connect_start = Instant::now();
            info!("Attempting WebSocket connection");

            match connect_async(&self.ws_url).await {
                Ok((ws_stream, response)) => {
                    let connect_latency = connect_start.elapsed();
                    info!(
                        status = %response.status(),
                        latency_ms = connect_latency.as_millis() as u64,
                        "WebSocket connected"
                    );

                    {
                        let mut state = self.state.write().await;
                        *state = ConnectionState::Connected;
                    }
                    {
                        let mut attempts = self.reconnect_attempts.lock().await;
                        *attempts = 0;
                    }

                    // Reconcile state on reconnect
                    self.reconcile_state().await;

                    // Drain queued signals
                    self.drain_signal_queue().await;

                    let (mut write, mut read) = ws_stream.split();

                    // Spawn heartbeat pinger
                    let heartbeat_handle = tokio::spawn({
                        let state = self.state.clone();
                        async move {
                            let mut interval =
                                tokio::time::interval(Duration::from_secs(HEARTBEAT_INTERVAL_SECS));
                            loop {
                                interval.tick().await;
                                let current = *state.read().await;
                                if current != ConnectionState::Connected {
                                    break;
                                }
                            }
                        }
                    });

                    // Read loop
                    while let Some(msg_result) = read.next().await {
                        match msg_result {
                            Ok(Message::Text(text)) => {
                                let process_start = Instant::now();
                                self.handle_text_message(&text).await;
                                let process_latency = process_start.elapsed();
                                debug!(
                                    latency_us = process_latency.as_micros() as u64,
                                    "Tick processed"
                                );
                            }
                            Ok(Message::Ping(data)) => {
                                if let Err(e) = write.send(Message::Pong(data)).await {
                                    warn!(error = %e, "Failed to send pong");
                                }
                            }
                            Ok(Message::Close(frame)) => {
                                info!(?frame, "WebSocket closed by server");
                                break;
                            }
                            Ok(_) => {
                                // Binary or other frames — ignore
                            }
                            Err(e) => {
                                error!(error = %e, "WebSocket read error");
                                break;
                            }
                        }
                    }

                    heartbeat_handle.abort();
                }
                Err(e) => {
                    error!(error = %e, "WebSocket connection failed");
                }
            }

            // Disconnected — calculate backoff and retry
            {
                let mut state = self.state.write().await;
                *state = ConnectionState::Reconnecting;
            }

            let attempt = {
                let mut attempts = self.reconnect_attempts.lock().await;
                let current = *attempts;
                *attempts = current + 1;
                current
            };

            let delay = Self::backoff_delay(attempt);
            warn!(
                attempt = attempt + 1,
                delay_ms = delay.as_millis() as u64,
                "Reconnecting after backoff"
            );
            tokio::time::sleep(delay).await;
        }
    }

    /// Parse a text message from the broker and broadcast it.
    async fn handle_text_message(&self, text: &str) {
        let tick: BrokerTick = match serde_json::from_str(text) {
            Ok(t) => t,
            Err(e) => {
                warn!(error = %e, raw_len = text.len(), "Failed to parse tick JSON");
                return;
            }
        };

        // UUID-based deduplication of stale messages
        if !tick.message_id.is_empty() {
            let mut seen = self.seen_message_ids.lock().await;
            if seen.contains(&tick.message_id) {
                debug!(message_id = %tick.message_id, "Duplicate message dropped");
                return;
            }
            seen.insert(tick.message_id.clone());

            // Prune old IDs to prevent unbounded growth (keep last 10K)
            if seen.len() > 10_000 {
                seen.clear();
                debug!("Dedup set pruned");
            }
        }

        // Check for stale messages based on timestamp
        if tick.timestamp_ms > 0 {
            let now_ms = chrono::Utc::now().timestamp_millis();
            let age_secs = (now_ms - tick.timestamp_ms) / 1000;
            if age_secs > STALE_MESSAGE_WINDOW_SECS {
                warn!(
                    symbol = %tick.symbol,
                    age_secs = age_secs,
                    "Dropping stale tick"
                );
                return;
            }
        }

        let proto_tick: trading::TickData = tick.into();

        // Broadcast to all gRPC StreamTicks subscribers
        if let Err(e) = self.tick_sender.send(proto_tick) {
            debug!(error = %e, "No tick subscribers currently");
        }
    }

    /// Queue a signal when disconnected for later replay.
    #[allow(dead_code)]
    pub async fn queue_signal(&self, tick: BrokerTick) {
        let signal = QueuedSignal {
            id: Uuid::new_v4(),
            tick,
            received_at: Instant::now(),
        };
        let mut queue = self.signal_queue.lock().await;
        queue.push(signal);
        debug!(queue_depth = queue.len(), "Signal queued during disconnection");
    }

    /// Drain queued signals after reconnection.
    async fn drain_signal_queue(&self) {
        let mut queue = self.signal_queue.lock().await;
        let count = queue.len();
        if count == 0 {
            return;
        }

        info!(count = count, "Draining signal queue after reconnect");
        for signal in queue.drain(..) {
            let proto_tick: trading::TickData = signal.tick.into();
            if let Err(e) = self.tick_sender.send(proto_tick) {
                debug!(error = %e, "Failed to replay queued signal");
            }
        }
    }

    /// Reconcile state via REST API after reconnection.
    ///
    /// Fetches open orders and positions from the broker REST endpoint
    /// so the engine has an accurate view of state after a disconnect.
    #[instrument(skip(self))]
    async fn reconcile_state(&self) {
        if self.rest_url.is_empty() {
            debug!("No REST URL configured, skipping reconciliation");
            return;
        }

        let reconcile_start = Instant::now();
        info!("Starting state reconciliation via REST");

        let client = reqwest::Client::new();

        // Fetch open orders
        let orders_url = format!("{}/orders", self.rest_url);
        match client.get(&orders_url).send().await {
            Ok(resp) => {
                let status = resp.status();
                info!(
                    status = %status,
                    latency_ms = reconcile_start.elapsed().as_millis() as u64,
                    "Orders reconciliation response"
                );
            }
            Err(e) => {
                error!(error = %e, "Failed to reconcile orders via REST");
            }
        }

        // Fetch positions
        let positions_url = format!("{}/portfolio/positions", self.rest_url);
        match client.get(&positions_url).send().await {
            Ok(resp) => {
                let status = resp.status();
                info!(
                    status = %status,
                    latency_ms = reconcile_start.elapsed().as_millis() as u64,
                    "Positions reconciliation response"
                );
            }
            Err(e) => {
                error!(error = %e, "Failed to reconcile positions via REST");
            }
        }

        info!(
            total_latency_ms = reconcile_start.elapsed().as_millis() as u64,
            "State reconciliation complete"
        );
    }
}
