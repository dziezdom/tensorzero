//! Infisical Client - Integration with Infisical secrets manager
//!
//! This module provides a client for fetching secrets from Infisical,
//! with support for:
//! - Universal Auth (Client ID + Client Secret)
//! - Automatic token refresh
//! - Background polling for secret updates
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
//! │ TensorZero      │────▶│ InfisicalClient│────▶│ Infisical   │
//! │ Gateway         │     │              │     │ API         │
//! └─────────────────┘     └──────────────┘     └─────────────┘
//!                              │
//!                              ▼
//!                         ┌──────────────┐
//!                         │ KeyPool      │
//!                         │ (cached keys)│
//!                         └──────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use tensorzero_core::infisical::{InfisicalClient, InfisicalConfig};
//!
//! let config = InfisicalConfig {
//!     site_url: "https://app.infisical.com".parse()?,
//!     client_id: "your-client-id".to_string(),
//!     client_secret: SecretString::new("your-client-secret".into()),
//!     project_id: "your-project-id".to_string(),
//!     environment: "production".to_string(),
//!     secret_path: "/api-keys".to_string(),
//! };
//!
//! let client = InfisicalClient::new(config, http_client);
//! let secrets = client.get_secrets().await?;
//! ```

use async_trait::async_trait;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use url::Url;

use crate::credentials::KeyProvider;
use crate::error::{Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for connecting to Infisical.
#[derive(Debug, Clone)]
pub struct InfisicalConfig {
    /// Infisical site URL (e.g., "https://app.infisical.com" for US,
    /// "https://eu.infisical.com" for EU, or self-hosted URL)
    pub site_url: Url,

    /// Machine Identity Client ID
    pub client_id: String,

    /// Machine Identity Client Secret
    pub client_secret: SecretString,

    /// Project ID where secrets are stored
    pub project_id: String,

    /// Environment slug (e.g., "dev", "staging", "prod")
    pub environment: String,

    /// Secret path within the project (e.g., "/", "/api-keys", "/llm-providers")
    pub secret_path: String,

    /// Optional: specific secret keys to fetch (if None, fetch all)
    pub secret_keys: Option<Vec<String>>,

    /// How often to refresh secrets in the background (default: 5 minutes)
    pub refresh_interval: Duration,
}

impl InfisicalConfig {
    /// Creates a new config with default refresh interval of 5 minutes.
    pub fn new(
        site_url: Url,
        client_id: String,
        client_secret: SecretString,
        project_id: String,
        environment: String,
        secret_path: String,
    ) -> Self {
        Self {
            site_url,
            client_id,
            client_secret,
            project_id,
            environment,
            secret_path,
            secret_keys: None,
            refresh_interval: Duration::from_secs(300), // 5 minutes
        }
    }

    /// Sets specific secret keys to fetch.
    pub fn with_secret_keys(mut self, keys: Vec<String>) -> Self {
        self.secret_keys = Some(keys);
        self
    }

    /// Sets the refresh interval.
    pub fn with_refresh_interval(mut self, interval: Duration) -> Self {
        self.refresh_interval = interval;
        self
    }
}

// ============================================================================
// API Types
// ============================================================================

/// Universal Auth login request
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LoginRequest<'a> {
    client_id: &'a str,
    client_secret: &'a str,
}

/// Universal Auth login response
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LoginResponse {
    access_token: String,
    expires_in: u64,
    #[expect(dead_code)]
    #[serde(rename = "accessTokenMaxTTL")]
    access_token_max_ttl: u64,
    #[expect(dead_code)]
    token_type: String,
}

/// Secrets list response (API v4)
#[derive(Debug, Deserialize)]
struct SecretsResponse {
    secrets: Vec<InfisicalSecret>,
    #[serde(default)]
    #[expect(dead_code)]
    imports: Vec<serde_json::Value>,
}

/// Individual secret from Infisical
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InfisicalSecret {
    #[expect(dead_code)]
    id: String,
    secret_key: String,
    secret_value: String,
    #[expect(dead_code)]
    #[serde(default)]
    secret_comment: Option<String>,
    #[expect(dead_code)]
    version: u32,
}

// ============================================================================
// Access Token Cache
// ============================================================================

/// Cached access token with expiry tracking.
#[derive(Debug)]
struct CachedToken {
    token: SecretString,
    obtained_at: Instant,
    expires_in: Duration,
}

impl CachedToken {
    fn new(token: String, expires_in: u64) -> Self {
        Self {
            token: SecretString::from(token),
            obtained_at: Instant::now(),
            expires_in: Duration::from_secs(expires_in),
        }
    }

    /// Check if token is expired or about to expire (within 60 seconds).
    fn is_expired(&self) -> bool {
        let elapsed = self.obtained_at.elapsed();
        // Refresh 60 seconds before actual expiry
        elapsed >= self.expires_in.saturating_sub(Duration::from_secs(60))
    }
}

// ============================================================================
// Infisical Client
// ============================================================================

/// Client for interacting with Infisical API.
///
/// Handles authentication, token refresh, and secret fetching.
pub struct InfisicalClient {
    config: InfisicalConfig,
    http_client: TensorzeroHttpClient,
    cached_token: RwLock<Option<CachedToken>>,
}

impl std::fmt::Debug for InfisicalClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InfisicalClient")
            .field("site_url", &self.config.site_url)
            .field("project_id", &self.config.project_id)
            .field("environment", &self.config.environment)
            .finish_non_exhaustive()
    }
}

impl InfisicalClient {
    /// Creates a new Infisical client.
    pub fn new(config: InfisicalConfig, http_client: TensorzeroHttpClient) -> Self {
        Self {
            config,
            http_client,
            cached_token: RwLock::new(None),
        }
    }

    /// Authenticates with Infisical and returns an access token.
    async fn authenticate(&self) -> Result<CachedToken, Error> {
        let login_url = self
            .config
            .site_url
            .join("/api/v1/auth/universal-auth/login")
            .map_err(|e| {
                Error::new(ErrorDetails::Config {
                    message: format!("Invalid Infisical URL: {e}"),
                })
            })?;

        let request = LoginRequest {
            client_id: &self.config.client_id,
            client_secret: self.config.client_secret.expose_secret(),
        };

        let response = self
            .http_client
            .post(login_url)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Failed to authenticate with Infisical: {e}"),
                    status_code: e.status(),
                    provider_type: "infisical".to_string(),
                    raw_request: None,
                    raw_response: None,
                })
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::new(ErrorDetails::InferenceClient {
                message: format!("Infisical authentication failed: {status}"),
                status_code: Some(status),
                provider_type: "infisical".to_string(),
                raw_request: None,
                raw_response: Some(body),
            }));
        }

        let response_text = response.text().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Failed to read Infisical login response: {e}"),
                provider_type: "infisical".to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;

        let login_response: LoginResponse = serde_json::from_str(&response_text).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Failed to parse Infisical login response: {e}"),
                provider_type: "infisical".to_string(),
                raw_request: None,
                raw_response: Some(response_text),
            })
        })?;

        Ok(CachedToken::new(
            login_response.access_token,
            login_response.expires_in,
        ))
    }

    /// Gets a valid access token, refreshing if necessary.
    async fn get_access_token(&self) -> Result<SecretString, Error> {
        // Check if we have a valid cached token
        {
            let cached = self.cached_token.read().await;
            if let Some(ref token) = *cached {
                if !token.is_expired() {
                    return Ok(token.token.clone());
                }
            }
        }

        // Need to refresh - acquire write lock
        let mut cached = self.cached_token.write().await;

        // Double-check after acquiring write lock
        if let Some(ref token) = *cached {
            if !token.is_expired() {
                return Ok(token.token.clone());
            }
        }

        // Authenticate and cache new token
        let new_token = self.authenticate().await?;
        let token_clone = new_token.token.clone();
        *cached = Some(new_token);

        Ok(token_clone)
    }

    /// Fetches secrets from Infisical.
    pub async fn get_secrets(&self) -> Result<Vec<(String, SecretString)>, Error> {
        let access_token = self.get_access_token().await?;

        let mut secrets_url = self.config.site_url.join("/api/v4/secrets").map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!("Invalid Infisical URL: {e}"),
            })
        })?;

        // Add query parameters
        secrets_url
            .query_pairs_mut()
            .append_pair("projectId", &self.config.project_id)
            .append_pair("environment", &self.config.environment)
            .append_pair("secretPath", &self.config.secret_path)
            .append_pair("viewSecretValue", "true")
            .append_pair("expandSecretReferences", "true");

        let response = self
            .http_client
            .get(secrets_url)
            .bearer_auth(access_token.expose_secret())
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Failed to fetch secrets from Infisical: {e}"),
                    status_code: e.status(),
                    provider_type: "infisical".to_string(),
                    raw_request: None,
                    raw_response: None,
                })
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::new(ErrorDetails::InferenceClient {
                message: format!("Failed to fetch Infisical secrets: {status}"),
                status_code: Some(status),
                provider_type: "infisical".to_string(),
                raw_request: None,
                raw_response: Some(body),
            }));
        }

        let secrets_response: SecretsResponse = response.json().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Failed to parse Infisical secrets response: {e}"),
                provider_type: "infisical".to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;

        // Filter secrets if specific keys were requested
        let secrets: Vec<(String, SecretString)> = secrets_response
            .secrets
            .into_iter()
            .filter(|s| {
                self.config
                    .secret_keys
                    .as_ref()
                    .map(|keys| keys.contains(&s.secret_key))
                    .unwrap_or(true)
            })
            .map(|s| (s.secret_key, SecretString::from(s.secret_value)))
            .collect();

        Ok(secrets)
    }

    /// Fetches a single secret by key.
    pub async fn get_secret(&self, key: &str) -> Result<Option<SecretString>, Error> {
        let secrets = self.get_secrets().await?;
        Ok(secrets.into_iter().find(|(k, _)| k == key).map(|(_, v)| v))
    }
}

// ============================================================================
// InfisicalKeyProvider - Implements KeyProvider trait
// ============================================================================

/// A KeyProvider that fetches API keys from Infisical.
///
/// This provider:
/// - Fetches multiple API keys from a single Infisical path
/// - Caches keys and refreshes them periodically
/// - Supports round-robin key selection via KeyPool
#[derive(Debug)]
pub struct InfisicalKeyProvider {
    client: Arc<InfisicalClient>,
    /// Cached secrets with last refresh time
    cache: RwLock<InfisicalCache>,
    /// Name of this provider for logging
    name: String,
    /// Round-robin index for key selection
    index: std::sync::atomic::AtomicUsize,
}

#[derive(Debug)]
struct InfisicalCache {
    secrets: Vec<SecretString>,
    last_refresh: Instant,
    refresh_interval: Duration,
}

impl InfisicalKeyProvider {
    /// Creates a new Infisical key provider.
    pub fn new(config: InfisicalConfig, http_client: TensorzeroHttpClient, name: String) -> Self {
        let refresh_interval = config.refresh_interval;
        Self {
            client: Arc::new(InfisicalClient::new(config, http_client)),
            cache: RwLock::new(InfisicalCache {
                secrets: Vec::new(),
                last_refresh: Instant::now() - refresh_interval, // Force initial refresh
                refresh_interval,
            }),
            name,
            index: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Refreshes the cached secrets from Infisical.
    async fn refresh_cache(&self) -> Result<(), Error> {
        let secrets = self.client.get_secrets().await?;

        let mut cache = self.cache.write().await;
        cache.secrets = secrets.into_iter().map(|(_, v)| v).collect();
        cache.last_refresh = Instant::now();

        tracing::info!(
            provider = %self.name,
            count = cache.secrets.len(),
            "Refreshed Infisical secrets"
        );

        Ok(())
    }

    /// Checks if cache needs refresh.
    async fn needs_refresh(&self) -> bool {
        let cache = self.cache.read().await;
        cache.last_refresh.elapsed() >= cache.refresh_interval || cache.secrets.is_empty()
    }
}

#[async_trait]
impl KeyProvider for InfisicalKeyProvider {
    async fn get_key(&self) -> Result<Option<SecretString>, Error> {
        if self.needs_refresh().await {
            self.refresh_cache().await?;
        }

        let cache = self.cache.read().await;
        if cache.secrets.is_empty() {
            return Ok(None);
        }

        // Round-robin key selection
        let idx = self
            .index
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % cache.secrets.len();
        Ok(cache.secrets.get(idx).cloned())
    }

    async fn refresh(&self) -> Result<(), Error> {
        self.refresh_cache().await
    }

    fn provider_name(&self) -> &str {
        &self.name
    }
}

impl InfisicalKeyProvider {
    /// Returns all cached keys (useful for round-robin via KeyPool).
    pub async fn get_all_keys(&self) -> Result<Vec<SecretString>, Error> {
        if self.needs_refresh().await {
            self.refresh_cache().await?;
        }

        let cache = self.cache.read().await;
        if cache.secrets.is_empty() {
            return Err(Error::new(ErrorDetails::ApiKeyMissing {
                provider_name: self.name.clone(),
                message: "No secrets found in Infisical".to_string(),
            }));
        }

        Ok(cache.secrets.clone())
    }
}

// ============================================================================
// Background Refresh Worker
// ============================================================================

/// Creates a background refresh loop for Infisical secrets.
///
/// This returns a Future that runs forever, periodically refreshing secrets.
/// The caller is responsible for spawning this on an appropriate executor
/// (e.g., via `TaskTracker::spawn`).
///
/// # Example
///
/// ```ignore
/// use tokio_util::task::TaskTracker;
///
/// let tracker = TaskTracker::new();
/// let provider = Arc::new(InfisicalKeyProvider::new(config, http_client, "openai".to_string()));
/// let refresh_interval = Duration::from_secs(300);
///
/// tracker.spawn(refresh_worker_loop(provider, refresh_interval));
/// ```
pub async fn refresh_worker_loop(provider: Arc<InfisicalKeyProvider>, interval: Duration) {
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        ticker.tick().await;

        if let Err(e) = provider.refresh().await {
            tracing::warn!(
                provider = %provider.provider_name(),
                error = %e,
                "Failed to refresh Infisical secrets"
            );
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cached_token_expiry() {
        // Fresh token should not be expired
        let token = CachedToken::new("test-token".to_string(), 120);
        assert!(!token.is_expired());

        // Token with 50 seconds elapsed out of 120 should NOT be expired
        // (refresh threshold is 60 seconds before expiry = at 60 seconds elapsed)
        let token = CachedToken {
            token: SecretString::from("test".to_string()),
            obtained_at: Instant::now() - Duration::from_secs(50),
            expires_in: Duration::from_secs(120),
        };
        assert!(!token.is_expired()); // 50 seconds elapsed, 10 seconds until refresh threshold

        // Token with 70 seconds elapsed out of 120 SHOULD be expired
        // (we're past the 60 second mark where we start refreshing)
        let token = CachedToken {
            token: SecretString::from("test".to_string()),
            obtained_at: Instant::now() - Duration::from_secs(70),
            expires_in: Duration::from_secs(120),
        };
        assert!(token.is_expired()); // 70 seconds elapsed, should refresh now

        // Token completely expired
        let token = CachedToken {
            token: SecretString::from("test".to_string()),
            obtained_at: Instant::now() - Duration::from_secs(130),
            expires_in: Duration::from_secs(120),
        };
        assert!(token.is_expired()); // Way past expiry
    }

    #[test]
    fn test_config_builder() {
        let config = InfisicalConfig::new(
            Url::parse("https://app.infisical.com").unwrap(),
            "client-id".to_string(),
            SecretString::from("client-secret".to_string()),
            "project-123".to_string(),
            "production".to_string(),
            "/api-keys".to_string(),
        )
        .with_secret_keys(vec!["OPENAI_API_KEY".to_string()])
        .with_refresh_interval(Duration::from_secs(60));

        assert_eq!(config.environment, "production");
        assert_eq!(config.secret_keys, Some(vec!["OPENAI_API_KEY".to_string()]));
        assert_eq!(config.refresh_interval, Duration::from_secs(60));
    }

    #[test]
    fn test_login_request_serialization() {
        let request = LoginRequest {
            client_id: "test-client",
            client_secret: "test-secret",
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("clientId"));
        assert!(json.contains("clientSecret"));
    }

    #[test]
    fn test_secrets_response_deserialization() {
        let json = r#"{
            "secrets": [
                {
                    "id": "123",
                    "secretKey": "API_KEY",
                    "secretValue": "sk-test-123",
                    "version": 1
                }
            ],
            "imports": []
        }"#;

        let response: SecretsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.secrets.len(), 1);
        assert_eq!(response.secrets[0].secret_key, "API_KEY");
        assert_eq!(response.secrets[0].secret_value, "sk-test-123");
    }
}
