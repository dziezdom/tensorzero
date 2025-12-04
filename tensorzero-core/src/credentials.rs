//! Credential management abstractions for TensorZero.
//!
//! This module provides the `KeyProvider` trait, which standardizes how API keys
//! and other credentials are fetched. It supports various strategies like:
//! - Static keys from environment variables or files
//! - Dynamic key pools with round-robin selection
//! - External secret managers (e.g., Infisical, AWS Secrets Manager)
//!
//! # Configuration
//!
//! Credentials can be configured in `tensorzero.toml` using the `CredentialSource` type:
//!
//! ```toml
//! # Static credential from environment variable (backward compatible)
//! [models.gpt4.providers.openai]
//! credential = { type = "static", env_var = "OPENAI_API_KEY" }
//!
//! # Multiple keys from environment variables with round-robin
//! [models.gpt4.providers.openai]
//! credential = { type = "env_list", vars = ["OPENAI_KEY_1", "OPENAI_KEY_2", "OPENAI_KEY_3"] }
//!
//! # External secret manager (Infisical)
//! [models.gpt4.providers.openai]
//! credential = { type = "infisical", project_id = "...", environment = "prod", path = "/openai-keys" }
//! ```
//!
//! # Example
//!
//! ```ignore
//! use tensorzero_core::credentials::{KeyProvider, StaticKeyProvider};
//! use secrecy::SecretString;
//!
//! // A simple static key provider
//! let provider = StaticKeyProvider::new(SecretString::new("my-api-key".into()));
//! let key = provider.get_key().await;
//! ```

use async_trait::async_trait;
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::error::{Error, ErrorDetails};

// ============================================================================
// CredentialSource - Configuration types for credentials in tensorzero.toml
// ============================================================================

/// Configuration for credential sources in `tensorzero.toml`.
///
/// This enum represents all possible ways to configure credentials for a model provider.
/// It is designed to be backward-compatible with the existing `CredentialLocation` while
/// adding support for new enterprise features like key pools and external secret managers.
///
/// # Backward Compatibility
///
/// The existing string-based syntax (e.g., `"env::OPENAI_API_KEY"`) continues to work
/// and is internally converted to `CredentialSource::Static`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CredentialSource {
    /// Static credential from a single environment variable.
    ///
    /// This is the default and most common configuration, equivalent to the
    /// existing `env::VAR_NAME` syntax.
    ///
    /// # Example
    /// ```toml
    /// credential = { type = "static", env_var = "OPENAI_API_KEY" }
    /// ```
    Static {
        /// The name of the environment variable containing the API key.
        env_var: String,
    },

    /// Static credential read from a file.
    ///
    /// The file path can be specified directly or via an environment variable.
    ///
    /// # Example
    /// ```toml
    /// credential = { type = "file", path = "/secrets/openai-key" }
    /// # or
    /// credential = { type = "file", path_env = "OPENAI_KEY_FILE" }
    /// ```
    File {
        /// Direct path to the credentials file.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<String>,
        /// Environment variable containing the path to the credentials file.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path_env: Option<String>,
    },

    /// Multiple API keys from environment variables with load balancing.
    ///
    /// Keys are selected using round-robin by default. This is useful for
    /// distributing load across multiple API keys to avoid rate limits.
    ///
    /// # Example
    /// ```toml
    /// credential = { type = "env_list", vars = ["KEY_1", "KEY_2", "KEY_3"] }
    /// ```
    EnvList {
        /// List of environment variable names containing API keys.
        vars: Vec<String>,
        /// Load balancing strategy (default: round_robin).
        #[serde(default)]
        strategy: LoadBalancingStrategy,
    },

    /// Credentials from Infisical secret manager.
    ///
    /// Supports fetching multiple secrets with a common prefix for key pooling.
    ///
    /// # Example
    /// ```toml
    /// credential = { type = "infisical", project_id = "...", environment = "prod", path = "/openai" }
    /// ```
    Infisical {
        /// Infisical project ID.
        project_id: String,
        /// Environment name (e.g., "dev", "staging", "prod").
        environment: String,
        /// Path to secrets in Infisical (e.g., "/openai-keys").
        path: String,
        /// Optional prefix filter for secret names (e.g., "KEY_" to match KEY_1, KEY_2, etc.).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        secret_name_prefix: Option<String>,
        /// Load balancing strategy when multiple secrets match (default: round_robin).
        #[serde(default)]
        strategy: LoadBalancingStrategy,
    },

    /// Dynamic credential resolved at inference time.
    ///
    /// The key must be provided in the inference request's credentials field.
    ///
    /// # Example
    /// ```toml
    /// credential = { type = "dynamic", key_name = "openai_key" }
    /// ```
    Dynamic {
        /// The name of the key to look up in inference credentials.
        key_name: String,
    },

    /// Use provider-specific SDK for credential resolution.
    ///
    /// For example, GCP Vertex AI can use application default credentials.
    ///
    /// # Example
    /// ```toml
    /// credential = { type = "sdk" }
    /// ```
    Sdk,

    /// No credentials required.
    ///
    /// # Example
    /// ```toml
    /// credential = { type = "none" }
    /// ```
    None,
}

/// Load balancing strategy for key pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadBalancingStrategy {
    /// Distribute requests evenly across keys in order.
    #[default]
    RoundRobin,
    /// Select a random key for each request.
    Random,
}

impl CredentialSource {
    /// Creates a static credential source from an environment variable name.
    pub fn from_env(env_var: impl Into<String>) -> Self {
        Self::Static {
            env_var: env_var.into(),
        }
    }

    /// Creates an env_list credential source from a list of environment variable names.
    pub fn from_env_list(vars: Vec<String>) -> Self {
        Self::EnvList {
            vars,
            strategy: LoadBalancingStrategy::default(),
        }
    }

    /// Checks if this source requires dynamic resolution at inference time.
    pub fn is_dynamic(&self) -> bool {
        matches!(self, Self::Dynamic { .. })
    }

    /// Checks if this source requires external API calls (e.g., Infisical).
    pub fn requires_external_fetch(&self) -> bool {
        matches!(self, Self::Infisical { .. })
    }
}

/// A trait for providing API keys with support for various strategies.
///
/// Implementations can provide:
/// - Single static keys
/// - Key pools with load balancing (e.g., round-robin)
/// - Dynamic key fetching from external secret managers
///
/// All implementations must be thread-safe (`Send + Sync`) to support
/// concurrent inference requests.
#[async_trait]
pub trait KeyProvider: Send + Sync + Debug {
    /// Retrieves a key according to the provider's internal strategy.
    ///
    /// For pool-based providers, this may implement round-robin or other
    /// load balancing strategies. The returned key is wrapped in `SecretString`
    /// to prevent accidental logging of sensitive data.
    ///
    /// Returns `None` if no key is available (e.g., provider configured without credentials).
    async fn get_key(&self) -> Result<Option<SecretString>, Error>;

    /// Refreshes the key pool from the underlying source.
    ///
    /// This is useful for providers that fetch keys from external services
    /// (e.g., Infisical, AWS Secrets Manager) and need periodic refresh.
    ///
    /// For static providers, this is typically a no-op.
    async fn refresh(&self) -> Result<(), Error>;

    /// Returns the name of this key provider for logging/debugging purposes.
    fn provider_name(&self) -> &str;
}

/// A static key provider that always returns the same key.
///
/// This is the simplest implementation, suitable for single API keys
/// loaded from environment variables or configuration files.
#[derive(Debug, Clone)]
pub struct StaticKeyProvider {
    key: SecretString,
    name: String,
}

impl StaticKeyProvider {
    /// Creates a new static key provider with the given key.
    pub fn new(key: SecretString) -> Self {
        Self {
            key,
            name: "static".to_string(),
        }
    }

    /// Creates a new static key provider with a custom name.
    pub fn with_name(key: SecretString, name: impl Into<String>) -> Self {
        Self {
            key,
            name: name.into(),
        }
    }
}

#[async_trait]
impl KeyProvider for StaticKeyProvider {
    async fn get_key(&self) -> Result<Option<SecretString>, Error> {
        Ok(Some(self.key.clone()))
    }

    async fn refresh(&self) -> Result<(), Error> {
        // Static keys don't need refresh
        Ok(())
    }

    fn provider_name(&self) -> &str {
        &self.name
    }
}

/// A key provider that returns no key.
///
/// Used for providers that don't require authentication or when
/// credentials are intentionally omitted.
#[derive(Debug, Clone, Default)]
pub struct NoKeyProvider;

#[async_trait]
impl KeyProvider for NoKeyProvider {
    async fn get_key(&self) -> Result<Option<SecretString>, Error> {
        Ok(None)
    }

    async fn refresh(&self) -> Result<(), Error> {
        Ok(())
    }

    fn provider_name(&self) -> &str {
        "none"
    }
}

/// A key provider that fetches keys dynamically from inference credentials.
///
/// This supports the existing `dynamic::key_name` pattern where keys are
/// passed at inference time rather than configured statically.
#[derive(Debug, Clone)]
pub struct DynamicKeyProvider {
    key_name: String,
}

impl DynamicKeyProvider {
    /// Creates a new dynamic key provider that will look up the given key name.
    pub fn new(key_name: impl Into<String>) -> Self {
        Self {
            key_name: key_name.into(),
        }
    }

    /// Returns the key name this provider looks up.
    pub fn key_name(&self) -> &str {
        &self.key_name
    }
}

#[async_trait]
impl KeyProvider for DynamicKeyProvider {
    async fn get_key(&self) -> Result<Option<SecretString>, Error> {
        // Dynamic keys are resolved at inference time via InferenceCredentials,
        // so this method returns an error indicating the key must be provided dynamically.
        Err(Error::new(ErrorDetails::ApiKeyMissing {
            provider_name: "dynamic".to_string(),
            message: format!(
                "Dynamic key '{}' must be provided in the inference request credentials",
                self.key_name
            ),
        }))
    }

    async fn refresh(&self) -> Result<(), Error> {
        // Dynamic keys are provided at inference time, nothing to refresh
        Ok(())
    }

    fn provider_name(&self) -> &str {
        "dynamic"
    }
}

/// A key provider with fallback support.
///
/// Tries the primary provider first, and if that fails, falls back to the secondary.
#[derive(Debug)]
pub struct FallbackKeyProvider {
    primary: Arc<dyn KeyProvider>,
    fallback: Arc<dyn KeyProvider>,
}

impl FallbackKeyProvider {
    /// Creates a new fallback key provider.
    pub fn new(primary: Arc<dyn KeyProvider>, fallback: Arc<dyn KeyProvider>) -> Self {
        Self { primary, fallback }
    }
}

#[async_trait]
impl KeyProvider for FallbackKeyProvider {
    async fn get_key(&self) -> Result<Option<SecretString>, Error> {
        match self.primary.get_key().await {
            Ok(key) => Ok(key),
            Err(e) => {
                tracing::warn!(
                    "Primary key provider '{}' failed, trying fallback '{}': {}",
                    self.primary.provider_name(),
                    self.fallback.provider_name(),
                    e
                );
                self.fallback.get_key().await
            }
        }
    }

    async fn refresh(&self) -> Result<(), Error> {
        // Try to refresh both, but don't fail if one fails
        let primary_result = self.primary.refresh().await;
        let fallback_result = self.fallback.refresh().await;

        // Return the first error if any
        primary_result?;
        fallback_result?;
        Ok(())
    }

    fn provider_name(&self) -> &str {
        "fallback"
    }
}

// ============================================================================
// RoundRobinKeyProvider - Key pool with round-robin selection
// ============================================================================

/// A key provider that rotates through multiple keys using round-robin.
///
/// This is useful for distributing load across multiple API keys to avoid
/// rate limits imposed by providers.
#[derive(Debug)]
pub struct RoundRobinKeyProvider {
    keys: Vec<SecretString>,
    counter: AtomicUsize,
    name: String,
}

impl RoundRobinKeyProvider {
    /// Creates a new round-robin key provider with the given keys.
    ///
    /// Returns an error if the keys list is empty.
    pub fn new(keys: Vec<SecretString>) -> Result<Self, Error> {
        if keys.is_empty() {
            return Err(Error::new(ErrorDetails::Config {
                message: "RoundRobinKeyProvider requires at least one key".to_string(),
            }));
        }
        Ok(Self {
            keys,
            counter: AtomicUsize::new(0),
            name: "round_robin".to_string(),
        })
    }

    /// Creates a new round-robin key provider with a custom name.
    pub fn with_name(keys: Vec<SecretString>, name: impl Into<String>) -> Result<Self, Error> {
        if keys.is_empty() {
            return Err(Error::new(ErrorDetails::Config {
                message: "RoundRobinKeyProvider requires at least one key".to_string(),
            }));
        }
        Ok(Self {
            keys,
            counter: AtomicUsize::new(0),
            name: name.into(),
        })
    }

    /// Returns the number of keys in the pool.
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }
}

#[async_trait]
impl KeyProvider for RoundRobinKeyProvider {
    async fn get_key(&self) -> Result<Option<SecretString>, Error> {
        let index = self.counter.fetch_add(1, Ordering::Relaxed) % self.keys.len();
        Ok(Some(self.keys[index].clone()))
    }

    async fn refresh(&self) -> Result<(), Error> {
        // Static key pool doesn't need refresh
        // For dynamic pools (e.g., from Infisical), override this method
        Ok(())
    }

    fn provider_name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// EnvListKeyProvider - Keys from multiple environment variables
// ============================================================================

/// A key provider that loads keys from multiple environment variables.
///
/// This provider reads keys from environment variables at construction time
/// and provides them using round-robin selection.
#[derive(Debug)]
pub struct EnvListKeyProvider {
    inner: RoundRobinKeyProvider,
    var_names: Vec<String>,
}

impl EnvListKeyProvider {
    /// Creates a new provider that reads keys from the specified environment variables.
    ///
    /// Missing environment variables are skipped with a warning.
    /// Returns an error if no valid keys are found.
    pub fn new(var_names: Vec<String>) -> Result<Self, Error> {
        let mut keys = Vec::new();
        let mut missing_vars = Vec::new();

        for var_name in &var_names {
            match std::env::var(var_name) {
                Ok(value) => keys.push(SecretString::from(value)),
                Err(_) => {
                    missing_vars.push(var_name.clone());
                    tracing::warn!("Environment variable '{}' is not set, skipping", var_name);
                }
            }
        }

        if keys.is_empty() {
            return Err(Error::new(ErrorDetails::ApiKeyMissing {
                provider_name: "env_list".to_string(),
                message: format!(
                    "No valid keys found. Missing environment variables: {missing_vars:?}"
                ),
            }));
        }

        Ok(Self {
            inner: RoundRobinKeyProvider::with_name(keys, "env_list")?,
            var_names,
        })
    }

    /// Returns the list of environment variable names.
    pub fn var_names(&self) -> &[String] {
        &self.var_names
    }

    /// Returns the number of successfully loaded keys.
    pub fn key_count(&self) -> usize {
        self.inner.key_count()
    }
}

#[async_trait]
impl KeyProvider for EnvListKeyProvider {
    async fn get_key(&self) -> Result<Option<SecretString>, Error> {
        self.inner.get_key().await
    }

    async fn refresh(&self) -> Result<(), Error> {
        // Environment variables are read at construction time
        // A full refresh would require recreating the provider
        Ok(())
    }

    fn provider_name(&self) -> &str {
        "env_list"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use secrecy::ExposeSecret;

    #[tokio::test]
    async fn test_static_key_provider() {
        let key = SecretString::new("test-key".into());
        let provider = StaticKeyProvider::new(key);

        let result = provider.get_key().await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().expose_secret(), "test-key");

        // Refresh should be a no-op
        assert!(provider.refresh().await.is_ok());
    }

    #[tokio::test]
    async fn test_no_key_provider() {
        let provider = NoKeyProvider;

        let result = provider.get_key().await.unwrap();
        assert!(result.is_none());

        // Refresh should be a no-op
        assert!(provider.refresh().await.is_ok());
    }

    #[tokio::test]
    async fn test_dynamic_key_provider() {
        let provider = DynamicKeyProvider::new("my_api_key");

        // Should return an error since no credentials are provided
        let result = provider.get_key().await;
        assert!(result.is_err());

        assert_eq!(provider.key_name(), "my_api_key");
    }

    #[tokio::test]
    async fn test_fallback_key_provider() {
        let primary = Arc::new(StaticKeyProvider::new(SecretString::new(
            "primary-key".into(),
        )));
        let fallback = Arc::new(StaticKeyProvider::new(SecretString::new(
            "fallback-key".into(),
        )));

        let provider = FallbackKeyProvider::new(primary, fallback);

        // Should return primary key
        let result = provider.get_key().await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().expose_secret(), "primary-key");
    }

    #[tokio::test]
    async fn test_fallback_when_primary_fails() {
        let primary = Arc::new(DynamicKeyProvider::new("missing_key"));
        let fallback = Arc::new(StaticKeyProvider::new(SecretString::new(
            "fallback-key".into(),
        )));

        let provider = FallbackKeyProvider::new(primary, fallback);

        // Should fall back to secondary
        let result = provider.get_key().await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().expose_secret(), "fallback-key");
    }

    #[tokio::test]
    async fn test_round_robin_key_provider() {
        let keys = vec![
            SecretString::new("key-1".into()),
            SecretString::new("key-2".into()),
            SecretString::new("key-3".into()),
        ];
        let provider = RoundRobinKeyProvider::new(keys).unwrap();

        // Should rotate through keys
        assert_eq!(
            provider.get_key().await.unwrap().unwrap().expose_secret(),
            "key-1"
        );
        assert_eq!(
            provider.get_key().await.unwrap().unwrap().expose_secret(),
            "key-2"
        );
        assert_eq!(
            provider.get_key().await.unwrap().unwrap().expose_secret(),
            "key-3"
        );
        // Should wrap around
        assert_eq!(
            provider.get_key().await.unwrap().unwrap().expose_secret(),
            "key-1"
        );
    }

    #[tokio::test]
    async fn test_round_robin_empty_keys() {
        let result = RoundRobinKeyProvider::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_credential_source_from_env() {
        let source = CredentialSource::from_env("OPENAI_API_KEY");
        assert_eq!(
            source,
            CredentialSource::Static {
                env_var: "OPENAI_API_KEY".to_string()
            }
        );
        assert!(!source.is_dynamic());
        assert!(!source.requires_external_fetch());
    }

    #[test]
    fn test_credential_source_from_env_list() {
        let source =
            CredentialSource::from_env_list(vec!["KEY_1".to_string(), "KEY_2".to_string()]);
        match source {
            CredentialSource::EnvList { vars, strategy } => {
                assert_eq!(vars, vec!["KEY_1", "KEY_2"]);
                assert_eq!(strategy, LoadBalancingStrategy::RoundRobin);
            }
            _ => panic!("Expected EnvList"),
        }
    }

    #[test]
    fn test_credential_source_dynamic() {
        let source = CredentialSource::Dynamic {
            key_name: "my_key".to_string(),
        };
        assert!(source.is_dynamic());
    }

    #[test]
    fn test_credential_source_infisical() {
        let source = CredentialSource::Infisical {
            project_id: "proj-123".to_string(),
            environment: "prod".to_string(),
            path: "/openai-keys".to_string(),
            secret_name_prefix: Some("KEY_".to_string()),
            strategy: LoadBalancingStrategy::RoundRobin,
        };
        assert!(source.requires_external_fetch());
        assert!(!source.is_dynamic());
    }

    #[test]
    fn test_credential_source_serialization() {
        let source = CredentialSource::Static {
            env_var: "OPENAI_API_KEY".to_string(),
        };
        let json = serde_json::to_string(&source).unwrap();
        assert!(json.contains("\"type\":\"static\""));
        assert!(json.contains("\"env_var\":\"OPENAI_API_KEY\""));

        // Round-trip
        let parsed: CredentialSource = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, source);
    }

    #[test]
    fn test_credential_source_deserialization_env_list() {
        let json = r#"{"type":"env_list","vars":["KEY_1","KEY_2","KEY_3"]}"#;
        let source: CredentialSource = serde_json::from_str(json).unwrap();
        match source {
            CredentialSource::EnvList { vars, strategy } => {
                assert_eq!(vars, vec!["KEY_1", "KEY_2", "KEY_3"]);
                assert_eq!(strategy, LoadBalancingStrategy::RoundRobin);
            }
            _ => panic!("Expected EnvList"),
        }
    }

    #[test]
    fn test_credential_source_deserialization_infisical() {
        let json = r#"{
            "type": "infisical",
            "project_id": "proj-123",
            "environment": "prod",
            "path": "/secrets",
            "secret_name_prefix": "OPENAI_"
        }"#;
        let source: CredentialSource = serde_json::from_str(json).unwrap();
        match source {
            CredentialSource::Infisical {
                project_id,
                environment,
                path,
                secret_name_prefix,
                strategy,
            } => {
                assert_eq!(project_id, "proj-123");
                assert_eq!(environment, "prod");
                assert_eq!(path, "/secrets");
                assert_eq!(secret_name_prefix, Some("OPENAI_".to_string()));
                assert_eq!(strategy, LoadBalancingStrategy::RoundRobin);
            }
            _ => panic!("Expected Infisical"),
        }
    }
}
