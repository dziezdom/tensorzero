//! Model Discovery - Auto-discovery of available models from providers
//!
//! This module provides the `ModelDiscovery` trait for querying available models
//! from LLM providers. This enables:
//!
//! - UI/Dashboard model browsing without manual configuration
//! - Config validation to verify model names exist before inference
//! - Dynamic model selection based on capabilities
//!
//! # Supported Providers
//!
//! | Provider | Support | Notes |
//! |----------|---------|-------|
//! | OpenAI | ✅ | `GET /v1/models` |
//! | Google AI Studio | ✅ | `GET /v1beta/models` with rich metadata |
//! | Azure OpenAI | ✅ | Lists deployments, not global models |
//! | Anthropic | ❌ | No public API, use hardcoded list |
//!
//! # Example
//!
//! ```ignore
//! use tensorzero_core::discovery::{ModelDiscovery, DiscoveredModel};
//!
//! let provider = OpenAIDiscovery::new(api_key);
//! let models = provider.list_models(&http_client).await?;
//!
//! for model in models {
//!     println!("{}: {} tokens", model.id, model.context_window.unwrap_or(0));
//! }
//! ```

use async_trait::async_trait;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

use crate::error::{Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;

// ============================================================================
// Core Types
// ============================================================================

/// Information about a discovered model from a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredModel {
    /// The model identifier used in API calls (e.g., "gpt-4o", "gemini-1.5-pro")
    pub id: String,

    /// Human-readable display name (e.g., "GPT-4o", "Gemini 1.5 Pro")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,

    /// Description of the model's capabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Maximum input context window in tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,

    /// Maximum output tokens the model can generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    /// Model capabilities
    pub capabilities: ModelCapabilities,

    /// Provider-specific metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Capabilities supported by a model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelCapabilities {
    /// Supports chat/conversation
    #[serde(default)]
    pub chat: bool,

    /// Supports text embeddings
    #[serde(default)]
    pub embeddings: bool,

    /// Supports vision/image input
    #[serde(default)]
    pub vision: bool,

    /// Supports function/tool calling
    #[serde(default)]
    pub function_calling: bool,

    /// Supports streaming responses
    #[serde(default)]
    pub streaming: bool,

    /// Supports JSON mode output
    #[serde(default)]
    pub json_mode: bool,

    /// Supports audio input
    #[serde(default)]
    pub audio_input: bool,

    /// Supports audio output
    #[serde(default)]
    pub audio_output: bool,
}

/// Trait for providers that support model discovery.
#[async_trait]
pub trait ModelDiscovery: Send + Sync {
    /// Returns the provider name for logging/error messages.
    fn provider_name(&self) -> &'static str;

    /// Lists all available models from this provider.
    ///
    /// # Arguments
    /// * `http_client` - HTTP client to use for API requests
    ///
    /// # Returns
    /// A list of discovered models with their capabilities.
    async fn list_models(
        &self,
        http_client: &TensorzeroHttpClient,
    ) -> Result<Vec<DiscoveredModel>, Error>;

    /// Checks if a specific model exists.
    ///
    /// Default implementation calls `list_models` and searches.
    /// Providers can override for more efficient single-model lookup.
    async fn model_exists(
        &self,
        http_client: &TensorzeroHttpClient,
        model_id: &str,
    ) -> Result<bool, Error> {
        let models = self.list_models(http_client).await?;
        Ok(models.iter().any(|m| m.id == model_id))
    }

    /// Gets detailed information about a specific model.
    ///
    /// Default implementation calls `list_models` and filters.
    /// Providers can override for direct model lookup.
    async fn get_model(
        &self,
        http_client: &TensorzeroHttpClient,
        model_id: &str,
    ) -> Result<Option<DiscoveredModel>, Error> {
        let models = self.list_models(http_client).await?;
        Ok(models.into_iter().find(|m| m.id == model_id))
    }
}

// ============================================================================
// OpenAI Discovery
// ============================================================================

/// OpenAI model discovery implementation.
///
/// Uses the `GET /v1/models` endpoint to list available models.
pub struct OpenAIDiscovery {
    api_key: SecretString,
    api_base: Option<url::Url>,
}

impl OpenAIDiscovery {
    /// Creates a new OpenAI discovery client.
    pub fn new(api_key: SecretString, api_base: Option<url::Url>) -> Self {
        Self { api_key, api_base }
    }

    fn get_models_url(&self) -> Result<url::Url, Error> {
        let base = self.api_base.clone().unwrap_or_else(|| {
            // SAFETY: This is a valid static URL
            #[expect(clippy::expect_used)]
            url::Url::parse("https://api.openai.com/v1/").expect("Invalid default URL")
        });

        let mut url = base;
        if !url.path().ends_with('/') {
            url.set_path(&format!("{}/", url.path()));
        }
        url.join("models").map_err(|e| {
            Error::new(ErrorDetails::InvalidBaseUrl {
                message: e.to_string(),
            })
        })
    }
}

/// OpenAI models list response
#[derive(Debug, Deserialize)]
struct OpenAIModelsResponse {
    data: Vec<OpenAIModelInfo>,
}

#[derive(Debug, Deserialize)]
struct OpenAIModelInfo {
    id: String,
    #[serde(default)]
    owned_by: Option<String>,
    #[serde(default)]
    #[expect(dead_code)]
    created: Option<i64>,
}

impl From<OpenAIModelInfo> for DiscoveredModel {
    fn from(info: OpenAIModelInfo) -> Self {
        // Infer capabilities from model name
        let id_lower = info.id.to_lowercase();
        let capabilities = ModelCapabilities {
            chat: id_lower.contains("gpt")
                || id_lower.contains("o1")
                || id_lower.contains("o3")
                || id_lower.contains("chatgpt"),
            embeddings: id_lower.contains("embedding"),
            vision: id_lower.contains("vision")
                || id_lower.contains("gpt-4o")
                || id_lower.contains("gpt-4-turbo"),
            function_calling: id_lower.contains("gpt-4") || id_lower.contains("gpt-3.5-turbo"),
            streaming: !id_lower.contains("embedding"),
            json_mode: id_lower.contains("gpt-4") || id_lower.contains("gpt-3.5-turbo"),
            audio_input: id_lower.contains("whisper") || id_lower.contains("gpt-4o-audio"),
            audio_output: id_lower.contains("tts") || id_lower.contains("gpt-4o-audio"),
        };

        DiscoveredModel {
            id: info.id,
            display_name: None,
            description: info.owned_by.map(|o| format!("Owned by: {o}")),
            context_window: None, // OpenAI doesn't return this in /models
            max_output_tokens: None,
            capabilities,
            metadata: None,
        }
    }
}

#[async_trait]
impl ModelDiscovery for OpenAIDiscovery {
    fn provider_name(&self) -> &'static str {
        "OpenAI"
    }

    async fn list_models(
        &self,
        http_client: &TensorzeroHttpClient,
    ) -> Result<Vec<DiscoveredModel>, Error> {
        let url = self.get_models_url()?;

        let response = http_client
            .get(url)
            .bearer_auth(self.api_key.expose_secret())
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Failed to fetch models: {e}"),
                    status_code: e.status(),
                    provider_type: "openai".to_string(),
                    raw_request: None,
                    raw_response: None,
                })
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::new(ErrorDetails::InferenceClient {
                message: format!("Failed to fetch models: {status}"),
                status_code: Some(status),
                provider_type: "openai".to_string(),
                raw_request: None,
                raw_response: Some(body),
            }));
        }

        let models_response: OpenAIModelsResponse = response.json().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Failed to parse models response: {e}"),
                provider_type: "openai".to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;

        Ok(models_response.data.into_iter().map(Into::into).collect())
    }
}

// ============================================================================
// Google AI Studio Discovery
// ============================================================================

/// Google AI Studio (Gemini) model discovery implementation.
///
/// Uses the `GET /v1beta/models` endpoint which returns rich metadata
/// including context window sizes and supported capabilities.
pub struct GoogleAIStudioDiscovery {
    api_key: SecretString,
}

impl GoogleAIStudioDiscovery {
    /// Creates a new Google AI Studio discovery client.
    pub fn new(api_key: SecretString) -> Self {
        Self { api_key }
    }

    fn get_models_url() -> Result<url::Url, Error> {
        url::Url::parse("https://generativelanguage.googleapis.com/v1beta/models").map_err(|e| {
            Error::new(ErrorDetails::InvalidBaseUrl {
                message: e.to_string(),
            })
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleModelInfo {
    name: String,
    #[serde(default)]
    display_name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    input_token_limit: Option<u32>,
    #[serde(default)]
    output_token_limit: Option<u32>,
    #[serde(default)]
    supported_generation_methods: Vec<String>,
}

/// Google AI Studio models list response
#[derive(Debug, Deserialize)]
struct GoogleModelsResponse {
    models: Vec<GoogleModelInfo>,
    #[serde(rename = "nextPageToken")]
    #[expect(dead_code)]
    next_page_token: Option<String>,
}

impl From<GoogleModelInfo> for DiscoveredModel {
    fn from(info: GoogleModelInfo) -> Self {
        // Extract model ID from "models/gemini-1.5-pro" format
        let id = info
            .name
            .strip_prefix("models/")
            .unwrap_or(&info.name)
            .to_string();

        let methods = &info.supported_generation_methods;
        let capabilities = ModelCapabilities {
            chat: methods.iter().any(|m| m == "generateContent"),
            embeddings: methods.iter().any(|m| m == "embedContent"),
            vision: info
                .description
                .as_ref()
                .map(|d| d.to_lowercase().contains("image") || d.to_lowercase().contains("vision"))
                .unwrap_or(false),
            function_calling: true, // Most Gemini models support this
            streaming: methods.iter().any(|m| m == "streamGenerateContent"),
            json_mode: true,
            audio_input: info
                .description
                .as_ref()
                .map(|d| d.to_lowercase().contains("audio"))
                .unwrap_or(false),
            audio_output: false,
        };

        DiscoveredModel {
            id,
            display_name: info.display_name,
            description: info.description,
            context_window: info.input_token_limit,
            max_output_tokens: info.output_token_limit,
            capabilities,
            metadata: None,
        }
    }
}

#[async_trait]
impl ModelDiscovery for GoogleAIStudioDiscovery {
    fn provider_name(&self) -> &'static str {
        "GoogleAIStudio"
    }

    async fn list_models(
        &self,
        http_client: &TensorzeroHttpClient,
    ) -> Result<Vec<DiscoveredModel>, Error> {
        let mut url = Self::get_models_url()?;
        url.query_pairs_mut()
            .append_pair("key", self.api_key.expose_secret())
            .append_pair("pageSize", "1000");

        let response = http_client.get(url).send().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                message: format!("Failed to fetch models: {e}"),
                status_code: e.status(),
                provider_type: "google_ai_studio_gemini".to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::new(ErrorDetails::InferenceClient {
                message: format!("Failed to fetch models: {status}"),
                status_code: Some(status),
                provider_type: "google_ai_studio_gemini".to_string(),
                raw_request: None,
                raw_response: Some(body),
            }));
        }

        let models_response: GoogleModelsResponse = response.json().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Failed to parse models response: {e}"),
                provider_type: "google_ai_studio_gemini".to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;

        Ok(models_response.models.into_iter().map(Into::into).collect())
    }
}

// ============================================================================
// Anthropic Discovery (Static List)
// ============================================================================

/// Anthropic model discovery - returns a static list since Anthropic
/// doesn't provide a public models API.
pub struct AnthropicDiscovery;

impl AnthropicDiscovery {
    /// Creates a new Anthropic discovery client.
    pub fn new() -> Self {
        Self
    }

    /// Returns the static list of known Anthropic models.
    fn known_models() -> Vec<DiscoveredModel> {
        vec![
            DiscoveredModel {
                id: "claude-sonnet-4-20250514".to_string(),
                display_name: Some("Claude Sonnet 4".to_string()),
                description: Some("Latest Claude Sonnet model".to_string()),
                context_window: Some(200_000),
                max_output_tokens: Some(64_000),
                capabilities: ModelCapabilities {
                    chat: true,
                    embeddings: false,
                    vision: true,
                    function_calling: true,
                    streaming: true,
                    json_mode: true,
                    audio_input: false,
                    audio_output: false,
                },
                metadata: None,
            },
            DiscoveredModel {
                id: "claude-opus-4-20250514".to_string(),
                display_name: Some("Claude Opus 4".to_string()),
                description: Some("Most capable Claude model".to_string()),
                context_window: Some(200_000),
                max_output_tokens: Some(32_000),
                capabilities: ModelCapabilities {
                    chat: true,
                    embeddings: false,
                    vision: true,
                    function_calling: true,
                    streaming: true,
                    json_mode: true,
                    audio_input: false,
                    audio_output: false,
                },
                metadata: None,
            },
            DiscoveredModel {
                id: "claude-3-5-sonnet-20241022".to_string(),
                display_name: Some("Claude 3.5 Sonnet".to_string()),
                description: Some("High-performance Claude 3.5 model".to_string()),
                context_window: Some(200_000),
                max_output_tokens: Some(8192),
                capabilities: ModelCapabilities {
                    chat: true,
                    embeddings: false,
                    vision: true,
                    function_calling: true,
                    streaming: true,
                    json_mode: true,
                    audio_input: false,
                    audio_output: false,
                },
                metadata: None,
            },
            DiscoveredModel {
                id: "claude-3-5-haiku-20241022".to_string(),
                display_name: Some("Claude 3.5 Haiku".to_string()),
                description: Some("Fast and efficient Claude 3.5 model".to_string()),
                context_window: Some(200_000),
                max_output_tokens: Some(8192),
                capabilities: ModelCapabilities {
                    chat: true,
                    embeddings: false,
                    vision: true,
                    function_calling: true,
                    streaming: true,
                    json_mode: true,
                    audio_input: false,
                    audio_output: false,
                },
                metadata: None,
            },
            DiscoveredModel {
                id: "claude-3-opus-20240229".to_string(),
                display_name: Some("Claude 3 Opus".to_string()),
                description: Some("Most capable Claude 3 model".to_string()),
                context_window: Some(200_000),
                max_output_tokens: Some(4096),
                capabilities: ModelCapabilities {
                    chat: true,
                    embeddings: false,
                    vision: true,
                    function_calling: true,
                    streaming: true,
                    json_mode: true,
                    audio_input: false,
                    audio_output: false,
                },
                metadata: None,
            },
        ]
    }
}

impl Default for AnthropicDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ModelDiscovery for AnthropicDiscovery {
    fn provider_name(&self) -> &'static str {
        "Anthropic"
    }

    async fn list_models(
        &self,
        _http_client: &TensorzeroHttpClient,
    ) -> Result<Vec<DiscoveredModel>, Error> {
        // Return static list - Anthropic doesn't have a public models API
        Ok(Self::known_models())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_model_capabilities_inference() {
        let model = OpenAIModelInfo {
            id: "gpt-4o".to_string(),
            owned_by: Some("openai".to_string()),
            created: Some(1234567890),
        };

        let discovered: DiscoveredModel = model.into();
        assert!(discovered.capabilities.chat);
        assert!(discovered.capabilities.vision);
        assert!(discovered.capabilities.streaming);
    }

    #[test]
    fn test_openai_embedding_model() {
        let model = OpenAIModelInfo {
            id: "text-embedding-3-small".to_string(),
            owned_by: None,
            created: None,
        };

        let discovered: DiscoveredModel = model.into();
        assert!(discovered.capabilities.embeddings);
        assert!(!discovered.capabilities.chat);
        assert!(!discovered.capabilities.streaming);
    }

    #[test]
    fn test_google_model_parsing() {
        let model = GoogleModelInfo {
            name: "models/gemini-1.5-pro".to_string(),
            display_name: Some("Gemini 1.5 Pro".to_string()),
            description: Some("Advanced multimodal model with image understanding".to_string()),
            input_token_limit: Some(2_000_000),
            output_token_limit: Some(8192),
            supported_generation_methods: vec![
                "generateContent".to_string(),
                "streamGenerateContent".to_string(),
            ],
        };

        let discovered: DiscoveredModel = model.into();
        assert_eq!(discovered.id, "gemini-1.5-pro");
        assert_eq!(discovered.context_window, Some(2_000_000));
        assert!(discovered.capabilities.chat);
        assert!(discovered.capabilities.streaming);
        assert!(discovered.capabilities.vision);
    }

    #[test]
    fn test_anthropic_static_models() {
        let discovery = AnthropicDiscovery::new();
        let models = AnthropicDiscovery::known_models();

        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.id.contains("claude-sonnet-4")));
        assert!(models.iter().all(|m| m.capabilities.chat));
        assert!(models.iter().all(|m| m.capabilities.function_calling));
        assert_eq!(discovery.provider_name(), "Anthropic");
    }

    #[test]
    fn test_model_capabilities_default() {
        let caps = ModelCapabilities::default();
        assert!(!caps.chat);
        assert!(!caps.embeddings);
        assert!(!caps.vision);
    }
}
