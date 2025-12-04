# TensorZero with Infisical Multi-Key Management

This example demonstrates how to run TensorZero with:
- **ClickHouse** for observability and inference logs
- **PostgreSQL** for structured data
- **Infisical** for secret management with multiple API keys
- **Automatic key rotation** on rate limits (429) and auth errors (401)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Infisical                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ /openai-keys│  │/gemini-keys │  │  /anthropic-keys        │  │
│  │ KEY_1=sk-.. │  │ KEY_1=AIza..│  │  KEY_1=sk-ant-...       │  │
│  │ KEY_2=sk-.. │  │ KEY_2=AIza..│  │  KEY_2=sk-ant-...       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Universal Auth
┌─────────────────────────────────────────────────────────────────┐
│                     TensorZero Gateway                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    KeyPool (per provider)                  │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                    │  │
│  │  │ Key 1   │──│ Key 2   │──│ Key 3   │── Round Robin      │  │
│  │  └─────────┘  └─────────┘  └─────────┘                    │  │
│  │       │            │            │                          │  │
│  │       └────────────┴────────────┘                          │  │
│  │              On 429/401: rotate to next key                │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │                                        │
         ▼                                        ▼
┌─────────────────┐                    ┌─────────────────┐
│   ClickHouse    │                    │   PostgreSQL    │
│  (observability)│                    │  (structured)   │
└─────────────────┘                    └─────────────────┘
```

## Quick Start

### 1. Set up Infisical

1. Create an account at [app.infisical.com](https://app.infisical.com)
2. Create a new project
3. Add your API keys as secrets:
   - Go to **Secrets** → Create folder `/openai-keys`
   - Add secrets: `KEY_1`, `KEY_2`, etc. with your OpenAI API keys
   - Repeat for `/gemini-keys` with Google AI Studio keys
4. Create a **Machine Identity**:
   - Go to **Project Settings** → **Machine Identities**
   - Create new identity with **Universal Auth**
   - Copy the **Client ID** and **Client Secret**

### 2. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit with your credentials
nano .env
```

Required variables:
```bash
INFISICAL_CLIENT_ID=your-client-id
INFISICAL_CLIENT_SECRET=your-client-secret
INFISICAL_PROJECT_ID=your-project-id
```

### 3. Start the Stack

```bash
# Start all services
docker compose up -d

# Check logs
docker compose logs -f gateway

# Verify health
curl http://localhost:3000/status
```

### 4. Test the API

```bash
# Simple chat request
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "chat",
    "input": {
      "messages": [
        {"role": "user", "content": "Hello! What is 2+2?"}
      ]
    }
  }'
```

## How Multi-Key Rotation Works

When you configure a `pool` credential source:

```toml
[models.gpt-4o.providers.openai.credentials]
pool = "openai_pool"
```

TensorZero will:

1. **Load all keys** from Infisical at startup
2. **Round-robin** between keys for each request
3. **Automatically rotate** to the next key on:
   - `429 Too Many Requests` (rate limit)
   - `401 Unauthorized` (invalid/expired key)
4. **Retry with exponential backoff** before giving up
5. **Periodically refresh** keys from Infisical (configurable)

### Benefits

- **Higher throughput**: Distribute requests across multiple API keys
- **Fault tolerance**: If one key is rate-limited, use another
- **Centralized management**: Update keys in Infisical, no container restarts
- **Security**: Keys never stored in config files or environment variables

## Configuration Reference

### Infisical Credential Source

```toml
[credentials.my_pool]
type = "infisical"
site_url = "https://app.infisical.com"  # or EU/self-hosted URL
client_id = { env = "INFISICAL_CLIENT_ID" }
client_secret = { env = "INFISICAL_CLIENT_SECRET" }
project_id = { env = "INFISICAL_PROJECT_ID" }
environment = "production"  # dev, staging, production
secret_path = "/openai-keys"  # path in Infisical
refresh_interval_secs = 300  # how often to refresh (default: 5 min)
```

### Using Pool in Model

```toml
[models.my-model.providers.my-provider.credentials]
pool = "my_pool"  # reference the credential source name
```

### Fallback Configuration

```toml
[models.gpt-4o]
routing = ["primary", "fallback"]

[models.gpt-4o.providers.primary.credentials]
pool = "openai_pool"  # Try Infisical keys first

[models.gpt-4o.providers.fallback.credentials]
env = "OPENAI_API_KEY"  # Fall back to env var
```

## Monitoring

### View Gateway Logs

```bash
docker compose logs -f gateway
```

Look for rotation events:
```
WARN Retrying inference with next key (attempt 2/5, status: 429)
```

### Access UI

Open http://localhost:4000 to view:
- Inference history
- Model performance
- Error rates

## Troubleshooting

### "No secrets found in Infisical"

- Check your `secret_path` matches the folder in Infisical
- Verify the Machine Identity has access to the project
- Check `environment` matches (dev/staging/production)

### "Failed to authenticate with Infisical"

- Verify `client_id` and `client_secret` are correct
- Check the Machine Identity is active (not expired)
- Ensure `site_url` is correct for your region

### Keys not refreshing

- Default refresh interval is 5 minutes
- Check gateway logs for refresh errors
- Verify network connectivity to Infisical

## Production Recommendations

1. **Use separate Machine Identities** for each environment
2. **Set appropriate refresh intervals** based on your key rotation policy
3. **Monitor key rotation events** in logs/metrics
4. **Use IP allowlisting** in Infisical for extra security
5. **Set up alerts** for high rate of 429/401 errors
