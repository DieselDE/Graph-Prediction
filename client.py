import os
from coingecko_sdk import Coingecko, AsyncCoingecko

# Initialize a single, reusable client. This should be imported and used application-wide.
# Load the production API key from the environment per project rules.
# Use the official environment variable name `COINGECKO_PRO_API_KEY`.
client = Coingecko(
    pro_api_key=os.environ.get("COINGECKO_DEMO_API_KEY"),
    environment="demo",
    max_retries=3,  # Rely on the SDK's built-in retry mechanism.
)

# Optional: Initialize a single async client for concurrent applications.
async_client = AsyncCoingecko(
    pro_api_key=os.environ.get("COINGECKO_DEMO_API_KEY"),
    environment="demo",
    max_retries=3,
)