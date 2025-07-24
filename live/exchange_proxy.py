# live/exchange_proxy.py
"""
A resilient proxy wrapper for the CCXT exchange object.

It uses the 'tenacity' library to automatically retry API calls that fail
due to temporary, recoverable network issues.
"""
import logging
import ccxt.async_support as ccxt
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

LOG = logging.getLogger(__name__)

# Define the specific, temporary errors we want to retry on.
# We should NOT retry on things like "Invalid API Key" or "Insufficient Funds".
RETRYABLE_EXCEPTIONS = (
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
    ccxt.RequestTimeout,
    ccxt.DDoSProtection,
)

class ExchangeProxy:
    """
    Wraps a ccxt exchange instance to provide automatic retries on network errors.
    """
    def __init__(self, exchange: ccxt.Exchange):
        self._exchange = exchange

    @property
    def markets(self):
        """Pass through to the underlying exchange's markets property."""
        return self._exchange.markets

    @retry(
        # Wait 2^x * 1 seconds between each retry, starting with 2s, up to 30s
        wait=wait_exponential(multiplier=1, min=2, max=30),
        # Stop trying after 5 attempts
        stop=stop_after_attempt(5),
        # Only retry if the exception is one of our defined temporary errors
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        # Log before retrying
        before_sleep=lambda state: LOG.warning(
            "Retrying API call %s due to %s. Attempt #%d",
            state.fn.__name__, state.outcome.exception(), state.attempt_number
        )
    )
    async def __getattr__(self, name):
        """
        This is the magic method. It intercepts any call to a method that
        doesn't exist on the Proxy (e.g., `fetch_ohlcv`, `create_order`)
        and calls it on the underlying exchange object, wrapped in our retry logic.
        """
        return await getattr(self._exchange, name)

    async def close(self):
        """Gracefully close the underlying exchange connection."""
        await self._exchange.close()