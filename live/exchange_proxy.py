# live/exchange_proxy.py
"""
A resilient proxy wrapper for the CCXT exchange object.

It uses the 'tenacity' library to automatically retry API calls that fail
due to temporary, recoverable network issues.
"""
import logging
import asyncio
from functools import wraps
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

    def __getattr__(self, name):
        """
        Intercepts any call to a method that doesn't exist on the Proxy,
        retrieves it from the underlying exchange object, and wraps it
        in our retry logic.
        """
        original_attr = getattr(self._exchange, name)

        if not callable(original_attr):
            return original_attr

        @wraps(original_attr)
        def wrapper(*args, **kwargs):
            # Define the retry decorator dynamically
            retry_decorator = retry(
                wait=wait_exponential(multiplier=1, min=2, max=30),
                stop=stop_after_attempt(5),
                retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
                before_sleep=lambda state: LOG.warning(
                    "Retrying API call %s due to %s. Attempt #%d",
                    name, state.outcome.exception(), state.attempt_number
                )
            )

            if asyncio.iscoroutinefunction(original_attr):
                # Apply decorator to an async function
                @retry_decorator
                async def async_call():
                    return await original_attr(*args, **kwargs)
                return async_call()
            else:
                # Apply decorator to a sync function
                @retry_decorator
                def sync_call():
                    return original_attr(*args, **kwargs)
                return sync_call()

        return wrapper

    async def close(self):
        """Gracefully close the underlying exchange connection."""
        await self._exchange.close()