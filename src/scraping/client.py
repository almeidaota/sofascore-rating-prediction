"""Thin HTTP client for the SofaScore REST API with rate limiting + retries."""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import RetryError, Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from .. import config
from .headers import DEFAULT_HEADERS


class SofaScoreError(RuntimeError):
    """Raised when a SofaScore request ultimately fails."""


class _NullRateLimiter:
    def wait(self) -> None:
        return None


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in config.HTTP_MAX_RETRIES_CODES
    return False


class SofaScoreClient:
    """Stateless-ish client; safe to reuse across a crawl run."""

    def __init__(
        self,
        *,
        base_url: str = config.BASE_URL,
        headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        rate_limiter: Any | None = None,
        timeout: float = config.REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        kwargs: dict[str, Any] = {
            "headers": headers or DEFAULT_HEADERS,
            "timeout": timeout,
            "follow_redirects": True,
        }
        if transport is not None:  # test seam (httpx.MockTransport)
            kwargs["transport"] = transport
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(**kwargs)
        self._limiter = rate_limiter or _NullRateLimiter()
        self._retry = Retrying(
            stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
            wait=wait_exponential(
                multiplier=config.RETRY_BACKOFF_SECONDS,
                min=config.RETRY_BACKOFF_SECONDS,
                max=30.0,
            ),
            retry=retry_if_exception(_should_retry),
            reraise=True,
        )

    # -- low level ----------------------------------------------------------
    def get_json(self, path: str, **params: Any) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        self._limiter.wait()
        try:
            response = self._retry(self._send, url, params)
        except (httpx.HTTPStatusError, httpx.TransportError, RetryError) as exc:
            raise SofaScoreError(f"request failed after retries: {url}") from exc
        try:
            return response.json()
        except ValueError as exc:
            raise SofaScoreError(f"non-JSON response from {url}") from exc

    def _send(self, url: str, params: dict[str, Any]) -> httpx.Response:
        response = self._http.get(url, params=params)
        response.raise_for_status()
        return response

    # -- endpoint helpers ---------------------------------------------------
    def get_seasons(self, tournament_id: int) -> list[dict[str, Any]]:
        payload = self.get_json(config.SEASONS_PATH.format(tournament_id=tournament_id))
        return payload["seasons"]

    def get_round_events(self, tournament_id: int, season_id: int, round_number: int) -> list[dict[str, Any]]:
        """All events of one round (paginated when ``hasNextPage`` is true)."""
        events: list[dict[str, Any]] = []
        page = 0
        while True:
            payload = self.get_json(
                config.EVENTS_ROUND_PATH.format(
                    tournament_id=tournament_id, season_id=season_id, round_number=round_number
                ),
                page=page,
            )
            events.extend(payload.get("events", []))
            if not payload.get("hasNextPage"):
                return events
            page += 1

    def get_event(self, event_id: int) -> dict[str, Any]:
        payload = self.get_json(config.EVENT_PATH.format(event_id=event_id))
        if "event" not in payload:
            raise SofaScoreError(f"unexpected event payload for id={event_id}")
        return payload["event"]

    def get_lineups(self, event_id: int) -> dict[str, Any]:
        return self.get_json(config.LINEUPS_PATH.format(event_id=event_id))

    def get_incidents(self, event_id: int) -> dict[str, Any]:
        return self.get_json(config.INCIDENTS_PATH.format(event_id=event_id))

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "SofaScoreClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
