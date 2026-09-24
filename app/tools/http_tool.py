import ipaddress
import urllib.parse
from typing import Any, Dict, Optional
import httpx
from pydantic import BaseModel, Field

from app.config import settings
from app.tools.registry import BaseTool, RiskLevel


class HttpToolInput(BaseModel):
    url: str = Field(..., description="Target URL to request (must match allowed domains).")
    method: str = Field(default="GET", description="HTTP method: GET or POST")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="URL query parameters.")
    data: Optional[Dict[str, Any]] = Field(default=None, description="JSON body payload for POST requests.")
    headers: Optional[Dict[str, str]] = Field(default_factory=dict, description="Optional request headers.")


class HttpToolOutput(BaseModel):
    status_code: int
    url: str
    data: Any
    headers: Dict[str, str]


class HttpTool(BaseTool):
    name: str = "http_tool"
    description: str = "Makes safe HTTP GET/POST requests against allowlisted API domains with timeout and SSRF protection."
    risk_level: RiskLevel = RiskLevel.MEDIUM_RISK
    input_schema = HttpToolInput
    output_schema = HttpToolOutput

    def _validate_url(self, url: str) -> None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Scheme '{parsed.scheme}' not allowed. Only http and https are supported.")

        hostname = parsed.hostname
        if not hostname:
            raise ValueError("Invalid URL: missing hostname.")

        # Check for localhost / loopback
        if hostname.lower() in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
            raise ValueError("Access to localhost / loopback addresses is forbidden for security.")

        # Check for private IP ranges (SSRF protection)
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                raise ValueError(f"Access to private IP address {hostname} is forbidden.")
        except ValueError:
            # Hostname is a domain, not a raw IP address
            pass

        # Check domain allowlist
        allowed = settings.ALLOWED_HTTP_DOMAINS
        domain_match = any(
            hostname == domain or hostname.endswith("." + domain)
            for domain in allowed
        )
        if not domain_match:
            raise ValueError(
                f"Domain '{hostname}' is not in the allowlist. Allowed domains: {', '.join(allowed)}"
            )

    async def run(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        self._validate_url(url)
        method_upper = method.upper()
        if method_upper not in ("GET", "POST"):
            raise ValueError(f"HTTP method '{method}' is not supported. Use GET or POST.")

        req_headers = {"User-Agent": "Krishna-Agent/1.0", **(headers or {})}

        async with httpx.AsyncClient(timeout=settings.TOOL_TIMEOUT_SECONDS, follow_redirects=True) as client:
            if method_upper == "GET":
                response = await client.get(url, params=params, headers=req_headers)
            else:
                response = await client.post(url, json=data, params=params, headers=req_headers)

            # Parse content
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    body = response.json()
                except Exception:
                    body = response.text
            else:
                body = response.text[:2000]

            return {
                "status_code": response.status_code,
                "url": str(response.url),
                "data": body,
                "headers": dict(response.headers)
            }
