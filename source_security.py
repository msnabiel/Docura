"""Validation and limits for caller supplied document sources."""

import ipaddress
import os
import re
import socket
from urllib.parse import urlsplit


MAX_SOURCE_BYTES = 10 * 1024 * 1024
MAX_ARCHIVE_BYTES = 30 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 100
UPLOAD_ID_PATTERN = re.compile(r"^upload:([0-9a-f]{32})$")


def validate_remote_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Only HTTP and HTTPS document URLs are supported")
    if parsed.username or parsed.password or parsed.port not in (None, 80, 443):
        raise ValueError("URL credentials and custom ports are not supported")
    allowed_hosts = {host.strip().lower() for host in os.getenv("DOCURA_ALLOWED_URL_HOSTS", "").split(",") if host.strip()}
    hostname = parsed.hostname.lower().rstrip(".")
    if hostname not in allowed_hosts:
        raise ValueError("Document URL host is not allowed")
    for address in socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM):
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("Document URL resolves to a private or reserved address")
    return url


def fetch_remote_document(url: str) -> tuple[bytes, str]:
    import requests

    validate_remote_url(url)
    with requests.get(url, stream=True, timeout=30, allow_redirects=False,
                      headers={"User-Agent": "Docura/1.0"}) as response:
        if 300 <= response.status_code < 400:
            raise ValueError("Document URL redirects are not supported")
        response.raise_for_status()
        content = bytearray()
        for chunk in response.iter_content(64 * 1024):
            content.extend(chunk)
            if len(content) > MAX_SOURCE_BYTES:
                raise ValueError("Document exceeds the size limit")
        filename = os.path.basename(urlsplit(url).path) or "document"
        if "." not in filename:
            content_type = response.headers.get("content-type", "").lower()
            extension = (
                ".pdf" if "pdf" in content_type else
                ".html" if "html" in content_type else
                ".json" if "json" in content_type else
                ".jpg" if "image" in content_type else ".txt"
            )
            filename += extension
        return bytes(content), filename


def read_bounded(response, limit: int = MAX_SOURCE_BYTES) -> bytes:
    chunks = []
    size = 0
    while True:
        chunk = response.read(min(64 * 1024, limit + 1 - size))
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise ValueError("Document exceeds the size limit")
        chunks.append(chunk)
    return b"".join(chunks)
