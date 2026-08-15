import base64
import json
from typing import Any, Dict, Union, Optional

from cryptography.fernet import Fernet
from passlib.context import CryptContext

from app.core.config import settings

import bcrypt

# ── Password hashing ──────────────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    """Hash a plain-text password with bcrypt."""
    pwd_bytes = plain.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode('utf-8')


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plain-text password against its bcrypt hash."""
    try:
        pwd_bytes = plain.encode('utf-8')[:72]
        return bcrypt.checkpw(pwd_bytes, hashed.encode('utf-8'))
    except Exception:
        return False


# ── Credential encryption ─────────────────────────────────────────────────────
def _get_fernet() -> Fernet:
    """
    Build a Fernet instance from ENCRYPTION_KEY.

    Accepts two formats:
    1. A valid 32-byte URL-safe base64 key (produced by ``Fernet.generate_key()``).
    2. An arbitrary string — it will be SHA-256-hashed to produce a stable 32-byte key.

    Raises ``ValueError`` at startup if the key is missing or empty, so misconfigured
    deployments fail loudly instead of silently falling back to a known dev key.
    """
    key = settings.ENCRYPTION_KEY
    if not key:
        raise ValueError(
            "ENCRYPTION_KEY is not set. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )

    # Try the key as-is first (valid 32-byte base64 url-safe string).
    try:
        return Fernet(key.encode() if isinstance(key, str) else key)
    except Exception:
        pass

    # Fall back: hash arbitrary string to a deterministic 32-byte key.
    import hashlib
    hashed = hashlib.sha256(key.encode()).digest()
    valid_key = base64.urlsafe_b64encode(hashed)
    return Fernet(valid_key)


def encrypt_credentials(data: Union[Dict[str, Any], str]) -> str:
    """Encrypt a string or dict of credentials into a ciphertext string."""
    fernet = _get_fernet()
    text = json.dumps(data) if isinstance(data, dict) else str(data)
    return fernet.encrypt(text.encode()).decode()


def decrypt_credentials(token: str) -> Union[Dict[str, Any], str]:
    """Decrypt a ciphertext string back into a Python dict or plain text."""
    if not token:
        return {}
    fernet = _get_fernet()
    decrypted = fernet.decrypt(token.encode()).decode()
    try:
        return json.loads(decrypted)
    except Exception:
        return decrypted


# ── Admin Auth Tokens ──────────────────────────────────────────────────────────
import hmac
import hashlib
import time

def create_admin_token(username: str, expires_in_seconds: int = 86400 * 7) -> str:
    """Create a signed admin access token (valid for 7 days by default)."""
    exp = int(time.time()) + expires_in_seconds
    payload = f"{username}:{exp}"
    signature = hmac.new(settings.ENCRYPTION_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    token_str = f"{payload}:{signature}"
    return base64.b64encode(token_str.encode()).decode()


def verify_admin_token(token: str) -> Optional[str]:
    """Verify admin access token. Returns username if valid, None otherwise."""
    if not token:
        return None
    try:
        decoded = base64.b64decode(token.encode()).decode()
        parts = decoded.split(":")
        if len(parts) != 3:
            return None
        username, exp_str, signature = parts[0], parts[1], parts[2]
        exp = int(exp_str)
        if time.time() > exp:
            return None
        expected_sig = hmac.new(settings.ENCRYPTION_KEY.encode(), f"{username}:{exp_str}".encode(), hashlib.sha256).hexdigest()
        if hmac.compare_digest(signature, expected_sig):
            return username
        return None
    except Exception:
        return None
