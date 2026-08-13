import base64
import json
from typing import Any, Dict, Union
from cryptography.fernet import Fernet
from app.core.config import settings


def _get_fernet() -> Fernet:
    key = settings.ENCRYPTION_KEY
    if not key or len(key) < 32:
        # Fallback deterministic key for dev if key invalid
        key = base64.urlsafe_b64encode(b"AIMED_DEV_SECRET_KEY_32_BYTES_01").decode()
    try:
        return Fernet(key.encode() if isinstance(key, str) else key)
    except Exception:
        # If user key is plain string, pad/hash to valid 32-byte base64 key
        import hashlib
        hashed = hashlib.sha256(key.encode()).digest()
        valid_key = base64.urlsafe_b64encode(hashed)
        return Fernet(valid_key)


def encrypt_credentials(data: Union[Dict[str, Any], str]) -> str:
    """Encrypt a string or dict of credentials into a ciphertext string."""
    fernet = _get_fernet()
    if isinstance(data, dict):
        text = json.dumps(data)
    else:
        text = str(data)
    return fernet.encrypt(text.encode()).decode()


def decrypt_credentials(token: str) -> Union[Dict[str, Any], str]:
    """Decrypt a ciphertext string back into python dict or text."""
    if not token:
        return {}
    fernet = _get_fernet()
    decrypted = fernet.decrypt(token.encode()).decode()
    try:
        return json.loads(decrypted)
    except Exception:
        return decrypted
