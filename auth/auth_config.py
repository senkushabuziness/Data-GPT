# auth/auth_config.py

import os
import chainlit as cl
from dotenv import load_dotenv
from typing import Dict, Optional

load_dotenv()

google_client_id = os.getenv("OAUTH_GOOGLE_CLIENT_ID")
google_client_secret = os.getenv("OAUTH_GOOGLE_CLIENT_SECRET")
github_client_id = os.getenv("OAUTH_GITHUB_CLIENT_ID")
github_client_secret = os.getenv("OAUTH_GITHUB_CLIENT_SECRET")

oauth_providers = []

if google_client_id and google_client_secret:
    oauth_providers.append({
        "id": "google",
        "name": "Google",
        "client_id": google_client_id.strip(),
        "client_secret": google_client_secret.strip(),
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "userinfo_url": "https://www.googleapis.com/oauth2/v1/userinfo",
        "scopes": ["openid", "email", "profile"]
    })

if github_client_id and github_client_secret:
    oauth_providers.append({
        "id": "github",
        "name": "GitHub",
        "client_id": github_client_id.strip(),
        "client_secret": github_client_secret.strip(),
        "authorize_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "userinfo_url": "https://api.github.com/user",
        "scopes": ["read:user", "user:email"]
    })

cl.oauth_providers = oauth_providers

@cl.oauth_callback
def oauth_callback(provider_id: str, token: str, raw_user_data: Dict[str, str], default_user: cl.User) -> Optional[cl.User]:
    if provider_id == "google":
        return cl.User(
            identifier=f"google:{raw_user_data.get('email')}",
            display_name=raw_user_data.get("name", raw_user_data.get("email")),
            metadata=raw_user_data
        )
    elif provider_id == "github":
        return cl.User(
            identifier=f"github:{raw_user_data.get('login')}",
            display_name=raw_user_data.get("name") or raw_user_data.get("login"),
            metadata=raw_user_data
        )
    return default_user
