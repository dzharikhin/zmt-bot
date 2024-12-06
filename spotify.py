import dataclasses
import os
from datetime import datetime, timedelta
from urllib.parse import urlsplit

import requests
from requests import Session
from requests_ratelimiter import LimiterAdapter


class SpotifyAuth(requests.auth.AuthBase):

    @dataclasses.dataclass
    class Token:
        value: str
        token_type: str
        expires_at: datetime

    def __init__(
        self,
        env_var_name_client_id: str,
        env_var_name_client_secret: str,
        speculative_update_lag_seconds: float = 60,
    ):
        self.env_var_name_client_id = env_var_name_client_id
        self.env_var_name_client_secret = env_var_name_client_secret
        self.speculative_update_lag = speculative_update_lag_seconds
        self._token: SpotifyAuth.Token | None = None

    def __call__(self, r: requests.Request):
        token = self._get_token()
        r.headers.update({"Authorization": f"{token.token_type} {token.value}"})
        return r

    def _get_token(self) -> Token:
        if not self._token or self._token.expires_at <= datetime.now():
            token_response = requests.post(
                "https://accounts.spotify.com/api/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "client_id": f"{os.getenv(self.env_var_name_client_id)}",
                    "client_secret": f"{os.getenv(self.env_var_name_client_secret)}",
                    "grant_type": "client_credentials",
                },
            ).json()
            self._token = SpotifyAuth.Token(
                token_response["access_token"],
                token_response["token_type"],
                datetime.now()
                + timedelta(seconds=token_response["expires_in"])
                - timedelta(seconds=self.speculative_update_lag),
            )
        return self._token


class SpotifySession(Session):

    def __init__(
        self,
        base_url: str,
        *,
        https_proxy: str | None = None,
        rpm_limit: float | None = None,
        env_var_name_client_id: str,
        env_var_name_client_secret: str,
    ):
        super().__init__()

        self.base_url = base_url
        self.https_proxy = https_proxy
        self.auth = SpotifyAuth(env_var_name_client_id, env_var_name_client_secret)
        self.hooks = {"response": lambda r, *args, **kwargs: r.raise_for_status()}
        if rpm_limit:
            adapter = LimiterAdapter(per_minute=rpm_limit)
            self.mount("http://", adapter)
            self.mount("https://", adapter)

    def request(self, method, url, *args, **kwargs):
        url_parts = urlsplit(url)
        joined_url = url if url_parts.netloc else self.base_url + url
        if self.https_proxy:
            kwargs.update({"proxies": {"https": self.https_proxy}})
        return super().request(method, joined_url, *args, **kwargs)
