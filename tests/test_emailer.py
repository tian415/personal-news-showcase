from datetime import date

import pytest

from news_assistant.emailer import EmailSettings, send_html


class FakeSMTP:
    def __init__(self, host: str, port: int) -> None:
        self.host, self.port, self.login_args, self.message = host, port, None, None
    def __enter__(self): return self
    def __exit__(self, *args): return None
    def login(self, user: str, password: str) -> None: self.login_args = (user, password)
    def send_message(self, message) -> None: self.message = message


def test_send_html_authenticates_and_addresses_message() -> None:
    instance = None
    def factory(host: str, port: int):
        nonlocal instance
        instance = FakeSMTP(host, port)
        return instance
    settings = EmailSettings(gmail_address="sender@gmail.com", gmail_app_password="secret", recipient="reader@gmail.com")
    send_html(
        "<h1>摘要</h1>",
        "今日新闻合集：https://example.com/2026-08-12.html",
        date(2026, 8, 12),
        settings,
        smtp_factory=factory,
    )
    assert instance.login_args == ("sender@gmail.com", "secret")
    assert instance.message["To"] == "reader@gmail.com"
    assert "摘要" in instance.message.get_body(preferencelist=("html",)).get_content()


def test_send_html_includes_digest_url_in_both_bodies() -> None:
    instance = None

    def factory(host: str, port: int):
        nonlocal instance
        instance = FakeSMTP(host, port)
        return instance

    settings = EmailSettings(
        gmail_address="sender@gmail.com",
        gmail_app_password="secret",
        recipient="reader@gmail.com",
    )
    digest_url = "https://example-user.github.io/personal-news-digest/"
    send_html(
        "<h1>摘要</h1>",
        f"今日新闻合集：{digest_url}",
        date(2026, 8, 13),
        settings,
        smtp_factory=factory,
    )

    plain = instance.message.get_body(preferencelist=("plain",)).get_content()
    rich = instance.message.get_body(preferencelist=("html",)).get_content()
    assert digest_url in plain
    assert "摘要" in rich


def test_email_settings_reject_empty_password() -> None:
    with pytest.raises(ValueError):
        EmailSettings(gmail_address="sender@gmail.com", gmail_app_password="", recipient="reader@gmail.com")
