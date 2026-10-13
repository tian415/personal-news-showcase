import smtplib
from collections.abc import Callable
from datetime import date
from email.message import EmailMessage

from pydantic import BaseModel, EmailStr, Field


class EmailSettings(BaseModel):
    gmail_address: EmailStr
    gmail_app_password: str = Field(min_length=1)
    recipient: EmailStr


def send_html(
    html: str,
    plain_text: str,
    digest_date: date,
    settings: EmailSettings,
    *,
    smtp_factory: Callable = smtplib.SMTP_SSL,
) -> None:
    message = EmailMessage()
    message["Subject"] = f"每日新闻 · {digest_date.isoformat()}"
    message["From"] = str(settings.gmail_address)
    message["To"] = str(settings.recipient)
    message.set_content(plain_text)
    message.add_alternative(html, subtype="html")
    with smtp_factory("smtp.gmail.com", 465) as smtp:
        smtp.login(str(settings.gmail_address), settings.gmail_app_password)
        smtp.send_message(message)
