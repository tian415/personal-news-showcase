"""Explicit local key setup; no plaintext files, model calls, or broad keychain scans."""

from __future__ import annotations

import os
import subprocess
import sys

_SERVICE = 'personal-news-assistant.deepseek'
_ACCOUNT = 'api-key'
_SECURITY = '/usr/bin/security'


class CredentialStoreError(RuntimeError):
    """A safe diagnostic without subprocess output or credential contents."""


def _load_keychain() -> str | None:
    if sys.platform != 'darwin':
        return None
    try:
        result = subprocess.run(
            [_SECURITY, 'find-generic-password', '-s', _SERVICE, '-a', _ACCOUNT, '-w'],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        raise CredentialStoreError('无法读取系统钥匙串；请检查本机钥匙串是否已解锁。') from None
    if result.returncode == 44:  # errSecItemNotFound; denial/lock is not absence.
        return None
    if result.returncode != 0 or not result.stdout.strip():
        raise CredentialStoreError('钥匙串读取失败或条目为空；不会覆盖现有条目。')
    return result.stdout.strip()


def load_deepseek_key() -> str | None:
    """Environment keeps priority for CI; macOS otherwise uses this app's item."""
    return os.environ.get('DEEPSEEK_API_KEY') or _load_keychain()


def configure_deepseek_key() -> str:
    """Let security prompt directly in the terminal, then verify read access."""
    if sys.platform != 'darwin':
        raise CredentialStoreError('钥匙串配置仅支持 macOS；其他系统请使用安全环境变量。')
    if _load_keychain() is not None:
        return 'already_configured'
    if not sys.stdin.isatty():
        raise CredentialStoreError('请在本机交互式终端配置，以便系统隐藏输入密钥。')
    print('将 API 密钥保存到本机钥匙串的程序专用条目；不会改动「密码」App 中的原条目。')
    try:
        # macOS security's documented secure input mode: -w is last with no value.
        # The secret is never in argv, Python output, or a shell command/history.
        result = subprocess.run(
            [_SECURITY, 'add-generic-password', '-s', _SERVICE, '-a', _ACCOUNT,
             '-l', 'Personal News Assistant / DeepSeek API', '-w'],
            check=False,
        )
    except OSError:
        raise CredentialStoreError('系统钥匙串配置失败；未启动任何模型请求。') from None
    if result.returncode != 0:
        raise CredentialStoreError('系统未完成钥匙串保存；未启动任何模型请求。')
    if _load_keychain() is None:
        raise CredentialStoreError('无法验证已保存的钥匙串条目；请检查后再执行模型请求。')
    return 'configured'
