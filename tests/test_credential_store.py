from __future__ import annotations

import subprocess
import sys

import pytest


def _module(monkeypatch):
    from news_assistant import credential_store
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    monkeypatch.setattr(credential_store.sys, 'platform', 'darwin')
    return credential_store


def test_environment_key_wins_without_touching_os_store(monkeypatch):
    module = _module(monkeypatch)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-env-key')
    monkeypatch.setattr(module.subprocess, 'run', lambda *_a, **_k: pytest.fail('no keychain access'))
    assert module.load_deepseek_key() == 'test-env-key'


def test_keychain_lookup_returns_secret_only_in_memory(monkeypatch, capsys):
    module = _module(monkeypatch)
    observed = []
    def run(args, **kwargs):
        observed.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout='test-os-key\n', stderr='')
    monkeypatch.setattr(module.subprocess, 'run', run)
    assert module.load_deepseek_key() == 'test-os-key'
    args, options = observed[0]
    assert args[0] == '/usr/bin/security'
    assert args[1] == 'find-generic-password'
    assert options['capture_output'] is True and not options.get('shell', False)
    assert 'test-os-key' not in repr(args) + capsys.readouterr().out


@pytest.mark.parametrize('code,expected_missing', [(44, True), (36, False), (1, False)])
def test_missing_item_is_distinct_from_denied_or_failed_access(monkeypatch, code, expected_missing):
    module = _module(monkeypatch)
    monkeypatch.setattr(module.subprocess, 'run', lambda args, **_k: subprocess.CompletedProcess(
        args, code, stdout='private-value', stderr='private-error'))
    if expected_missing:
        assert module.load_deepseek_key() is None
    else:
        with pytest.raises(module.CredentialStoreError) as error:
            module.load_deepseek_key()
        assert 'private-' not in str(error.value)


def test_other_platforms_do_not_invoke_mac_security(monkeypatch):
    module = _module(monkeypatch)
    monkeypatch.setattr(module.sys, 'platform', 'linux')
    monkeypatch.setattr(module.subprocess, 'run', lambda *_a, **_k: pytest.fail('mac only'))
    assert module.load_deepseek_key() is None


def test_configuration_uses_native_hidden_prompt_without_secret_in_command(monkeypatch):
    module = _module(monkeypatch)
    monkeypatch.setattr(module.sys.stdin, 'isatty', lambda: True)
    commands = []
    def run(args, **kwargs):
        commands.append(args)
        if args[1] == 'find-generic-password':
            return subprocess.CompletedProcess(args, 0 if len(commands) == 3 else 44,
                stdout='stored-test-key\n' if len(commands) == 3 else '', stderr='')
        assert args[1] == 'add-generic-password'
        assert args[-1] == '-w'  # Native prompt, never an argv password value.
        assert '-U' not in args and '-A' not in args and '-T' not in args
        assert 'input' not in kwargs and not kwargs.get('shell', False)
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(module.subprocess, 'run', run)
    assert module.configure_deepseek_key() == 'configured'
    assert len(commands) == 3
    assert all('stored-test-key' not in arg for command in commands for arg in command)


def test_existing_key_is_not_overwritten(monkeypatch):
    module = _module(monkeypatch)
    def run(args, **kwargs):
        assert args[1] == 'find-generic-password'
        return subprocess.CompletedProcess(args, 0, stdout='existing', stderr='')
    monkeypatch.setattr(module.subprocess, 'run', run)
    assert module.configure_deepseek_key() == 'already_configured'


def test_noninteractive_missing_configuration_cannot_prompt_or_write(monkeypatch):
    module = _module(monkeypatch)
    monkeypatch.setattr(module.sys.stdin, 'isatty', lambda: False)
    def run(args, **kwargs):
        assert args[1] == 'find-generic-password'
        return subprocess.CompletedProcess(args, 44, stdout='', stderr='')
    monkeypatch.setattr(module.subprocess, 'run', run)
    with pytest.raises(module.CredentialStoreError, match='交互式'):
        module.configure_deepseek_key()


def test_configure_command_stores_only_and_never_calls_model(monkeypatch, capsys):
    from news_assistant import cli
    module = _module(monkeypatch)
    monkeypatch.setattr(module, 'configure_deepseek_key', lambda: 'configured')
    monkeypatch.setattr(cli.httpx, 'Client', lambda *_a, **_k: pytest.fail('configuration is offline'))
    monkeypatch.setattr(sys, 'argv', ['news-assistant', 'configure-key'])
    assert cli.main() == 0
    assert '钥匙串' in capsys.readouterr().out
