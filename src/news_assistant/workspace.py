from __future__ import annotations

import json
import os
import shutil
import tempfile
from importlib.resources import files
from pathlib import Path

import yaml
from pydantic import ValidationError

from .models import Preferences
from .source_registry import SourceRegistry

_README = """# 我的新闻工作区

来源配置、偏好和反馈属于本工作区。生成的阅读页与私有状态分开保存。

## 先体验

无需密钥、网络或模型费用的合成演示：

```sh
news-assistant demo --output work/demo
```

只访问公共 RSS，使用来源摘要，不调用模型、不发送邮件：

```sh
news-assistant build --sources config/source_registry.yaml --preferences config/preferences.json --source-only --output public
```

来源由 config/source_registry.yaml 管理；production_enabled=false 的来源不进入默认日报。
config/preferences.json 初始为中立偏好，主题和来源权重范围 0.5 至 2.0。

## 验证内容审阅和实际反馈

```sh
news-assistant audit-sources --capture-candidates --output work/snapshot.json
news-assistant prepare-review --snapshot work/snapshot.json --output work/review.json
```

审阅表初始全部待审，无默认合格分。审阅时填审阅者、时间、分数、类型和理由，保留内容指纹和原文引句。
也可用 assess-review 命令免费预检模型请求；加 --execute 才会发送一次付费请求，失败不自动重试。
模型只提出判断，不等于完成独立事实核验。macOS 可先运行 news-assistant configure-key，把 API key 存入程序专用钥匙串，后续自动读取；其他系统使用安全环境变量 DEEPSEEK_API_KEY。配置不联网，保存密钥不等于授权收费；勿把密钥写在这里。

```sh
news-assistant preview-reviewed --snapshot work/snapshot.json --review work/review.json --preferences config/preferences.json --output local-reader --state-dir work/local-state --run-id local-001
```

用本机静态服务打开 local-reader，阅读后确认准确日期页，再导入网页导出的反馈文件：

```sh
news-assistant confirm-local --state-dir work/local-state --run-id local-001 --page local-reader/YYYY-MM-DD.html
news-assistant import-feedback --channel local --state-dir work/local-state --public-dir local-reader --input YOUR_DOWNLOADED_FEEDBACK.json
```

日期和下载路径需替换。下一轮使用新的快照、审阅和 run-id，反馈实际影响本地选文；撤销可恢复。
本地确认不能冒充线上发布；普通 build 和生产仍使用 legacy。私有评分与偏好不得放入阅读目录。

## 可选服务与升级

去掉 build 的 --source-only 才会调用摘要模型。邮件另需 GMAIL_ADDRESS、GMAIL_APP_PASSWORD、NEWS_RECIPIENT，且必须显式执行发送命令。
来源可能失效，定期 audit-sources 检查。当前安装仅在 macOS 验证，Linux/Windows 尚未实机验收。
升级前把 config 和 work 备份到自己的私有位置，再安装新版包；不要重复 init 覆盖旧工作区。
GitHub Actions 的私有状态分支方案需使用私有运行仓库，不能直接在公开 fork 写入个人状态。
"""

_GITIGNORE = """.env
.env.*
public/
local-reader/
work/
__pycache__/
*.pyc
"""


def _resource_text(name: str) -> str:
    resource = files("news_assistant").joinpath("resources", name)
    try:
        return resource.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        module_path = Path(__file__)
        source_checkout = module_path.parents[2]
        expected_package = source_checkout / "src/news_assistant"
        if name == "source_registry.yaml" and (
            module_path.parent == expected_package
            and (source_checkout / "pyproject.toml").is_file()
        ):
            return (source_checkout / "config/source_registry.yaml").read_text(
                encoding="utf-8"
            )
        raise FileNotFoundError(f"packaged resource is missing: {name}") from error


def _validated_templates() -> tuple[str, str]:
    source_registry = _resource_text("source_registry.yaml")
    preferences = _resource_text("preferences.json")
    try:
        SourceRegistry.model_validate(yaml.safe_load(source_registry))
    except (ValueError, TypeError, yaml.YAMLError) as error:
        raise ValueError("packaged source registry is invalid") from error
    try:
        Preferences.model_validate(json.loads(preferences))
    except (ValueError, TypeError, json.JSONDecodeError, ValidationError) as error:
        raise ValueError("packaged preferences are invalid") from error
    return source_registry, preferences


def _absolute_without_resolving_links(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symbolic_link_path(target: Path) -> None:
    for component in (target, *target.parents):
        if component.is_symlink():
            raise ValueError(f"workspace path contains a symbolic link: {component}")


def _validate_target(target: Path) -> bool:
    _reject_symbolic_link_path(target)
    parent = target.parent
    if not parent.exists() or not parent.is_dir():
        raise FileNotFoundError(f"workspace parent directory does not exist: {parent}")
    if not target.exists():
        return False
    if not target.is_dir():
        raise FileExistsError(f"workspace target is not a directory: {target}")
    if any(target.iterdir()):
        raise FileExistsError(f"workspace target is not empty: {target}")
    return True


def initialize_workspace(target: Path) -> dict[str, object]:
    """Create an isolated, reusable workspace without network or credential access."""

    target = _absolute_without_resolving_links(target)
    target_was_empty_directory = _validate_target(target)
    source_registry, preferences = _validated_templates()

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.init-", dir=target.parent))
    try:
        (staging / "config").mkdir()
        (staging / "work/private-state").mkdir(parents=True)
        (staging / "config/source_registry.yaml").write_text(
            source_registry, encoding="utf-8"
        )
        (staging / "config/preferences.json").write_text(preferences, encoding="utf-8")
        (staging / "README.md").write_text(_README, encoding="utf-8")
        (staging / ".gitignore").write_text(_GITIGNORE, encoding="utf-8")

        _reject_symbolic_link_path(target)
        if target_was_empty_directory:
            if any(target.iterdir()):
                raise FileExistsError(f"workspace target is not empty: {target}")
            target.rmdir()
        elif target.exists():
            raise FileExistsError(f"workspace target appeared during initialization: {target}")
        try:
            os.replace(staging, target)
        except OSError as error:
            if target_was_empty_directory and not target.exists():
                try:
                    target.mkdir()
                except OSError as restore_error:
                    error.add_note(
                        f"failed to restore original empty directory: {restore_error}"
                    )
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return {
        "workspace": str(target),
        "created_files": [
            ".gitignore",
            "README.md",
            "config/preferences.json",
            "config/source_registry.yaml",
        ],
        "created_directories": ["config", "work/private-state"],
        "source_registry": "config/source_registry.yaml",
        "preferences": "config/preferences.json",
        "private_state": "work/private-state",
    }
