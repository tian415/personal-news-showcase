"""Deterministic, unabridged readable RSS text for prompting and quote validation."""

from html.parser import HTMLParser

_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "div",
        "dl",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
)


class _SourceTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._ignored_depth = 0

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self._parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style"}:
            self._ignored_depth += 1
        elif not self._ignored_depth and (tag == "br" or tag in _BLOCK_TAGS):
            self._append_break()

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            if self._ignored_depth:
                self._ignored_depth -= 1
        elif not self._ignored_depth and tag in _BLOCK_TAGS:
            self._append_break()

    def _append_break(self) -> None:
        if self._parts and not self._parts[-1].endswith("\n"):
            self._parts.append("\n")

    def text(self) -> str:
        lines = [" ".join(line.split()) for line in "".join(self._parts).splitlines()]
        return "\n".join(line for line in lines if line)


def readable_source_text(value: str) -> str:
    parser = _SourceTextParser()
    parser.feed(value)
    parser.close()
    return parser.text()
