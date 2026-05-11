from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence
from urllib import error, request


DEFAULT_INPUT = Path("/Users/ziyang/Downloads/S2.txt")
DEFAULT_ZSHRC = Path.home() / ".zshrc"
START_MARKER = "# >>> Design Scientist S2_API_KEY >>>"
END_MARKER = "# <<< Design Scientist S2_API_KEY <<<"
QUOTE_CHARS = "\"'\u2018\u2019\u201c\u201d"
SMOKE_URL = "https://api.semanticscholar.org/graph/v1/paper/search?query=protein%20design&limit=1&fields=title"


class S2KeyConfigError(ValueError):
    """Raised for non-secret configuration errors."""


def _strip_outer_noise(value: str) -> str:
    previous = None
    while previous != value:
        previous = value
        value = value.strip().strip(QUOTE_CHARS).strip()
    return value


def parse_s2_api_key(text: str) -> str:
    value = _strip_outer_noise(text)
    if not value:
        raise S2KeyConfigError("Input did not contain an S2 API key.")

    if value.startswith("export "):
        value = value.removeprefix("export ").strip()

    if "=" in value:
        name, raw_value = value.split("=", 1)
        if name.strip() != "S2_API_KEY":
            raise S2KeyConfigError("Expected a raw key or an S2_API_KEY assignment.")
        value = raw_value

    value = _strip_outer_noise(value)
    if not value:
        raise S2KeyConfigError("Input did not contain an S2 API key.")
    if not value.isascii():
        raise S2KeyConfigError("S2 API key must contain only ASCII characters after trimming quotes.")
    if any(char.isspace() for char in value):
        raise S2KeyConfigError("S2 API key must not contain whitespace.")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise S2KeyConfigError("S2 API key contains invalid control characters.")

    return value


def _zsh_single_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _marker_block(key: str) -> str:
    return f"{START_MARKER}\nexport S2_API_KEY={_zsh_single_quote(key)}\n{END_MARKER}\n"


def update_zshrc(zshrc: Path, key: str) -> None:
    zshrc = zshrc.expanduser()
    existing = zshrc.read_text(encoding="utf-8") if zshrc.exists() else ""
    block = _marker_block(key)

    start = existing.find(START_MARKER)
    end = existing.find(END_MARKER)
    if (start == -1) != (end == -1) or (start != -1 and end < start):
        raise S2KeyConfigError("Existing S2_API_KEY marker block in zshrc is incomplete.")

    if start == -1:
        separator = "" if not existing or existing.endswith("\n") else "\n"
        updated = existing + separator + block
    else:
        end += len(END_MARKER)
        if end < len(existing) and existing[end] == "\n":
            end += 1
        updated = existing[:start] + block + existing[end:]

    zshrc.parent.mkdir(parents=True, exist_ok=True)
    zshrc.write_text(updated, encoding="utf-8")


def update_shell_config(zshrc: Path, key: str) -> Path:
    preferred = zshrc.expanduser()
    try:
        update_zshrc(preferred, key)
        return preferred
    except PermissionError:
        if preferred != DEFAULT_ZSHRC.expanduser():
            raise
        fallback = preferred.with_name(".zshenv")
        update_zshrc(fallback, key)
        return fallback


def smoke_check(key: str) -> tuple[bool, str]:
    api_request = request.Request(
        SMOKE_URL,
        headers={
            "Accept": "application/json",
            "x-api-key": key,
        },
    )
    try:
        with request.urlopen(api_request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
            records = payload.get("data", []) if isinstance(payload, dict) else []
            return True, f"Semantic Scholar smoke status={response.status} record_count={len(records)}"
    except error.HTTPError:
        return False, "Semantic Scholar smoke error=HTTPError"
    except error.URLError:
        return False, "Semantic Scholar smoke error=URLError"
    except TimeoutError:
        return False, "Semantic Scholar smoke error=TimeoutError"
    except json.JSONDecodeError:
        return False, "Semantic Scholar smoke error=JSONDecodeError"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Configure the Semantic Scholar S2_API_KEY for Design Scientist.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to a file containing the S2 API key.")
    parser.add_argument("--zshrc", type=Path, default=DEFAULT_ZSHRC, help="Path to the zshrc file to update.")
    parser.add_argument("--smoke", action="store_true", help="Send a minimal Semantic Scholar request after updating.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        key = parse_s2_api_key(args.input.expanduser().read_text(encoding="utf-8"))
        configured_path = update_shell_config(args.zshrc, key)
    except S2KeyConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Error: could not read or update S2 API key configuration ({exc.__class__.__name__}).", file=sys.stderr)
        return 1

    print(f"Configured S2_API_KEY marker block in {configured_path}.")

    if args.smoke:
        ok, message = smoke_check(key)
        print(message)
        if not ok:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
