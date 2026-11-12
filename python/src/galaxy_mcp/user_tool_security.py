import re
from typing import Any, TypedDict


class UserToolSecurityDiagnostic(TypedDict):
    path: str
    code: str
    message: str


_TEMPLATE_BLOCK_RE = re.compile(r"\$\(([\s\S]*?)\)")
_DIRECT_INPUT_EXPR_RE = re.compile(r"^\s*inputs\.([A-Za-z_][A-Za-z0-9_]*)\s*$")
_SCRIPT_SUFFIXES = (".bash", ".cjs", ".js", ".mjs", ".pl", ".py", ".r", ".rb", ".sh")


def _direct_input_refs(value: Any) -> set[str]:
    if not isinstance(value, str):
        return set()
    refs: set[str] = set()
    for block in _TEMPLATE_BLOCK_RE.findall(value):
        match = _DIRECT_INPUT_EXPR_RE.fullmatch(block)
        if match:
            refs.add(match.group(1))
    return refs


def user_tool_security_diagnostics(tool: dict[str, Any]) -> list[UserToolSecurityDiagnostic]:
    text_inputs = {
        item["name"]
        for item in tool.get("inputs", [])
        if isinstance(item, dict)
        and item.get("type") == "text"
        and isinstance(item.get("name"), str)
    }
    diagnostics: list[UserToolSecurityDiagnostic] = []

    def check(value: Any, path: str) -> None:
        for name in sorted(_direct_input_refs(value) & text_inputs):
            diagnostics.append(
                {
                    "path": path,
                    "code": "dynamic_tool.unsafe_text_interpolation",
                    "message": (
                        f"free-text input 'inputs.{name}' is substituted as executable text; "
                        "serialize it into a config file with JSON.stringify and read it as data"
                    ),
                }
            )

    check(tool.get("shell_command"), "shell_command")
    for index, configfile in enumerate(tool.get("configfiles", [])):
        if not isinstance(configfile, dict):
            continue
        filename = configfile.get("filename")
        if isinstance(filename, str) and filename.lower().endswith(_SCRIPT_SUFFIXES):
            check(configfile.get("content"), f"configfiles.{index}.content")
    return diagnostics
