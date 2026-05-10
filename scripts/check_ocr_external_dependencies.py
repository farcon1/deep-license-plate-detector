from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def _check_python_module(module_name: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "name": module_name,
        "available": False,
        "version": "",
        "error": "",
    }

    try:
        module = importlib.import_module(module_name)
        result["available"] = True
        result["version"] = str(getattr(module, "__version__", "unknown"))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def _check_tesseract_binary() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "name": "tesseract_binary",
        "available": False,
        "path": "",
        "version": "",
        "error": "",
    }

    path = shutil.which("tesseract")
    if not path:
        result["error"] = "tesseract executable not found in PATH"
        return result

    result["path"] = path

    try:
        proc = subprocess.run(
            ["tesseract", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
        output = (proc.stdout or proc.stderr or "").strip()
        first_line = output.splitlines()[0] if output else ""
        result["available"] = proc.returncode == 0
        result["version"] = first_line
        if proc.returncode != 0:
            result["error"] = output
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def _markdown_table(rows: List[Dict[str, Any]]) -> str:
    header = "| Компонент | Доступен | Версия / путь | Ошибка |\n|---|---:|---|---|"
    body = []

    for row in rows:
        name = str(row.get("name", ""))
        available = "да" if row.get("available") else "нет"
        version = str(row.get("version", ""))

        if row.get("path"):
            version = f"{version}; {row.get('path')}" if version else str(row.get("path"))

        error = str(row.get("error", "")).replace("\n", " ")
        body.append(f"| {name} | {available} | {version} | {error} |")

    return "\n".join([header, *body])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check external OCR baseline dependencies: Tesseract, EasyOCR, PaddleOCR."
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="outputs/ocr_external_baselines/dependency_check.json",
        help="Path to save JSON dependency check report.",
    )
    args = parser.parse_args()

    checks: List[Dict[str, Any]] = [
        _check_python_module("pytesseract"),
        _check_tesseract_binary(),
        _check_python_module("easyocr"),
        _check_python_module("paddleocr"),
        _check_python_module("paddle"),
    ]

    try:
        import paddle

        checks.append(
            {
                "name": "paddle_cuda",
                "available": bool(paddle.device.is_compiled_with_cuda()),
                "version": f"device={paddle.device.get_device()}",
                "error": "",
            }
        )
    except Exception as exc:
        checks.append(
            {
                "name": "paddle_cuda",
                "available": False,
                "version": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps({"checks": checks}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n# OCR external baseline dependency check\n")
    print(_markdown_table(checks))
    print(f"\nJSON saved to: {out_json}")


if __name__ == "__main__":
    main()