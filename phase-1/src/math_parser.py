from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


NUMBER_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "too": "2",
    "to": "2",
    "three": "3",
    "four": "4",
    "for": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "ate": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}

VARIABLE_WORDS = {
    "x": "x",
    "ex": "x",
    "axe": "x",
    "y": "y",
    "why": "y",
    "z": "z",
    "zed": "z",
    "zee": "z",
    "a": "a",
    "b": "b",
    "be": "b",
    "bee": "b",
    "c": "c",
    "see": "c",
    "d": "d",
    "e": "e",
    "n": "n",
    "m": "m",
    "t": "t",
}

GREEK_WORDS = {
    "alpha": r"\alpha",
    "beta": r"\beta",
    "theta": r"\theta",
    "lambda": r"\lambda",
    "pi": r"\pi",
}

FUNCTION_WORDS = {
    "sine": r"\sin",
    "sin": r"\sin",
    "cosine": r"\cos",
    "cos": r"\cos",
    "tangent": r"\tan",
    "tan": r"\tan",
    "log": r"\log",
}


@dataclass
class ParseResult:
    spoken: str
    normalized: str
    latex: str
    warnings: list[str]


def normalize_spoken(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("−", " minus ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_math_text(text: str) -> ParseResult:
    normalized = normalize_spoken(text)
    tokens = normalized.split()
    output: list[str] = []
    warnings: list[str] = []
    i = 0

    while i < len(tokens):
        token = tokens[i]
        next_token = tokens[i + 1] if i + 1 < len(tokens) else ""

        if _match(tokens, i, ["is", "equal", "to"]):
            output.append("=")
            i += 3
            continue

        if _match(tokens, i, ["equal", "to"]):
            output.append("=")
            i += 2
            continue

        if token in {"equals", "equal"}:
            output.append("=")
            i += 1
            continue

        if token == "plus":
            output.append("+")
            i += 1
            continue

        if token == "minus":
            output.append("-")
            i += 1
            continue

        if token in {"times", "multiplied"}:
            output.append(r"\times")
            i += 2 if token == "multiplied" and next_token == "by" else 1
            continue

        if _match(tokens, i, ["divided", "by"]):
            output.append("/")
            i += 2
            continue

        if token == "by":
            output.append("/")
            warnings.append("Used '/' for standalone 'by'. Say 'divided by' for clearer intent.")
            i += 1
            continue

        if token == "over":
            output.append("/")
            warnings.append("Used '/' for 'over'. Grouped fractions will need stronger parsing.")
            i += 1
            continue

        if token == "squared":
            _apply_power(output, "2", warnings)
            i += 1
            continue

        if token == "cubed":
            _apply_power(output, "3", warnings)
            i += 1
            continue

        if _match(tokens, i, ["to", "the", "power", "of"]):
            power, consumed = _read_atom(tokens, i + 4)
            if power:
                _apply_power(output, power, warnings)
                i += 4 + consumed
            else:
                warnings.append("Could not read exponent after 'to the power of'.")
                i += 4
            continue

        if _match(tokens, i, ["square", "root", "of"]):
            atom, consumed = _read_atom(tokens, i + 3)
            if atom:
                output.append(r"\sqrt{" + atom + "}")
                i += 3 + consumed
            else:
                output.append(r"\sqrt{}")
                warnings.append("Could not read term after 'square root of'.")
                i += 3
            continue

        atom, consumed = _read_atom(tokens, i)
        if atom:
            output.append(atom)
            i += consumed
            continue

        warnings.append(f"Unrecognized token: '{token}'")
        output.append(token)
        i += 1

    latex = _format_latex(output)
    return ParseResult(
        spoken=text,
        normalized=normalized,
        latex=latex,
        warnings=warnings,
    )


def _read_atom(tokens: list[str], index: int) -> tuple[Optional[str], int]:
    if index >= len(tokens):
        return None, 0

    token = tokens[index]

    if token in NUMBER_WORDS:
        return NUMBER_WORDS[token], 1

    if re.fullmatch(r"\d+", token):
        return token, 1

    if token in VARIABLE_WORDS:
        return VARIABLE_WORDS[token], 1

    if token in GREEK_WORDS:
        return GREEK_WORDS[token], 1

    if token in FUNCTION_WORDS:
        return FUNCTION_WORDS[token], 1

    return None, 0


def _apply_power(output: list[str], power: str, warnings: list[str]) -> None:
    if not output:
        warnings.append(f"Dropped exponent ^{power} because there was no previous term.")
        return

    previous = output.pop()
    output.append(f"{previous}^{power}")


def _format_latex(parts: list[str]) -> str:
    text = " ".join(parts)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _match(tokens: list[str], index: int, phrase: list[str]) -> bool:
    return tokens[index : index + len(phrase)] == phrase


if __name__ == "__main__":
    import sys

    phrase = " ".join(sys.argv[1:]) or "x squared minus y squared is equal to four"
    result = parse_math_text(phrase)
    print(result.latex)
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")
