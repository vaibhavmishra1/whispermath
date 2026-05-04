# LaTeX Verbalizer Prompt

You convert LaTeX math expressions into natural spoken math variants.

The goal is to create training data for a model that maps spoken math text back to LaTeX.

Return only valid JSON with this shape:

```json
{
  "variants": [
    {
      "spoken": "x squared minus y squared equals four",
      "style": "casual",
      "notes": "compact common speech"
    }
  ],
  "quality": {
    "usable": true,
    "reason": "short algebraic equation"
  }
}
```

Rules:

- Generate spoken text only, not explanations.
- Spoken text should sound like a person saying the formula aloud.
- Generate 3 to 6 variants when the formula is usable.
- Include both casual and explicit variants when useful.
- Preserve mathematical meaning.
- Prefer plain lowercase speech.
- Do not include punctuation unless it helps speech clarity.
- Do not invent a different formula.
- If the LaTeX is too complex, malformed, or too notation-heavy for reliable speech, set `usable` to false and return no variants.

Few-shot examples:

LaTeX:

```latex
x^2 - y^2 = 4
```

JSON:

```json
{
  "variants": [
    {
      "spoken": "x squared minus y squared equals four",
      "style": "casual",
      "notes": "common algebra speech"
    },
    {
      "spoken": "x to the power of two minus y to the power of two is equal to four",
      "style": "explicit",
      "notes": "power spoken explicitly"
    },
    {
      "spoken": "the difference of x squared and y squared equals four",
      "style": "teacher",
      "notes": "natural classroom phrasing"
    }
  ],
  "quality": {
    "usable": true,
    "reason": "simple algebraic equation"
  }
}
```

LaTeX:

```latex
\frac{x + 1}{y - 2}
```

JSON:

```json
{
  "variants": [
    {
      "spoken": "x plus one over y minus two",
      "style": "ambiguous",
      "notes": "natural but grouping is ambiguous"
    },
    {
      "spoken": "the quantity x plus one over the quantity y minus two",
      "style": "explicit",
      "notes": "grouping is clear"
    },
    {
      "spoken": "fraction with numerator x plus one and denominator y minus two",
      "style": "formal",
      "notes": "fraction structure is explicit"
    }
  ],
  "quality": {
    "usable": true,
    "reason": "short fraction with clear spoken variants"
  }
}
```

LaTeX:

```latex
\int_0^1 x^2 dx
```

JSON:

```json
{
  "variants": [
    {
      "spoken": "integral from zero to one of x squared d x",
      "style": "casual",
      "notes": "common calculus speech"
    },
    {
      "spoken": "the integral from zero to one of x to the power of two with respect to x",
      "style": "explicit",
      "notes": "more formal speech"
    }
  ],
  "quality": {
    "usable": true,
    "reason": "standard calculus expression"
  }
}
```
