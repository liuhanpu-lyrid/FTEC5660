#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    from pydantic import BaseModel, Field
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    class ReceiptExtraction(BaseModel):
        final_payment: str = Field(
            description=(
                "The final amount actually paid by the customer AFTER discounts "
                "and AFTER rounding. Return only the HKD monetary amount, "
                "without a currency symbol."
            )
        )
        subtotal: str = Field(
            description=(
                "The receipt SUBTOTAL after discounts but BEFORE rounding. "
                "Return only the HKD monetary amount, without a currency symbol."
            )
        )
        discounts: list[str] = Field(
            description=(
                "All actual discount, promotion, coupon, member, app, "
                "packaging-damage, or percentage-discount monetary amounts. "
                "Return each monetary discount as a positive HKD amount. "
                "Do not include percentages and do not include rounding."
            )
        )
        rounding: str | None = Field(
            description=(
                "The ROUNDING adjustment shown on the receipt, including its sign. "
                "Return null if no rounding line exists."
            )
        )

    llm = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
        max_tokens=800,
        timeout=60,
        max_retries=2,
        reasoning_effort="none",
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a precise information-extraction system for Hong Kong supermarket receipts.

Your job is to READ the receipt, not to calculate totals across multiple receipts.

Carefully distinguish these concepts:

1. final_payment
   - The amount the customer ultimately paid AFTER discounts and AFTER ROUNDING.
   - It may appear next to a payment method such as OCTOPUS, VISA, Mastercard,
     card payment, cash payment, NET TOTAL, AMOUNT DUE, or similar.
   - Do NOT confuse it with SUBTOTAL, cash tendered, amount received, change,
     card balance, loyalty points, receipt number, date, time, or item quantity.

2. subtotal
   - The subtotal AFTER discounts have already been applied but BEFORE ROUNDING.
   - Prefer an amount explicitly labelled SUBTOTAL or an equivalent subtotal line.

3. discounts
   - Extract every actual monetary discount / promotion / coupon amount.
   - Include member discounts, app discounts, promotional reductions,
     packaging-damage discounts and percentage discounts.
   - Always return the MONETARY discount amount as a positive value.
   - Example: if the receipt says "5% OFF   -5.39", return "5.39".
   - Never return "5" from the 5% percentage.
   - ROUNDING is NOT a discount.
   - If there are no discounts, return an empty list.

4. rounding
   - Extract the receipt's ROUNDING adjustment separately.
   - Preserve whether it is positive or negative.
   - ROUNDING must never be included in discounts.

Copy monetary values accurately from the receipt.
Do not invent values that are not supported by the receipt.
""".strip(),
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": (
                            "Extract the required financial information from this "
                            "single supermarket receipt."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "{image_url}",
                            "detail": "high",
                        },
                    },
                ],
            ),
        ]
    )

    structured_llm = llm.with_structured_output(
        ReceiptExtraction,
        method="function_calling",
    )

    return prompt | structured_llm


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    def to_decimal(value: Any) -> Decimal:
        text = str(value).strip()
        text = (
            text.replace("HK$", "")
            .replace("$", "")
            .replace(",", "")
            .strip()
        )

        try:
            return Decimal(text)
        except InvalidOperation as exc:
            raise ValueError(
                f"Invalid monetary value returned by model: {value!r}"
            ) from exc

    total_paid = Decimal("0.00")
    total_without_discount = Decimal("0.00")

    for image in images:
        receipt = chain.invoke(
            {
                "image_url": image_data_url(image)
            }
        )

        final_payment = to_decimal(receipt.final_payment)
        subtotal = to_decimal(receipt.subtotal)

        discount_total = sum(
            (
                abs(to_decimal(discount))
                for discount in receipt.discounts
            ),
            Decimal("0.00"),
        )

        total_paid += final_payment
        total_without_discount += subtotal + discount_total

    total_paid = total_paid.quantize(Decimal("0.01"))
    total_without_discount = total_without_discount.quantize(
        Decimal("0.01")
    )

    return {
        QUERY_1: f"HK${total_paid:.2f}",
        QUERY_2: f"HK${total_without_discount:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
