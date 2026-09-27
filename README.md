# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution

```mermaid
flowchart TD
    A[Receipt Images] --> B[DeepSeek V4 Flash Vision]
    B --> C[Structured Receipt Extraction]
    C --> D[Final Payment]
    C --> E[Subtotal]
    C --> F[Discounts]
    C --> G[Rounding]
    D --> H[Python Decimal Aggregation]
    E --> H
    F --> H
    H --> I[Question 1: Total Amount Paid]
    H --> J[Question 2: Total Without Discounts]
    I --> K[Strict HKD Output]
    J --> K
```

The solution processes each receipt independently using the required `deepseek-v4-flash-vision-exp` model through LangChain. The vision model extracts structured fields for the final payment, subtotal, discounts, and rounding. The model is used only for visual and semantic extraction, while all financial aggregation is performed deterministically in Python using `Decimal`. For the first query, the program sums the final payments after rounding. For the second query, it adds all extracted discount amounts back to each subtotal while deliberately excluding rounding. The final responses are formatted as a single HKD amount so that they satisfy the automated grader's output requirements.

> to students: please fill your solution description here.

