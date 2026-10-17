"""Adapted from https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook

No response cache; model changed to jev-latest; notebook display becomes terminal output.
make_client/load_fixture are supplied by the end-to-end runner.
"""


def run(make_client, make_async_client, load_fixture):
    import re
    from decimal import Decimal

    import phonenumbers
    from typesafe_sdk import Choice, Noul

    typesafe_model = "jev-latest"
    none = "none"  # the escape hatch on every selection: "none of the candidates fits"

    # base_url defaults to https://api.typesafe.ai/ ; the env override points at another deployment.
    ts = make_client()

    email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    phone_re = re.compile(r"\(?\+?\d[\d\s()\-.]{6,}\d")
    money_re = re.compile(r"[$€£¥]\s?\d[\d,]*(?:\.\d{2})?")

    def find(pattern: re.Pattern, text: str) -> list[str]:
        """Code-side candidate finder: recall-tuned regex, deduped, in document order."""
        seen: set[str] = set()
        out: list[str] = []
        for match in pattern.findall(text):
            span = match.strip()
            if span and span not in seen:
                seen.add(span)
                out.append(span)
        return out

    def pick(document: str, candidates: list[str], question: str) -> dict:
        """TypeSafe selects which found span plays the role. Returns {choice, confidence}.

        The options ARE the candidate spans, so ``choice`` is a verbatim copy of one of them (or the
        ``none`` hatch) - the model chooses, code owns the string."""
        criteria = {c: None for c in candidates} | {none: "None of these is the requested value."}
        answer = ts.system_one(
            state=document,
            questions={"pick": Choice(instructions=question, criteria=criteria)},
            model=typesafe_model,
        ).answers["pick"]
        return {"choice": answer.choice, "confidence": answer.confidence}

    def classify(document: str, question: str, options: list[str]) -> dict:
        """A small Choice over a fixed label set (currency, country, ...). Returns {choice, confidence}."""
        answer = ts.system_one(
            state=document,
            questions={"q": Choice(instructions=question, criteria={o: None for o in options})},
            model=typesafe_model,
        ).answers["q"]
        return {"choice": answer.choice, "confidence": answer.confidence}

    def is_true(document: str, question: str) -> float:
        """A yes/no Noul. Returns P(yes)."""
        return (
            ts.system_one(
                state=document,
                questions={"q": Noul(instructions=question)},
                model=typesafe_model,
            )
            .answers["q"]
            .noul
        )

    email_doc = """From: Dana Whit <dana.whit@acme-corp.com>
    To: billing@acme-corp.com
    Cc: orders@acme-corp.com
    Reply-To: dana.personal@gmail.com

    Hi team - please don't use the billing alias for this one. Send my receipt to my
    personal address instead. Thanks, Dana."""

    emails = find(email_re, email_doc)
    receipt = pick(
        email_doc, emails, "Which email address does the sender want their receipt sent to?"
    )
    sender = pick(
        email_doc, emails, "Which email address did this message come from (the From line)?"
    )

    print("candidates :", emails)
    # code copies the picked value verbatim and normalizes (lowercase); it never re-types it
    print(f"receipt -> : {receipt['choice'].lower():<28} (conf {receipt['confidence']:.2f})")
    print(f"sender  -> : {sender['choice'].lower():<28} (conf {sender['confidence']:.2f})")

    phone_doc = """Reach our San Francisco office at these numbers: main desk (415) 555-0199,
    billing fax (415) 555-0142, and my direct cell (415) 555-0177. Call the cell if it's urgent."""

    phones = find(phone_re, phone_doc)
    mobile = pick(phone_doc, phones, "Which of these is the direct mobile / cell number?")
    region = classify(
        phone_doc,
        "In what country is this office located?",
        ["US", "GB", "DE", "FR", "CA", "AU"],
    )

    # code copies the picked value and normalizes it with the model-supplied country
    parsed = phonenumbers.parse(mobile["choice"], region["choice"])
    e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)

    print("candidates :", phones)
    print(f"mobile  -> : {mobile['choice']}  (conf {mobile['confidence']:.2f})")
    print(f"country -> : {region['choice']}  (conf {region['confidence']:.2f})")
    print(f"E.164   -> : {e164}")

    money_doc = """Invoice INV-2087.
    Subtotal: $1,200.00
    Sales tax: $115.50
    Total due: $1,315.50
    A $50.00 courtesy credit from last month has already been applied."""

    amounts = find(money_re, money_doc)
    currency = classify(
        money_doc,
        "What currency are these amounts in?",
        ["USD", "EUR", "GBP", "JPY", "CAD"],
    )
    total = pick(money_doc, amounts, "Which amount is the total the customer must pay?")
    credit = pick(money_doc, amounts, "Which amount is the courtesy credit that was applied?")

    def to_decimal(value: str) -> Decimal:
        """Copy the picked value and parse the number in code (US grouping/decimal here)."""
        return Decimal(re.sub(r"[^\d.]", "", value))

    for label, chosen in [("total due", total), ("credit", credit)]:
        is_credit = is_true(
            money_doc,
            f"Is the amount {chosen['choice']} a credit or refund to the customer, not a charge?",
        )
        kind = "credit" if is_credit > 0.5 else "charge"
        print(
            f"{label:<10}: {chosen['choice']:<10} -> {to_decimal(chosen['choice'])} {currency['choice']} "
            f"({kind}, P(credit)={is_credit:.2f})"
        )
    print("\ncandidates :", amounts)
