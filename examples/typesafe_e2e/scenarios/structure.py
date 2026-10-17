"""Adapted from https://docs.typesafe.ai/cookbooks/autoformat

No response cache; model changed to jev-latest; notebook display becomes terminal output.
make_client/load_fixture are supplied by the end-to-end runner.
"""


def run(make_client, make_async_client, load_fixture):
    import re
    from time import perf_counter

    from typesafe_sdk import Choice, Noul, NoulCriteria

    typesafe_model = "jev-latest"
    client = make_client()

    gist = (
        "https://gist.githubusercontent.com/eugene-shvarts/6df7daf97233bf92bcdd6b386a0fa561"
        "/raw/5da03690611fb6ddcbaabdb91fb9f91d9751b113/build-memo.txt"
    )

    def fetch_document(url: str) -> str:
        return load_fixture(url)

    raw = fetch_document(gist)
    print(raw[:560])

    def to_lines(text: str) -> list[dict]:
        lines, gap = [], False
        for raw in text.split("\n"):
            stripped = re.sub(r"[\t ]+", " ", raw).strip()
            if not stripped:
                gap = bool(lines)  # a leading blank is not a break
                continue
            lines.append({"text": stripped, "gap": gap})
            gap = False
        return lines

    def tag(items: list[dict], prefix: str) -> str:
        return "\n".join(
            f"{chr(10) if item['gap'] else ''}{prefix}{i:03d}| {item['text']}"
            for i, item in enumerate(items)
        )

    def line_id(i: int) -> str:
        return f"L{i:03d}"

    def block_id(i: int) -> str:
        return f"B{i:03d}"

    lines = to_lines(raw)
    print(f"{len(lines)} non-blank lines. The model sees, e.g.:")
    print("\n".join(tag(lines, "L").splitlines()[19:24]))

    def join_question(i: int) -> Noul:
        return Noul(
            instructions=f"Does line {line_id(i)} pick up mid-sentence, continuing a sentence left unfinished at the end of line {line_id(i - 1)}?",
            criteria=NoulCriteria(
                true="The line starts in the middle of a sentence that began on the previous line - the line break tore the sentence apart",
                false="The line begins a new sentence, item, heading, or thought of its own",
            ),
        )

    def stitch(wording: str = "mid-sentence") -> dict:
        make = join_question
        questions = {line_id(i): make(i) for i in range(1, len(lines)) if not lines[i]["gap"]}
        started = perf_counter()
        response = client.system_one(
            state=tag(lines, "L"), questions=questions, model=typesafe_model
        )
        return {
            "joins": [
                response.answers[line_id(i)].noul if line_id(i) in response.answers else 0.0
                for i in range(len(lines))
            ],
            "seconds": round(perf_counter() - started, 2),
            "usage": [response.usage.input_tokens, response.usage.output_tokens],
        }

    result = stitch()
    print(
        f"{sum(1 for line in lines if not line['gap']) - 1} pair questions, one request, "
        f"{result['seconds']}s"
    )

    join_after_dangling, join_after_terminal = 0.2, 0.5

    def ends_terminal(text: str) -> bool:
        return re.search(r'[.!?:;…]["\')\]]*$', text) is not None

    def merge(joins: list[float]) -> list[dict]:
        blocks = []
        for i, line in enumerate(lines):
            bar = (
                join_after_terminal
                if i and ends_terminal(lines[i - 1]["text"])
                else join_after_dangling
            )
            if blocks and not line["gap"] and joins[i] >= bar:
                blocks[-1]["text"] += " " + line["text"]
                blocks[-1]["lines"].append(i)
            else:
                blocks.append({"text": line["text"], "lines": [i], "gap": line["gap"]})
        return blocks

    blocks = merge(result["joins"])
    healed = len(lines) - len(blocks)
    print(f"{len(lines)} lines -> {len(blocks)} blocks ({healed} line breaks healed)")
    for i, block in enumerate(blocks):
        n = len(block["lines"])
        print(f"{block_id(i)}  {n} line{'s' if n > 1 else ' '}  {block['text'][:62]}")

    type_criteria = {
        "heading": "A short label or title that names the document or the section that follows it - not a full sentence of content",
        "paragraph": "Running prose: one or more complete sentences of explanatory or narrative text",
        "list_item": "One entry in a list of parallel items - an ingredient, a feature, a task, an attendee; reads as one of several sibling entries",
        "quote": "Words attributed to a person or source - quoted speech, a citation, an excerpt someone else wrote",
        "code": "Computer code, a shell command, terminal output, or a config snippet meant to be read verbatim",
        "callout": "A warning, tip, or important note that interrupts the flow to flag something the reader must not miss",
    }
    hlevel_criteria = {
        "title": "The title of the whole document",
        "section": "A major section heading within the document",
        "subsection": "A minor heading nested under a section",
    }
    callout_criteria = {
        "note": "Neutral extra information the reader should be aware of",
        "tip": "A helpful suggestion or shortcut that makes things easier",
        "warning": "A caution about something that can go wrong or cause harm",
    }

    heading_max_chars = 90  # longer blocks can't render as headings, so don't ask

    def classify_questions(texts: list[str]) -> dict:
        questions = {}
        for i, text in enumerate(texts):
            bid = block_id(i)
            questions[f"type_{bid}"] = Choice(
                instructions=f"What kind of content is block {bid}?", criteria=type_criteria
            )
            if len(text) <= heading_max_chars:
                questions[f"hlevel_{bid}"] = Choice(
                    instructions=f"As a heading, what level would block {bid} occupy in this document's structure?",
                    criteria=hlevel_criteria,
                )
            questions[f"step_{bid}"] = Noul(
                instructions=f"Is block {bid} an instruction in a sequence where the order of the items matters?",
                criteria=NoulCriteria(
                    true="It is one step of a procedure - the items around it must happen in order",
                    false="Order is irrelevant - it is a loose collection, or not a list item at all",
                ),
            )
            questions[f"callout_{bid}"] = Choice(
                instructions=f"What kind of aside is block {bid}?", criteria=callout_criteria
            )
        return questions

    def classify(texts: list[str], gaps: list[bool]) -> dict:
        tagged = tag([{"text": t, "gap": g} for t, g in zip(texts, gaps)], "B")
        questions = classify_questions(texts)
        started = perf_counter()
        response = client.system_one(state=tagged, questions=questions, model=typesafe_model)
        judgments = []
        for i in range(len(texts)):
            bid = block_id(i)
            type_answer = response.answers[f"type_{bid}"]
            hlevel = response.answers.get(f"hlevel_{bid}")
            judgments.append(
                {
                    "type": type_answer.choice,
                    "confidence": type_answer.confidence,
                    "probabilities": type_answer.probabilities,
                    "hlevel": hlevel.choice if hlevel else "section",
                    "step": response.answers[f"step_{bid}"].noul,
                    "callout": response.answers[f"callout_{bid}"].choice,
                }
            )
        return {
            "judgments": judgments,
            "n_questions": len(questions),
            "seconds": round(perf_counter() - started, 2),
            "usage": [response.usage.input_tokens, response.usage.output_tokens],
        }

    classified = classify([b["text"] for b in blocks], [b["gap"] for b in blocks])
    for block, judgment in zip(blocks, classified["judgments"]):
        block.update(judgment)
    print(
        f"{classified['n_questions']} questions about {len(blocks)} blocks, one request, "
        f"{classified['seconds']}s\n"
    )
    print(f"{'block':<6}{'type':<11}{'conf':<6}{'companion used':<18}text")
    for i, b in enumerate(blocks):
        companion = {
            "heading": f"level={b['hlevel']}",
            "list_item": f"step={b['step']:.2f}",
            "callout": f"kind={b['callout']}",
        }.get(b["type"], "-")
        print(
            f"{block_id(i):<6}{b['type']:<11}{b['confidence']:.2f}  {companion:<18}{b['text'][:46]}"
        )

    step_threshold = 0.5
    heading_mark = {"title": "#", "section": "##", "subsection": "###"}
    callout_mark = {"note": "NOTE", "tip": "TIP", "warning": "WARNING"}

    def to_markdown(blocks: list[dict]) -> str:
        groups = []
        for b in blocks:
            if b["type"] in ("list_item", "code") and groups and groups[-1][0] == b["type"]:
                groups[-1][1].append(b)
            else:
                groups.append((b["type"], [b]))
        parts = []
        for kind, items in groups:
            if kind == "list_item":
                ordered = sum(b["step"] for b in items) / len(items) >= step_threshold
                parts.append(
                    "\n".join(
                        f"{n + 1}. {b['text']}" if ordered else f"- {b['text']}"
                        for n, b in enumerate(items)
                    )
                )
            elif kind == "code":
                parts.append("```\n" + "\n".join(b["text"] for b in items) + "\n```")
            elif kind == "heading":
                parts.append(f"{heading_mark[items[0]['hlevel']]} {items[0]['text']}")
            elif kind == "quote":
                parts.append(f"> {items[0]['text']}")
            elif kind == "callout":
                parts.append(f"> [!{callout_mark[items[0]['callout']]}]\n> {items[0]['text']}")
            else:
                parts.append(items[0]["text"])
        return "\n\n".join(parts) + "\n"

    markdown = to_markdown(blocks)
    print(markdown)
