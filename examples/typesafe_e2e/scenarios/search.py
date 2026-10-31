"""Adapted from https://docs.typesafe.ai/cookbooks/semantic_find

No response cache; model changed to jev-latest; notebook display becomes terminal output.
make_client/load_fixture are supplied by the end-to-end runner.
"""


def run(make_client, make_async_client, load_fixture):

    from typesafe_sdk import Choice, Noul, NoulCriteria

    typesafe_model = "jev-latest"

    client = make_client()

    gist = (
        "https://gist.githubusercontent.com/eugene-shvarts/900632789a24983d5678ffd508dd01f6"
        "/raw/cf9c2ab422d568deade949ef0a06bed6896964b9/github-tos.txt"
    )

    def fetch_document(url: str) -> str:
        return load_fixture(url)

    lines = fetch_document(gist).splitlines()

    def line_id(i: int) -> str:
        return f"L{i:03d}"

    document = "\n".join(f"{line_id(i)}| {line}" for i, line in enumerate(lines))

    def where_question(query: str) -> Choice:
        return Choice(
            instructions=f'Which line of the document contains the answer to: "{query}"?',
            criteria={line_id(i): None for i in range(len(lines))},
        )

    def exists_question(query: str) -> Noul:
        return Noul(
            instructions=f'Does any line of the document address or answer: "{query}"?',
            criteria=NoulCriteria(
                true="At least one line of the document states or directly implies the answer",
                false="No line of the document addresses this",
            ),
        )

    def _find(
        model: str,
        state: str,
        where: Choice,
        exists: Noul,
    ) -> dict:
        response = client.system_one(
            state=state,
            questions={"where": where, "exists": exists},
            model=model,
        )
        probabilities = response.answers["where"].probabilities
        return {
            "exists": response.answers["exists"].noul,
            "relevance": [probabilities.get(line_id(i), 0.0) for i in range(len(lines))],
        }

    def find(query: str) -> dict:
        return _find(
            typesafe_model,
            document,
            where_question(query),
            exists_question(query),
        )

    found, absent = 0.7, 0.35  # present answers typically read >=0.9, absent <=0.05

    def verdict(exists: float) -> str:
        if exists >= found:
            return "answered in this document"
        return "not in this document" if exists < absent else "partially addressed"

    def show(query: str, top: int = 4) -> dict:
        result = find(query)
        print(f'"{query}"')
        print(f"  exists {result['exists']:.2f} -> {verdict(result['exists'])}")
        ranked = sorted(range(len(lines)), key=lambda i: result["relevance"][i], reverse=True)
        for i in ranked[:top]:
            bar = "#" * max(1, round(result["relevance"][i] * 12))
            preview = lines[i][:58].rstrip()
            print(f"  {line_id(i)}  {result['relevance'][i]:.2f}  {bar:<12}  {preview}")
        return result

    print(f"{len(lines)} lines, {len(document):,} characters\n")
    show("who owns the code I upload?")
    print()
    show("can GitHub kick me off the platform without warning?")
    print()
    show("do I have to take disputes to arbitration?", top=2)
    print()
    show("can minors use GitHub with parental permission?", top=2)
