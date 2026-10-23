"""From https://docs.typesafe.ai/sdk/python; only client factory is injected."""


def run(make_client, make_async_client, load_fixture):
    from typesafe_sdk import Choice, Noul, Score

    async def main() -> None:
        async with make_async_client() as client:
            response = await client.system_one(
                state={"document": "I was charged twice. Please fix this ASAP."},
                questions={
                    "billing": Noul(instructions="Is this ticket about billing?"),
                    "tone": Choice(
                        instructions="What is the customer's tone?",
                        criteria={"calm": None, "frustrated": None, "angry": None},
                    ),
                    "urgency": Score(
                        instructions="How urgent is this ticket?",
                        criteria=["can wait", "this week", "today"],
                    ),
                },
            )

        print(response.nouls["billing"].noul)
        print(response.choices["tone"].choice)
        print(response.scores["urgency"].score)

    import asyncio

    asyncio.run(main())
