# TypeSafe / Jev local end-to-end verification

This opt-in suite makes **17 real Jev API calls** and checks their complete path through a local TraceRoot development stack. It requires the DECISION ingestion/UI support merged in [traceroot#2538](https://github.com/traceroot-ai/traceroot/pull/2538). It is separate from the offline CI tests.

## Scenarios

| Example | Real calls | Source |
| --- | ---: | --- |
| Synchronous quickstart | 1 | https://docs.typesafe.ai/sdk/python |
| Asynchronous quickstart | 1 | https://docs.typesafe.ai/sdk/python |
| Line-by-line semantic search | 4 | https://docs.typesafe.ai/cookbooks/semantic_find |
| Structure recovery: stitch then classify/render | 2 | https://docs.typesafe.ai/cookbooks/autoformat |
| Email, phone and invoice value extraction | 9 | https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook |

The scenario code is adapted from these official examples. Changes: wrap each example in a callable; inject a client that delegates to the real SDK and records its real results; remove the response cache and notebook/playground display; use `jev-latest` instead of the older cookbook model; fetch pinned document fixtures with SHA-256 checks. The structure cookbook's optional alternative-wording appendix is omitted. No network response is mocked.

The suite checks exactly one DECISION span per call, original request state/questions, complete response answers, resolved model, API token counts, backend-derived total tokens, backend-calculated input-only Jev price, requested alias metadata, and session/user propagation. Stored input/output are compared with the complete emitted JSON, including the large search document. The actual API HTTP response is compared with the SDK result. A separate logic-only replay of the captured real results checks example output, ignoring elapsed-time text; it makes no additional API calls and is not a second independent live run.

## Run against local `make dev`

1. Start current `traceroot-ai/traceroot` with `make dev`. Create a test project and a project API key in that **local** instance. A cloud project key cannot authenticate to the local instance.
2. Put `TYPESAFE_API_KEY` and the **local** `TRACEROOT_API_KEY` in a private env file. Do not commit it.
3. Build this PR and install its wheel into an empty environment without its lockfile:

```bash
uv build --wheel
uv venv /tmp/jev-proof --python 3.11
uv pip install --python /tmp/jev-proof/bin/python 'dist/traceroot-0.3.0-py3-none-any.whl[typesafe]' phonenumbers
uv pip check --python /tmp/jev-proof/bin/python
```

4. From outside the SDK checkout, invoke the runner using that environment. The scenario code stays in the checkout but the imported SDK must come from the installed wheel:

```bash
cd /tmp
/tmp/jev-proof/bin/python /path/to/traceroot-py/scripts/verify_typesafe_e2e.py \
  --env-file /private/jev-local.env \
  --host-url http://localhost:8000 \
  --git-ref <sdk-commit> \
  --platform-ref <local-platform-commit> \
  --output /tmp/jev-evidence
```

Expected result: five scenarios, 17 real calls, and 17 priced DECISION spans. Output contains the JSON report, each example's terminal output, a raw-observation checkpoint, and the two pinned public document fixtures. If read-back fails after the calls finish, rerun the same command with `--verify-only` to recheck the stored traces without spending more API calls. The runner refuses a non-local host. Review the report and use its trace links to capture the local UI. API keys are never included in the report.

Jev pricing is checked as input tokens × $0.042 / 1,000,000, with free output, matching the existing `jev-1.13.0` price row. If pricing or alias resolution changes, investigate the mismatch and record the actual supported version/rate rather than accepting an arbitrary nonzero cost.
