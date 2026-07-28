"""DS-1: client-generated stable ids (ULID with typed prefixes)."""

import re

from traceroot.eval.ids import new_dataset_id, new_run_id, new_test_case_id, ulid

_ULID = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")  # Crockford base32, 26 chars


def test_ulid_shape():
    u = ulid()
    assert _ULID.match(u), u


def test_prefixes():
    assert new_dataset_id().startswith("ds_")
    assert new_test_case_id().startswith("tc_")
    assert new_run_id().startswith("run_")
    assert _ULID.match(new_dataset_id().split("_", 1)[1])


def test_unique():
    ids = {new_test_case_id() for _ in range(1000)}
    assert len(ids) == 1000


def test_time_sortable():
    # ULIDs generated later sort >= earlier ones (millisecond time prefix).
    a = ulid()
    b = ulid()
    # same-ms ties are allowed; never earlier-after-later beyond the random tail
    assert a[:10] <= b[:10]
