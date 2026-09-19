"""End-to-end tests for the /weight-pick-plan endpoint."""

from decimal import Decimal

from fastapi.testclient import TestClient

from app.main import app
from app.summary import KG_TO_LB

client = TestClient(app)

GS = "\x1d"
GTIN_OK = "09501101530003"  # valid Mod-10 check digit
GTIN2 = "00000000000017"  # second valid GTIN


def plan(scans, demands, target_unit=None, as_of=None):
    body = {"scans": scans, "demands": demands}
    if target_unit is not None:
        body["target_unit"] = target_unit
    if as_of is not None:
        body["as_of"] = as_of
    return client.post("/weight-pick-plan", json=body)


def scan(gtin=GTIN_OK, expiry="250131", batch=None, weight="3103002000"):
    """Build an element-string scan with one 01/17/weight and optional 10.

    The default weight 3103 002000 is exactly 2 kg.
    """
    parts = [f"01{gtin}", f"17{expiry}"]
    if batch is not None:
        parts.append(f"10{batch}{GS}")
    parts.append(weight)
    return "".join(parts)


def demand(gtin=GTIN_OK, quantity="1", min_remaining_days=None):
    d = {"gtin": gtin, "quantity": quantity}
    if min_remaining_days is not None:
        d["min_remaining_days"] = min_remaining_days
    return d


# ---------------------------------------------------------------- happy path


def test_fefo_allocates_earliest_expiry_first():
    resp = plan(
        [
            scan(batch="LATE", expiry="250331"),
            scan(batch="EARLY", expiry="250131"),
            scan(batch="MID", expiry="250228"),
        ],
        [demand(quantity="5")],
    )
    assert resp.status_code == 200
    d = resp.json()["demands"][0]
    assert [(a["batch"], a["expiry"], a["weight"]) for a in d["allocations"]] == [
        ("EARLY", "250131", 2.0),
        ("MID", "250228", 2.0),
        ("LATE", "250331", 1.0),
    ]
    assert d["fulfilled"] == 5.0
    assert d["shortage"] == 0.0


def test_last_group_may_be_partially_allocated():
    resp = plan(
        [scan(batch="A"), scan(batch="B", expiry="250228")],
        [demand(quantity="3")],
    )
    d = resp.json()["demands"][0]
    assert [(a["batch"], a["weight"]) for a in d["allocations"]] == [
        ("A", 2.0),
        ("B", 1.0),
    ]
    assert d["fulfilled"] == 3.0
    assert d["shortage"] == 0.0


def test_shortage_when_supply_insufficient():
    resp = plan([scan(), scan()], [demand(quantity="10")])
    d = resp.json()["demands"][0]
    # identical scans share one (gtin, batch, expiry) group of 4 kg
    assert d["allocations"] == [{"batch": None, "expiry": "250131", "weight": 4.0}]
    assert d["fulfilled"] == 4.0
    assert d["shortage"] == 6.0


def test_empty_scans_produce_full_shortage():
    resp = plan([], [demand(quantity="2.5"), demand(gtin=GTIN2, quantity="1")])
    assert resp.status_code == 200
    body = resp.json()
    assert body["target_unit"] == "kg"
    first, second = body["demands"]
    assert first["fulfilled"] == 0.0
    assert first["shortage"] == 2.5
    assert first["allocations"] == []
    assert second["fulfilled"] == 0.0
    assert second["shortage"] == 1.0
    assert second["allocations"] == []


def test_empty_demands_and_scans():
    resp = plan([], [])
    assert resp.status_code == 200
    assert resp.json() == {"target_unit": "kg", "demands": []}


def test_demand_for_unscanned_gtin_is_full_shortage():
    resp = plan([scan()], [demand(gtin=GTIN2, quantity="2")])
    d = resp.json()["demands"][0]
    assert d["fulfilled"] == 0.0
    assert d["shortage"] == 2.0
    assert d["allocations"] == []


def test_day_00_expiry_sorts_as_last_day_of_month():
    # 250200 denotes 2025-02-28, so 250215 expires first.
    resp = plan(
        [scan(batch="MONTH_END", expiry="250200"), scan(batch="MID", expiry="250215")],
        [demand(quantity="3")],
    )
    d = resp.json()["demands"][0]
    assert [a["batch"] for a in d["allocations"]] == ["MID", "MONTH_END"]


def test_day_00_ties_with_explicit_last_day_keep_first_seen_order():
    # 2025 is not a leap year: 250200 and 250228 are both 2025-02-28.
    resp = plan(
        [scan(batch="ZERO", expiry="250200"), scan(batch="EXPLICIT", expiry="250228")],
        [demand(quantity="3")],
    )
    d = resp.json()["demands"][0]
    assert [a["batch"] for a in d["allocations"]] == ["ZERO", "EXPLICIT"]


def test_day_00_leap_year_month_end():
    # 2024 is a leap year: 240200 and 240229 are both 2024-02-29.
    resp = plan(
        [scan(batch="EXPLICIT", expiry="240229"), scan(batch="ZERO", expiry="240200")],
        [demand(quantity="3")],
    )
    d = resp.json()["demands"][0]
    assert [a["batch"] for a in d["allocations"]] == ["EXPLICIT", "ZERO"]


def test_same_expiry_date_keeps_batch_first_seen_order():
    resp = plan(
        [scan(batch="B2"), scan(batch="B1"), scan(batch="B3")],
        [demand(quantity="5")],
    )
    d = resp.json()["demands"][0]
    assert [a["batch"] for a in d["allocations"]] == ["B2", "B1", "B3"]


def test_same_gtin_batch_expiry_scans_are_aggregated():
    resp = plan(
        [scan(batch="A"), scan(batch="A", weight="3103001000")],
        [demand(quantity="2.5")],
    )
    d = resp.json()["demands"][0]
    assert d["allocations"] == [{"batch": "A", "expiry": "250131", "weight": 2.5}]
    assert d["shortage"] == 0.0


def test_same_batch_with_different_expiries_forms_two_groups():
    resp = plan(
        [scan(batch="A", expiry="250228"), scan(batch="A", expiry="250131")],
        [demand(quantity="3")],
    )
    allocs = resp.json()["demands"][0]["allocations"]
    assert [(a["expiry"], a["weight"]) for a in allocs] == [
        ("250131", 2.0),
        ("250228", 1.0),
    ]


def test_zero_weight_group_is_skipped():
    resp = plan(
        [scan(batch="EMPTY", weight="3103000000"), scan(batch="FULL", expiry="250228")],
        [demand(quantity="1")],
    )
    allocs = resp.json()["demands"][0]["allocations"]
    assert allocs == [{"batch": "FULL", "expiry": "250228", "weight": 1.0}]


def test_demands_for_different_gtins_are_independent():
    resp = plan(
        [scan(gtin=GTIN2, expiry="250228"), scan()],
        [demand(quantity="3"), demand(gtin=GTIN2, quantity="3")],
    )
    assert resp.status_code == 200
    first, second = resp.json()["demands"]
    assert first["gtin"] == GTIN_OK
    assert first["fulfilled"] == 2.0
    assert first["shortage"] == 1.0
    assert second["gtin"] == GTIN2
    assert second["fulfilled"] == 2.0
    assert second["shortage"] == 1.0


def test_digital_link_scans_accepted():
    resp = plan(
        ["https://id.gs1.org/01/09501101530003/17/250131/3103/002000?10=B1"],
        [demand(quantity="2")],
    )
    assert resp.status_code == 200
    alloc = resp.json()["demands"][0]["allocations"][0]
    assert alloc == {"batch": "B1", "expiry": "250131", "weight": 2.0}


def test_serial_number_ai_21_is_ignored():
    resp = plan(
        [f"01{GTIN_OK}1725013121SN0001{GS}3103002000"],
        [demand(quantity="2")],
    )
    assert resp.status_code == 200
    assert resp.json()["demands"][0]["fulfilled"] == 2.0


def test_response_shape():
    resp = plan([scan(batch="A")], [demand(quantity="1")])
    body = resp.json()
    assert set(body) == {"target_unit", "demands"}
    d = body["demands"][0]
    assert set(d) == {
        "gtin",
        "quantity",
        "fulfilled",
        "shortage",
        "excluded_weight",
        "allocations",
    }
    assert set(d["allocations"][0]) == {"batch", "expiry", "weight"}


# ------------------------------------------------------- units and conversion


def test_lb_scans_converted_for_kg_target():
    # 25.13 lb * 0.45359237 = 11.3987762581 kg, demanded exactly.
    resp = plan([scan(weight="3202002513")], [demand(quantity="11.3987762581")])
    d = resp.json()["demands"][0]
    assert d["fulfilled"] == 11.398776  # six-place rendering
    assert d["shortage"] == 0.0


def test_kg_scan_fulfills_lb_demand_exactly():
    resp = plan(
        [scan(weight="3100000001")],  # 1 kg
        [demand(quantity=str(KG_TO_LB))],
        target_unit="lb",
    )
    d = resp.json()["demands"][0]
    assert d["shortage"] == 0.0
    assert d["fulfilled"] == 2.204623  # rounded only at the response


def test_mixed_units_aggregate_before_allocation():
    # one group: 2 kg + 25.13 lb = 13.3987762581 kg available
    resp = plan(
        [scan(batch="A"), scan(batch="A", weight="3202002513")],
        [demand(quantity="13")],
    )
    d = resp.json()["demands"][0]
    assert d["allocations"] == [{"batch": "A", "expiry": "250131", "weight": 13.0}]
    assert d["shortage"] == 0.0


def test_default_target_unit_is_kg():
    resp = plan([scan()], [demand(quantity="1")])
    assert resp.json()["target_unit"] == "kg"


# ------------------------------------------------- shelf-life threshold


def test_as_of_excludes_inventory_expiring_before_threshold():
    resp = plan(
        [scan(batch="OLD", expiry="250131"), scan(batch="FRESH", expiry="250228")],
        [demand(quantity="3")],
        as_of="2025-02-01",
    )
    assert resp.status_code == 200
    d = resp.json()["demands"][0]
    # OLD (2025-01-31) is before as_of and cannot fulfil the demand.
    assert d["allocations"] == [{"batch": "FRESH", "expiry": "250228", "weight": 2.0}]
    assert d["fulfilled"] == 2.0
    assert d["shortage"] == 1.0
    assert d["excluded_weight"] == 2.0


def test_expiry_on_threshold_boundary_is_eligible():
    # as_of 2025-01-15 + 16 days = 2025-01-31: the boundary date itself is usable.
    resp = plan(
        [scan(batch="EDGE", expiry="250131")],
        [demand(quantity="2", min_remaining_days=16)],
        as_of="2025-01-15",
    )
    d = resp.json()["demands"][0]
    assert d["fulfilled"] == 2.0
    assert d["shortage"] == 0.0
    assert d["excluded_weight"] == 0.0


def test_one_day_beyond_boundary_is_excluded():
    # as_of 2025-01-15 + 17 days = 2025-02-01: 2025-01-31 no longer qualifies.
    resp = plan(
        [scan(batch="EDGE", expiry="250131")],
        [demand(quantity="2", min_remaining_days=17)],
        as_of="2025-01-15",
    )
    d = resp.json()["demands"][0]
    assert d["allocations"] == []
    assert d["fulfilled"] == 0.0
    assert d["shortage"] == 2.0
    assert d["excluded_weight"] == 2.0


def test_min_remaining_days_defaults_to_zero():
    # Without the field the threshold is as_of itself; expiry on as_of qualifies.
    resp = plan([scan(expiry="250131")], [demand(quantity="2")], as_of="2025-01-31")
    d = resp.json()["demands"][0]
    assert d["fulfilled"] == 2.0
    assert d["excluded_weight"] == 0.0


def test_min_remaining_days_without_as_of_filters_nothing():
    resp = plan([scan(expiry="250131")], [demand(quantity="2", min_remaining_days=90)])
    d = resp.json()["demands"][0]
    assert d["fulfilled"] == 2.0
    assert d["excluded_weight"] == 0.0


def test_day_00_expiry_compares_as_month_end():
    # 250200 is 2025-02-28: it survives a 2025-02-28 threshold but not 2025-03-01.
    resp = plan([scan(expiry="250200")], [demand(quantity="2")], as_of="2025-02-28")
    assert resp.json()["demands"][0]["fulfilled"] == 2.0
    resp = plan([scan(expiry="250200")], [demand(quantity="2")], as_of="2025-03-01")
    d = resp.json()["demands"][0]
    assert d["fulfilled"] == 0.0
    assert d["excluded_weight"] == 2.0


def test_day_00_leap_year_month_end_against_threshold():
    # 240200 is 2024-02-29 (leap year): eligible on 2024-02-29, not on 2024-03-01.
    resp = plan([scan(expiry="240200")], [demand(quantity="2")], as_of="2024-02-29")
    assert resp.json()["demands"][0]["fulfilled"] == 2.0
    resp = plan([scan(expiry="240200")], [demand(quantity="2")], as_of="2024-03-01")
    assert resp.json()["demands"][0]["excluded_weight"] == 2.0


def test_excluded_weight_sums_all_excluded_groups():
    resp = plan(
        [
            scan(batch="E1", expiry="250131"),
            scan(batch="E2", expiry="250200"),  # 2025-02-28
            scan(batch="K1", expiry="250331"),
            scan(batch="K2", expiry="250331", weight="3103001000"),
        ],
        [demand(quantity="2.5")],
        as_of="2025-03-01",
    )
    d = resp.json()["demands"][0]
    assert d["excluded_weight"] == 4.0  # E1 (2 kg) + E2 (2 kg)
    # eligible groups keep FEFO / first-seen order and allow a partial take
    assert [(a["batch"], a["weight"]) for a in d["allocations"]] == [
        ("K1", 2.0),
        ("K2", 0.5),
    ]
    assert d["fulfilled"] == 2.5
    assert d["shortage"] == 0.0


def test_excluded_weight_converts_units_and_rounds_half_even():
    # 25.13 lb = 11.3987762581 kg -> 11.398776 at six places.
    resp = plan([scan(weight="3202002513")], [demand(quantity="1")], as_of="2025-02-01")
    d = resp.json()["demands"][0]
    assert d["fulfilled"] == 0.0
    assert d["shortage"] == 1.0
    assert d["excluded_weight"] == 11.398776
    assert '"excluded_weight":11.398776' in resp.text


def test_excluded_weight_is_zero_without_as_of():
    resp = plan([scan()], [demand(quantity="1")])
    assert '"excluded_weight":0.000000' in resp.text


def test_thresholds_are_per_demand():
    resp = plan(
        [scan(expiry="250131"), scan(gtin=GTIN2, expiry="250131")],
        [
            demand(quantity="2", min_remaining_days=60),  # threshold 2025-04-01
            demand(gtin=GTIN2, quantity="2"),  # threshold 2025-01-31
        ],
        as_of="2025-01-31",
    )
    first, second = resp.json()["demands"]
    assert first["fulfilled"] == 0.0
    assert first["shortage"] == 2.0
    assert first["excluded_weight"] == 2.0
    assert second["fulfilled"] == 2.0
    assert second["excluded_weight"] == 0.0


def test_pick_plan_direct_call_with_as_of():
    from datetime import date

    from app.pickplan import pick_plan

    result = pick_plan(
        [scan(expiry="250131"), scan(batch="B", expiry="250228")],
        "kg",
        [(GTIN_OK, Decimal("3"), 0)],
        as_of=date(2025, 2, 1),
    )
    d = result["demands"][0]
    assert d["excluded_weight"] == Decimal("2.000")
    assert d["fulfilled"] == Decimal("2.000")
    assert d["shortage"] == Decimal("1.000")
    assert not isinstance(d["excluded_weight"], float)


# ------------------------------------------------- exact Decimal arithmetic


def test_allocation_math_is_exact_decimal():
    from app.pickplan import pick_plan

    result = pick_plan(
        [scan(weight="3202002513")],  # 11.3987762581 kg exactly
        "kg",
        [(GTIN_OK, Decimal("11.3987762581"))],
    )
    d = result["demands"][0]
    assert d["fulfilled"] == Decimal("11.3987762581")
    assert d["shortage"] == Decimal(0)
    assert d["allocations"][0]["weight"] == Decimal("11.3987762581")
    assert d["fulfilled"] + d["shortage"] == d["quantity"]
    assert not isinstance(d["fulfilled"], float)


def test_json_float_quantity_is_exact():
    # 2.513 as a JSON number must not acquire binary-float noise.
    resp = plan([scan(weight="3103002513")], [{"gtin": GTIN_OK, "quantity": 2.513}])
    d = resp.json()["demands"][0]
    assert d["fulfilled"] == 2.513
    assert d["shortage"] == 0.0


def test_weights_render_as_six_place_json_numbers():
    resp = plan([scan(weight="3103002513")], [demand(quantity="2.513")])
    assert '"quantity":2.513000' in resp.text
    assert '"fulfilled":2.513000' in resp.text
    assert '"shortage":0.000000' in resp.text
    assert '"weight":2.513000' in resp.text
    assert '"fulfilled":"2.513000"' not in resp.text


def test_half_even_rounding_only_at_response():
    # 0.0000005 kg -> nearest even at six places: 0.000000
    resp = plan([scan(weight="3107000005")], [demand(quantity="0.0000005")])
    assert '"fulfilled":0.000000' in resp.text
    # 0.0000015 kg -> half-even rounds up to 0.000002
    resp = plan([scan(weight="3107000015")], [demand(quantity="0.0000015")])
    assert '"fulfilled":0.000002' in resp.text


# ---------------------------------------------------------------- scan errors


def test_missing_expiry_rejected():
    resp = plan([f"01{GTIN_OK}3103002000"], [demand()])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["scan_index"] == 0
    assert detail["error"]["error"] == "MISSING_AI"
    assert "17" in detail["error"]["message"]


def test_duplicate_expiry_rejected():
    resp = plan([f"01{GTIN_OK}17250131172501313103002000"], [demand()])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["scan_index"] == 0
    assert detail["error"]["error"] == "DUPLICATE_AI"
    assert detail["error"]["offset"] == 24  # offset of the second AI 17


def test_missing_gtin_rejected():
    resp = plan(["172501313103002000"], [demand()])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["scan_index"] == 0
    assert detail["error"]["error"] == "MISSING_AI"
    assert "01" in detail["error"]["message"]


def test_missing_weight_rejected():
    resp = plan([f"01{GTIN_OK}17250131"], [demand()])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["scan_index"] == 0
    assert detail["error"]["error"] == "MISSING_AI"
    assert "310n/320n" in detail["error"]["message"]


def test_duplicate_weight_rejected():
    resp = plan([f"01{GTIN_OK}1725013131030020003202002513"], [demand()])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["scan_index"] == 0
    assert detail["error"]["error"] == "DUPLICATE_AI"
    assert "3103" in detail["error"]["found"]
    assert "3202" in detail["error"]["found"]


def test_parser_error_reports_scan_index_and_original_error():
    resp = plan([scan(), "0109501101530004172501313103002000"], [demand()])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["scan_index"] == 1
    assert detail["error"]["error"] == "INVALID_CHECK_DIGIT"
    assert detail["error"]["offset"] == 15
    assert set(detail) == {"scan_index", "error"}


# -------------------------------------------------------------- demand errors


def test_duplicate_demand_gtin_rejected():
    resp = plan([scan()], [demand(quantity="1"), demand(quantity="2")])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["demand_index"] == 1
    assert detail["error"]["error"] == "DUPLICATE_GTIN"
    assert detail["error"]["found"] == GTIN_OK
    assert set(detail) == {"demand_index", "error"}


def test_duplicate_demand_gtin_checked_before_scans():
    resp = plan(["not-a-gs1-string"], [demand(), demand()])
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"]["error"] == "DUPLICATE_GTIN"


def test_negative_min_remaining_days_rejected_with_demand_index():
    resp = plan(
        [scan()],
        [demand(), demand(gtin=GTIN2, quantity="1", min_remaining_days=-1)],
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["demand_index"] == 1
    assert detail["error"]["error"] == "INVALID_MIN_REMAINING_DAYS"
    assert detail["error"]["found"] == -1
    assert set(detail) == {"demand_index", "error"}


def test_negative_min_remaining_days_checked_before_scans():
    resp = plan(["not-a-gs1-string"], [demand(min_remaining_days=-3)])
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"]["error"] == "INVALID_MIN_REMAINING_DAYS"
    assert resp.json()["detail"]["demand_index"] == 0


# ---------------------------------------------------- request model validation


def test_missing_demands_field_is_422():
    resp = client.post("/weight-pick-plan", json={"scans": []})
    assert resp.status_code == 422


def test_missing_scans_field_is_422():
    resp = client.post("/weight-pick-plan", json={"demands": []})
    assert resp.status_code == 422


def test_zero_or_negative_quantity_is_422():
    assert plan([], [demand(quantity="0")]).status_code == 422
    assert plan([], [demand(quantity="-1")]).status_code == 422


def test_unknown_target_unit_is_422():
    resp = plan([], [demand()], target_unit="stone")
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "target_unit"]


def test_invalid_as_of_is_422():
    for bad in (
        "not-a-date",
        "2025-13-01",  # no such month
        "2025-02-30",  # no such day
        "20250228",  # missing dashes
        "2025-02-28T00:00:00",  # datetime, not a bare date
        "28/02/2025",
    ):
        resp = plan([], [demand()], as_of=bad)
        assert resp.status_code == 422, bad
        assert resp.json()["detail"][0]["loc"] == ["body", "as_of"]


def test_valid_as_of_boundary_dates_accepted():
    # leap day and month-end are real dates and pass validation
    assert plan([], [], as_of="2024-02-29").status_code == 200
    assert plan([], [], as_of="2025-01-31").status_code == 200


def test_non_string_scan_entry_is_422():
    resp = client.post(
        "/weight-pick-plan",
        json={"scans": [123], "demands": [demand()]},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "scans", 0]
