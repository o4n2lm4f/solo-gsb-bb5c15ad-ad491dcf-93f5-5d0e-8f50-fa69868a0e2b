"""FEFO pick-planning of net weights across parsed GS1 scans.

Each scan is fed through :func:`app.gs1.parse` and must then contain:

* exactly one AI 01 (GTIN),
* exactly one AI 17 (expiry date; day ``00`` denotes the last day of the
  month, per the GS1 General Specifications),
* zero or one AI 10 (batch/lot) -- optional,
* exactly one weight element of the 310n (kg) / 320n (lb) family.

Any parser failure, missing/duplicated required AI or multiple weight
elements make the whole request fail with
:class:`app.summary.WeightSummaryError` whose ``detail`` names the
offending scan index -- the same 422 shape as ``/weight-summary``.

Valid scans are aggregated by (GTIN, batch, AI 17 value) in first-seen
order; every weight is rebuilt with :class:`decimal.Decimal` from the
original six digits and converted with the fixed factors of
:mod:`app.summary`.  Each demand is then satisfied FEFO: the GTIN's
groups are walked by ascending *actual* expiry date (day ``00`` -> last
day of the month), ties keep the batches' first-appearance order (the
sort is stable), and the last group reached may be allocated partially.

When the request carries an ``as_of`` date, each demand additionally
declares a non-negative ``min_remaining_days`` threshold (default 0):
only groups whose actual expiry date is on or after
``as_of + min_remaining_days`` may fulfil that demand (the boundary
date itself stays eligible).  Excluded groups never count towards
``fulfilled`` -- the shortfall simply lands in ``shortage`` -- and
their combined weight is reported per demand as ``excluded_weight``.

All conversion, aggregation, comparison and allocation arithmetic stays
in :class:`Decimal`; rounding to six places happens only when the
response is rendered.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from decimal import Decimal, localcontext

from .gs1 import GS1ParseError, Element, decode_weight, parse
from .summary import (
    ARITHMETIC_PRECISION,
    ScanValidationError,
    WeightSummaryError,
    _failure,
    _to_target,
)

REQUIRED_AI_TEXT = (
    "exactly one AI 01 (GTIN), exactly one AI 17 (expiry date), "
    "at most one AI 10 (batch/lot), and exactly one weight AI 310n/320n"
)


class DemandValidationError(Exception):
    """A demand violated the request shape (e.g. a duplicated GTIN).

    ``detail`` is the complete 422 payload: ``demand_index`` plus the
    validation error under ``error``, mirroring the scan-level shape.
    """

    def __init__(self, detail: dict):
        super().__init__(detail["error"]["message"])
        self.detail = detail


def _expiry_date(yymmdd: str) -> date:
    """Actual calendar date of an AI 17 value; day ``00`` = last day of month."""
    year = 2000 + int(yymmdd[0:2])
    month = int(yymmdd[2:4])
    day = int(yymmdd[4:6])
    if day == 0:
        day = calendar.monthrange(year, month)[1]
    return date(year, month, day)


def _threshold_date(as_of: date | None, min_days: int) -> date | None:
    """Earliest eligible actual expiry date for one demand.

    ``None`` (no filtering) when the request carries no ``as_of``.  A
    ``min_days`` so large that the addition overflows simply excludes
    everything: AI 17 years are 20YY, so :data:`date.max` is beyond any
    real expiry.
    """
    if as_of is None:
        return None
    try:
        return as_of + timedelta(days=min_days)
    except OverflowError:
        return date.max


def _extract(
    index: int, elements: list[Element]
) -> tuple[str, str | None, str, Decimal, str]:
    """Pull (GTIN, batch or None, AI 17 value, exact weight, unit) from one scan."""
    gtins = [e for e in elements if e.ai == "01"]
    expiries = [e for e in elements if e.ai == "17"]
    batches = [e for e in elements if e.ai == "10"]
    weights = [e for e in elements if e.unit is not None]

    if len(gtins) != 1:
        if not gtins:
            raise ScanValidationError(
                "MISSING_AI",
                f"scan {index}: missing required AI 01 (GTIN)",
                expected=REQUIRED_AI_TEXT,
            )
        raise ScanValidationError(
            "DUPLICATE_AI",
            f"scan {index}: AI 01 (GTIN) appears {len(gtins)} times; exactly one is required",
            expected="exactly one AI 01 (GTIN)",
            found=f"{len(gtins)} AI 01 elements",
            offset=gtins[1].offset,
        )

    if len(expiries) != 1:
        if not expiries:
            raise ScanValidationError(
                "MISSING_AI",
                f"scan {index}: missing required AI 17 (expiry date)",
                expected=REQUIRED_AI_TEXT,
            )
        raise ScanValidationError(
            "DUPLICATE_AI",
            f"scan {index}: AI 17 (expiry date) appears {len(expiries)} times; exactly one is required",
            expected="exactly one AI 17 (expiry date)",
            found=f"{len(expiries)} AI 17 elements",
            offset=expiries[1].offset,
        )

    if len(batches) > 1:
        raise ScanValidationError(
            "DUPLICATE_AI",
            f"scan {index}: AI 10 (batch/lot) appears {len(batches)} times; at most one is allowed",
            expected="at most one AI 10 (batch/lot)",
            found=f"{len(batches)} AI 10 elements",
            offset=batches[1].offset,
        )

    if len(weights) != 1:
        if not weights:
            raise ScanValidationError(
                "MISSING_AI",
                f"scan {index}: missing required weight AI 310n/320n",
                expected=REQUIRED_AI_TEXT,
            )
        ais = ", ".join(e.ai for e in weights)
        raise ScanValidationError(
            "DUPLICATE_AI",
            f"scan {index}: {len(weights)} weight elements ({ais}); "
            "exactly one 310n/320n is required",
            expected="exactly one weight AI 310n (kg) or 320n (lb)",
            found=ais,
            offset=weights[1].offset,
        )

    weight = weights[0]
    value = decode_weight(weight.value, int(weight.ai[3]))
    return (
        gtins[0].value,
        batches[0].value if batches else None,
        expiries[0].value,
        value,
        weight.unit,
    )


def _check_demands(demands: list[tuple[str, Decimal, int]]) -> None:
    """Reject negative thresholds and GTINs requested more than once."""
    seen: dict[str, int] = {}
    for demand_index, (gtin, _quantity, min_days) in enumerate(demands):
        if min_days < 0:
            raise DemandValidationError(
                {
                    "demand_index": demand_index,
                    "error": {
                        "error": "INVALID_MIN_REMAINING_DAYS",
                        "message": (
                            f"demand {demand_index}: min_remaining_days must "
                            f"be a non-negative integer, got {min_days}"
                        ),
                        "expected": "non-negative integer min_remaining_days",
                        "found": min_days,
                    },
                }
            )
        if gtin in seen:
            first = seen[gtin]
            raise DemandValidationError(
                {
                    "demand_index": demand_index,
                    "error": {
                        "error": "DUPLICATE_GTIN",
                        "message": (
                            f"demand {demand_index}: GTIN '{gtin}' is already "
                            f"requested by demand {first}; each GTIN may "
                            "appear only once"
                        ),
                        "expected": "unique GTIN across all demands",
                        "found": gtin,
                    },
                }
            )
        seen[gtin] = demand_index


def pick_plan(
    scans: list[str],
    target_unit: str,
    demands: list[tuple[str, Decimal] | tuple[str, Decimal, int]],
    as_of: date | None = None,
) -> dict:
    """Aggregate scan weights per GTIN and satisfy each demand FEFO.

    ``demands`` is a list of ``(gtin, quantity)`` or ``(gtin, quantity,
    min_remaining_days)`` tuples with ``quantity`` a :class:`Decimal`
    already expressed in ``target_unit``; a GTIN may appear only once
    and ``min_remaining_days`` (default 0) must be a non-negative
    integer.  When ``as_of`` is given, only groups whose actual expiry
    date is on or after ``as_of + min_remaining_days`` may fulfil the
    demand.  Returns ``{"target_unit": ..., "demands": [{gtin, quantity,
    fulfilled, shortage, excluded_weight, allocations}, ...]}`` in
    demand order, where every numeric value is an unrounded
    :class:`Decimal` in the target unit and each allocation is
    ``{"batch", "expiry", "weight"}`` with ``expiry`` the raw AI 17
    value of the source group.  ``excluded_weight`` is the total weight
    of the demand's GTIN held back by the shelf-life threshold (zero
    when ``as_of`` is absent).
    """
    normalized = [
        (demand[0], demand[1], demand[2] if len(demand) > 2 else 0)
        for demand in demands
    ]
    _check_demands(normalized)

    groups: dict[tuple[str, str | None, str], dict] = {}
    order: list[tuple[str, str | None, str]] = []

    with localcontext() as ctx:
        ctx.prec = ARITHMETIC_PRECISION
        for index, scan in enumerate(scans):
            try:
                elements = parse(scan)
            except GS1ParseError as exc:
                raise _failure(index, exc) from exc
            try:
                gtin, batch, expiry, weight, unit = _extract(index, elements)
            except ScanValidationError as exc:
                raise _failure(index, exc) from exc

            key = (gtin, batch, expiry)
            converted = _to_target(weight, unit, target_unit)
            if key not in groups:
                order.append(key)
                groups[key] = {
                    "gtin": gtin,
                    "batch": batch,
                    "expiry": expiry,
                    "expiry_date": _expiry_date(expiry),
                    "available": Decimal(0),
                }
            groups[key]["available"] += converted

        by_gtin: dict[str, list[dict]] = {}
        for key in order:
            by_gtin.setdefault(key[0], []).append(groups[key])
        for gtin_groups in by_gtin.values():
            # FEFO: ascending actual expiry date.  The sort is stable, so
            # groups expiring on the same date keep their first-seen order.
            gtin_groups.sort(key=lambda group: group["expiry_date"])

        results = []
        for gtin, quantity, min_days in normalized:
            threshold = _threshold_date(as_of, min_days)
            remaining = quantity
            excluded = Decimal(0)
            allocations = []
            for group in by_gtin.get(gtin, []):
                if threshold is not None and group["expiry_date"] < threshold:
                    # Not enough remaining shelf life for this demand: the
                    # group never counts towards fulfilled.  Groups are
                    # FEFO-sorted, so excluded ones form a prefix and all
                    # of them are totalled here before allocation starts.
                    excluded += group["available"]
                    continue
                if remaining <= 0:
                    break
                take = min(remaining, group["available"])
                if take <= 0:
                    continue  # group of zero-weight scans contributes nothing
                allocations.append(
                    {
                        "batch": group["batch"],
                        "expiry": group["expiry"],
                        "weight": take,
                    }
                )
                remaining -= take
            results.append(
                {
                    "gtin": gtin,
                    "quantity": quantity,
                    "fulfilled": quantity - remaining,
                    "shortage": remaining,
                    "excluded_weight": excluded,
                    "allocations": allocations,
                }
            )

    return {"target_unit": target_unit, "demands": results}
