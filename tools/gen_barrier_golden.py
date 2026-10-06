#!/usr/bin/env python3
"""Generate tests/golden/barrier_reference.inc from the Python reference library.

Usage (from the repo root, with the Python package importable):

    python tools/gen_barrier_golden.py > tests/golden/barrier_reference.inc

Each row records the inputs of a European barrier valuation (BSM analytical
engine) plus the present value, the auto-selected theta (Black-Scholes PDE
identity in the continuation region, short-circuits when triggered at
inception) and the NUMERICAL (bump-and-revalue, default bumps) delta, gamma,
vega and rho. The C++ suite replays every row through ``dp::OptionValuation``
and requires agreement to ~1e-11 (PV) and ~1e-8 (greeks).
"""

from __future__ import annotations

import datetime as dt
import sys

import derivatives_pricing as dp

PRICING_DATE = dt.datetime(2025, 1, 1)
UP = dp.BarrierDirection.UP
DOWN = dp.BarrierDirection.DOWN
IN = dp.BarrierAction.IN
OUT = dp.BarrierAction.OUT
CALL = dp.OptionType.CALL
PUT = dp.OptionType.PUT
AT_HIT = dp.RebateTiming.AT_HIT
AT_EXPIRY = dp.RebateTiming.AT_EXPIRY

TYPES = [(direction, action, ot) for direction in (UP, DOWN) for action in (IN, OUT)
         for ot in (CALL, PUT)]


def type_tag(direction, action, ot):
    return f"{'U' if direction is UP else 'D'}{'I' if action is IN else 'O'}{ot.name[0]}"


def row(label, *, spot, strike, barrier, direction, action, option_type, rate=0.05,
        div_yield=0.02, vol=0.25, maturity_days=365, num_observations=0, rebate=0.0,
        rebate_at_hit=True):
    maturity = PRICING_DATE + dt.timedelta(days=maturity_days)
    md = dp.MarketData(PRICING_DATE, dp.DiscountCurve.flat(rate), currency="USD")
    q_curve = dp.DiscountCurve.flat(div_yield) if div_yield != 0.0 else None
    ud = dp.UnderlyingData(initial_value=spot, volatility=vol, market_data=md,
                           dividend_curve=q_curve)
    spec = dp.BarrierSpec(
        option_type=option_type, exercise_type=dp.ExerciseType.EUROPEAN, strike=strike,
        maturity=maturity, barrier=barrier, direction=direction, action=action,
        monitoring=(dp.BarrierMonitoring.DISCRETE if num_observations
                    else dp.BarrierMonitoring.CONTINUOUS),
        rebate=rebate, rebate_timing=AT_HIT if rebate_at_hit else AT_EXPIRY,
        num_observations=num_observations or None,
    )
    ov = dp.OptionValuation(underlying=ud, spec=spec, pricing_method=dp.PricingMethod.ANALYTICAL)
    values = (
        ov.present_value(),
        ov.theta(),
        ov.delta(greek_calc_method=dp.GreekCalculationMethod.NUMERICAL),
        ov.gamma(greek_calc_method=dp.GreekCalculationMethod.NUMERICAL),
        ov.vega(greek_calc_method=dp.GreekCalculationMethod.NUMERICAL),
        ov.rho(greek_calc_method=dp.GreekCalculationMethod.NUMERICAL),
    )
    inputs = (
        f'{{"{label}", {spot}, {strike}, {barrier}, {rate}, {div_yield}, {vol}, '
        f"{maturity_days}, {num_observations}, {rebate}, "
        f"{'true' if rebate_at_hit else 'false'}, "
        f"{'true' if direction is UP else 'false'}, {'true' if action is IN else 'false'}, "
        f"{'true' if option_type is CALL else 'false'}, "
    )
    return inputs + ", ".join(repr(v) for v in values) + "},"


def main() -> None:
    rows = []

    # All 8 types, both branches of the H<=K / H>K formula table.
    for direction, action, ot in TYPES:
        barrier = 115.0 if direction is UP else 85.0
        strikes = (105.0, 130.0) if direction is UP else (95.0, 70.0)
        for strike in strikes:
            tag = type_tag(direction, action, ot)
            rows.append(row(f"{tag}_K{strike:g}", spot=100.0, strike=strike, barrier=barrier,
                            direction=direction, action=action, option_type=ot))

    # Rebates: knock-outs at hit and at expiry, knock-ins at expiry.
    for direction, action, ot in TYPES:
        barrier = 115.0 if direction is UP else 85.0
        tag = type_tag(direction, action, ot)
        # Knock-in rebates must use AT_EXPIRY (contract validation).
        timings = (True, False) if action is OUT else (False,)
        for at_hit in timings:
            suffix = "hit" if at_hit else "exp"
            rows.append(row(f"{tag}_R3_{suffix}", spot=100.0, strike=100.0, barrier=barrier,
                            direction=direction, action=action, option_type=ot, rebate=3.0,
                            rebate_at_hit=at_hit))

    # Discrete monitoring (Broadie-Glasserman-Kou adjustment), 12 observations.
    for direction, action, ot in TYPES:
        barrier = 115.0 if direction is UP else 85.0
        tag = type_tag(direction, action, ot)
        rows.append(row(f"{tag}_m12", spot=100.0, strike=100.0, barrier=barrier,
                        direction=direction, action=action, option_type=ot,
                        num_observations=12))

    # Volatility / maturity / zero-yield variants.
    for direction, action, ot in [(UP, OUT, CALL), (UP, IN, PUT), (DOWN, OUT, PUT),
                                  (DOWN, IN, CALL)]:
        barrier = 120.0 if direction is UP else 80.0
        tag = type_tag(direction, action, ot)
        rows.append(row(f"{tag}_v0.4_T180", spot=100.0, strike=100.0, barrier=barrier,
                        direction=direction, action=action, option_type=ot, vol=0.4,
                        maturity_days=180, div_yield=0.0))

    # Spot close to the barrier: exercises the clamped numerical spot bump.
    rows.append(row("UOC_near", spot=100.0, strike=95.0, barrier=102.0, direction=UP,
                    action=OUT, option_type=CALL))
    rows.append(row("DOP_near", spot=100.0, strike=105.0, barrier=98.5, direction=DOWN,
                    action=OUT, option_type=PUT))

    # Triggered at inception (continuous monitoring, spot past the barrier).
    rows.append(row("trig_UOC_R0", spot=100.0, strike=95.0, barrier=99.0, direction=UP,
                    action=OUT, option_type=CALL))
    rows.append(row("trig_UOC_R3_hit", spot=100.0, strike=95.0, barrier=99.0, direction=UP,
                    action=OUT, option_type=CALL, rebate=3.0, rebate_at_hit=True))
    rows.append(row("trig_UOC_R3_exp", spot=100.0, strike=95.0, barrier=99.0, direction=UP,
                    action=OUT, option_type=CALL, rebate=3.0, rebate_at_hit=False))
    rows.append(row("trig_DIC", spot=100.0, strike=95.0, barrier=101.0, direction=DOWN,
                    action=IN, option_type=CALL))

    out = sys.stdout
    out.write("// Generated by tools/gen_barrier_golden.py from derivatives_pricing "
              f"{dp.__version__}. Do not edit.\n")
    out.write("// Columns: label, spot, strike, barrier, rate, div_yield, vol, maturity_days,\n")
    out.write("//          num_observations (0 = continuous), rebate, rebate_at_hit, is_up,\n")
    out.write("//          is_in, is_call, pv, theta_auto, delta_num, gamma_num, vega_num, "
              "rho_num\n")
    out.write("// NOLINTBEGIN\n")
    out.write("inline constexpr BarrierGoldenRow kBarrierGolden[] = {\n")
    out.write("\n".join(rows))
    out.write("\n};\n// NOLINTEND\n")


if __name__ == "__main__":
    main()
