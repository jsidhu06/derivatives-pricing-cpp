#!/usr/bin/env python3
"""Generate tests/golden/pde_barrier_reference.inc from the Python reference.

Usage (from the repo root, with the Python package importable):

    python tools/gen_pde_barrier_golden.py > tests/golden/pde_barrier_reference.inc

Each row records the inputs of a barrier valuation priced on the PDE grid
(default ``PDEParams.for_barriers`` parameters on both sides) plus the
present value, the native GRID delta/gamma/theta, and NUMERICAL vega and rho.
The C++ suite replays every row through ``dp::OptionValuation`` and requires
agreement to ~1e-10 (PV, grid greeks) and ~1e-8 (numerical greeks).
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
NUM = dp.GreekCalculationMethod.NUMERICAL

TYPES = [(direction, action, ot) for direction in (UP, DOWN) for action in (IN, OUT)
         for ot in (CALL, PUT)]


def type_tag(direction, action, ot):
    return f"{'U' if direction is UP else 'D'}{'I' if action is IN else 'O'}{ot.name[0]}"


def row(label, *, spot, strike, barrier, direction, action, option_type, american=False,
        rate=0.05, div_yield=0.02, vol=0.25, maturity_days=365, num_observations=0,
        rebate=0.0, rebate_at_hit=True, div_amount=0.0, div_day=0):
    maturity = PRICING_DATE + dt.timedelta(days=maturity_days)
    md = dp.MarketData(PRICING_DATE, dp.DiscountCurve.flat(rate), currency="USD")
    q_curve = dp.DiscountCurve.flat(div_yield) if div_yield != 0.0 else None
    divs = [(PRICING_DATE + dt.timedelta(days=div_day), div_amount)] if div_amount else None
    ud = dp.UnderlyingData(initial_value=spot, volatility=vol, market_data=md,
                           dividend_curve=q_curve, discrete_dividends=divs)
    spec = dp.BarrierSpec(
        option_type=option_type,
        exercise_type=dp.ExerciseType.AMERICAN if american else dp.ExerciseType.EUROPEAN,
        strike=strike, maturity=maturity, barrier=barrier, direction=direction, action=action,
        monitoring=(dp.BarrierMonitoring.DISCRETE if num_observations
                    else dp.BarrierMonitoring.CONTINUOUS),
        rebate=rebate,
        rebate_timing=(dp.RebateTiming.AT_HIT if rebate_at_hit else dp.RebateTiming.AT_EXPIRY),
        num_observations=num_observations or None,
    )
    ov = dp.OptionValuation(underlying=ud, spec=spec, pricing_method=dp.PricingMethod.PDE_FD)
    values = (
        ov.present_value(),
        ov.delta(),
        ov.gamma(),
        ov.theta(),
        ov.vega(greek_calc_method=NUM),
        ov.rho(greek_calc_method=NUM),
    )
    inputs = (
        f'{{"{label}", {spot}, {strike}, {barrier}, {rate}, {div_yield}, {vol}, '
        f"{maturity_days}, {num_observations}, {rebate}, "
        f"{'true' if rebate_at_hit else 'false'}, "
        f"{'true' if direction is UP else 'false'}, {'true' if action is IN else 'false'}, "
        f"{'true' if option_type is CALL else 'false'}, {'true' if american else 'false'}, "
        f"{div_amount}, {div_day}, "
    )
    return inputs + ", ".join(repr(v) for v in values) + "},"


def main() -> None:
    rows = []

    # All 8 types, European and American, continuous monitoring.
    for direction, action, ot in TYPES:
        barrier = 115.0 if direction is UP else 85.0
        for american in (False, True):
            tag = type_tag(direction, action, ot)
            rows.append(row(f"{tag}{'_am' if american else ''}", spot=100.0, strike=100.0,
                            barrier=barrier, direction=direction, action=action,
                            option_type=ot, american=american))

    # Discrete monitoring (12 observations), half-step barrier anchoring.
    for direction, action, ot in TYPES:
        barrier = 115.0 if direction is UP else 85.0
        tag = type_tag(direction, action, ot)
        rows.append(row(f"{tag}_m12", spot=100.0, strike=100.0, barrier=barrier,
                        direction=direction, action=action, option_type=ot,
                        num_observations=12))

    # Rebates: knock-outs at hit and at expiry, knock-ins at expiry.
    for direction, ot in [(UP, CALL), (DOWN, PUT)]:
        barrier = 115.0 if direction is UP else 85.0
        rows.append(row(f"{type_tag(direction, OUT, ot)}_R3_hit", spot=100.0, strike=100.0,
                        barrier=barrier, direction=direction, action=OUT, option_type=ot,
                        rebate=3.0))
        rows.append(row(f"{type_tag(direction, OUT, ot)}_R3_exp", spot=100.0, strike=100.0,
                        barrier=barrier, direction=direction, action=OUT, option_type=ot,
                        rebate=3.0, rebate_at_hit=False))
        rows.append(row(f"{type_tag(direction, IN, ot)}_R3_exp", spot=100.0, strike=100.0,
                        barrier=barrier, direction=direction, action=IN, option_type=ot,
                        rebate=3.0, rebate_at_hit=False))

    # American knock-in with rebate: full two-surface coupled solve.
    rows.append(row("DIP_am_R2", spot=100.0, strike=105.0, barrier=85.0, direction=DOWN,
                    action=IN, option_type=PUT, american=True, rebate=2.0,
                    rebate_at_hit=False))

    # Discrete cash dividends (jump conditions on both barrier surfaces).
    rows.append(row("DOC_div", spot=100.0, strike=100.0, barrier=85.0, direction=DOWN,
                    action=OUT, option_type=CALL, div_amount=1.5, div_day=120,
                    div_yield=0.0))
    rows.append(row("UIC_am_div", spot=100.0, strike=100.0, barrier=115.0, direction=UP,
                    action=IN, option_type=CALL, american=True, div_amount=1.0,
                    div_day=200, div_yield=0.0))

    # Volatility / maturity variants and a near-barrier case.
    rows.append(row("UOC_v0.4_q0", spot=100.0, strike=100.0, barrier=125.0, direction=UP,
                    action=OUT, option_type=CALL, vol=0.4, div_yield=0.0))
    rows.append(row("DOP_T180", spot=100.0, strike=100.0, barrier=82.0, direction=DOWN,
                    action=OUT, option_type=PUT, maturity_days=180))
    rows.append(row("UOC_near", spot=100.0, strike=95.0, barrier=102.0, direction=UP,
                    action=OUT, option_type=CALL))

    # Triggered at inception.
    rows.append(row("trig_UOC_R3_hit", spot=100.0, strike=95.0, barrier=99.0, direction=UP,
                    action=OUT, option_type=CALL, rebate=3.0))
    rows.append(row("trig_UOC_R3_exp", spot=100.0, strike=95.0, barrier=99.0, direction=UP,
                    action=OUT, option_type=CALL, rebate=3.0, rebate_at_hit=False))
    rows.append(row("trig_DIC_am", spot=100.0, strike=95.0, barrier=101.0, direction=DOWN,
                    action=IN, option_type=CALL, american=True))

    out = sys.stdout
    out.write("// Generated by tools/gen_pde_barrier_golden.py from derivatives_pricing "
              f"{dp.__version__}. Do not edit.\n")
    out.write("// Default PDEParams.for_barriers parameters on both sides.\n")
    out.write("// Columns: label, spot, strike, barrier, rate, div_yield, vol, maturity_days,\n")
    out.write("//          num_observations (0 = continuous), rebate, rebate_at_hit, is_up,\n")
    out.write("//          is_in, is_call, is_american, div_amount, div_day,\n")
    out.write("//          pv, delta_grid, gamma_grid, theta_grid, vega_num, rho_num\n")
    out.write("// NOLINTBEGIN\n")
    out.write("inline constexpr PdeBarrierGoldenRow kPdeBarrierGolden[] = {\n")
    out.write("\n".join(rows))
    out.write("\n};\n// NOLINTEND\n")


if __name__ == "__main__":
    main()
