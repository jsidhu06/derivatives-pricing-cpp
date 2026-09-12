#!/usr/bin/env python3
"""Generate tests/golden/pde_reference.inc from the Python reference library.

Usage (repo root):

    .venv-goldenref/bin/python tools/gen_pde_golden.py > tests/golden/pde_reference.inc

Each row records a PDE_FD valuation's inputs plus PV, grid Greeks (delta,
gamma, theta) and NUMERICAL (default-bump) vega and rho. Maturities are whole
days so the chrono and pandas date handling coincide exactly.
"""

from __future__ import annotations

import datetime as dt
import itertools
import sys

import derivatives_pricing as dp

PRICING_DATE = dt.datetime(2025, 1, 1)
GRID = dp.GreekCalculationMethod.GRID
NUM = dp.GreekCalculationMethod.NUMERICAL

METHOD = {"IM": dp.PDEMethod.IMPLICIT, "EX": dp.PDEMethod.EXPLICIT,
          "EH": dp.PDEMethod.EXPLICIT_HULL, "CN": dp.PDEMethod.CRANK_NICOLSON}
SPACE = {"S": dp.PDESpaceGrid.SPOT, "L": dp.PDESpaceGrid.LOG_SPOT}
SOLVER = {"IN": dp.PDEEarlyExercise.INTRINSIC, "GS": dp.PDEEarlyExercise.GAUSS_SEIDEL}


def row(label, *, spot, strike=100.0, rate, div_yield, vol, maturity_days, spot_steps,
        time_steps, method, space, solver="GS", rannacher=2, option_type, exercise,
        div_amount=0.0, div_day=0, control_variate=False):
    maturity = PRICING_DATE + dt.timedelta(days=maturity_days)
    md = dp.MarketData(PRICING_DATE, dp.DiscountCurve.flat(rate), currency="USD")
    q_curve = dp.DiscountCurve.flat(div_yield) if div_yield != 0.0 else None
    divs = [(PRICING_DATE + dt.timedelta(days=div_day), div_amount)] if div_amount else None
    ud = dp.UnderlyingData(initial_value=spot, volatility=vol, market_data=md,
                           dividend_curve=q_curve, discrete_dividends=divs)
    spec = dp.VanillaSpec(option_type=option_type, exercise_type=exercise,
                          strike=strike, maturity=maturity)
    params = dp.PDEParams(spot_steps=spot_steps, time_steps=time_steps,
                          method=METHOD[method], space_grid=SPACE[space],
                          american_solver=SOLVER[solver], rannacher_steps=rannacher,
                          control_variate_european=control_variate)
    ov = dp.OptionValuation(ud, spec, dp.PricingMethod.PDE_FD, params=params)
    values = [ov.present_value(),
              ov.delta(greek_calc_method=GRID), ov.gamma(greek_calc_method=GRID),
              ov.theta(greek_calc_method=GRID),
              ov.vega(greek_calc_method=NUM), ov.rho(greek_calc_method=NUM)]
    flags = ["true" if option_type is dp.OptionType.CALL else "false",
             "true" if exercise is dp.ExerciseType.AMERICAN else "false",
             "true" if control_variate else "false"]
    nums = ", ".join(repr(float(v)) for v in values)
    return (f'    {{"{label}", {spot!r}, {strike!r}, {rate!r}, {div_yield!r}, {vol!r}, '
            f'{maturity_days}, {spot_steps}, {time_steps}, "{method}", "{space}", "{solver}", '
            f'{rannacher}, {div_amount!r}, {div_day}, {", ".join(flags)}, {nums}}},')


def main() -> None:
    C, P = dp.OptionType.CALL, dp.OptionType.PUT
    EU, AM = dp.ExerciseType.EUROPEAN, dp.ExerciseType.AMERICAN
    rows = []

    # Scheme / grid matrix (explicit schemes only on the log grid — the spot
    # grid trips the CFL guard at these settings, covered by C++ unit tests).
    combos = [("CN", "S"), ("CN", "L"), ("IM", "S"), ("IM", "L"), ("EX", "L"), ("EH", "L")]
    for (method, space), ot, ex in itertools.product(combos, [C, P], [EU, AM]):
        solver = "IN" if method in ("EX", "EH") and ex is AM else "GS"
        rows.append(row(f"{method}_{space}_{ot.name[0]}_{ex.name[:2]}", spot=100.0, rate=0.05,
                        div_yield=0.0, vol=0.2, maturity_days=365, spot_steps=120,
                        time_steps=200, method=method, space=space, solver=solver,
                        option_type=ot, exercise=ex))

    # Moneyness / dividend-yield / vol variants on the default CN spot grid.
    for spot, q, vol, ot, ex in itertools.product(
            [85.0, 110.0], [0.0, 0.03], [0.35], [C, P], [EU, AM]):
        rows.append(row(f"S{spot:g}_q{q:g}_v{vol:g}_{ot.name[0]}_{ex.name[:2]}", spot=spot,
                        rate=0.04, div_yield=q, vol=vol, maturity_days=365, spot_steps=120,
                        time_steps=160, method="CN", space="S", option_type=ot, exercise=ex))

    # American INTRINSIC solver vs PSOR.
    for ot in (C, P):
        rows.append(row(f"solver_IN_{ot.name[0]}", spot=100.0, rate=0.05, div_yield=0.02,
                        vol=0.25, maturity_days=365, spot_steps=120, time_steps=160,
                        method="CN", space="S", solver="IN", option_type=ot, exercise=AM))

    # Rannacher off.
    rows.append(row("rannacher0_C_EU", spot=100.0, rate=0.05, div_yield=0.0, vol=0.2,
                    maturity_days=365, spot_steps=120, time_steps=160, method="CN",
                    space="S", rannacher=0, option_type=C, exercise=EU))

    # Discrete dividends (jump conditions), both grids.
    for space, ot, ex in itertools.product(["S", "L"], [C, P], [EU, AM]):
        rows.append(row(f"div_{space}_{ot.name[0]}_{ex.name[:2]}", spot=100.0, rate=0.05,
                        div_yield=0.0, vol=0.2, maturity_days=365, spot_steps=120,
                        time_steps=160, method="CN", space=space, option_type=ot,
                        exercise=ex, div_amount=1.5, div_day=180))

    # Short/long maturities.
    for days, ot in itertools.product([30, 730], [C, P]):
        rows.append(row(f"T{days}_{ot.name[0]}_AM", spot=100.0, rate=0.05, div_yield=0.0,
                        vol=0.2, maturity_days=days, spot_steps=120, time_steps=160,
                        method="CN", space="S", option_type=ot, exercise=AM))

    # Control variate.
    for ot in (C, P):
        rows.append(row(f"cv_{ot.name[0]}", spot=100.0, strike=105.0, rate=0.04, div_yield=0.0,
                        vol=0.25, maturity_days=365, spot_steps=120, time_steps=160,
                        method="CN", space="S", option_type=ot, exercise=AM,
                        control_variate=True))

    out = sys.stdout
    out.write("// Generated by tools/gen_pde_golden.py from derivatives_pricing "
              f"{dp.__version__}. Do not edit.\n")
    out.write("// Columns: label, spot, strike, rate, div_yield, vol, maturity_days, "
              "spot_steps, time_steps,\n//          method, space, solver, rannacher, "
              "div_amount, div_day, is_call, is_american, control_variate,\n")
    out.write("//          pv, delta_grid, gamma_grid, theta_grid, vega_num, rho_num\n")
    out.write("// NOLINTBEGIN\n")
    out.write("inline constexpr PdeGoldenRow kPdeGolden[] = {\n")
    out.write("\n".join(rows))
    out.write("\n};\n// NOLINTEND\n")


if __name__ == "__main__":
    main()
