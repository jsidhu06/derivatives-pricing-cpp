#!/usr/bin/env python3
"""Generate tests/golden/conventions_reference.inc from the Python reference library.

Usage (repo root):

    .venv-goldenref/bin/python tools/gen_conventions_golden.py > tests/golden/conventions_reference.inc

Market conventions through the deterministic engines: term-structure rate
and dividend curves (piecewise-constant forwards), ACT/360 / ACT/365.25 /
30/360 day counts, intraday pricing and maturity timestamps, and cash
dividends on a non-flat curve — for vanillas and a few barriers on BSM,
binomial and PDE. Each row records PV and the engine-default greeks; a
greek (or PV) the engine refuses is emitted as std::nullopt, and the C++
test requires the same refusal.

The curve knots are emitted into the .inc so both sides build identical
curves from one source.
"""

from __future__ import annotations

import datetime as dt
import sys

import derivatives_pricing as dp

BASE = dt.datetime(2025, 1, 1)

CURVE_TIMES = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]
FORWARDS = {
    "up": [0.02, 0.03, 0.04, 0.05, 0.055],  # upward-sloping rates
    "inv": [0.06, 0.055, 0.045, 0.035, 0.03],  # inverted rates
    "q_up": [0.0, 0.01, 0.02, 0.025, 0.03],  # rising dividend yield
}
DCC = {
    "ACT365F": dp.DayCountConvention.ACT_365F,
    "ACT360": dp.DayCountConvention.ACT_360,
    "ACT365_25": dp.DayCountConvention.ACT_365_25,
    "THIRTY360": dp.DayCountConvention.THIRTY_360_US,
}
ENGINE = {"BSM": dp.PricingMethod.BSM, "TREE": dp.PricingMethod.BINOMIAL,
          "PDE": dp.PricingMethod.PDE_FD}
DIRECTION = {"U": dp.BarrierDirection.UP, "D": dp.BarrierDirection.DOWN}
ACTION = {"I": dp.BarrierAction.IN, "O": dp.BarrierAction.OUT}

# Engine settings (the C++ test mirrors them).
TREE_BARRIER_STEPS = 400
PDE_BARRIER_STEPS = 300


def curve(name: str) -> dp.DiscountCurve | None:
    if name == "none":
        return None
    return dp.DiscountCurve.from_forwards(CURVE_TIMES, FORWARDS[name])


def valuation(r):
    pricing = BASE + dt.timedelta(minutes=r["pricing_minutes"])
    maturity = BASE + dt.timedelta(days=r["maturity_days"], minutes=r["maturity_minutes"])
    md = dp.MarketData(pricing, curve(r["curve"]), currency="USD",
                       day_count_convention=DCC[r["dcc"]])
    divs = None
    if r["div_amount"]:
        divs = [(BASE + dt.timedelta(days=r["div_day"]), r["div_amount"])]
    ud = dp.UnderlyingData(initial_value=100.0, volatility=0.25, market_data=md,
                           dividend_curve=curve(r["q_curve"]), discrete_dividends=divs)
    option_type = dp.OptionType.CALL if r["call"] else dp.OptionType.PUT
    exercise = dp.ExerciseType.AMERICAN if r["american"] else dp.ExerciseType.EUROPEAN
    params = None
    if r["barrier"]:
        monitoring = (dp.BarrierMonitoring.DISCRETE if r["observations"]
                      else dp.BarrierMonitoring.CONTINUOUS)
        spec = dp.BarrierSpec(
            option_type=option_type, exercise_type=exercise, strike=100.0, maturity=maturity,
            barrier=r["barrier"], direction=DIRECTION[r["direction"]],
            action=ACTION[r["action"]], monitoring=monitoring,
            num_observations=r["observations"] or None)
        if r["engine"] == "TREE":
            params = dp.BinomialParams(num_steps=TREE_BARRIER_STEPS)
        elif r["engine"] == "PDE":
            params = dp.PDEParams.for_barriers(
                monitoring=monitoring, spot_steps=PDE_BARRIER_STEPS,
                time_steps=PDE_BARRIER_STEPS, method=dp.PDEMethod.CRANK_NICOLSON,
                american_solver=dp.PDEEarlyExercise.GAUSS_SEIDEL)
    else:
        spec = dp.VanillaSpec(option_type=option_type, exercise_type=exercise, strike=100.0,
                              maturity=maturity)
    return dp.OptionValuation(ud, spec, ENGINE[r["engine"]], params=params)


def attempt(fn):
    try:
        return repr(float(fn()))
    except dp.DerivativesPricingError:
        return "std::nullopt"


def emit(r) -> str:
    try:
        ov = valuation(r)
    except dp.DerivativesPricingError:
        ov = None
    values = ["std::nullopt"] * 6
    if ov is not None:
        values = [attempt(ov.present_value), attempt(ov.delta), attempt(ov.gamma),
                  attempt(ov.theta), attempt(ov.vega), attempt(ov.rho)]
    b = lambda x: "true" if x else "false"  # noqa: E731
    return (f'    {{"{r["label"]}", "{r["engine"]}", "{r["curve"]}", "{r["q_curve"]}", '
            f'"{r["dcc"]}", {r["pricing_minutes"]}, {r["maturity_days"]}, '
            f'{r["maturity_minutes"]}, {r["div_amount"]!r}, {r["div_day"]}, {b(r["call"])}, '
            f'{b(r["american"])}, {r["barrier"]!r}, "{r["direction"]}", "{r["action"]}", '
            f'{r["observations"]}, {", ".join(values)}}},')


def row(label, engine, **kw):
    r = dict(label=label, engine=engine, curve="up", q_curve="none", dcc="ACT365F",
             pricing_minutes=0, maturity_days=365, maturity_minutes=0, div_amount=0.0,
             div_day=0, call=True, american=False, barrier=0.0, direction="U", action="O",
             observations=0)
    r.update(kw)
    return r


def scenarios():
    rows = []
    engines = [("BSM", False), ("TREE", False), ("TREE", True), ("PDE", False), ("PDE", True)]
    # Term structures: rate curve x dividend curve.
    for c in ("up", "inv"):
        for q in ("none", "q_up"):
            for engine, am in engines:
                for call in (True, False):
                    rows.append(row(f"{c}_{q}_{engine}_{'A' if am else 'E'}{'C' if call else 'P'}",
                                    engine, curve=c, q_curve=q, american=am, call=call,
                                    maturity_days=548))
    # Day counts (30/360 is non-uniform on a calendar grid: the tree refuses it).
    for dcc in ("ACT360", "ACT365_25", "THIRTY360"):
        for engine, am in (("BSM", False), ("TREE", True), ("PDE", True)):
            rows.append(row(f"{dcc}_{engine}_{'A' if am else 'E'}P", engine, dcc=dcc,
                            american=am, call=False, q_curve="q_up"))
    # Intraday timestamps: 09:30 pricing to a 16:00 expiry, three months and one week out.
    for days, tag in ((90, "3m"), (7, "1w")):
        for engine, am in (("BSM", False), ("TREE", True), ("PDE", True)):
            rows.append(row(f"intraday_{tag}_{engine}_{'A' if am else 'E'}C", engine,
                            pricing_minutes=9 * 60 + 30, maturity_days=days,
                            maturity_minutes=16 * 60, american=am, q_curve="q_up"))
    # Cash dividend on the upward curve.
    for engine, am in engines:
        rows.append(row(f"cash_div_{engine}_{'A' if am else 'E'}P", engine, call=False,
                        american=am, div_amount=2.0, div_day=120))
    # Barriers on non-flat rate and dividend curves.
    barriers = [("UOC_cont", True, 125.0, "U", "O", 0), ("DIP_cont", False, 85.0, "D", "I", 0),
                ("DOC_m12", True, 90.0, "D", "O", 12)]
    for name, call, h, d, a, obs in barriers:
        for engine in ("BSM", "TREE", "PDE"):
            rows.append(row(f"{name}_{engine}_E", engine, call=call, q_curve="q_up", barrier=h,
                            direction=d, action=a, observations=obs))
    for engine in ("TREE", "PDE"):
        rows.append(row(f"DOP_cont_{engine}_A", engine, call=False, american=True,
                        q_curve="q_up", barrier=85.0, direction="D", action="O"))
    return rows


def main() -> None:
    rows = [emit(r) for r in scenarios()]
    fmt = lambda xs: ", ".join(repr(float(x)) for x in xs)  # noqa: E731
    print(f"// Generated by tools/gen_conventions_golden.py from derivatives_pricing "
          f"{dp.__version__}. Do not edit.")
    print("// Columns: label, engine, curve, q_curve, dcc, pricing_minutes, maturity_days,")
    print("//          maturity_minutes, div_amount, div_day, is_call, is_american, barrier,")
    print("//          direction, action, observations, pv, delta, gamma, theta, vega, rho")
    print("// std::nullopt = the engine refuses that value; engine-default greek methods.")
    print(f"// Engine settings: barrier trees num_steps={TREE_BARRIER_STEPS}; barrier PDEs "
          f"for_barriers(...) with spot/time steps {PDE_BARRIER_STEPS}, Crank-Nicolson,")
    print("// Gauss-Seidel; vanillas use default params.")
    print("// NOLINTBEGIN")
    print(f"inline constexpr std::array<double, {len(CURVE_TIMES)}> kCurveTimes{{{fmt(CURVE_TIMES)}}};")
    for name, fwd in FORWARDS.items():
        print(f"inline constexpr std::array<double, {len(fwd)}> kForwards_{name}{{{fmt(fwd)}}};")
    print("inline constexpr ConventionsGoldenRow kConventionsGolden[] = {")
    for r in rows:
        print(r)
    print("};")
    print("// NOLINTEND")
    print(f"// {len(rows)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
