#include "dp/contracts.hpp"
#include "dp/dates.hpp"
#include "dp/enums.hpp"

#include <iostream>

using namespace dp;

int main() {
    const VanillaSpec spec({
        .option_type = OptionType::Call,
        .exercise_type = ExerciseType::European,
        .strike = 100.0,
        .maturity = make_datetime(2026, 1, 1),
        .currency = "USD",
    });

    std::cout << to_string(spec.exercise_type()) << '\n';
    std::cout << to_string(spec.option_type()) << " K=" << spec.strike()
              << " T=" << to_string(spec.maturity()) << '\n';
    return 0;
}
