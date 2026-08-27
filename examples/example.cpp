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

    // Designated initialisers must follow the declaration order of the
    // struct (option_type, exercise_type, strike, maturity, ...).
    const VanillaSpec::Params params{
        .option_type = OptionType::Put,
        .exercise_type = ExerciseType::American,
        .strike = 95.0,
        .maturity = make_datetime(2026, 12, 31),
    };
    const VanillaSpec vanilla(params);
    std::cout << to_string(vanilla.option_type()) << " " << to_string(vanilla.exercise_type())
              << " K=" << vanilla.strike() << " T=" << to_string(vanilla.maturity()) << '\n';

    return 0;
}
