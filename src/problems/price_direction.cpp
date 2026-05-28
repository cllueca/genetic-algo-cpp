#include <problems/price_direction.hpp>

#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

namespace data::price_direction {

    float get_sma(
        const std::vector<StockPrice>& prices,
        int position,
        int period
    ) {
        if (position < period - 1)
            throw std::runtime_error("Not enough data to calculate SMA");

        float total = 0.0f;

        for (int i = position - period + 1; i <= position; ++i)
            total += prices[i].nav;

        return total / static_cast<float>(period);
    }

    float get_ema(
        const std::vector<StockPrice>& prices,
        int position,
        int period
    ) {
        if (position < period - 1)
            throw std::runtime_error("Not enough data to calculate EMA");

        float k = 2.0f / (static_cast<float>(period) + 1.0f);

        // Seed EMA with SMA
        float ema = get_sma(prices, period - 1, period);

        for (int i = period; i <= position; ++i)
            ema = prices[i].nav * k + ema * (1.0f - k);

        return ema;
    }

    float get_rsi(
        const std::vector<StockPrice>& prices,
        int position,
        int period
    ) {
        if (position < period)
            throw std::runtime_error("Not enough data to calculate RSI");

        // Seed: simple mean of first `period` up/down moves
        float avg_gain = 0.0f;
        float avg_loss = 0.0f;

        for (int i = position - period + 1; i <= position - period + period; ++i) {
            float change = prices[i].nav - prices[i - 1].nav;
            if (change > 0.0f) avg_gain += change;
            else               avg_loss -= change;   // store as positive
        }
        avg_gain /= static_cast<float>(period);
        avg_loss /= static_cast<float>(period);

        // Wilder smoothing for remaining bars
        for (int i = position - period + period + 1; i <= position; ++i) {
            float change = prices[i].nav - prices[i - 1].nav;
            float gain   = (change > 0.0f) ?  change : 0.0f;
            float loss   = (change < 0.0f) ? -change : 0.0f;

            avg_gain = (avg_gain * static_cast<float>(period - 1) + gain) / static_cast<float>(period);
            avg_loss = (avg_loss * static_cast<float>(period - 1) + loss) / static_cast<float>(period);
        }

        if (avg_loss < 1e-10f) return 1.0f;   // all gains — RSI = 100 → 1.0

        float rs  = avg_gain / avg_loss;
        float rsi = 100.0f / (1.0f + rs);     // rsi in [0, 100], inverted form: 100 - 100/(1+rs)
        // Correction: standard RSI = 100 - 100/(1+RS)
        rsi = 100.0f - (100.0f / (1.0f + rs));

        return rsi / 100.0f;   // normalise to [0, 1]
    }

    float get_volatility(
        const std::vector<StockPrice>& prices,
        int position,
        int period
    ) {
        if (position < period)
            throw std::runtime_error("Not enough data to calculate volatility");

        std::vector<float> log_returns;
        log_returns.reserve(period);

        for (int i = position - period + 1; i <= position; ++i)
            log_returns.push_back(std::log(prices[i].nav / prices[i - 1].nav));

        float mean = std::accumulate(log_returns.begin(), log_returns.end(), 0.0f)
                    / static_cast<float>(period);

        float variance = 0.0f;
        for (float r : log_returns)
            variance += (r - mean) * (r - mean);
        variance /= static_cast<float>(period);   // population std-dev

        return std::sqrt(variance);
    }

    std::vector<float> StockPrice::to_features() const {
        return {
            (ema20  > 0.0f) ? (nav / ema20  - 1.0f) : 0.0f,
            (ema50  > 0.0f) ? (nav / ema50  - 1.0f) : 0.0f,
            (ema200 > 0.0f) ? (nav / ema200 - 1.0f) : 0.0f,
            momentum5,
            momentum20,
            rsi14,
            volatility20,
        };
    }

    std::vector<StockPrice> load_data(
        const std::filesystem::path& file_path
    ) {
        std::ifstream ifs(file_path);
        if (!ifs.is_open())
            throw std::runtime_error("Failed to open CSV file: " + file_path.string());

        std::vector<StockPrice> prices;
        std::string line;

        // Skip header
        std::getline(ifs, line);

        while (std::getline(ifs, line)) {

            if (line.empty()) continue;

            std::stringstream ss(line);
            std::string date, nav_str;
            std::getline(ss, date, ',');
            std::getline(ss, nav_str, ',');

            StockPrice price{};
            price.nav = std::stof(nav_str);
            prices.push_back(price);
        }

        const int n = static_cast<int>(prices.size());

        for (int i = 0; i < n; ++i) {
            // EMA trend indicators
            if (i >= 19)  prices[i].ema20  = get_ema(prices, i, 20);
            if (i >= 49)  prices[i].ema50  = get_ema(prices, i, 50);
            if (i >= 199) prices[i].ema200 = get_ema(prices, i, 200);

            // Momentum: rate of change (nav[T] / nav[T-N]) - 1
            if (i >= 5)  prices[i].momentum5  = (prices[i].nav / prices[i - 5].nav)  - 1.0f;
            if (i >= 20) prices[i].momentum20 = (prices[i].nav / prices[i - 20].nav) - 1.0f;

            // RSI(14) — needs at least 14 prior bars for diffs, so index >= 14
            if (i >= 14) prices[i].rsi14 = get_rsi(prices, i, 14);

            // Volatility(20) — needs 20 prior bars for log-returns
            if (i >= 20) prices[i].volatility20 = get_volatility(prices, i, 20);
        }

        for (int i = 0; i < n - 1; ++i)
            prices[i].direction = (prices[i + 1].nav > prices[i].nav);

        // Remove all entries without all emas and smas calculated
        prices.erase(prices.begin(), prices.begin()+199);
        prices.pop_back();

        return prices;
    }

    std::string StockPrice::to_string() const {
        std::ostringstream oss;
        oss << *this;
        return oss.str();
    }

    std::ostream& operator<<(std::ostream& os, const StockPrice& f) {
        os << "StockPrice {\n"
        << "  nav:          " << f.nav          << "\n"
        << "  ema20:        " << f.ema20         << "\n"
        << "  ema50:        " << f.ema50         << "\n"
        << "  ema200:       " << f.ema200        << "\n"
        << "  momentum5:    " << f.momentum5     << "\n"
        << "  momentum20:   " << f.momentum20    << "\n"
        << "  rsi14:        " << f.rsi14         << "\n"
        << "  volatility20: " << f.volatility20  << "\n"
        << "  direction:    " << (f.direction ? "UP" : "DOWN/FLAT") << "\n"
        << "}";
        return os;
    }
}