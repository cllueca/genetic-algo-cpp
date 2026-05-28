#pragma once

#include <filesystem>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Raw price record — stores NAV, computed indicators, and the direction label.
//
// direction: true  → next day's NAV > today's NAV  (price went up)
//            false → next day's NAV ≤ today's NAV  (price flat or down)
//
// The last entry in any loaded dataset has direction = false by convention
// (no "next day" exists), and should be excluded from training.
// ─────────────────────────────────────────────────────────────────────────────
namespace data::price_direction {

    struct StockPrice {
    public:
        float nav = 0.0f;

        // ── Trend indicators (exponential moving averages) ─────────────────────
        float ema20  = 0.0f;   // short-term trend
        float ema50  = 0.0f;   // medium-term trend
        float ema200 = 0.0f;   // long-term trend

        // ── Momentum indicators ───────────────────────────────────────────────────
        // Rate-of-change over N days: (nav[T] / nav[T-N]) - 1
        float momentum5  = 0.0f;   // ~1-week momentum
        float momentum20 = 0.0f;   // ~1-month momentum

        // ── Mean-reversion / overbought-oversold ──────────────────────────────────
        // Classic 14-period RSI, clamped to [0, 1] for network compatibility.
        float rsi14 = 0.0f;

        // ── Volatility ────────────────────────────────────────────────────────────
        // Rolling 20-day standard deviation of daily log-returns.
        // High volatility = noisy regime; low volatility = trending regime.
        float volatility20 = 0.0f;

        // ── Label ────────────────────────────────────────────────────────────────
        bool direction = false;   // true → price went UP the next day

        // ── Normalised feature vector ready for network input ─────────────────────
        // Returns: { nav/ema20-1, nav/ema50-1, nav/ema200-1,
        //            momentum5, momentum20, rsi14, volatility20 }
        // All values are dimensionless and roughly centred near zero.
        std::vector<float> to_features() const;

        // String conversion that reuses operator<<
        std::string to_string() const;

    private:
        std::vector<std::string> available_sources = {
            "finect",
            "yfinance"
        };
    };


    // ─────────────────────────────────────────────────────────────────────────────
    // Low-level indicator helpers (operate on raw nav values via StockPrice vec)
    // ─────────────────────────────────────────────────────────────────────────────

    float get_sma(const std::vector<StockPrice>& prices, int position, int period);

    float get_ema(const std::vector<StockPrice>& prices, int position, int period);

    float get_rsi(const std::vector<StockPrice>& prices, int position, int period = 14);

    // Rolling standard deviation of log-returns over `period` days
    // log_return[i] = ln(nav[i] / nav[i-1])
    // volatility    = std_dev of the last `period` log-returns
    float get_volatility(const std::vector<StockPrice>& prices, int position, int period = 20);

    // ─────────────────────────────────────────────────────────────────────────────
    // Data loading
    // ─────────────────────────────────────────────────────────────────────────────

    std::ostream& operator<<(std::ostream& os, const StockPrice& f);

    std::vector<StockPrice> load_data(const std::filesystem::path& file_path);
}