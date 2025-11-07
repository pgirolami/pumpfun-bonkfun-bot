-- Query to calculate PnL for each wallet balance
-- For each wallet_balance row:
-- 1. Computes PnL as the difference between the current balance and the first (earliest) balance
-- 2. Sums realized_pnl_sol_decimal from positions where exit_ts <= wallet_balance.timestamp

SELECT
    SUBSTRING(wb.wallet_pubkey, 1, 8) AS wallet,
    wb.timestamp,
    datetime(wb.timestamp / 1000, 'unixepoch', 'localtime') || '.' || printf('%03d', wb.timestamp % 1000) AS timestamp_datetime,
    wb.balance_sol,
    wb.balance_lamports,
    wb.run_id,
    (
        SELECT wb_first.balance_sol
        FROM wallet_balances wb_first
        WHERE wb_first.wallet_pubkey = wb.wallet_pubkey
        ORDER BY wb_first.timestamp ASC
        LIMIT 1
    ) AS first_balance_sol,
    wb.balance_sol - (
        SELECT wb_first.balance_sol
        FROM wallet_balances wb_first
        WHERE wb_first.wallet_pubkey = wb.wallet_pubkey
        ORDER BY wb_first.timestamp ASC
        LIMIT 1
    ) AS wallet_pnl,
    COALESCE(
        (
            SELECT SUM(p.realized_pnl_sol_decimal)
            FROM positions p
            WHERE p.exit_ts IS NOT NULL
              AND p.realized_pnl_sol_decimal IS NOT NULL
              AND p.exit_ts <= wb.timestamp
        ),
        0.0
    ) AS cumulative_realized_pnl
FROM
    wallet_balances wb
ORDER BY
    wb.wallet_pubkey,
    wb.timestamp ASC;

