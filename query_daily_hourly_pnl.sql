-- Query to show positions statistics per day and hour
-- Shows: number of positions, percent winning, avg PnL, and sum PnL

SELECT
    strftime('%Y-%m-%d %H:00:00', datetime(entry_ts / 1000, 'unixepoch')) AS day_hour,
    COUNT(*) AS num_positions,
    ROUND(
        SUM(CASE WHEN realized_pnl_sol_decimal > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    ) AS percent_winning,
    ROUND(AVG(realized_pnl_sol_decimal), 6) AS avg_realized_pnl_sol_decimal,
    ROUND(SUM(realized_pnl_sol_decimal), 6) AS sum_realized_pnl_sol_decimal
FROM
    positions
WHERE
    realized_pnl_sol_decimal IS NOT NULL  -- Only include closed positions with calculated PnL
    and entry_ts>=1761992748876
GROUP BY
    strftime('%Y-%m-%d %H:00:00', datetime(entry_ts / 1000, 'unixepoch'))
ORDER BY
    day_hour DESC;

