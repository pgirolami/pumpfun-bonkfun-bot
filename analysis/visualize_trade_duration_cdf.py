"""
Visualize CDF (Cumulative Distribution Function) of buy and sell trade durations.

This script queries trade durations from a SQLite database and creates
an interactive plotly chart showing the empirical CDF for both buy and sell trades,
along with the theoretical normal distribution CDF for comparison.

Usage:
    # Option 1: Run as a script
    uv run analysis/visualize_trade_duration_cdf.py
    
    # Option 2: Run interactively in VSCode
    # Install Jupyter extension, then use "Run Cell" buttons above # %% markers
"""

# %%
# Cell 1: Imports and setup
import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path to import ui.shared
sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.shared import query_trade_durations

# %%
# Cell 2: Configuration
# Database path - update this to point to your database
# The database name pattern is: bot-{name}_{wallet_short}_{mode}.db
database_name = "sniper1-prod_WmeerW84"

# Try to find the database in the data directory
project_root = Path(__file__).parent.parent
data_dir = project_root / "data"

# Look for database files matching the pattern
db_files = list(data_dir.glob(f"bot-{database_name}*.db"))
if not db_files:
    # Try without the bot- prefix
    db_files = list(data_dir.glob(f"{database_name}*.db"))

if not db_files:
    print(f"Warning: No database found matching '{database_name}' in {data_dir}")
    print(f"Please update the 'database_path' variable below with the full path to your database.")
    database_path = None
else:
    # Use the first matching database (or specify which one if multiple)
    database_path = str(db_files[0])
    if len(db_files) > 1:
        print(f"Found {len(db_files)} matching databases:")
        for i, db in enumerate(db_files):
            print(f"  {i+1}. {db}")
        print(f"Using: {database_path}")

# Alternative: Specify the full path directly
# database_path = "/Users/philippe/Documents/crypto/pumpfun-bonkfun-bot/data/bot-sniper1-prod_WmeerW84_dryrun.db"

# Start timestamp (Unix epoch milliseconds) - 0 means all data
start_timestamp = 0  # Change this to filter by time if needed


# %%
# Cell 3: Helper functions
def calculate_empirical_cdf(data: list[float]) -> tuple[list[float], list[float]]:
    """Calculate empirical CDF from data.
    
    Args:
        data: List of values
        
    Returns:
        Tuple of (sorted_values, cdf_values) where cdf_values[i] is the CDF at sorted_values[i]
    """
    if not data:
        return ([], [])
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    cdf_values = [(i + 1) / n for i in range(n)]
    
    return (sorted_data, cdf_values)


def calculate_smoothed_empirical_cdf(
    data: list[float], num_points: int = 200
) -> tuple[list[float], list[float]]:
    """Calculate kernel-smoothed empirical CDF from data.
    
    Uses a Gaussian kernel to smooth the empirical CDF, providing a smoother curve
    while maintaining the overall shape of the data distribution.
    
    Args:
        data: List of values
        num_points: Number of points to evaluate the smoothed CDF at
        
    Returns:
        Tuple of (x_values, smoothed_cdf_values)
    """
    import math
    import statistics
    
    if not data:
        return ([], [])
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    # Calculate bandwidth using Silverman's rule of thumb
    if n < 2:
        return (sorted_data, [1.0] * n)
    
    std_dev = statistics.stdev(sorted_data) if n > 1 else 0.0
    iqr = sorted_data[int(n * 0.75)] - sorted_data[int(n * 0.25)] if n > 1 else 0.0
    
    # Use Silverman's rule: h = 0.9 * min(σ, IQR/1.34) * n^(-1/5)
    if std_dev > 0 and iqr > 0:
        bandwidth = 0.9 * min(std_dev, iqr / 1.34) * (n ** (-1/5))
    elif std_dev > 0:
        bandwidth = 0.9 * std_dev * (n ** (-1/5))
    elif iqr > 0:
        bandwidth = 0.9 * (iqr / 1.34) * (n ** (-1/5))
    else:
        bandwidth = (sorted_data[-1] - sorted_data[0]) / 10.0 if len(sorted_data) > 1 else 1.0
    
    # Ensure minimum bandwidth
    bandwidth = max(bandwidth, (sorted_data[-1] - sorted_data[0]) / 100.0) if len(sorted_data) > 1 else 1.0
    
    # Create evaluation points
    min_val = sorted_data[0]
    max_val = sorted_data[-1]
    range_val = max_val - min_val
    x_values = [
        min_val - 0.1 * range_val + (max_val - min_val + 0.2 * range_val) * i / (num_points - 1)
        for i in range(num_points)
    ]
    
    # Calculate smoothed CDF using kernel smoothing
    # CDF(x) = (1/n) * sum(Φ((x - x_i) / h)) where Φ is standard normal CDF
    smoothed_cdf = []
    for x in x_values:
        cdf_sum = 0.0
        for data_point in sorted_data:
            z = (x - data_point) / bandwidth
            # Standard normal CDF: Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))
            cdf_sum += 0.5 * (1 + math.erf(z / math.sqrt(2)))
        smoothed_cdf.append(cdf_sum / n)
    
    return (x_values, smoothed_cdf)


def calculate_normal_cdf(x: float, mean: float, std_dev: float) -> float:
    """Calculate normal distribution CDF at point x.
    
    Args:
        x: Value at which to evaluate CDF
        mean: Mean of the normal distribution
        std_dev: Standard deviation of the normal distribution
        
    Returns:
        CDF value (between 0 and 1)
    """
    import math
    
    if std_dev == 0:
        return 1.0 if x >= mean else 0.0
    
    z_score = (x - mean) / std_dev
    return 0.5 * (1 + math.erf(z_score / math.sqrt(2)))


def fit_lognormal(durations: list[float], method: str = "moments") -> tuple[float, float]:
    """Fit log-normal distribution to durations.
    
    Args:
        durations: List of duration values
        method: Fitting method:
            - "moments" (default): Standard method of moments
            - "robust": Uses median/IQR for steeper fit (good for bottom/middle)
            - "robust_tail": Hybrid approach - robust median with tail-aware sigma
                (better fit at top while maintaining good fit at bottom/middle)
        
    Returns:
        Tuple of (mu, sigma) where mu and sigma are parameters of the log-normal
        (mean and std dev of the underlying normal distribution of log values)
    """
    import math
    import statistics
    
    if not durations:
        return (0.0, 1.0)
    
    log_durations = [math.log(x) for x in durations if x > 0]
    if not log_durations:
        return (0.0, 1.0)
    
    if method == "robust":
        # Use median and IQR for more robust estimation (less affected by outliers)
        # This typically gives a steeper (lower sigma) distribution
        mu = statistics.median(log_durations)
        sorted_log = sorted(log_durations)
        n = len(sorted_log)
        q1_idx = max(0, int(n * 0.25))
        q3_idx = min(n - 1, int(n * 0.75))
        iqr = sorted_log[q3_idx] - sorted_log[q1_idx]
        # For normal distribution: IQR ≈ 1.35 * sigma, so sigma ≈ IQR / 1.35
        sigma = iqr / 1.35 if iqr > 0 else statistics.stdev(log_durations) if len(log_durations) > 1 else 1.0
        # This typically gives a smaller sigma, making the distribution steeper
    elif method == "robust_tail":
        # Hybrid approach: use robust median for mu, but estimate sigma to better match the tail
        # This version gives more weight to tail percentiles for a more rounded tail
        mu = statistics.median(log_durations)
        sorted_log = sorted(log_durations)
        n = len(sorted_log)
        
        # Get percentiles for better tail estimation
        q1_idx = max(0, int(n * 0.25))
        q3_idx = min(n - 1, int(n * 0.75))
        p90_idx = min(n - 1, int(n * 0.90))
        p95_idx = min(n - 1, int(n * 0.95))
        p99_idx = min(n - 1, int(n * 0.99))
        
        q1 = sorted_log[q1_idx]
        q3 = sorted_log[q3_idx]
        p90 = sorted_log[p90_idx]
        p95 = sorted_log[p95_idx]
        p99 = sorted_log[p99_idx] if n > 10 else p95
        
        # Estimate sigma from IQR (for middle fit)
        iqr = q3 - q1
        sigma_iqr = iqr / 1.35 if iqr > 0 else 0.0
        
        # Estimate sigma from tail percentiles (for top fit - more rounded)
        # For normal distribution: P90 ≈ μ + 1.28σ, P95 ≈ μ + 1.65σ, P99 ≈ μ + 2.33σ
        sigma_p90 = (p90 - mu) / 1.28 if (p90 - mu) > 0 else 0.0
        sigma_p95 = (p95 - mu) / 1.65 if (p95 - mu) > 0 else 0.0
        sigma_p99 = (p99 - mu) / 2.33 if (p99 - mu) > 0 else 0.0
        
        # Use the maximum tail-based sigma to ensure rounded tails
        sigma_tail = max(sigma_p90, sigma_p95, sigma_p99) if any([sigma_p90 > 0, sigma_p95 > 0, sigma_p99 > 0]) else 0.0
        
        # Combine: use more conservative weighting (70/30) to maintain steepness
        # This keeps the distribution steep while still improving tail fit slightly
        if sigma_iqr > 0 and sigma_tail > 0:
            # Use 70/30 weighting (IQR/tail) to maintain steepness
            sigma = 0.7 * sigma_iqr + 0.3 * sigma_tail
            # Allow sigma to go up to 1.5x IQR-based (more conservative)
            sigma = min(sigma, sigma_iqr * 1.5)
        elif sigma_iqr > 0:
            sigma = sigma_iqr
        elif sigma_tail > 0:
            sigma = sigma_tail
        else:
            # Fallback to standard deviation
            sigma = statistics.stdev(log_durations) if len(log_durations) > 1 else 1.0
        
        # Ensure sigma is reasonable but conservative (maintain steepness)
        # Allow sigma to be up to 1.5x IQR-based (more conservative than before)
        sigma = max(sigma_iqr * 0.8, min(sigma, sigma_iqr * 1.5)) if sigma_iqr > 0 else sigma
    else:
        # Standard method of moments
        mu = statistics.mean(log_durations)
        sigma = statistics.stdev(log_durations) if len(log_durations) > 1 else 1.0
    
    return (mu, sigma)


def calculate_lognormal_cdf(x: float, mu: float, sigma: float) -> float:
    """Calculate log-normal distribution CDF at point x.
    
    Args:
        x: Value at which to evaluate CDF
        mu: Mean of the underlying normal distribution (of log values)
        sigma: Standard deviation of the underlying normal distribution (of log values)
        
    Returns:
        CDF value (between 0 and 1)
    """
    import math
    
    if x <= 0:
        return 0.0
    
    if sigma == 0:
        return 1.0 if math.log(x) >= mu else 0.0
    
    z_score = (math.log(x) - mu) / sigma
    return 0.5 * (1 + math.erf(z_score / math.sqrt(2)))


def fit_truncated_lognormal(durations: list[float]) -> tuple[float, float, float]:
    """Fit truncated log-normal distribution to durations.
    
    The distribution is truncated at the maximum observed value to prevent
    unrealistic tail extension when there are no outliers. Uses slightly larger
    sigma to create more rounding at the top before truncation.
    
    Args:
        durations: List of duration values
        
    Returns:
        Tuple of (mu, sigma, truncation_point) where truncation_point is the maximum value
    """
    import math
    import statistics
    
    if not durations:
        return (0.0, 1.0, 1.0)
    
    log_durations = [math.log(x) for x in durations if x > 0]
    if not log_durations:
        return (0.0, 1.0, 1.0)
    
    # Use robust method for fitting (median and IQR)
    mu = statistics.median(log_durations)
    sorted_log = sorted(log_durations)
    n = len(sorted_log)
    
    q1_idx = max(0, int(n * 0.25))
    q3_idx = min(n - 1, int(n * 0.75))
    p90_idx = min(n - 1, int(n * 0.90))
    p95_idx = min(n - 1, int(n * 0.95))
    
    q1 = sorted_log[q1_idx]
    q3 = sorted_log[q3_idx]
    p90 = sorted_log[p90_idx]
    p95 = sorted_log[p95_idx]
    
    iqr = q3 - q1
    sigma_iqr = iqr / 1.35 if iqr > 0 else statistics.stdev(log_durations) if len(log_durations) > 1 else 1.0
    
    # Estimate sigma from tail percentiles to create more rounding
    # For normal distribution: P90 ≈ μ + 1.28σ, P95 ≈ μ + 1.65σ
    sigma_p90 = (p90 - mu) / 1.28 if (p90 - mu) > 0 else 0.0
    sigma_p95 = (p95 - mu) / 1.65 if (p95 - mu) > 0 else 0.0
    
    # Use a weighted combination: 50% IQR (for middle fit) + 50% tail (for rounding)
    # This creates more rounding at the top before truncation
    # Give more weight to tail to ensure proper rounding
    if sigma_iqr > 0 and sigma_p95 > 0:
        sigma = 0.5 * sigma_iqr + 0.5 * max(sigma_p90, sigma_p95)
        # Allow up to 1.5x IQR for better tail rounding
        sigma = min(sigma, sigma_iqr * 1.5)
    elif sigma_iqr > 0:
        # If no tail estimate, use IQR but allow some expansion
        sigma = sigma_iqr * 1.2  # Slightly larger for rounding
    else:
        sigma = sigma_iqr
    
    # Set truncation point: use maximum observed value (not P99)
    # This ensures the CDF reaches 1.0 at the actual data maximum
    max_log = sorted_log[-1]
    truncation_point = math.exp(max_log)
    
    return (mu, sigma, truncation_point)


def calculate_truncated_lognormal_cdf(x: float, mu: float, sigma: float, truncation_point: float) -> float:
    """Calculate truncated log-normal distribution CDF at point x.
    
    The CDF is normalized so that F(truncation_point) = 1.0.
    
    Args:
        x: Value at which to evaluate CDF
        mu: Mean of the underlying normal distribution (of log values)
        sigma: Standard deviation of the underlying normal distribution (of log values)
        truncation_point: Maximum value where CDF reaches 1.0
        
    Returns:
        CDF value (between 0 and 1)
    """
    import math
    
    if x <= 0:
        return 0.0
    
    if x >= truncation_point:
        return 1.0
    
    if sigma == 0:
        return 1.0 if math.log(x) >= mu else 0.0
    
    # Calculate normal CDF at x and at truncation point
    z_x = (math.log(x) - mu) / sigma
    z_trunc = (math.log(truncation_point) - mu) / sigma
    
    cdf_x = 0.5 * (1 + math.erf(z_x / math.sqrt(2)))
    cdf_trunc = 0.5 * (1 + math.erf(z_trunc / math.sqrt(2)))
    
    # Normalize: F_truncated(x) = F(x) / F(truncation_point)
    if cdf_trunc > 0:
        return cdf_x / cdf_trunc
    else:
        return 1.0 if x >= truncation_point else 0.0


def fit_exponential(durations: list[float]) -> float:
    """Fit exponential distribution to durations.
    
    Args:
        durations: List of duration values
        
    Returns:
        Lambda (rate parameter) = 1 / mean
    """
    import statistics
    
    if not durations:
        return 1.0
    
    mean = statistics.mean(durations)
    if mean == 0:
        return 1.0
    
    return 1.0 / mean


def calculate_exponential_cdf(x: float, lam: float) -> float:
    """Calculate exponential distribution CDF at point x.
    
    Args:
        x: Value at which to evaluate CDF
        lam: Rate parameter (lambda)
        
    Returns:
        CDF value (between 0 and 1)
    """
    import math
    
    if x <= 0:
        return 0.0
    
    return 1.0 - math.exp(-lam * x)


def fit_weibull(durations: list[float]) -> tuple[float, float]:
    """Fit Weibull distribution to durations using percentile-based method for steeper fit.
    
    Uses percentiles to estimate parameters, which tends to give a steeper (higher k) distribution
    compared to method of moments.
    
    Args:
        durations: List of duration values
        
    Returns:
        Tuple of (k, lambda) where k is shape parameter and lambda is scale parameter
    """
    import math
    import statistics
    
    if not durations or len(durations) < 2:
        return (1.0, 1.0)
    
    sorted_durations = sorted(durations)
    n = len(sorted_durations)
    
    # Use percentiles for more robust estimation
    p50_idx = max(0, int(n * 0.50))
    p90_idx = min(n - 1, int(n * 0.90))
    
    p50 = sorted_durations[p50_idx]
    p90 = sorted_durations[p90_idx]
    
    # For Weibull distribution:
    # P50 = λ * (ln(2))^(1/k)
    # P90 = λ * (ln(10/9))^(1/k)  # Actually, P90 means 90% below, so we need ln(1/0.1) = ln(10)
    # Actually: F(x) = 1 - exp(-(x/λ)^k)
    # So: P50: 0.5 = 1 - exp(-(p50/λ)^k) => (p50/λ)^k = ln(2)
    # P90: 0.9 = 1 - exp(-(p90/λ)^k) => (p90/λ)^k = ln(10)
    
    # From these: (p90/p50)^k = ln(10)/ln(2) = log_2(10) ≈ 3.32
    # So: k = ln(ln(10)/ln(2)) / ln(p90/p50) = ln(3.32) / ln(p90/p50)
    
    if p50 > 0 and p90 > p50:
        ratio = p90 / p50
        if ratio > 1.0:
            # k = ln(ln(10)/ln(2)) / ln(p90/p50)
            k = math.log(math.log(10) / math.log(2)) / math.log(ratio)
            # Clamp k to reasonable range, but allow higher values for steeper distributions
            k = max(1.0, min(20.0, k))  # Higher max for steeper fits
        else:
            k = 5.0  # Default to high k (steep) if ratio is too small
    else:
        # Fallback to method of moments but with higher k
        mean = statistics.mean(durations)
        variance = statistics.variance(durations) if len(durations) > 1 else 0.0
        if variance == 0 or mean == 0:
            return (5.0, mean if mean > 0 else 1.0)
        
        cv = math.sqrt(variance) / mean
        # For steeper fit, use higher k
        k = max(3.0, min(15.0, 2.0 / cv if cv > 0 else 5.0))
    
    # Estimate lambda from P50 (median)
    # P50 = λ * (ln(2))^(1/k)
    # So: λ = P50 / (ln(2))^(1/k)
    if p50 > 0:
        lam = p50 / (math.log(2) ** (1.0 / k))
    else:
        mean = statistics.mean(durations)
        # Fallback: use mean with approximation
        # For high k, Γ(1 + 1/k) ≈ 1, so λ ≈ mean
        lam = mean
    
    lam = max(0.001, lam)  # Ensure positive
    
    return (k, lam)


def calculate_weibull_cdf(x: float, k: float, lam: float) -> float:
    """Calculate Weibull distribution CDF at point x.
    
    Args:
        x: Value at which to evaluate CDF
        k: Shape parameter
        lam: Scale parameter
        
    Returns:
        CDF value (between 0 and 1)
    """
    import math
    
    if x <= 0:
        return 0.0
    
    return 1.0 - math.exp(-((x / lam) ** k))


def fit_loglogistic(durations: list[float]) -> tuple[float, float]:
    """Fit log-logistic distribution to durations.
    
    Log-logistic is similar to log-normal but can be steeper. It's the distribution
    of a random variable whose logarithm has a logistic distribution.
    
    Args:
        durations: List of duration values
        
    Returns:
        Tuple of (alpha, beta) where alpha is scale and beta is shape parameter
    """
    import math
    import statistics
    
    if not durations:
        return (1.0, 1.0)
    
    log_durations = [math.log(x) for x in durations if x > 0]
    if not log_durations:
        return (1.0, 1.0)
    
    # Method of moments for log-logistic
    # E[X] = alpha * π / beta / sin(π / beta) for beta > 1
    # We'll use a simpler approach: estimate from log values
    mu_log = statistics.median(log_durations)  # Use median for robustness
    # Estimate beta from the spread of log values
    sorted_log = sorted(log_durations)
    n = len(sorted_log)
    p25 = sorted_log[max(0, int(n * 0.25))]
    p75 = sorted_log[min(n - 1, int(n * 0.75))]
    
    # For log-logistic, the IQR in log space relates to beta
    # Approximate: beta ≈ 2.2 / (p75 - p25) for typical cases
    spread = p75 - p25
    if spread > 0:
        beta = 2.2 / spread
        beta = max(0.5, min(10.0, beta))  # Clamp to reasonable range
    else:
        beta = 2.0
    
    # Alpha is the scale parameter (median in original scale)
    alpha = math.exp(mu_log)
    
    return (alpha, beta)


def calculate_loglogistic_cdf(x: float, alpha: float, beta: float) -> float:
    """Calculate log-logistic distribution CDF at point x.
    
    Args:
        x: Value at which to evaluate CDF
        alpha: Scale parameter
        beta: Shape parameter
        
    Returns:
        CDF value (between 0 and 1)
    """
    import math
    
    if x <= 0:
        return 0.0
    
    # Log-logistic CDF: F(x) = 1 / (1 + (x/alpha)^(-beta))
    # = (x/alpha)^beta / (1 + (x/alpha)^beta)
    ratio = x / alpha
    if ratio <= 0:
        return 0.0
    
    return (ratio ** beta) / (1 + (ratio ** beta))


def fit_gamma(durations: list[float]) -> tuple[float, float]:
    """Fit Gamma distribution to durations using method of moments.
    
    Args:
        durations: List of duration values
        
    Returns:
        Tuple of (alpha, beta) where alpha is shape parameter and beta is rate parameter
    """
    import math
    import statistics
    
    if not durations or len(durations) < 2:
        return (1.0, 1.0)
    
    mean = statistics.mean(durations)
    variance = statistics.variance(durations) if len(durations) > 1 else 0.0
    
    if variance == 0 or mean == 0:
        return (1.0, 1.0 / mean if mean > 0 else 1.0)
    
    # Method of moments:
    # E[X] = alpha / beta
    # Var[X] = alpha / beta²
    # Solving: alpha = E[X]² / Var[X], beta = E[X] / Var[X]
    alpha = (mean * mean) / variance
    beta = mean / variance
    
    # Clamp to reasonable ranges
    alpha = max(0.1, min(100.0, alpha))
    beta = max(0.001, min(1000.0, beta))
    
    return (alpha, beta)


def calculate_gamma_cdf(x: float, alpha: float, beta: float) -> float:
    """Calculate Gamma distribution CDF at point x.
    
    Uses the regularized incomplete gamma function approximation.
    
    Args:
        x: Value at which to evaluate CDF
        alpha: Shape parameter
        beta: Rate parameter
        
    Returns:
        CDF value (between 0 and 1)
    """
    import math
    
    if x <= 0:
        return 0.0
    
    # Gamma CDF: P(X <= x) = γ(alpha, beta*x) / Γ(alpha)
    # where γ is the lower incomplete gamma function
    # We'll use an approximation
    
    # For integer alpha, we can use the Poisson CDF relationship
    # For non-integer, use series expansion or approximation
    
    # Simplified approximation using chi-square relationship
    # If alpha is large, use normal approximation
    if alpha > 30:
        # Normal approximation
        mean = alpha / beta
        std_dev = math.sqrt(alpha) / beta
        z_score = (x - mean) / std_dev
        return 0.5 * (1 + math.erf(z_score / math.sqrt(2)))
    
    # For smaller alpha, use series expansion of incomplete gamma
    # γ(alpha, x) = x^alpha * sum_{k=0}^∞ (-x)^k / (k! * (alpha + k))
    # This is computationally expensive, so we'll use a simpler approximation
    
    # Use the relationship: Gamma(alpha, beta*x) = P(alpha, beta*x) * Gamma(alpha)
    # where P is the regularized incomplete gamma function
    # Approximation using continued fraction or series
    
    # For practical purposes, use a numerical integration approximation
    # or use scipy if available. For now, use a simplified approach:
    
    # If alpha is close to 1, it's approximately exponential
    if abs(alpha - 1.0) < 0.1:
        return calculate_exponential_cdf(x, beta)
    
    # Use approximation based on chi-square (when alpha = n/2, beta = 1/2)
    # For general case, use transformation
    z = beta * x
    
    # Simple approximation: use series for small z, asymptotic for large z
    if z < alpha:
        # Use series expansion (first few terms)
        result = 0.0
        term = 1.0
        for k in range(100):  # Limit iterations
            result += term
            if k >= alpha - 1:
                break
            term *= z / (alpha + k)
            if abs(term) < 1e-10:
                break
        # Normalize
        try:
            gamma_alpha = math.gamma(alpha)
            return (z ** alpha) * result / gamma_alpha
        except (ValueError, OverflowError):
            # Fallback to simpler approximation
            return min(1.0, max(0.0, 1.0 - math.exp(-z)))
    else:
        # Large z: use asymptotic expansion or complement
        # P(alpha, z) ≈ 1 - z^(alpha-1) * e^(-z) / Gamma(alpha) * (1 + (alpha-1)/z + ...)
        try:
            complement = (z ** (alpha - 1)) * math.exp(-z) / math.gamma(alpha)
            return max(0.0, min(1.0, 1.0 - complement))
        except (ValueError, OverflowError):
            # Fallback
            return min(1.0, max(0.0, 1.0 - math.exp(-z / alpha)))


# %%
# Cell 4: Main function
def main() -> None:
    """Main function to run the CDF visualization."""
    if database_path is None:
        print("Please set the database_path variable in Cell 2.")
        return
    
    if not Path(database_path).exists():
        print(f"Database file not found: {database_path}")
        return
    
    print(f"Loading trade durations from: {database_path}")
    print(f"Start timestamp filter: {start_timestamp} (0 = all data)")
    
    # Query trade durations
    buy_durations_ms, sell_durations_ms = query_trade_durations(database_path, start_timestamp)
    
    print(f"\nFound {len(buy_durations_ms)} buy trades and {len(sell_durations_ms)} sell trades")
    
    if not buy_durations_ms and not sell_durations_ms:
        print("No trade durations found. Make sure the database contains successful trades.")
        return
    
    # Calculate empirical CDFs
    buy_sorted, buy_cdf = calculate_empirical_cdf(buy_durations_ms)
    sell_sorted, sell_cdf = calculate_empirical_cdf(sell_durations_ms)
    
    # Calculate distribution parameters
    import statistics
    
    buy_mean = statistics.mean(buy_durations_ms) if buy_durations_ms else 0.0
    buy_std = statistics.stdev(buy_durations_ms) if len(buy_durations_ms) > 1 else 0.0
    
    sell_mean = statistics.mean(sell_durations_ms) if sell_durations_ms else 0.0
    sell_std = statistics.stdev(sell_durations_ms) if len(sell_durations_ms) > 1 else 0.0
    
    # Fit alternative distributions
    # Keep only: log-normal (moments), log-normal (robust_tail), truncated log-normal, Weibull, Gamma
    buy_lognormal_mu, buy_lognormal_sigma = fit_lognormal(buy_durations_ms, method="moments")
    buy_lognormal_tail_mu, buy_lognormal_tail_sigma = fit_lognormal(buy_durations_ms, method="robust_tail")
    buy_trunc_mu, buy_trunc_sigma, buy_trunc_point = fit_truncated_lognormal(buy_durations_ms)
    buy_weibull_k, buy_weibull_lam = fit_weibull(buy_durations_ms)
    buy_gamma_alpha, buy_gamma_beta = fit_gamma(buy_durations_ms)
    
    sell_lognormal_mu, sell_lognormal_sigma = fit_lognormal(sell_durations_ms, method="moments")
    sell_lognormal_tail_mu, sell_lognormal_tail_sigma = fit_lognormal(sell_durations_ms, method="robust_tail")
    sell_trunc_mu, sell_trunc_sigma, sell_trunc_point = fit_truncated_lognormal(sell_durations_ms)
    sell_weibull_k, sell_weibull_lam = fit_weibull(sell_durations_ms)
    sell_gamma_alpha, sell_gamma_beta = fit_gamma(sell_durations_ms)
    
    print(f"\nBuy trade durations:")
    print(f"  Mean: {buy_mean:.2f} ms")
    print(f"  Std Dev: {buy_std:.2f} ms")
    print(f"  Min: {min(buy_durations_ms) if buy_durations_ms else 0:.2f} ms")
    print(f"  Max: {max(buy_durations_ms) if buy_durations_ms else 0:.2f} ms")
    print(f"\n  Fitted distributions:")
    print(f"    Log-normal (moments): μ={buy_lognormal_mu:.4f}, σ={buy_lognormal_sigma:.4f}")
    print(f"    Log-normal (robust_tail): μ={buy_lognormal_tail_mu:.4f}, σ={buy_lognormal_tail_sigma:.4f}")
    print(f"    Truncated log-normal: μ={buy_trunc_mu:.4f}, σ={buy_trunc_sigma:.4f}, trunc={buy_trunc_point:.2f} ms")
    print(f"    Weibull: k={buy_weibull_k:.4f}, λ={buy_weibull_lam:.2f}")
    print(f"    Gamma: α={buy_gamma_alpha:.4f}, β={buy_gamma_beta:.6f}")
    
    print(f"\nSell trade durations:")
    print(f"  Mean: {sell_mean:.2f} ms")
    print(f"  Std Dev: {sell_std:.2f} ms")
    print(f"  Min: {min(sell_durations_ms) if sell_durations_ms else 0:.2f} ms")
    print(f"  Max: {max(sell_durations_ms) if sell_durations_ms else 0:.2f} ms")
    print(f"\n  Fitted distributions:")
    print(f"    Log-normal (moments): μ={sell_lognormal_mu:.4f}, σ={sell_lognormal_sigma:.4f}")
    print(f"    Log-normal (robust_tail): μ={sell_lognormal_tail_mu:.4f}, σ={sell_lognormal_tail_sigma:.4f}")
    print(f"    Truncated log-normal: μ={sell_trunc_mu:.4f}, σ={sell_trunc_sigma:.4f}, trunc={sell_trunc_point:.2f} ms")
    print(f"    Weibull: k={sell_weibull_k:.4f}, λ={sell_weibull_lam:.2f}")
    print(f"    Gamma: α={sell_gamma_alpha:.4f}, β={sell_gamma_beta:.6f}")
    
    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Buy Trade Durations CDF", "Sell Trade Durations CDF"),
        horizontal_spacing=0.15,
    )
    
    # Plot buy CDF
    if buy_durations_ms:
        # Empirical CDF (raw)
        fig.add_trace(
            go.Scatter(
                x=buy_sorted,
                y=buy_cdf,
                mode="lines",
                name="Empirical CDF (Buy)",
                line=dict(color="#1f77b4", width=2),
                hovertemplate="Duration: %{x:.2f} ms<br>CDF: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        
        # Kernel-smoothed empirical CDF
        buy_smooth_x, buy_smooth_cdf = calculate_smoothed_empirical_cdf(buy_durations_ms)
        fig.add_trace(
            go.Scatter(
                x=buy_smooth_x,
                y=buy_smooth_cdf,
                mode="lines",
                name="Smoothed Empirical CDF (Buy)",
                line=dict(color="#1f77b4", width=3, dash="dot"),
                hovertemplate="Duration: %{x:.2f} ms<br>CDF: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        
        # Theoretical distribution CDFs
        x_range = [
            max(0, min(buy_sorted) - 0.1 * (max(buy_sorted) - min(buy_sorted))),
            max(buy_sorted) + 0.1 * (max(buy_sorted) - min(buy_sorted)),
        ]
        x_theoretical = [
            x_range[0] + (x_range[1] - x_range[0]) * i / 200
            for i in range(201)
        ]
        
        # Log-normal CDF (standard)
        if buy_lognormal_sigma > 0:
            y_lognormal = [calculate_lognormal_cdf(x, buy_lognormal_mu, buy_lognormal_sigma) for x in x_theoretical]
            fig.add_trace(
                go.Scatter(
                    x=x_theoretical,
                    y=y_lognormal,
                    mode="lines",
                    name="Log-Normal (Buy)",
                    line=dict(color="#9467bd", width=2, dash="dot"),
                    hovertemplate="Duration: %{x:.2f} ms<br>CDF: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        
        # Log-normal CDF (robust_tail - better tail fit with conservative weighting)
        if buy_lognormal_tail_sigma > 0:
            y_lognormal_tail = [calculate_lognormal_cdf(x, buy_lognormal_tail_mu, buy_lognormal_tail_sigma) for x in x_theoretical]
            fig.add_trace(
                go.Scatter(
                    x=x_theoretical,
                    y=y_lognormal_tail,
                    mode="lines",
                    name="Log-Normal Tail-Aware (Buy)",
                    line=dict(color="#2ca02c", width=2, dash="dash"),
                    hovertemplate="Duration: %{x:.2f} ms<br>CDF: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        
        # Truncated log-normal CDF (no tail extension)
        if buy_trunc_sigma > 0:
            y_trunc = [calculate_truncated_lognormal_cdf(x, buy_trunc_mu, buy_trunc_sigma, buy_trunc_point) for x in x_theoretical]
            fig.add_trace(
                go.Scatter(
                    x=x_theoretical,
                    y=y_trunc,
                    mode="lines",
                    name="Truncated Log-Normal (Buy)",
                    line=dict(color="#d62728", width=3, dash="dashdot"),
                    hovertemplate="Duration: %{x:.2f} ms<br>CDF: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        
        # Weibull CDF (improved fitting for steeper distribution)
        if buy_weibull_lam > 0:
            y_weibull = [calculate_weibull_cdf(x, buy_weibull_k, buy_weibull_lam) for x in x_theoretical]
            fig.add_trace(
                go.Scatter(
                    x=x_theoretical,
                    y=y_weibull,
                    mode="lines",
                    name="Weibull (Buy)",
                    line=dict(color="#e377c2", width=2, dash="longdash"),
                    hovertemplate="Duration: %{x:.2f} ms<br>CDF: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        
        # Gamma CDF
        if buy_gamma_beta > 0:
            y_gamma = [calculate_gamma_cdf(x, buy_gamma_alpha, buy_gamma_beta) for x in x_theoretical]
            fig.add_trace(
                go.Scatter(
                    x=x_theoretical,
                    y=y_gamma,
                    mode="lines",
                    name="Gamma (Buy)",
                    line=dict(color="#7f7f7f", width=2, dash="longdashdot"),
                    hovertemplate="Duration: %{x:.2f} ms<br>CDF: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
    
    # Plot sell CDF
    if sell_durations_ms:
        # Empirical CDF (raw)
        fig.add_trace(
            go.Scatter(
                x=sell_sorted,
                y=sell_cdf,
                mode="lines",
                name="Empirical CDF (Sell)",
                line=dict(color="#2ca02c", width=2),
                hovertemplate="Duration: %{x:.2f} ms<br>CDF: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        
        # Kernel-smoothed empirical CDF
        sell_smooth_x, sell_smooth_cdf = calculate_smoothed_empirical_cdf(sell_durations_ms)
        fig.add_trace(
            go.Scatter(
                x=sell_smooth_x,
                y=sell_smooth_cdf,
                mode="lines",
                name="Smoothed Empirical CDF (Sell)",
                line=dict(color="#2ca02c", width=3, dash="dot"),
                hovertemplate="Duration: %{x:.2f} ms<br>CDF: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        
        # Theoretical distribution CDFs
        x_range = [
            max(0, min(sell_sorted) - 0.1 * (max(sell_sorted) - min(sell_sorted))),
            max(sell_sorted) + 0.1 * (max(sell_sorted) - min(sell_sorted)),
        ]
        x_theoretical = [
            x_range[0] + (x_range[1] - x_range[0]) * i / 200
            for i in range(201)
        ]
        
        # Log-normal CDF (standard)
        if sell_lognormal_sigma > 0:
            y_lognormal = [calculate_lognormal_cdf(x, sell_lognormal_mu, sell_lognormal_sigma) for x in x_theoretical]
            fig.add_trace(
                go.Scatter(
                    x=x_theoretical,
                    y=y_lognormal,
                    mode="lines",
                    name="Log-Normal (Sell)",
                    line=dict(color="#9467bd", width=2, dash="dot"),
                    hovertemplate="Duration: %{x:.2f} ms<br>CDF: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=2,
            )
        
        # Log-normal CDF (robust_tail - better tail fit with conservative weighting)
        if sell_lognormal_tail_sigma > 0:
            y_lognormal_tail = [calculate_lognormal_cdf(x, sell_lognormal_tail_mu, sell_lognormal_tail_sigma) for x in x_theoretical]
            fig.add_trace(
                go.Scatter(
                    x=x_theoretical,
                    y=y_lognormal_tail,
                    mode="lines",
                    name="Log-Normal Tail-Aware (Sell)",
                    line=dict(color="#2ca02c", width=2, dash="dash"),
                    hovertemplate="Duration: %{x:.2f} ms<br>CDF: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=2,
            )
        
        # Truncated log-normal CDF (no tail extension)
        if sell_trunc_sigma > 0:
            y_trunc = [calculate_truncated_lognormal_cdf(x, sell_trunc_mu, sell_trunc_sigma, sell_trunc_point) for x in x_theoretical]
            fig.add_trace(
                go.Scatter(
                    x=x_theoretical,
                    y=y_trunc,
                    mode="lines",
                    name="Truncated Log-Normal (Sell)",
                    line=dict(color="#d62728", width=3, dash="dashdot"),
                    hovertemplate="Duration: %{x:.2f} ms<br>CDF: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=2,
            )
        
        # Weibull CDF (improved fitting for steeper distribution)
        if sell_weibull_lam > 0:
            y_weibull = [calculate_weibull_cdf(x, sell_weibull_k, sell_weibull_lam) for x in x_theoretical]
            fig.add_trace(
                go.Scatter(
                    x=x_theoretical,
                    y=y_weibull,
                    mode="lines",
                    name="Weibull (Sell)",
                    line=dict(color="#e377c2", width=2, dash="longdash"),
                    hovertemplate="Duration: %{x:.2f} ms<br>CDF: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=2,
            )
        
        # Gamma CDF
        if sell_gamma_beta > 0:
            y_gamma = [calculate_gamma_cdf(x, sell_gamma_alpha, sell_gamma_beta) for x in x_theoretical]
            fig.add_trace(
                go.Scatter(
                    x=x_theoretical,
                    y=y_gamma,
                    mode="lines",
                    name="Gamma (Sell)",
                    line=dict(color="#7f7f7f", width=2, dash="longdashdot"),
                    hovertemplate="Duration: %{x:.2f} ms<br>CDF: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=2,
            )
    
    # Update layout
    database_label = Path(database_path).stem
    fig.update_layout(
        title=f"Trade Duration CDF - {database_label}<br><sub>Start timestamp: {start_timestamp if start_timestamp > 0 else 'All data'}</sub>",
        height=600,
        showlegend=True,
        template="plotly_white",
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Duration (ms)", row=1, col=1)
    fig.update_xaxes(title_text="Duration (ms)", row=1, col=2)
    
    # Update y-axis labels
    fig.update_yaxes(title_text="CDF", row=1, col=1)
    fig.update_yaxes(title_text="CDF", row=1, col=2)
    
    # Show the plot
    fig.show()
    
    print("\nChart displayed. Close the browser window to exit.")


if __name__ == "__main__":
    main()

