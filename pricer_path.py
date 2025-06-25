# pricer_path.py

import pandas as pd
import numpy as np
import math
from scipy.stats import norm
import config

# ─────────────────────── Chargement historique depuis CSV ───────────────────────

def load_historical_data_from_csv() -> pd.DataFrame:
    """
    Charge la série historique 'Close' du ticker config.HIST_TICKER
    depuis le CSV sp500_price_volume.csv.
    Renvoie un DataFrame ['Date','Close'].
    """
    # Lire le CSV multi-index (tickers / champs)
    df = pd.read_csv(
        config.CSV_PATH,
        sep=';',
        skiprows=2,
        header=[0,1],
        index_col=0,
        parse_dates=True,
        encoding='utf-8'
    )
    # Extraire la colonne PX_LAST
    px = df.xs('PX_LAST', axis=1, level=1)
    if config.HIST_TICKER not in px.columns:
        raise ValueError(f"Ticker historique {config.HIST_TICKER} non trouvé dans le CSV")
    series = px[config.HIST_TICKER].dropna()
    return pd.DataFrame({
        'Date':  series.index,
        'Close': series.values
    }).reset_index(drop=True)

# Ancienne API
load_historical_data = load_historical_data_from_csv

# Charger une seule fois pour estimate_mu_sigma
_historical_df = load_historical_data()

# ─────────────────────── Estimation de μ / σ ───────────────────────

def estimate_mu_sigma(start_date: str, end_date: str) -> tuple[float, float]:
    """
    Calcule mu et sigma des rendements log sur la période [start_date, end_date]
    à partir du ticker historique chargé globalement.
    """
    d0 = pd.to_datetime(start_date)
    d1 = pd.to_datetime(end_date)
    dfw = _historical_df[
        (_historical_df['Date'] >= d0) &
        (_historical_df['Date'] <= d1)
    ]
    if len(dfw) < 2:
        raise ValueError(f"Pas assez de données entre {start_date} et {end_date}")
    prices = dfw['Close'].values
    lr = np.diff(np.log(prices))
    sigma = np.std(lr, ddof=1)
    mu = np.mean(lr) + 0.5 * sigma**2
    return mu, sigma

def _estimate_mu_sigma_series(series: pd.Series,
                              start_date: str,
                              end_date: str) -> tuple[float, float]:
    """
    Calcule mu et sigma des rendements log d'une pd.Series indexée par date
    sur la période [start_date, end_date].
    """
    ser = series.loc[start_date:end_date].dropna()
    if len(ser) < 2:
        raise ValueError(f"Pas assez de données pour la série entre {start_date} et {end_date}")
    lr = np.diff(np.log(ser.values))
    sigma = np.std(lr, ddof=1)
    mu = np.mean(lr) + 0.5 * sigma**2
    return mu, sigma

# ─────────────────────── Pricers d’options et simulateurs ───────────────────────

def binomial_crr_call(S0, K, T, r, sigma, N):
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)
    ST = np.array([S0 * (u**j) * (d**(N-j)) for j in range(N+1)])
    C = np.maximum(ST - K, 0.0)
    for i in range(N-1, -1, -1):
        C = np.exp(-r * dt) * (p * C[1:] + (1-p) * C[:-1])
    return C[0]

def trinomial_call(S0, K, T, r, sigma, N):
    dt = T / N
    v = sigma * math.sqrt(3 * dt)
    u = math.exp(v)
    d = 1 / u
    pu = 1/6 + (r - 0.5*sigma**2) * math.sqrt(dt / (12 * sigma**2))
    pd = 1/6 - (r - 0.5*sigma**2) * math.sqrt(dt / (12 * sigma**2))
    pm = 1 - pu - pd

    M = 2*N + 1
    ST = np.zeros(M)
    for k in range(M):
        move = k - N
        ST[k] = S0 * (u**move) if move>=0 else S0 * (d**(-move))
    C = np.maximum(ST - K, 0.0)
    for i in range(N, 0, -1):
        Ci = np.zeros(2*(i-1) + 1)
        for k in range(len(Ci)):
            Ci[k] = np.exp(-r*dt) * (
                pd * C[k] + pm * C[k+1] + pu * C[k+2]
            )
        C = Ci
    return C[0]

def bs_explicit_fd_call(S_max, K, T, r, sigma, M, N):
    dS = S_max / M
    dt = T / N
    grid = np.zeros((M+1, N+1))
    S = np.linspace(0, S_max, M+1)
    grid[:, -1] = np.maximum(S - K, 0)
    grid[0, :] = 0
    grid[-1, :] = S_max - K * np.exp(-r * dt * np.arange(N+1)[::-1])
    for n in reversed(range(N)):
        for i in range(1, M):
            a = 0.5*dt*(sigma**2*i**2 - r*i)
            b = 1 - dt*(sigma**2*i**2 + r)
            c = 0.5*dt*(sigma**2*i**2 + r*i)
            grid[i,n] = a*grid[i-1,n+1] + b*grid[i,n+1] + c*grid[i+1,n+1]
    return lambda x: np.interp(x, S, grid[:,0])

def simulate_heston(S0, V0, r, kappa, theta, xi, rho, T, steps):
    dt = T / steps
    S = np.zeros(steps)
    V = np.zeros(steps)
    S[0], V[0] = S0, V0
    for t in range(1, steps):
        z1 = np.random.normal()
        z2 = rho*z1 + math.sqrt(1-rho**2)*np.random.normal()
        V[t] = abs(V[t-1] + kappa*(theta-V[t-1])*dt + xi*math.sqrt(V[t-1]*dt)*z2)
        S[t] = S[t-1] * math.exp((r - 0.5*V[t-1])*dt + math.sqrt(V[t-1]*dt)*z1)
    return S

def heston_mc_call(S0, V0, K, T, r, kappa, theta, xi, rho, steps, n_paths):
    payoffs = []
    for _ in range(n_paths):
        path = simulate_heston(S0, V0, r, kappa, theta, xi, rho, T, steps)
        payoffs.append(max(path[-1] - K, 0))
    return math.exp(-r*T) * np.mean(payoffs)

def carr_madan_call_price(S0, K, T, r, sigma, alpha=1.5, N=4096, B=800):
    # avoid S0 ≤ 0 in the log-domain
    eps = 1e-8
    S0  = max(S0, eps)

    eta = B / N
    v   = np.arange(N) * eta

    def cf(u):
        # now safe: S0 >= eps > 0
        return np.exp(
            1j * u * (np.log(S0) + (r - 0.5 * sigma**2) * T)
            - 0.5 * sigma**2 * u**2 * T
        )

    psi = (
        np.exp(-r * T)
        * cf(v - 1j * (alpha + 1))
        / (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v)
    )

    x = np.exp(1j * v * np.log(K)) * psi * eta * np.exp(-1j * B * v / 2)
    y = np.fft.fft(x).real

    # mask k-values to |k| ≤ 20 to avoid overflow
    kvec_full = np.log(K) + (np.arange(N) - N / 2) * eta
    mask      = np.abs(kvec_full) <= 20
    kvec      = kvec_full[mask]
    y_sel     = y[mask]

    call_vals = y_sel * np.exp(-alpha * kvec) / math.pi
    call_vals = np.nan_to_num(call_vals, nan=0.0, posinf=0.0, neginf=0.0)

    strikes = np.exp(kvec)
    prices  = dict(zip(strikes, call_vals))

    return prices



def mc_call_antithetic(S0, K, T, r, sigma, n_paths):
    payoffs = []
    for _ in range(n_paths//2):
        z = np.random.normal()
        for sign in (+1, -1):
            ST = S0 * math.exp((r-0.5*sigma**2)*T + sigma*math.sqrt(T)*(sign*z))
            payoffs.append(max(ST - K, 0))
    return math.exp(-r*T) * np.mean(payoffs)

def bjerksund_stensland_call(S0, K, T, r, sigma):
    beta = (0.5 - r/sigma**2) + math.sqrt((r/sigma**2 - 0.5)**2 + 2*r/sigma**2)
    B_inf = beta/(beta-1)*K
    B0 = max(K, r/(r+beta*sigma**2)*B_inf)
    def h(S, t):
        d1 = (math.log(S/B0) + (r+0.5*sigma**2)*(T-t)) / (sigma*math.sqrt(T-t))
        return S*norm.cdf(d1) - B0*math.exp(-r*(T-t))*norm.cdf(d1-sigma*math.sqrt(T-t))
    return (S0-K) if S0>=B0 else h(S0, 0)

# ─────────────────────── Simulateurs de trajectoires de base ───────────────────────

def simulate_bs_path(S0, mu, sigma, seq_len, dt=1/252):
    path = np.zeros(seq_len)
    path[0] = S0
    for t in range(1, seq_len):
        z = np.random.normal()
        path[t] = path[t-1] * math.exp((mu - 0.5*sigma**2)*dt + sigma*math.sqrt(dt)*z)
    return path

def simulate_merton_path(S0, mu, sigma, lamb, mu_j, sigma_j, seq_len, dt=1/252):
    path = np.zeros(seq_len)
    path[0] = S0
    kappa = math.exp(mu_j+0.5*sigma_j**2) - 1
    for t in range(1, seq_len):
        z = np.random.normal()
        nj = np.random.poisson(lamb*dt)
        jump = np.sum(np.random.normal(mu_j, sigma_j, size=nj)) if nj>0 else 0
        path[t] = path[t-1] * math.exp((mu - 0.5*sigma**2 - lamb*kappa)*dt + sigma*math.sqrt(dt)*z + jump)
    return path

def simulate_monte_carlo_paths_historical(df_hist: pd.DataFrame,
                                         n_paths: int,
                                         seq_len: int) -> np.ndarray:
    prices = df_hist["Close"].values
    lr     = np.diff(np.log(prices))
    paths  = np.zeros((n_paths, seq_len))
    for i in range(n_paths):
        paths[i,0] = prices[-1]
        for t in range(1, seq_len):
            eps = np.random.choice(lr)
            paths[i,t] = paths[i,t-1] * math.exp(eps)
    return paths

def simulate_euler(S0, r, mu, sigma, seq_len, dt=1/252):
    path = np.zeros(seq_len)
    path[0] = S0
    for t in range(1, seq_len):
        z = np.random.normal()
        path[t] = path[t-1] + r*path[t-1]*dt + sigma*path[t-1]*math.sqrt(dt)*z
    return path

# ─────────────────────── Génération multi-ticker vs single-ticker ───────────────────────

def generate_pricer_paths_for_ticker(pricer_name: str,
                                     ticker: str,
                                     start_date: str,
                                     end_date: str,
                                     seq_len: int,
                                     n_paths: int = 1) -> np.ndarray:
    """
    Simule n_paths trajectoires de longueur seq_len pour un ticker donné.
    Calibrage mu/sigma sur la période start_date→end_date pour ce ticker.
    """
    # Lire la série PX_LAST pour le ticker
    df = pd.read_csv(
        config.CSV_PATH,
        sep=';',
        skiprows=2,
        header=[0,1],
        index_col=0,
        parse_dates=True,
        encoding='utf-8'
    )
    px = df.xs('PX_LAST', axis=1, level=1)
    if ticker not in px.columns:
        raise ValueError(f"Ticker {ticker} non trouvé dans le CSV")
    series = px[ticker].dropna()

    # Calibrer mu, sigma localement
    mu, sigma = _estimate_mu_sigma_series(series, start_date, end_date)

    # Point de départ
    S0 = series.loc[start_date:end_date].values[-1]
    r  = 0.01

    # Simulations
    paths = np.zeros((n_paths, seq_len))
    for i in range(n_paths):
        if pricer_name == "BlackScholes":
            paths[i] = simulate_bs_path(S0, mu, sigma, seq_len)
        elif pricer_name == "Merton":
            paths[i] = simulate_merton_path(S0, mu, sigma, 0.1, -0.05, 0.2, seq_len)
        elif pricer_name == "MonteCarlo":
            df_hist = series.to_frame(name='Close').reset_index()
            paths[i] = simulate_monte_carlo_paths_historical(df_hist, 1, seq_len)[0]
        elif pricer_name == "EulerExplicit":
            paths[i] = simulate_euler(S0, r, mu, sigma, seq_len)
        elif pricer_name == "BinomialCRR":
            paths[i] = _build_autoregr_path(binomial_crr_call, S0, r, sigma, seq_len)
        elif pricer_name == "Trinomial":
            paths[i] = _build_autoregr_path(trinomial_call, S0, r, sigma, seq_len)
        elif pricer_name == "ExplicitFD":
            # utiliser M=N=200 par défaut
            f = bs_explicit_fd_call(S0*2, 100, 1.0, r, sigma, M=200, N=200)
            arr = np.zeros(seq_len)
            price = f(S0)
            for t in range(seq_len):
                arr[t] = price
                price = f(price)
            paths[i] = arr
        elif pricer_name == "HestonMC":
            paths[i] = simulate_heston(S0, 0.04, r, 2.0, 0.04, 0.1, -0.7, 1.0, seq_len)
        elif pricer_name == "CarrMadan":
            arr   = np.zeros(seq_len)
            price = S0
            for t in range(seq_len):
                # recompute the full CM surface at the current underlying
                pm_dict    = carr_madan_call_price(price, 100, 1.0, r, sigma)
                strikes    = np.array(list(pm_dict.keys()))
                call_prices= np.array(list(pm_dict.values()))
                # interpolate the call price at K=100 exactly
                price      = np.interp(100.0, strikes, call_prices)
                arr[t]     = price
            paths[i] = arr

        elif pricer_name == "AntitheticMC":
            arr = np.zeros(seq_len)
            price = mc_call_antithetic(S0, 100, 1.0, r, sigma, 100000)
            for t in range(seq_len):
                arr[t] = price
                price = mc_call_antithetic(price, 100, 1.0, r, sigma, 100000)
            paths[i] = arr
        elif pricer_name == "BjerksundStensland":
            arr = np.zeros(seq_len)
            price = bjerksund_stensland_call(S0, 100, 1.0, r, sigma)
            for t in range(seq_len):
                arr[t] = price
                price = bjerksund_stensland_call(price, 100, 1.0, r, sigma)
            paths[i] = arr
        else:
            raise ValueError(f"Pricer inconnu : {pricer_name}")
        # pour pricers auto-régressifs, S0 est mis à jour dans _build_autoregr_path
    return paths

def _build_autoregr_path(pricer_fn, S0, r, sigma, seq_len):
    arr = np.zeros(seq_len)
    price = pricer_fn(S0, K=100, T=1.0, r=r, sigma=sigma, N=100)
    for t in range(seq_len):
        arr[t] = price
        price = pricer_fn(price, K=100, T=1.0, r=r, sigma=sigma, N=100)
    return arr

# Pour compatibilité avec l'ancienne API generate_pricer_paths()
def generate_pricer_paths(pricer_name, start_date, end_date, seq_len, n_paths=1):
    """
    Simule n_paths trajectoires pour le ticker historique défini dans config.
    """
    return generate_pricer_paths_for_ticker(
        pricer_name,
        config.HIST_TICKER,
        start_date,
        end_date,
        seq_len,
        n_paths
    )
