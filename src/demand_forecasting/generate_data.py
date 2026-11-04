"""Generate a synthetic but realistic dataset for "Harbour Fresh", a fictional grocery chain with
10 stores in Northern Ireland. All data is synthetic; no real company or customer data is used.

Built-in effects (the "known truth" the models should recover):
  * day-of-week pattern (Saturday peak; Sunday trading 1-6 pm only, so Sunday is lower)
  * yearly seasonality by category, and strong Christmas, Easter and Shrove Tuesday effects
  * stores closed on Christmas Day and Easter Sunday (Northern Ireland large-shop rules)
  * temperature: ice cream, water and burgers sell more when warm; soup sells more when cold
  * promotions: price cuts lift sales via a price elasticity; storable goods dip after a promotion
  * payday (last Friday of the month) uplift, chain growth, one declining and one newly launched product
  * stockouts: on some days sales are capped below demand, and the stockout flag is recorded
"""

from __future__ import annotations

from datetime import date

import holidays
import numpy as np
import pandas as pd
from dateutil.easter import easter

from .config import END, RAW, SAMPLE, SEED, START

STORES = [  # store_id, name, size factor, temperature offset (deg C)
    ("S01", "Belfast City Centre", 1.60, 0.4), ("S02", "Belfast Lisburn Road", 1.15, 0.4),
    ("S03", "Lisburn", 1.05, 0.2), ("S04", "Newry", 0.95, 0.3), ("S05", "Derry~Londonderry", 1.20, -0.2),
    ("S06", "Bangor", 0.90, 0.1), ("S07", "Ballymena", 0.85, -0.4), ("S08", "Craigavon", 1.00, 0.0),
    ("S09", "Coleraine", 0.75, -0.5), ("S10", "Enniskillen", 0.65, -0.6),
]

# sku, name, category, regular price, unit cost, shelf life (days), base units/day (size 1.0),
# temperature effect per deg C above 12, yearly amplitude, peak day of year, promo elasticity, NB dispersion
PRODUCTS = [
    ("P01", "White sliced bread 800g", "Bakery", 1.35, 0.62, 3, 95, 0.000, 0.04, 350, -1.6, 0.06),
    ("P02", "Wholemeal bread 800g", "Bakery", 1.55, 0.70, 3, 48, 0.000, 0.04, 20, -1.6, 0.08),
    ("P03", "Soda farls 4 pack", "Bakery", 1.40, 0.55, 2, 40, -0.004, 0.10, 20, -1.8, 0.10),
    ("P04", "Croissants 4 pack", "Bakery", 1.90, 0.80, 2, 22, 0.000, 0.05, 200, -2.2, 0.15),
    ("P05", "Bananas (each)", "Fresh produce", 0.22, 0.12, 5, 160, 0.000, 0.03, 100, -1.4, 0.05),
    ("P06", "Strawberries 400g", "Fresh produce", 2.50, 1.30, 3, 30, 0.020, 0.55, 185, -2.4, 0.15),
    ("P07", "Salad bag 120g", "Fresh produce", 1.10, 0.48, 4, 35, 0.018, 0.35, 190, -2.0, 0.12),
    ("P08", "Potatoes 2.5kg", "Fresh produce", 2.20, 1.10, 14, 42, -0.006, 0.12, 15, -1.5, 0.08),
    ("P09", "Semi-skimmed milk 2L", "Dairy", 1.65, 0.95, 7, 120, 0.000, 0.03, 350, -1.2, 0.04),
    ("P10", "Natural yoghurt 500g", "Dairy", 1.20, 0.55, 10, 26, 0.004, 0.10, 20, -2.0, 0.12),
    ("P11", "Cheddar 400g", "Dairy", 3.75, 2.10, 30, 34, 0.000, 0.14, 355, -2.3, 0.10),
    ("P12", "Oat drink 1L", "Dairy", 1.80, 0.85, 30, 20, 0.000, 0.05, 20, -1.9, 0.12),
    ("P13", "Vanilla ice cream 2L", "Frozen", 3.00, 1.40, 365, 10, 0.075, 0.55, 200, -2.6, 0.30),
    ("P14", "Frozen peas 1kg", "Frozen", 1.60, 0.70, 365, 22, 0.000, 0.08, 10, -2.0, 0.12),
    ("P15", "Frozen pizza", "Frozen", 2.75, 1.20, 365, 28, -0.004, 0.10, 10, -2.4, 0.18),
    ("P16", "Beef burgers 4 pack", "Meat", 3.50, 1.85, 6, 18, 0.055, 0.50, 190, -2.5, 0.25),
    ("P17", "Chicken breast 600g", "Meat", 4.25, 2.60, 6, 38, 0.000, 0.05, 190, -2.0, 0.08),
    ("P18", "Pork sausages 8 pack", "Meat", 2.40, 1.20, 6, 32, -0.004, 0.12, 15, -2.1, 0.10),
    ("P19", "Tomato soup 400g", "Ambient", 0.95, 0.38, 540, 30, -0.040, 0.30, 15, -2.0, 0.15),
    ("P20", "Pasta 500g", "Ambient", 0.85, 0.32, 720, 36, 0.000, 0.06, 20, -2.2, 0.12),
    ("P21", "Tea bags 80", "Ambient", 2.95, 1.45, 540, 24, -0.004, 0.06, 15, -2.6, 0.12),
    ("P22", "Breakfast cereal 500g", "Ambient", 2.60, 1.15, 270, 30, 0.000, 0.04, 250, -2.5, 0.12),
    ("P23", "Pancake mix", "Ambient", 1.25, 0.50, 270, 2.5, 0.000, 0.00, 1, -1.5, 0.40),
    ("P24", "Still water 6x500ml", "Drinks", 2.10, 0.85, 365, 26, 0.060, 0.40, 195, -2.2, 0.20),
    ("P25", "Cola 2L", "Drinks", 1.99, 0.80, 270, 40, 0.025, 0.15, 200, -2.8, 0.15),
    ("P26", "Orange juice 1L", "Drinks", 1.75, 0.80, 21, 28, 0.000, 0.05, 15, -2.2, 0.12),
    ("P27", "Mince pies 6 pack", "Seasonal", 1.75, 0.70, 21, 0.0, 0.000, 0.00, 1, -2.0, 0.20),
    ("P28", "Easter egg (large)", "Seasonal", 4.50, 2.20, 120, 0.0, 0.000, 0.00, 1, -1.8, 0.25),
    ("P29", "Specialty blue cheese 150g", "Dairy", 3.25, 1.70, 21, 1.4, 0.000, 0.10, 355, -2.0, 0.35),
    ("P30", "Washing-up liquid 450ml", "Household", 1.30, 0.55, 1000, 14, 0.000, 0.03, 10, -2.4, 0.15),
]
STORABLE = {"Ambient", "Household", "Frozen", "Drinks"}


def _temperature(rng, days: pd.DatetimeIndex, n_stores: int) -> np.ndarray:
    doy = days.dayofyear.values
    norm = 9.5 + 6.0 * np.sin(2 * np.pi * (doy - 110) / 365.25)
    anom = np.zeros(len(days))
    for t in range(1, len(days)):
        anom[t] = 0.8 * anom[t - 1] + rng.normal(0, 1.3)
    local = rng.normal(0, 0.6, (len(days), n_stores))
    return norm[:, None] + anom[:, None] + local


def _promotions(rng, days: pd.DatetimeIndex, products) -> tuple[np.ndarray, np.ndarray]:
    """Chain-wide promotions: 4-6 per product per year, 7 or 14 days, 20-35% off."""
    n = len(days)
    disc = np.zeros((n, len(products)))
    post = np.zeros((n, len(products)), bool)
    for j, p in enumerate(products):
        if p[0] in ("P27", "P28", "P23"):
            continue
        t = int(rng.integers(5, 40))
        while t < n:
            length = int(rng.choice([7, 14], p=[0.7, 0.3]))
            d = float(rng.choice([0.2, 0.25, 0.33]))
            disc[t:t + length, j] = d
            post[t + length:t + length + 7, j] = True
            t += length + int(rng.integers(45, 90))
    return disc, post


def generate(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = pd.date_range(START, END, freq="D")
    n_d, n_s, n_p = len(days), len(STORES), len(PRODUCTS)
    years = sorted(set(days.year))
    ni_hol = holidays.UK(subdiv="NIR", years=years)
    hol_name = np.array([ni_hol.get(d.date(), "") for d in days], dtype=object)
    easter_days = {pd.Timestamp(easter(y)) for y in years}
    closed = np.array([(d.month == 12 and d.day == 25) or d in easter_days for d in days])

    temp = _temperature(rng, days, n_s) + np.array([s[3] for s in STORES])[None, :]
    disc, post = _promotions(rng, days, PRODUCTS)

    dow = days.day_of_week.values                              # Monday = 0
    dow_f = np.array([0.92, 0.90, 0.93, 0.98, 1.13, 1.30, 0.84])[dow]
    doy = days.dayofyear.values
    t_years = (days - days[0]).days.values / 365.25
    last_fri = np.array([d.day_of_week == 4 and (d + pd.Timedelta(days=7)).month != d.month for d in days])
    payday = np.where(last_fri, 1.06, 1.0) * np.where(np.roll(last_fri, 1), 1.04, 1.0)

    xmas = np.ones(n_d)
    for i, d in enumerate(days):
        if d.month == 12 and 18 <= d.day <= 24:
            xmas[i] = 1.25 + 0.06 * (d.day - 18) if d.day < 24 else 1.45
        elif d.month == 12 and d.day == 26:
            xmas[i] = 0.75
        elif d.month == 12 and d.day >= 27 or (d.month == 1 and d.day <= 2):
            xmas[i] = 0.88
        elif d.month == 1 and d.day <= 20:
            xmas[i] = 0.93
    bank = np.where(hol_name != "", 1.08, 1.0) * np.where(np.roll(hol_name != "", -1), 1.10, 1.0)

    size = np.array([s[2] for s in STORES])
    store_sku = rng.lognormal(0, 0.18, (n_s, n_p))
    demand = np.zeros((n_d, n_s, n_p))
    for j, (sku, _, cat, price, cost, shelf, base, tb, amp, peak, el, disp) in enumerate(PRODUCTS):
        season = 1 + amp * np.cos(2 * np.pi * (doy - peak) / 365.25)
        trend = np.exp(0.03 * t_years)
        if sku == "P25":                                      # cola: declining 6% a year
            trend = np.exp(-0.06 * t_years)
        if sku == "P12":                                      # oat drink launched March 2024
            launch = (days - pd.Timestamp("2024-03-01")).days.values
            trend = np.where(launch < 0, 0.0, 1 - np.exp(-np.maximum(launch, 0) / 60)) * np.exp(0.10 * t_years)
        mult = season * trend * dow_f * payday * xmas * bank
        if cat == "Bakery" or sku in ("P09",):
            mult = mult * np.where(dow == 6, 0.85, 1.0)        # fresh basics even lower on short Sundays
        if sku == "P27":                                      # mince pies: 1 Nov - 31 Dec, ramping up
            in_season = (days.month >= 11).astype(float)
            ramp = np.clip((doy - 305) / 50, 0, 1)
            mult = 45 * in_season * (0.25 + 0.75 * ramp) * dow_f * xmas
        if sku == "P28":                                      # Easter eggs: 6 weeks before Easter
            mult = np.zeros(n_d)
            for e in easter_days:
                dd = (e - days).days.values
                w = (dd > 0) & (dd <= 42)
                mult[w] = 30 * np.exp(-dd[w] / 9) * dow_f[w]
        if sku == "P23":                                      # pancake mix: Shrove Tuesday spike
            boost = np.ones(n_d)
            for e in easter_days:
                st = e - pd.Timedelta(days=47)
                dd = (st - days).days.values
                boost[(dd >= 0) & (dd <= 6)] = 1 + 14 * np.exp(-dd[(dd >= 0) & (dd <= 6)] / 1.5)
            mult = mult * boost
        temp_f = np.exp(tb * (temp - 12))                    # (days, stores)
        promo_f = (1 - disc[:, j]) ** el
        dip = np.where(post[:, j] & (cat in STORABLE), 0.85, 1.0)
        scale = (base if base > 0 else 1.0)
        demand[:, :, j] = scale * mult[:, None] * temp_f * (promo_f * dip)[:, None] * size[None, :] * store_sku[None, :, j]

    demand[closed] = 0.0
    # Negative binomial noise around the mean (gamma-Poisson mixture)
    disp = np.array([p[11] for p in PRODUCTS])[None, None, :]
    lam = rng.gamma(1 / disp, demand * disp + 1e-12)
    units_demand = rng.poisson(lam)

    # Stockouts: availability failures cap sales; more likely during promotions
    p_out = 0.012 + 0.05 * (disc > 0)[:, None, :]
    out = (rng.random((n_d, n_s, n_p)) < p_out) & (units_demand > 2) & ~closed[:, None, None]
    frac = rng.uniform(0.3, 0.85, (n_d, n_s, n_p))
    units = np.where(out, np.floor(units_demand * frac), units_demand).astype(int)

    d_idx, s_idx, p_idx = np.meshgrid(np.arange(n_d), np.arange(n_s), np.arange(n_p), indexing="ij")
    reg_price = np.array([p[3] for p in PRODUCTS])
    sales = pd.DataFrame({
        "date": days[d_idx.ravel()],
        "store_id": np.array([s[0] for s in STORES])[s_idx.ravel()],
        "sku": np.array([p[0] for p in PRODUCTS])[p_idx.ravel()],
        "units": units.ravel(),
        "price": np.round(reg_price[p_idx.ravel()] * (1 - disc[d_idx.ravel(), p_idx.ravel()]), 2),
        "on_promo": (disc[d_idx.ravel(), p_idx.ravel()] > 0).astype(int),
        "stockout": out.ravel().astype(int),
        "store_closed": closed[d_idx.ravel()].astype(int),
    })
    sales = sales.sort_values(["store_id", "sku", "date"]).reset_index(drop=True)
    # Products are only listed once launched (oat drink) or in season (seasonal lines)
    launched = ~((sales["sku"] == "P12") & (sales["date"] < "2024-03-01"))
    sales = sales[launched].reset_index(drop=True)

    products = pd.DataFrame([p[:6] for p in PRODUCTS],
                            columns=["sku", "product", "category", "regular_price", "unit_cost", "shelf_life_days"])
    stores = pd.DataFrame([s[:3] for s in STORES], columns=["store_id", "store", "size_index"])
    weather = pd.DataFrame({"date": np.repeat(days, n_s), "store_id": np.tile([s[0] for s in STORES], n_d),
                            "temp_c": np.round(temp.ravel(), 1)})
    calendar = pd.DataFrame({"date": days, "holiday": hol_name, "store_closed": closed.astype(int)})
    truth = pd.DataFrame({"sku": [p[0] for p in PRODUCTS], "promo_elasticity": [p[10] for p in PRODUCTS],
                          "temp_effect_per_degree": [p[7] for p in PRODUCTS]})
    # Expected demand (the noise-free mean) is kept ONLY to measure the best accuracy any model could
    # reach; no model ever reads it.
    expected = pd.DataFrame({"date": days[d_idx.ravel()], "store_id": np.array([s[0] for s in STORES])[s_idx.ravel()],
                             "sku": np.array([p[0] for p in PRODUCTS])[p_idx.ravel()],
                             "expected_units": np.round(demand.ravel(), 3)})
    return {"sales": sales, "products": products, "stores": stores, "weather": weather,
            "calendar": calendar, "known_truth": truth, "expected_demand": expected}


def write(seed: int = SEED) -> dict[str, pd.DataFrame]:
    data = generate(seed)
    RAW.mkdir(parents=True, exist_ok=True)
    SAMPLE.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        df.to_csv(RAW / f"{name}.csv", index=False)
    s = data["sales"]
    s[(s.store_id == "S01") & (s.date >= "2025-12-01")].to_csv(SAMPLE / "sales_sample_S01_dec2025.csv", index=False)
    return data


def load() -> dict[str, pd.DataFrame]:
    if not (RAW / "sales.csv").exists():
        return write()
    out = {n: pd.read_csv(RAW / f"{n}.csv", keep_default_na=False if n == "calendar" else True)
           for n in ("sales", "products", "stores", "weather", "calendar", "known_truth", "expected_demand")}
    for n in ("sales", "weather", "calendar", "expected_demand"):
        out[n]["date"] = pd.to_datetime(out[n]["date"])
    return out


if __name__ == "__main__":
    d = write()
    print({k: len(v) for k, v in d.items()})
