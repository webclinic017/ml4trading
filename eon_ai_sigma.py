import math
import pandas as pd
from zipline import run_algorithm
from zipline.data import bundles
from zipline.errors import SymbolNotFound
from zipline.pipeline import Pipeline, CustomFactor, factors, filters
from zipline.pipeline.domain import US_EQUITIES
from zipline.pipeline.filters import StaticAssets
from zipline.pipeline.data import Column, DataSet, USEquityPricing
from zipline.pipeline.loaders.frame import DataFrameLoader
from zipline.api import schedule_function, order_target_percent, date_rules, time_rules, record, attach_pipeline, \
    pipeline_output, get_open_orders, cancel_order
from zipline.finance import commission
from collections import defaultdict
from pypfopt.hierarchical_portfolio import HRPOpt
from sigma import TradeTwoSigma

now = pd.Timestamp('2022-3-4', tz='UTC')
DATA_STORE = 'assets_3_7_22.h5'
sigma_value = TradeTwoSigma('f81d5fdc-89fb-4795-b35a-13792a76c203').calculate_portfolio(
    asofDate=now - pd.Timedelta(days=2))
kept_pos = TradeTwoSigma('f81d5fdc-89fb-4795-b35a-13792a76c203').list_position(reformat=True)

def load_predictions(bundle):
    predictions = (pd.read_hdf(DATA_STORE, 'lgb/train/01')
                   .append(pd.read_hdf(DATA_STORE, 'lgb/test/01')))
    tickers = predictions.index.get_level_values('symbol').unique().tolist()
    predictions = (predictions.loc[~predictions.index.duplicated()]
                   .iloc[:, :10]
                   .mean(1)
                   .sort_index()
                   .dropna()
                   .to_frame('prediction'))
    check = True
    while check:
        try:
            assets = bundle.asset_finder.lookup_symbols(tickers, as_of_date=None)
            check = False
        except SymbolNotFound as e:
            del_ticker = str(e).split("'")[1]
            print(del_ticker)
            tickers.remove(del_ticker)
    sigma_sids = defaultdict()
    for t in tickers:
        if t in TradeTwoSigma().universe:
            try: sigma_sids[t] = TradeTwoSigma().lookup_figi(str(t))['FIGI']
            except TypeError: tickers.remove(t)
        else:
            print(t)
            tickers.remove(t)
    predicted_sids = pd.Int64Index([asset.sid for asset in assets])
    ticker_map = dict(zip(tickers, predicted_sids))
    print('---Above are deleted tickers---')
    return (predictions
            .unstack('symbol')
            .rename(columns=ticker_map)
            .prediction
            .tz_localize('UTC')), assets, sigma_sids
predictions, assets, sigma_sids = load_predictions(bundles.load('sep'))
dates = predictions.index.get_level_values('date')


class SignalData(DataSet):
    predictions = Column(dtype=float)
    domain = US_EQUITIES
signal_loader = {SignalData.predictions: DataFrameLoader(SignalData.predictions, predictions)}


class MLSignal(CustomFactor):
    """Converting signals to Factor, so we can rank and filter in Pipeline"""
    inputs = [SignalData.predictions]
    window_length = 1

    def compute(self, today, assets, out, predictions):
        out[:] = predictions


def compute_signals():
    signals = MLSignal()
    return Pipeline(columns={
        'longs': signals.top(25, mask=signals > 0),
        'shorts': signals.bottom(25, mask=signals < 0)},
        screen=StaticAssets(assets))


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    output = pipeline_output('signals')['longs'].astype(int)
    context.longs = output[output != 0].index
    if len(context.longs) < context.min_positions:
        context.divest = set(context.portfolio.positions.keys())
    else:
        context.divest = context.portfolio.positions.keys() - context.longs
    df = (output.append(pipeline_output('signals')['shorts'].astype(int).mul(-1)))
    holdings = df[df != 0]
    other = df[df == 0]
    other = other[~other.index.isin(holdings.index) & ~other.index.duplicated()]
    context.trades = holdings.append(other)
    assert len(context.trades.index.unique()) == len(context.trades)

def rebalance_hierarchical_risk_parity(context, data):
    """
    Execute orders according to schedule_function() date & time rules.
    Uses PyPortfolioOpt to optimize weights
    """
    for symbol, open_orders in get_open_orders().items():
        for open_order in open_orders:
            cancel_order(open_order)

    for asset in context.divest:
        order_target_percent(asset, target=0)

    if len(context.longs) > context.min_positions:
        returns = (data.history(context.longs, fields='price',
                                bar_count=252 + 1,  # for 1 year of returns
                                frequency='1d')
                   .pct_change()
                   .dropna(how='any'))
        try:
            hrp_weights = HRPOpt(returns=returns).optimize()
            for asset, target in hrp_weights.items():
                order_target_percent(asset=asset, target=target)
        except ValueError: pass


def rebalance_hrp_sigma(context, data):
    for symbol, open_orders in get_open_orders().items():
        for open_order in open_orders:
            cancel_order(open_order)
    for asset in context.divest:
        TradeTwoSigma('f81d5fdc-89fb-4795-b35a-13792a76c203').submit_order(f"{sigma_sids[str(asset).split('[')[1].split(']')[0]]}", 0)

    if len(context.longs) > context.min_positions:
        prices = data.history(context.longs, fields='price',
                              bar_count=252 + 1,  # for 1 year of returns
                              frequency='1d')
        returns = prices.pct_change().dropna(how='any')
        hrp_weights = HRPOpt(returns=returns).optimize()
        import pickle
        with open('hrp_opt.pickle', 'wb') as f:
            pickle.dump(hrp_weights, f)
        for asset, weight in hrp_weights.items():
            share = math.floor(weight * TradeTwoSigma('f81d5fdc-89fb-4795-b35a-13792a76c203'
                                                      ).BASE_CAPITAL / prices[asset].dropna().tail(1))
            # share = math.floor(weight_ * sigma_value / prices[asset].dropna().tail(1))
            ticker = sigma_sids[str(asset).split('[')[1].split(']')[0]]
            if ticker in kept_pos.keys():
                share -= kept_pos[ticker]['totalShares']
            TradeTwoSigma('f81d5fdc-89fb-4795-b35a-13792a76c203').submit_order(f"{ticker}", share)


def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    record(leverage=context.account.leverage,
           longs=context.longs,
           shorts=context.shorts)


def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    context.min_positions = 20
    context.trades = pd.Series()
    context.longs = context.shorts = 0
    context.universe = assets
    context.set_commission(commission.PerShare(cost=TradeTwoSigma().COMMISSION, min_trade_cost=None))
    schedule_function(rebalance_hrp_sigma, date_rules.every_day(), time_rules.market_open(hours=1, minutes=30))
    schedule_function(record_vars, date_rules.every_day(), time_rules.market_close())
    pipeline = compute_signals()
    attach_pipeline(pipeline, 'signals')

results = run_algorithm(start=now,
                        end=now,
                        initialize=initialize,
                        before_trading_start=before_trading_start,
                        capital_base=1e8,
                        data_frequency='daily',
                        bundle='sep',
                        custom_loader=signal_loader)  # need to modify zipline