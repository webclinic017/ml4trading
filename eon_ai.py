#!/usr/bin/env python
from __future__ import absolute_import, unicode_literals

# SaaS Live
# Version: 03.1
# Simple:
#      No partial fills.
#      No multiple day rolls

""" January 30, 2019  HotChili Analytics Software as a Service (SaaS) Algorithm.  Done especially for that sector of the stock market companies that use software and cloud assets to support other entities.  The stocks are handpicked and comprise the trading universe, ususally around 30 stocks.  The algo can operate in either long/short or long mode only.
#The economic thesis is that SaaS companies value is very hard to determine by the stock market.  This creates a somewhat volatile group as varying forces try and determine the direction of these companies.  While that is happening, there is differences in price and large swings in valuation.  This algorithm uses that uncertainty and trades using a mean reversion calculation to trade daily.  Its been showing great returns in a two year backtest as the sector has done well and been up over the last two years.  The trade should continue into the future as the cloud is only getting bigger with more data, Internet of Things and addtional capabilities.(video comes to mind).
February 13, 2019
#### Modification to allow only QQQ as a short ETF to hedge the portfolio.  Mod takes place in the routine set_weights. approx line 412.  The prorate was also commented out in the shor area of the rebalance routine. approx line 473"""

#Algorithm:     hca_SaaS_v03-01-01
#Authors:       ajjc, jjc
#Date:          2019-04-15
#ChangeLog:     Added in prorate to reduce single position concentrations.
#Date:          2019-02-13
#ChangeLog:     Added a QQQ short only modification as requested to allow for hedging.
#Date:    :     2020-02-22

#Algorithm:     hca_eonum_v01a
#Authors:       ajjc
#Date:          2021-12-02
#ChangeLog:     Modified to incorporate Eonum AI algo.
#                  1. Find StaticAssets from h5 file(put in hca_segsecm_config.py)
#                  2. Stitch new factors in from eonum backtest (parities.py)
#                  3. Run backtest using long/short HRP portfolio opt.
#                  4. Integrate into existing hca_segsecm.py by creating long-short factor 
#                     from HRP opt of asset selection portfolio consisting of side_weightsvalues (1=long/-1=short)
#                  5. Run live-algo on paper account.

from zipline.pipeline.data import USEquityPricing as USEP
from zipline.pipeline.factors import Returns, AverageDollarVolume
from zipline.pipeline import Pipeline

from zipline import run_algorithm
from zipline.api import symbol, symbols, attach_pipeline, date_rules, order_target_percent, pipeline_output, record, schedule_function

from zipline import extension_args as  ext_args  # Access to commandline arguments

from zipline.finance import commission, slippage

from zipline.pipeline import Pipeline
from zipline.pipeline import factors, filters, classifiers
from zipline.pipeline.factors import CustomFactor, SimpleMovingAverage, AverageDollarVolume, Returns
from zipline.pipeline.filters import StaticAssets
from zipline.pipeline.data import USEquityPricing

# Eonum
#from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.data import Column, DataSet
#from zipline.pipeline.domain import US_EQUITIES
#from zipline.pipeline.filters import StaticAssets
from zipline.pipeline.loaders.frame import DataFrameLoader

from zipline.data import bundles

import zipline.utils.events
from zipline.utils.events import (
    EventManager,
    make_eventrule,
    date_rules,
    time_rules,
    calendars,
    AfterOpen,
    BeforeClose
)

from zipline.api import get_open_orders, order, cancel_order
from zipline.api import (slippage,
                         commission,
                         set_slippage,
                         set_commission,
                         record,
                         sid)
import pandas as pd
import numpy as np
import scipy.stats as stats

from pypfopt.hierarchical_portfolio import HRPOpt
from collections import OrderedDict


from six import viewkeys
#import matplotlib.pyplot as plt
from datetime import datetime
import pytz
from pytz import timezone as _tz  # Python only does once, makes this portable.
                                    #   Move to top of algo for better efficiency.
import sys, os, inspect
from distutils.util import strtobool
import logbook
import pathlib as pl

from pathlib import Path
import pprint as pp

def get_script_dir(follow_symlinks=True):
    if getattr(sys, 'frozen', False): # py2exe, PyInstaller, cx_Freeze
        path = os.path.abspath(sys.executable)
    else:
        path = inspect.getabsfile(get_script_dir)
    if follow_symlinks:
        path = os.path.realpath(path)
    return os.path.dirname(path)

path = get_script_dir()

# aws_ec2
from os import getenv
HCA_RELEASE_STRAT_DIR = getenv("HCA_RELEASE_STRAT_DIR", path)
sys.path.append(os.path.abspath(HCA_RELEASE_STRAT_DIR))
LOCAL_ZL_LIVE_PATH = HCA_RELEASE_STRAT_DIR #Set up local execution path


LOCAL_BOT_LIVE_PATH=Path(HCA_RELEASE_STRAT_DIR).parent / "telegram"
sys.path.append(os.path.abspath(LOCAL_BOT_LIVE_PATH))

sys.path.append(os.path.abspath(LOCAL_ZL_LIVE_PATH))
print ("Added to sys.path: LOCAL_ZL_LIVE_PATH = {} LOCAL_BOT_LIVE_PATH = {}".format(LOCAL_ZL_LIVE_PATH, LOCAL_BOT_LIVE_PATH))


# This has all configuration information.
import eon_ai_config as scfg

'''
StratList20211202   =
[
    'EonumTech',
     'SaasTech',   
]
'''

### Param Handling Section
ExtParams           = ext_args.__dict__ #Pick up External Command Line Custom Parameters(for this run only)
AvailableStrategies = [x for x in scfg.py['strats']['segm_sect'].keys()]
# Default Strategy is the first in the list.
if ExtParams and ExtParams["algo_name"]:
    if ExtParams["algo_name"] in AvailableStrategies:
        ThisStrat = ExtParams["algo_name"]
        print("Using External Parameter Strategy: {}".format(ThisStrat))

else:
    ThisStrat           = AvailableStrategies[0] #Default
    print("Using Default Strategy: {}".format(ThisStrat))
ParamList           = [x for x in scfg.py['strats']['segm_sect'][ThisStrat].keys()]
NonNumericParams =['algo', 'algo_name', 'IS_LIVE', 'PSF', 'IS_PERSIST']
#Start:AlgoParams
AlgoParms = scfg.py['strats']['segm_sect'][ThisStrat] #Pick up default parameters from scfg fot this strategy.
############## Default Parameter Constants from scfg file #############
SHORT_PCT       = 1.0 - AlgoParms['LONG_PCT']  #
VOL_DAYS = 5 #Usual-3 #2 #2=Daily, 10=Weekly

# 
# Number of max longs and shorts allowed in Class Strategy.
# For backtesting long-only, set shorts to 0.
# Also, can use AlgoParms['LONG_PCT']=1.0 for Live-trading, as that'll give you a weighted short portfolio that isn't executed.
STRAT_MAX_LONGS  = 200
STRAT_MAX_SHORTS = 200 

#Eonum factor parameters
NUM_LONGS_FACT   =  40 #Number of assets
NUM_SHORTS_FACT  =  40
MIN_PRICE_FACT   =  10     #avg min price
MIN_VOLUME_FACT  =  250000 #avg min volume


TRACK_ORDERS_ON = False # First call to track_orders will init that object and set this flag. (hack for __contains__ on TradingAlgorithm)
PVR_ON          = False # First call to pvr will init that object and set this flag. (hack for __contains__ on TradingAlgorithm)
DEBUG           = scfg.py['DEBUG'] #Printing debug
# Reset Params with external commandline changes.(e.g. -x name=value )

IsExtVarSet = set(ParamList) and set(ExtParams)
for key in IsExtVarSet:
    if key not in NonNumericParams:
        AlgoParms[key] = float(ExtParams[key])
    else:
        if key == 'IS_LIVE': # run algo live with broker
            AlgoParms[key]=bool(strtobool(ExtParams[key]))
        if key == 'IS_PERSIST': # remove state file
            AlgoParms[key]=bool(strtobool(ExtParams[key]))
        if key == 'algo_name':
            AlgoParms[key]=str(ExtParams[key])
        if key == 'PSF': #Portfolio Summary File(PSF) for Bot communication
            AlgoParms[key]=str(ExtParams[key])

SHORT_PCT       = 1.0 - AlgoParms['LONG_PCT']

#print(["{}={}".format(x, AlgoParms[x]) for x in IsExtVarSet])
print("---------------Current Algo Params after CmdLine Adjustment:--------------")
print([(n,v) for n, v in AlgoParms.items()])
#print(*(f"{n} {v}" for n, v in AlgoParms.items()), sep="\n")

if DEBUG:
    cfg_list =  [(x,y,scfg.py['strats'][x][y]) for x in scfg.py['strats'] for y in scfg.py['strats'][x] ]
    print("OriginalDefaultParams:")
    print(cfg_list)
    
def create_logs_path(context, is_live, log_root_pth): # return log Path(). Create dirs if needed.
    if is_live:
        today        = datetime.today() #Current day live
        name_sim     = "Live"
    else:
        today        = context.get_datetime(tz=None) #simulation day
        name_sim     = "Sim"

    p            = pl.Path(LOCAL_ZL_LIVE_PATH, 'Data', name_sim)
    pa            = pl.Path(LOCAL_ZL_LIVE_PATH, 'Data', name_sim, 'algostats')
    p.mkdir(exist_ok=True, parents=True)
    pa.mkdir(exist_ok=True, parents=True)
    
    return today, name_sim, p, pa
    
if not AlgoParms['IS_PERSIST']: # remove state file
    pass  
        
#End:AlgoParams

###################### Custom Factors #####################################
def price_sim(data, stock, is_live=True):
    if not is_live:
        price_lb = data.history(stock, fields='open', bar_count=2, frequency='1d')
        if type(stock) is list :
            price = price_lb.transpose().iloc[:,0] #Last trading day's open for array of stocks
            if np.isnan(price).any():
                price = data.current(stock, 'open')     
        else:
            price = price_lb[0] #Last trading day's open for single stock        
    else:
        price = data.current(stock, 'price')
    return price 


def prorate(dist, max_alloc=0.1):
    if (dist is None) or (len(dist) == 0):  #Pass on empty dist
        return dist
    cur_pr  = np.ma.array(dist)
    anymore = cur_pr[cur_pr>=max_alloc].count()
    while(anymore > 0):
        dist_m           = cur_pr
        redist           = (dist_m[cur_pr>=max_alloc]).sum() - (max_alloc)*len(dist_m[dist_m>=max_alloc])
        prd_mask         = np.ma.greater_equal(dist_m, max_alloc)
        prd              = np.ma.array(dist_m, mask = prd_mask)
        prd_norm         = prd/prd.sum()
        delta_redistrib  = redist*prd_norm
        cur_pr           = delta_redistrib + prd
        anymore = (cur_pr[cur_pr>=max_alloc]).count()
        #cur_pr[cur_pr >=max_alloc] = ma.masked

    final_pr = cur_pr.data
    final_pr[cur_pr.mask] =  max_alloc
    #final_pr[final_pr>=max_alloc] = max_alloc
    final_normed = final_pr/final_pr.sum()
    print("prorate: len={} sum={}".format(len(final_normed), final_normed.sum()))
    return final_normed
    # wts_pr = prorate(allocation, max_alloc= 2.0 * context.maxportfoliobin)  put this somewhere

# Logging. Following imports are not approved in Quantopian
####################################################################################


log_format = "{record.extra[algo_dt]}  {record.message}"

zipline_logging = logbook.NestedSetup([
    logbook.StreamHandler(sys.stdout, level=logbook.INFO, format_string=log_format),
    logbook.StreamHandler(sys.stdout, level=logbook.DEBUG, format_string=log_format),
    logbook.StreamHandler(sys.stdout, level=logbook.WARNING, format_string=log_format),
    logbook.StreamHandler(sys.stdout, level=logbook.NOTICE, format_string=log_format),
    logbook.StreamHandler(sys.stdout, level=logbook.ERROR, format_string=log_format),
    #logbook.StreamHandler(sys.stderr, level=logbook.ERROR, format_string=log_format),
])
zipline_logging.push_application()
log = logbook.Logger('Main Logger')


#from pytz    import timezone as tz
def minut(context): #Minute of trading day
    dt = context.get_datetime().astimezone(pytz.timezone('US/Eastern'))
    return (dt.hour * 60) + dt.minute - 570
###################### Global Constants ###################################

# Start:Eonum Predictions/Assets Modules --------------------------------

if AlgoParms['UNIVERSE'] is None:
    from zipline.utils.paths import data_root
    from zipline.utils.run_algo import loaders
    from zipline.errors import SymbolNotFound
    
    #from alphatools.research import loaders
    
    def load_predictions(bundle):
        predictions = (pd.read_hdf(DATA_STORE, 'lgb/train/01')
                       .append(pd.read_hdf(DATA_STORE, 'lgb/test/01') ))
                               # ajjc-2021-12-08  .drop('y_test', axis=1)))
        predictions = (predictions.loc[~predictions.index.duplicated()]
                       .iloc[:, :10]
                       .mean(1)
                       .sort_index()
                       .dropna()
                       .to_frame('prediction'))
        tickers = predictions.index.get_level_values('symbol').unique().tolist()
    
        check = True
        while check:
            try:
                assets = bundle.asset_finder.lookup_symbols(tickers, as_of_date=None)
                check = False
            except SymbolNotFound as e:
                del_ticker = str(e).split("'")[1]
                print(del_ticker)
                tickers.remove(del_ticker)
        predicted_sids = pd.Int64Index([asset.sid for asset in assets])
        ticker_map = dict(zip(tickers, predicted_sids))
        print('---Above are deleted tickers---')
        return (predictions
                .unstack('symbol')
                .rename(columns=ticker_map)
                .prediction
                .tz_localize('UTC')), assets
    #DATA_STORE = '/home/ajjc/hca/strats-eonum/newassets.h5'
    #DATA_STORE = '/home/ajjc/hca/strats-eonum/sefpassets.h5' #2021-12-07
    #DATA_STORE = pl.Path(HCA_RELEASE_STRAT_DIR, 'assets_1213.h5') #2021-12-07
    DATA_STORE = pl.Path(HCA_RELEASE_STRAT_DIR, 'ASSETS_1227.h5') #2021-12-07
    predictions_df, assets = load_predictions(bundles.load('sharadar-eqfd'))
    AlgoParms['UNIVERSE'] = assets
    #orig predictions, assets = load_predictions(bundles.load('sep'))
    dates = predictions_df.index.get_level_values('date')
    print(f'PredictionsDB: min-date={dates.min()} max-date={dates.max()}')
    
    class SectorEmul(CustomFactor):
        """ Emulates a Sector factor """
        inputs = [USEquityPricing.close]
        window_length = 1
        def compute(self, today, assets, out, prices):
            out[:] = (prices[-1] > 0)
    
    class SignalData(DataSet):
        predictions = Column(dtype=float)
       # domain = US_EQUITIES
       
    #  ----- Forward Fill from last date of predictions to current date(utcnow()--------------
    #signal_loader = {SignalData.predictions:DataFrameLoader(SignalData.predictions, predictions)}
    #https://www.py4u.net/discuss/153517
    def f(df):
        dates = pd.date_range(predictions_df.index.max(), pd.Timestamp(pd.datetime.today(),tz='UTC')).union(predictions_df.index)
        d = df.sort_index().reindex(dates, method='ffill')
        return d.reset_index().set_index('index')
    
    #ajjc Comment out for no forward fill.   
    #predictions_df = f(predictions_df)
    
    name='predictions'
    loaders[SignalData.get_column(name)] = DataFrameLoader(SignalData.get_column(name), predictions_df )
    signal_loader=loaders
    
    class MLSignal(CustomFactor):
        """Converting signals to Factor, so we can rank and filter in Pipeline"""
        inputs = [SignalData.predictions] #.latest
        window_length = 1
    
        def compute(self, today, assets, out, predictions):
            out[:] = predictions
    
    
    def compute_signals():
        signals = MLSignal()
        universe = StaticAssets(assets)
        
        min_price = MIN_PRICE_FACT       # None
        min_volume = MIN_VOLUME_FACT #None
        
        sec_emul = SectorEmul()
        
        if min_price is not None:
            price_thres = SimpleMovingAverage(inputs=[USEquityPricing.close],
                                        window_length=21, mask=universe)
            universe &= (price_thres >= min_price)
    
        if min_volume is not None:
            volume_thres = SimpleMovingAverage(inputs=[USEquityPricing.volume],
                                         window_length=21, mask=universe)
            universe &= (volume_thres >= min_volume)
                
        return Pipeline(columns={
            'longs': signals.top(NUM_LONGS_FACT, mask=signals > 0),
            'shorts': signals.bottom(NUM_SHORTS_FACT, mask=signals < 0),
            'group' : sec_emul,
            'is_price_thres_':(price_thres >= min_price),
            'is_volume_thres':(volume_thres >= min_volume),
            'price_thres':price_thres ,
            'volume_thres':volume_thres,
            
            },
            screen=universe)
    
    #print(assets)

# End:Eonum Predictions/Assets Modules    ----------------------------------

###################### Custom Factors #####################################

class Volatility(CustomFactor):
    inputs = [Returns(window_length=VOL_DAYS)] #3

    def compute(self, today, assets, out, returns):
        out[:] = np.nanstd(returns, axis=0)

def MeanReversion(mask, days=VOL_DAYS):#2
    dmean_returns = Returns(window_length=days, mask=mask).demean(mask=mask)
    ret = SimpleMovingAverage(inputs=[dmean_returns], window_length=days, mask=mask)
    vol = Volatility(inputs=[dmean_returns], window_length=days, mask=mask)
    return -ret/vol

class JoinFactors(CustomFactor):
    #inputs = [factor1, factor2, ...]
    window_length = 1

    def compute(self, today, assets, out, *inputs):
        array = np.concatenate(inputs, axis=0)
        out[:] = np.nansum(array, axis=0)
        out[ np.all(np.isnan(array), axis=0) ] = np.nan

def make_MeanReversionBySector(mask):
    PCAs = []
    #sector = Sector(mask=mask)
    # #sector = Sector()
    #for sector_code in Sector.SECTOR_NAMES.keys():
    # #for sector_code in sector.names.index:
        # #sector_mask = sector.eq(sector_code)
        # #pca = MeanReversion(mask=sector_mask)
        # #pca.window_safe = True
        # #PCAs.append(pca)
    pca = MeanReversion(mask=mask)
    pca.window_safe = True
    PCAs.append(pca)
    return JoinFactors(mask=mask, inputs=PCAs)
###########################################################################

class ExposureMngr(object):
    """
    Keep track of leverage and long/short exposure

    One Class to rule them all, One Class to define them,
    One Class to monitor them all and in the bytecode bind them

    Usage:
    Define your targets at initialization: I want leverage 1.3  and 60%/40% Long/Short balance
       context.exposure = ExposureMngr(target_leverage = 1.3,
                                       target_long_exposure_perc = 0.60,
                                       target_short_exposure_perc = 0.40)

    update internal state (open orders and positions)
      context.exposure.update(context, data)

    After update is called, you can access the following information:

    how much cash available for trading
      context.exposure.get_available_cash(consider_open_orders = True)
    get long and short available cash as two distinct values
      context.exposure.get_available_cash_long_short(consider_open_orders = True)

    same as account.leverage but this keeps track of open orders
      context.exposure.get_current_leverage(consider_open_orders = True)

    sum of long and short positions current value
      context.exposure.get_exposure(consider_open_orders = True)
    get long and short position values as two distinct values
      context.exposure.get_long_short_exposure(consider_open_orders = True)
    get long and short exposure as percentage
      context.exposure.get_long_short_exposure_pct(consider_open_orders = True,  consider_unused_cash = True)
    """
    def __init__(self, target_leverage = 1.0, target_long_exposure_perc = 0.50, target_short_exposure_perc = 0.50):
        self.target_leverage            = target_leverage
        self.target_long_exposure_perc  = target_long_exposure_perc
        self.target_short_exposure_perc = target_short_exposure_perc

        self.long_actual_dist = {}
        self.short_actual_dist = {}

        self.short_exposure             = 0.0
        self.long_exposure              = 0.0
        self.open_order_short_exposure  = 0.0
        self.open_order_long_exposure   = 0.0

    def get_current_leverage(self, context, consider_open_orders = False): #True
        curr_cash = context.portfolio.cash - (self.short_exposure * 2) #(self.short_exposure * 2)
        if consider_open_orders:
            curr_cash -= self.open_order_short_exposure
            curr_cash -= self.open_order_long_exposure
        curr_leverage = (context.portfolio.portfolio_value - curr_cash) / context.portfolio.portfolio_value
        return curr_leverage

    def get_exposure(self, context, consider_open_orders = False):
        long_exposure, short_exposure = self.get_long_short_exposure(context, consider_open_orders)
        return long_exposure + short_exposure

    def get_long_short_exposure(self, context, consider_open_orders = False):
        long_exposure         = self.long_exposure
        short_exposure        = self.short_exposure
        if consider_open_orders:
            long_exposure  += self.open_order_long_exposure
            short_exposure += self.open_order_short_exposure
        return (long_exposure, short_exposure)

    def get_long_short_exposure_pct(self, context, consider_open_orders = False, consider_unused_cash = False):
        long_exposure, short_exposure = self.get_long_short_exposure(context, consider_open_orders)
        total_cash = long_exposure + short_exposure
        if consider_unused_cash:
            total_cash += self.get_available_cash(context, consider_open_orders)
        long_exposure_pct   = long_exposure  / total_cash if total_cash > 0 else 0
        short_exposure_pct  = short_exposure / total_cash if total_cash > 0 else 0
        return (long_exposure_pct, short_exposure_pct)

    def get_available_cash(self, context, consider_open_orders = False):
        curr_cash = context.portfolio.cash - (self.short_exposure * 2)
        if consider_open_orders:
            curr_cash -= self.open_order_short_exposure
            curr_cash -= self.open_order_long_exposure
        leverage_cash = context.portfolio.portfolio_value * (self.target_leverage - 1.0)
        return curr_cash + leverage_cash

    def get_available_cash_long_short(self, context, consider_open_orders = False):
        total_available_cash  = self.get_available_cash(context, consider_open_orders)
        long_exposure         = self.long_exposure
        short_exposure        = self.short_exposure
        if consider_open_orders:
            long_exposure  += self.open_order_long_exposure
            short_exposure += self.open_order_short_exposure
        current_exposure       = long_exposure + short_exposure + total_available_cash
        target_long_exposure  = current_exposure * self.target_long_exposure_perc
        target_short_exposure = current_exposure * self.target_short_exposure_perc
        long_available_cash   = target_long_exposure  - long_exposure
        short_available_cash  = target_short_exposure - short_exposure
        return (long_available_cash, short_available_cash)

    def update(self, context, data):
        #
        # calculate cash needed to complete open orders
        #
        self.open_order_short_exposure  = 0.0
        self.open_order_long_exposure   = 0.0
        if AlgoParms['IS_LIVE']: #Open orders are delayed as only orders are at EOD           
            for stock, orders in  list(get_open_orders().items()): #old: iteritems
                price = price_sim(data, stock, is_live=AlgoParms['IS_LIVE'])
                if np.isnan(price):
                    continue
                amount = 0 if stock not in context.portfolio.positions else context.portfolio.positions[stock].amount
                for oo in orders:
                    order_amount = oo.amount - oo.filled
                    if order_amount < 0 and amount <= 0:
                        self.open_order_short_exposure += (price * -order_amount)
                    elif order_amount > 0 and amount >= 0:
                        self.open_order_long_exposure  += (price * order_amount)

        #
        # calculate long/short positions exposure
        #

        self.short_exposure = 0.0
        self.long_exposure  = 0.0
        for stock, position in list(context.portfolio.positions.items()):  #old:iteritems
            amount = position.amount
            if amount == 0:
                continue ###ajjc should not matter, yet not sure
            if not data.can_trade(stock):
                context.portfolio_manual[str(stock.symbol)] = amount
                log.info('Actual Portfolio: Security: {} Amount: {}:  NOT TRADING: Added to manual_portfolio.'.format(str(stock.symbol), amount))
                continue
            #last_sale_price = data.current(stock, 'price')  ## ajjc is 0 when run 2nd time orig: =position.last_sale_price
            last_sale_price = price_sim(data, stock, is_live=AlgoParms['IS_LIVE'])
            
            ### if pd.isnull(last_sale_price):
            ###     last_sale_price = 0.0  # ajjc: Work around for not getting price. Remove from computation
            stock_volume = (last_sale_price * amount)
            if amount < 0:
                self.short_exposure += -stock_volume
                self.short_actual_dist[stock] = stock_volume/context.portfolio.portfolio_value
            elif amount > 0:
                self.long_exposure  += stock_volume
                self.long_actual_dist[stock] = stock_volume/context.portfolio.portfolio_value
        #log.info('self.long_actual_dist:self.long_actual_dist:\n{}'.format(self.long_actual_dist))
        ### Check code for prate validation: 
        #   log.info('self.long_actual_dist:>PRATE_PCT\n')
        # [print(key, value, value > AlgoParms['PRATE_PCT']) for key, value in self.long_actual_dist.items()]
        if self.long_actual_dist:
            log.info('sum(long_actual_dist)={}'.format(np.array(list(self.long_actual_dist.values())).sum()))
        if self.short_actual_dist:
            log.info('sum(short_actual_dist)={}'.format(np.array(list(self.short_actual_dist.values())).sum()))

class OrderMngr(object):
    """
    Buy/sell order manager
    """

    def __init__(self, sec_volume_limit_perc = None, min_shares_order = 1):
        '''
        sec_volume_limit_perc : max percentage of stock volume per minute to fill with our orders
        min_shares_order      : min number of shares to order per stock (this is because brokers
                                have a minimum fees other than a cost per share)
        '''
        self.sec_volume_limit_perc = sec_volume_limit_perc
        self.min_shares_order      = min_shares_order
        self.order_queue           = {}
        self.orders_signature      = 0

    def set_orders(self, orders):
        self.order_queue = orders

    def process_order_queue(self, context,  data):
        '''
        Scan order queue and perform orders: the order queue allows to spread orders along the
        day to avoid excessive slipapge
        '''
        if not self.order_queue:
            return
        self.orders_signature = 0
        
        #price = data.current(list(self.order_queue.keys()), 'price')
        price = price_sim(data, list(self.order_queue.keys()), is_live=AlgoParms['IS_LIVE'])
        
        ###ajjc
        if self.sec_volume_limit_perc is not None: #Only need volume if not buying full traunch of share.
            ###volume_history = data.history(list(self.order_queue.keys()), fields='volume', bar_count=6, frequency='1d')
            volume_history = data.history(list(self.order_queue.keys()), fields='volume', bar_count=30, frequency='1m')
            volume = volume_history.iloc[:-1].mean() # Last row is current day, which has volume 0
            #volume = volume_history.mean()

        for sec, amount in list(self.order_queue.items()):
            amount = round(amount)
            if amount == 0 or np.isnan(price[sec]):
                if np.isnan(price[sec]):
                    log.info("NAN price: price[{}]={}".format(sec, price[sec]))
                del self.order_queue[sec] 
                continue
            
            ###ajjc 2021-01-05  In Sim(bi=weekly/weekly), have open orders days after order event...which messes up new rebalance.
            ## # Commented out below 'continue' block due to sim open orders event firing protocol.
            #if get_open_orders(sec):
                #continue
            ### ajjc 2020-03-20
            ### ajjc 2020-03-26 : Promlem with Sim vs Live: can_trade:Sim==Tue is default can_trade:Live==False
            # ajjc: Comment out for now.
            #if data.can_trade(sec): # opposite sense?
                #log.info('Cannot Trade Security {} '.format(str(sec)))
                #continue
            if self.sec_volume_limit_perc is not None:
                max_share_allowed = round(self.sec_volume_limit_perc * volume[sec])
                if max_share_allowed < self.min_shares_order:
                    max_share_allowed = self.min_shares_order
                allowed_amount = min(amount, max_share_allowed) if amount > 0 else max(amount, -max_share_allowed)
                if abs(amount - allowed_amount) >= self.min_shares_order:
                    amount = allowed_amount
            # Flash Crash abatement + Add in StopLoss Order code here.
            ord_type = "NONE"
            if amount >= 0:
                ord_type = "BUY"
                adj_price = price[sec]*(1.0 + AlgoParms['LIMIT_ORDER_PCT']) #BUY
            else: # amount < 0
                ord_type = "SELL"
                adj_price = price[sec]*(1.0 - AlgoParms['LIMIT_ORDER_PCT'])  #SELL

            order(sec, amount, limit_price = adj_price)
            self.orders_signature += amount
            stock_volume = (price[sec] * amount)
            pct_actual =  stock_volume/context.portfolio.portfolio_value
            
            self.order_queue[sec] -= amount
            
            if DEBUG:
                log.info('Security {} ORDER: {} executed for (amount={} price={} limit_price={} price_value={})'.format(str(sec), ord_type, amount, price[sec], adj_price, (price[sec]*amount)))
                if self.sec_volume_limit_perc is not None:
                    log.debug('{} $ {} volume {} %% order {} shares remaining {}'.format(str(sec), (price[sec]*amount), float(amount)/volume[sec], amount, self.order_queue[sec]))
        log.notice("c.ORDERS_SIGN = {} ORDERS_SIGN = {} ".format(context.orders_signature, self.orders_signature))
        if abs(context.orders_signature - self.orders_signature)>0:
            log.notice("Error: ORDERS_SIGNATURE mismatch = {}".format(context.orders_signature-self.orders_signature))
        return

    def has_open_orders(self, sec):
        if get_open_orders(sec):
            return True
        if sec in self.order_queue:
            return True
        return False

    def cancel_open_orders(self, data):
        for sec, amount in list(self.order_queue.items()):
            amount = round(amount)
            if amount == 0:
                del self.order_queue[sec]
                log.warn('Security {} had queued orders (amount={}): now removed'.format(str(sec),amount))
                continue
        ### ajjc: Bug???
        for security in get_open_orders():
           for order in get_open_orders(security):
               cancel_order(order)
               log.warn('Security {} had open orders: now cancelled'.format(str(security)))
        self.order_queue = {}

    def clear_positions(self, context, data, security = None):
        if security is not None: # clear security positions
            if get_open_orders(security):
                return
            if not data.can_trade(security):
                return
            if security in list(self.order_queue):
                del self.order_queue[security]
            #price = data.current(security, 'price')
            price = price_sim(data, security, is_live=AlgoParms['IS_LIVE'])
            
            order_target_percent(security, 0, style=LimitOrder(price))
        else: # clear all positions
            for stock in context.portfolio.positions:
                if stock is None:
                    continue
                self.clear_positions(context, data, stock)

def get_weights(pipe_out, rank_cols, max_long_sec, max_short_sec, group_neutral):
    if group_neutral:
        pipe_out = pipe_out[rank_cols + ['group']]
    else:
        pipe_out = pipe_out[rank_cols]
    pipe_out = pipe_out.replace([np.inf, -np.inf], np.nan)
    pipe_out = pipe_out.dropna()
    def to_weights(factor, is_long_short):
        if is_long_short:
            demeaned_vals = factor - factor.mean()
            return demeaned_vals/demeaned_vals.abs().sum()
        else:
            return factor/factor.abs().sum()
    #
    # rank stocks so that we can select long/short ones
    #
    weights = pd.Series(0., index=pipe_out.index)
    for rank_col in rank_cols:
        if not group_neutral: # rank regardless of sector code
            weights += to_weights(pipe_out[rank_col], max_short_sec > 0)
            #weights += to_weights(pipe_out[rank_col], True)
        else: # weight each sector equally
            weights += pipe_out.groupby(['group'])[rank_col].apply(to_weights, max_short_sec > 0)
    if not group_neutral: # rank regardless of sector/group code
        longs  = weights[ weights > 0 ]
        shorts = weights[ weights < 0 ].abs()
        if max_long_sec:
            longs  = longs.sort_values(ascending=False).head(max_long_sec)
        if max_short_sec:
            shorts = shorts.sort_values(ascending=False).head(max_short_sec)
    else: # weight each group/sector equally
        sectors = pipe_out['group'].unique()
        num_sectors = len(sectors)
        longs  = pd.Series()
        shorts = pd.Series()
        for current_sector in sectors:
            _w = weights[ pipe_out['group'] == current_sector ]
            _longs  = _w[ _w > 0 ]
            _shorts = _w[ _w < 0 ].abs()
            if max_long_sec:
                _longs  = _longs.sort_values(ascending=False).head(max_long_sec/num_sectors)
            if max_short_sec:
                _shorts = _shorts.sort_values(ascending=False).head(max_short_sec/num_sectors)
            _longs /=  _longs.sum()
            _shorts /= _shorts.sum()
            longs  = longs.append( _longs )
            shorts = shorts.append( _shorts )

    longs  = longs[ longs > 0 ]
    shorts = shorts[ shorts > 0 ]
    longs  /= longs.sum()
    shorts /= shorts.sum()
    return longs, shorts

def add_positions(d1, d2):
    return { k:(d1.get(k,0)+d2.get(k,0)) for k in set(d1) | set(d2) if (d1.get(k,0)+d2.get(k,0)) != 0 }

def diff_positions(d1, d2):
    return { k:(d1.get(k,0)-d2.get(k,0)) for k in set(d1) | set(d2) if (d1.get(k,0)-d2.get(k,0)) != 0 }

class Strategy(object):

    def __init__(self, rebalance_days, max_long_sec, max_short_sec, group_neutral, factors, context):
        self.rebalance_days  = rebalance_days
        self.max_long_sec    = max_long_sec
        self.max_short_sec   = max_short_sec
        self.group_neutral   = group_neutral
        self.factors         = factors
        self.shorts = None
        self.longs = None
        self.shorts_port_pct = None
        self.longs_port_pct = None

        
        self.curr_day = -1
        self.days = {}
        self.fact_trade_dist = {}
        self.context = context
        #ajjc: fix issue of over-weighting a single symbol and under ordering 
        self.curr_positions = {}

    def set_weights(self, pipeline_output):
       if pipeline_output.empty:
           return
       lngs, shrts = get_weights(pipeline_output, self.factors,
                                              self.max_long_sec,self.max_short_sec,
                                              self.group_neutral)
       self.longs  = lngs
       self.shorts = shrts
       
       # Init 
       self.longss_port_pct = pd.Series([])
       self.shorts_port_pct = pd.Series([])

       if len(self.shorts) > 0:
           wts    = np.array(self.shorts)
           wts_pr = prorate(wts, max_alloc= AlgoParms['PRATE_PCT'])
           self.shorts_port_pct = pd.Series(wts_pr, index=self.shorts.index)

       if len(self.longs) > 0:
           wts    = np.array(self.longs)
           wts_pr = prorate(wts, max_alloc=AlgoParms['PRATE_PCT'])
           self.longs_port_pct = pd.Series(wts_pr, index=self.longs.index)

       long_pct = AlgoParms['LONG_PCT']
       short_pct = 1.0 - AlgoParms['LONG_PCT']
       leverage_factor = AlgoParms['LEVERAGE_FAC']
       
       if len(self.longs) >0:
           self.longs_port_pct  =   long_pct * leverage_factor * self.longs_port_pct
       if len(self.shorts) >0:
           self.shorts_port_pct = -short_pct * leverage_factor * self.shorts_port_pct

       universe = (self.longs_port_pct.index | self.shorts_port_pct.index)
       # save today's long and short portfolio actual percentages, after prorate clip, along with cash to spend per symbol.
       portpct_df = pd.DataFrame(0, index= [x.symbol for x in universe], columns=['shorts_pct','longs_pct'])

       spct_s        = pd.Series({name.symbol: val for name, val in self.shorts_port_pct.items()}, name='shorts_pct')
       lpct_s         = pd.Series({name.symbol: val for name, val in self.longs_port_pct.items()}, name='longs_pct')


       portpct_df.update(spct_s)
       portpct_df.update(lpct_s)

       portpct_df['portfolio_pct'] = portpct_df.longs_pct  + portpct_df.shorts_pct
       portpct_df=portpct_df.drop(['shorts_pct', 'longs_pct'], axis=1)
       portpct_df.index.name='asset'

       if DEBUG:
           log.info ('longs  weighted (length {}, sum {}, long_dist\n{}):\n'.format(len(self.longs.index), self.longs.sum(), self.longs))
           log.info ('shrts weighted (length {}, sum {}, shrt_dist\n{}):\n'.format(len(self.shorts.index), self.shorts.sum(), self.shorts))
       # Prepare trade distribution targets for reports output
       self.fact_trade_dist = pipeline_output.copy(deep=True)
       self.fact_trade_dist["short_dist"] = 0.0
       self.fact_trade_dist["long_dist"]  = 0.0
       self.shorts.name="short_dist"
       self.longs.name="long_dist"
       self.fact_trade_dist.update(self.shorts)
       self.fact_trade_dist.update(self.longs)
       self.fact_trade_dist.index.name = 'asset'

       #self.fact_trade_dist.sort_values(by='mr', ascending=False)
       
       today, name_sim, p, pa = create_logs_path(self.context,AlgoParms['IS_LIVE'], LOCAL_ZL_LIVE_PATH)    
       
       total_summary_file = p / f"portfolio_pct_total-{today:%Y-%m-%d-%H-%M-%S}.csv"
       (portpct_df.to_csv(total_summary_file,float_format='%.4f'))
       total_summary_file_cur = pa /  f"portfolio_pct_total.csv"
       (portpct_df.to_csv(total_summary_file_cur,float_format='%.4f'))

       summary_file = p / f"factor_dist_{today:%Y-%m-%d-%H-%M-%S}.csv"
       (self.fact_trade_dist.to_csv(summary_file,float_format='%.4f'))
       summary_file_cur = pa / f"factor_dist.csv"
       (self.fact_trade_dist.to_csv(summary_file_cur,float_format='%.4f'))

    def expected_positions(self):
        expected_positions = {}
        for day, pos in list(self.days.items()):
            expected_positions = add_positions(expected_positions, pos)
        return expected_positions

    def fix_positions(self, missing_positions):
        # called before rebalance, so self.curr_day is previous day
        prev_day = self.curr_day

        missing_positions = dict(missing_positions) # copy

        if prev_day in self.days:
            # update yesterday positions with actual ones
            prev_pos = self.days[prev_day]
            for sec, amount in list(missing_positions.items()):
                if sec in prev_pos:
                    if amount > 0 and prev_pos[sec] > 0:
                        fixed_amount = min(prev_pos[sec], amount)
                        prev_pos[sec] -= fixed_amount
                        missing_positions[sec] -= fixed_amount
                    elif amount < 0 and prev_pos[sec] < 0:
                        fixed_amount = max(prev_pos[sec], amount)
                        prev_pos[sec] -= fixed_amount
                        missing_positions[sec] -= fixed_amount
        return missing_positions

    def rebal_strat(self, data, long_cash, short_cash):
        #
        # Move to next rebalancing day
        #
        self.curr_day = (self.curr_day+1) % self.rebalance_days

        #
        # Get the positions we previously entered for this day slot
        #
        prev_positions = self.days[self.curr_day] if self.curr_day in self.days else {}

        #
        # we share the available cash between every trading day
        #
        today_long_cash  = long_cash / self.rebalance_days
        today_short_cash = short_cash / self.rebalance_days
        
        if    (today_long_cash < 0) or (today_short_cash < 0):
            log.debug( 'rebal_strat: Curr_day {} today_long_cash {} today_short_cash {}'.format(self.curr_day, today_long_cash, today_short_cash) )

        #
        # calculate new positions
        #
        new_positions = {}

        universe = (self.longs.index | self.shorts.index)

        #current_price = data.current(list(universe), 'price')

        if today_short_cash > 0:
            log.debug( 'rebal_strat: WARNING: Will trade SHORT: today_short_cash {}'.format(today_short_cash) )

            wts=np.array(self.shorts)
            wts_pr = prorate(wts, max_alloc= AlgoParms['PRATE_PCT'])
            self.shorts = pd.Series(wts_pr, index=self.shorts.index)

            #current_price = data.current(list(self.shorts.index), 'price')
            current_price = price_sim(data, list(self.shorts.index), is_live=AlgoParms['IS_LIVE'])
            
            for sec in self.shorts.index:
                try:
                    amount = - (self.shorts[sec] * today_short_cash / current_price[sec])
                    if not np.isfinite(amount):
                        log.info("BAD_Short_Order:amount_not_finite:{} Probably no subscription. Set order amt=0".format(sec))
                        amount = 0  # Don't by or sell bad amount
                    
                except:
                    amount = 0
                    log.info("BAD_Short_Order_amount_bad_type:{}. Set order amt=0".format(sec))
                    
                new_positions[sec] = round(amount)

        if today_long_cash > 0:
            wts=np.array(self.longs)
            wts_pr = prorate(wts, max_alloc=AlgoParms['PRATE_PCT'])
            self.longs = pd.Series(wts_pr, index=self.longs.index)

            #current_price = data.current(list(self.longs.index), 'price')
            current_price = price_sim(data, list(self.longs.index), is_live=AlgoParms['IS_LIVE'])
            
            for sec in self.longs.index:
                try:
                    amount = self.longs[sec] * today_long_cash / current_price[sec]
                    if not np.isfinite(amount):
                        log.info("BAD_Long_Order:amount_not_finite:{} Probably no subscription. Set order amt=0".format(sec))
                        amount = 0  # Don't by or sell bad amount
                    
                except:
                    amount = 0
                    log.info("BAD_Long_Order_amount_bad_type:{}. Set order amt=0".format(sec))
                
                new_positions[sec] = round(amount) #Set any bad or missing amount values to 0. Most robust way.

        self.curr_positions = { sec:position.amount for sec, position in list(self.context.portfolio.positions.items()) }
        self.days[self.curr_day] = self.curr_positions
        #original self.days[self.curr_day] = new_positions
        
        # save today's long and short portfolio actual percentages, after prorate clip, along with cash to spend per symbol.
        portpct_df = pd.DataFrame(0, index= [x.symbol for x in universe], columns=['longs_pct','shorts_pct'])

        spct_s        = pd.Series({name.symbol: val for name, val in self.shorts.items()}, name='shorts_pct')
        lpct_s         = pd.Series({name.symbol: val for name, val in self.longs.items()}, name='longs_pct')


        portpct_df.update(spct_s)
        portpct_df.update(lpct_s)

        portpct_df['long_pct'] = AlgoParms['LONG_PCT']
        portpct_df['short_pct'] = 1.0 - AlgoParms['LONG_PCT']
        portpct_df['leverage_factor'] = AlgoParms['LEVERAGE_FAC']

        portpct_df['pct_port_total'] = portpct_df.longs_pct * portpct_df.leverage_factor * portpct_df.long_pct - portpct_df.shorts_pct * portpct_df.leverage_factor * portpct_df.short_pct

        portpct_df['long_cash'] = today_long_cash
        portpct_df['short_cash'] = today_short_cash
        portpct_df['cash_port'] = portpct_df.longs_pct * today_long_cash - portpct_df.shorts_pct * today_short_cash

        today, name_sim, p, pa = create_logs_path(self.context, AlgoParms['IS_LIVE'], LOCAL_ZL_LIVE_PATH)    
        summary_file = p / f"portfolio_percent-{today:%Y-%m-%d-%H-%M-%S}.csv"
        portpct_df.to_csv(summary_file,float_format='%.4f')
        summary_file_cur = pa / f"portfolio_percent.csv"
        portpct_df.to_csv(summary_file_cur,float_format='%.4f')

        self.shorts = None
        self.longs = None

        return new_positions, prev_positions

def high_volume_universe(context, top_liquid, min_price = None, min_volume = None):
    """
    Computes a security universe of liquid stocks and filtering out
    hard to trade ones
    Returns
    -------
    high_volume_tradable - zipline.pipeline.filter
    """

    if top_liquid == "QTradableStocksUS":
        universe = QTradableStocksUS()
    elif top_liquid == 500:
        universe = Q500US()
    elif top_liquid == 1500:
        universe = Q1500US()
    elif top_liquid == 3000:
        universe = Q3000US()
    elif top_liquid == 30:
        universe =  context.my_etfs
    else:
        universe = filters.make_us_equity_universe(
            target_size=top_liquid,
            rankby=factors.AverageDollarVolume(window_length=200),
            mask=filters.default_us_equity_universe_mask(),
            groupby=Sector(),
            max_group_weight=0.3,
            smoothing_func=lambda f: f.downsample('month_start'),
        )

    if min_price is not None:
        price = SimpleMovingAverage(inputs=[USEquityPricing.close],
                                    window_length=21, mask=universe)
        universe &= (price >= min_price)

    if min_volume is not None:
        volume = SimpleMovingAverage(inputs=[USEquityPricing.volume],
                                     window_length=21, mask=universe)
        universe &= (volume >= min_volume)

    return universe

def make_pipeline(context):

    universe = high_volume_universe(context, top_liquid=context.universe_size)
    pipe = Pipeline()
    pipe.set_screen(universe)

    #
    # Add grouping factors
    #
    if context.group_neutral:
        group = Sector(mask=universe) # any group you like
        pipe.add(group, "group")

    #
    # Add any custom factor here
    #
    mr = make_MeanReversionBySector(mask=universe)
    pipe.add(mr, "mr")

    return pipe

def initialize(context):
    #Added per RH request: PS, WORK, ZM, and PD
    context.my_etfs_sym = scfg.py['strats']['segm_sect'][ThisStrat]['UNIVERSE']
    #context.my_etfs     = (StaticAssets(symbols(*context.my_etfs_sym)))
    ### context.my_etfs     = (StaticAssets(context.my_etfs_sym))
    context.my_etfs     = context.my_etfs_sym

    today, name_sim, p, pa = create_logs_path(context,AlgoParms['IS_LIVE'], LOCAL_ZL_LIVE_PATH)    
    
    cli_args_run_file_ts = p / f"cli_args_run_file-{name_sim}-{today:%Y-%m-%d-%H-%M-%S}.sh"
    cli_args_run_file = pa / f"cli_args_run_file-{name_sim}.sh"
    # Write CommandLine Params files(timestamped and current).
    with open(cli_args_run_file_ts, 'w') as f:
        f.write('\n'.join(sys.argv[:]))
    with open(cli_args_run_file, 'w') as f:
        f.write('\n'.join(sys.argv[:]))
    
    context.cli_file =   cli_args_run_file #Command Line file for this simulation.
    
    #)
    #
    # Algo configuration
    #
    if not AlgoParms['IS_LIVE']:
        #IB Tiered <300k shares
        set_commission(commission.PerShare(cost=0.0035, min_trade_cost=0.35))
        #Q set_commission(commission.PerShare(cost=0.00075, min_trade_cost=0))
        set_slippage(slippage.FixedSlippage(spread=0.00))
    #set_slippage(slippage.FixedBasisPointsSlippage( ))
    ###set_commission(commission.PerShare(cost=0.001, min_trade_cost=0))
    #set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1))
    #Dan Whitnable
    #set_commission(commission.PerShare(cost=0.00, min_trade_cost=0.0))
    #set_slippage(slippage.VolumeShareSlippage(volume_limit=.1, price_impact=0.1))

    #Convention: Execute Orders the first minute AFTER
    
    
    # start:Eonum
    context.min_positions = 9 #20    
    # end:Eonum


    context.MINUTES_TO_REBAL = AlgoParms['MINUTES_TO_REBAL']
    context.IS_LIVE          = AlgoParms['IS_LIVE']
    context.all_orders       = {}
    context.portfolio_manual = {}
    context.TRACK_ORDERS_ON  = TRACK_ORDERS_ON
    context.PVR_ON           = PVR_ON
    context.PSF_name         = AlgoParms['PSF']
    context.STOPLOSS_THR     = 0.1  #Default: Exit asset on 10% loss
    
    context.order_signature = 0 # Sum of amount of all orders(+/-) as a rebalance invariant

    context.exposure = ExposureMngr(target_leverage = AlgoParms['LEVERAGE_FAC'], #1.0 , #1.3, #1.0
                                    target_long_exposure_perc = AlgoParms['LONG_PCT'], #0.2, #1.0, #0.50,
                                    target_short_exposure_perc = SHORT_PCT) #0.50)

    context.order_mngr = OrderMngr(sec_volume_limit_perc = None, # Order all shares in one chunk, no volume limit
                                   min_shares_order      = 1)    # This makes sense only with volume limit

    #context.universe_size = "QTradableStocksUS"
    context.universe_size = 30
    context.PCT_FUNDED = 10  # Percent of initial cash that algo is Funded with
    #context.group_neutral = True
    ###ajjc 
    context.group_neutral = False
    #2
    s1 = Strategy(rebalance_days=1, max_long_sec=STRAT_MAX_LONGS, max_short_sec=STRAT_MAX_SHORTS, #300
                  group_neutral=context.group_neutral,
                  #factors=["longs", "shorts"], context=context)
                  factors=["mr"], context=context)

    context.strategies = [s1]

    #
    # Algo logic starts
    # Put this in before_trading_start if using persisten context read from state_filename
    
    ###orig attach_pipeline(make_pipeline(context), 'factors')
    
    pipeline = compute_signals()
    attach_pipeline(pipeline, 'signals')
    
    ### ajjc live
    do_pvr = True
    if not AlgoParms['IS_LIVE']: # Run Backtest Simulation
        # Buy&Hold
        schedule_function(func=rebalance,
                          #date_rule=date_rules.month_start(days_offset=4),
                         date_rule=date_rules.week_start(days_offset=3),
                         #date_rule=date_rules.every_day(),
                         time_rule=time_rules.market_open(minutes=380),
                         half_days=True)

        cancel_open_orders = lambda context, data: context.order_mngr.cancel_open_orders(data)
        schedule_function(cancel_open_orders, date_rules.week_start(days_offset=2), time_rules.market_close())

    pass

def exit_on_StopLoss(context, data):
    positions = context.portfolio.positions
    open_orders = get_open_orders()
    Stop_loss = context.STOPLOSS_THR
    for security in positions:
        current_price = positions[security].last_sale_price
        cost_price    = positions[security].cost_basis
        returns       = (current_price/cost_price)-1
        if (returns < -Stop_loss) and security not in open_orders:
            order_target_percent(security, 0)
            
# Will be called on every trade event for the securities you specify.
def sync_portfolio_to_broker(context, data):
    log.info("___CurrZiplinPosBef: {}".format(context.portfolio.positions)) #BUG: This is a Criticalupdate...
    if AlgoParms['IS_LIVE']:
        log.info("___CurrBrokerPosCur: {}".format(context.broker.positions)) # Look=Hook for sync of context.portfolio to context.broker.portfolio
    for x in list(context.portfolio.positions):
        #ajjc: zlb: BUG: Clean out null portfolio values. Handle this generically in zipline-broker in some way
        amt_x_port = context.portfolio.positions[x].amount
        if amt_x_port == 0: 
            del context.portfolio.positions[x]    
    log.info("___CurrZiplinPosAft: {}".format(context.portfolio.positions)) #BUG: This is a Criticalupdate...


def handle_data(context, data):
    time_now = minut(context)
    time_till_trade = time_now -context.MINUTES_TO_REBAL
    log.info("___handle_data: {} = Current Trading Minute".format(time_now))

    
    if np.isnan(context.portfolio.portfolio_value):
        portfolio_value = 0.0
    else:
        portfolio_value = context.portfolio.portfolio_value
        
    if DEBUG:
        log.info('handle_data: time_till_trade={} curr_min={} portval={}'.format(time_till_trade,time_now, int(portfolio_value)))
    #if not context.ORDERS_DONE and (len(context.all_orders) >0) and (time_till_trade > 0) :
        #context.ORDERS_DONE = True
        #execute_orders(context, data)
    pass
    #context.order_mngr.process_order_queue(context, data)

#def cancel_open_orders(context, data):
#    context.order_mngr.cancel_open_orders(data)
# Eonum portfolio optimization

def optimize_weights(prices, short=False):
    """Uses PyPortfolioOpt to optimize weights"""
    returns = expected_returns.mean_historical_return(prices=prices, frequency=252)
    cov = risk_models.sample_cov(prices=prices, frequency=252)

    # get weights that maximize the Sharpe ratio
    # using solver SCS which produces slightly fewer errors than default
    # see https://github.com/robertmartin8/PyPortfolioOpt/issues/221
    ef = EfficientFrontier(expected_returns=returns,
                           cov_matrix=cov,
                           weight_bounds=(0, 1),
                           solver='SCS')

    weights = ef.max_sharpe()
    if short:
        return {asset: -weight for asset, weight in ef.clean_weights().items()}
    else:
        return ef.clean_weights()
    
def rebalance_hierarchical_risk_parity(context, data):
    #context.assets_all - longs+shorts index of pipeline
    hrp_weights_longs  = OrderedDict()
    hrp_weights_shorts = OrderedDict()
    
    if len(context.longs) > context.min_positions:
        returns_longs = (data.history(context.longs, fields='price',
                                bar_count=252 + 1,  # for 1 year of returns
                                frequency='1d')
                   .pct_change()
                   .dropna(how='all'))
        hrp_weights_longs = HRPOpt(returns=returns_longs).optimize()
        
    if len(context.shorts) > context.min_positions:
        returns_shorts = (data.history(context.shorts, fields='price',
                                bar_count=252 + 1,  # for 1 year of returns
                                frequency='1d')
                   .pct_change()
                   .dropna(how='all'))
        hrp_weights_shorts = HRPOpt(returns=returns_shorts).optimize()
        
    hrp_weights = OrderedDict(list(hrp_weights_longs.items()) + list(hrp_weights_shorts.items()))       
        
    return hrp_weights

def before_trading_start(context, data):
    context.ORDERS_DONE       = False #No Orders done yet today
    context.orders_done_today = False #Flag for daily pnl file support (ajjc removed:12/22/2021)
    context.REBALANCE_DONE    = False #No Orders done yet today
    context.all_orders = {}
    ### ajjc live
    if AlgoParms['IS_LIVE']:
        #schedule_function(rebalance, date_rules.every_day(),time_rule=time_rules.market_open())
        #schedule_function(rebalance, date_rules.every_day(),time_rule=time_rules.every_minute())
        for i in range(1, 391):
            #Daily schedule_function(rebalance, date_rules.every_day(), time_rules.market_open(minutes=i))
            #Weekley
            schedule_function(func=rebalance,
                              #date_rule=date_rules.month_start(days_offset=4),
                             ###Weekly date_rule=date_rules.week_start(days_offset=4),
                             ###Daily
                             date_rule=date_rules.every_day(),
                             time_rule=time_rules.market_open(minutes=i),
                             half_days=True)
            
        sync_portfolio_to_broker(context, data)
        
        pass
    current_time = context.get_datetime('US/Eastern')
    #NanTime      = pd.Timestamp('2020-05-01',).tz_localize('US/Eastern')
    #if current_time > NanTime:
    #    print("currtime {}  > StopTime {}".format(current_time, NanTime))
    if DEBUG:
        log.debug( 'Time:before_trading_start:US/Eastern {}'.format(current_time ))
    ### original  pipeout = pipeline_output('factors')
    #pipeout = pipeout[np.isfinite(pipeout).all(1)]
    ##print(pipeout.isna(),pipeout.isnull(),pipeout.dropna)

    #Eonum
    pipeout = pipeline_output('signals')
    if pipeout.empty:
        log.info( 'WARNING: Pipeline is empty today {}'.format(current_time ))
        return
    pipeout        = pipeout[pipeout[['longs','shorts']].any(axis=1)] #Only want assets with either a long or short
    context.longs  = pipeout[pipeout.longs].index
    context.shorts = pipeout[pipeout.shorts].index
    context.assets_all = pipeout.index
    
    hrp_opt = rebalance_hierarchical_risk_parity(context, data)
    if len(hrp_opt) ==0:
        pipeout['hrp'] = 0
        pipeout['mr']  = 0
    else:
        pipeout['hrp'] = pd.DataFrame.from_dict(hrp_opt, orient='index')
        # Create long/short factor to feed into execution modules.
        pipeout['mr'] =(pipeout.longs.apply(int) - pipeout.shorts.apply(int)) * pipeout['hrp']

    if DEBUG:
        print(pipeout)
        print ('Basket of stocks {}'.format(len(pipeout)))
    #Set today's weights for long/short execution of algo.
    
    for strategy in context.strategies:
        strategy.set_weights(pipeout)
    #order_book = {}
    #order_book = rebalance(context, data)
    #context.all_orders.update(order_book)
    if DEBUG:
        log.debug( 'strategy.longs {}'.format(strategy.longs))
        log.debug( 'strategy.shorts {}'.format(strategy.shorts))
        log.debug( 'strategy.expected_positions {}'.format(strategy.expected_positions()))
        #log.debug( 'context.all_orders {}'.format(context.all_orders))
    
    today, name_sim, p, pa = create_logs_path(context,AlgoParms['IS_LIVE'], LOCAL_ZL_LIVE_PATH)    

    #p            = pl.Path(LOCAL_ZL_LIVE_PATH, 'Data', name_sim, 'algostats')
    #p.mkdir(exist_ok=True, parents=True)
    #today        = datetime.today() #Current day live

    daily_port_date_traded_file = pa / f"daily_port_date_traded-{name_sim}.csv"
            
    return
    #execute_orders(context,data)

def rebalance(context, data):
    # This will be run once in befor_trading_start, to get new portfolio distribution from the factor,
    # and every minute during the trading session.  It will only actually rebalance/order the portfolio ONCE during the trading day,
    #and return without doing anything otherwise
    #log.info('rebalance: num_trading_minutes = {} num_portfolio_assets = {}'.format(minut(context), len(list(context.portfolio.positions))))
    #log.debug("-------- all_orders={}".format(context.all_orders))
    # context.MINUTES_TO_REBAL = current day market minutes to rebalnace(e.g MINUTES_TO_REBAL=60 is trade after 60 min of market open)
    time_now = minut(context)
    time_till_trade = time_now -context.MINUTES_TO_REBAL
    if np.isnan(context.portfolio.portfolio_value):
        portfolio_value = 0.0
    else:
        portfolio_value = context.portfolio.portfolio_value
    if True: #DEBUG
        log.info('rebalance_top: time_till_trade={} curr_min={} portval={}'.format(time_till_trade, time_now, int(portfolio_value)))
        try:
            track_orders(context, data)
            pvr(context, data)
            pass
        except:
            log.info('rebalance_top: track_orders/pvr except: time_till_trade={} curr_min={} portval={}'.format(time_till_trade, time_now, int(portfolio_value)))
            pass
    if AlgoParms['IS_LIVE']:
        if ((not context.ORDERS_DONE) 
             and (len(context.all_orders) >0) 
             and (time_till_trade > 0)):
            log.info('IS_LIVE:{} execute_orders: time_till_trade={} curr_min={} portval={}'.format(AlgoParms['IS_LIVE'], time_till_trade, time_now, int(portfolio_value)))

            sync_portfolio_to_broker(context, data)

            context.ORDERS_DONE = True
            context.orders_done_today = True
            brk_acct_bo =context.broker.get_account_from_broker()
            pp.pprint("IB-account: BeforeOrders:{} {}".format(time_now, brk_acct_bo))
            pp.pprint("IB-account: BeforeOrders:context.latest_no_subscription_list={}".format(context.broker.latest_no_subscription_list))
            execute_orders(context, data)
            time_now = minut(context)
            brk_acct_ao =context.broker.get_account_from_broker()
            pp.pprint("IB-account: AfterOrders:{} {}".format(time_now, brk_acct_ao))
            pp.pprint("IB-account: AfterOrders:context.latest_no_subscription_list={}".format(context.broker.latest_no_subscription_list))
            
            
        else:
            log.info('IS_LIVE:{} Caution: OrdersDoneForTheDay: Any other orders(including Dump) will kick off pattern-trading checks time_till_trade={} curr_min={} portval={}'.format(AlgoParms['IS_LIVE'], time_till_trade, time_now, int(portfolio_value)))
            
    else:
        oo_list = list(get_open_orders().values())
        if len(oo_list) > 0:
            log.info("rebalance:Open orders--skip rebalance when NOT LIVE")
            ### log.info(f'rebalance: oo_list{oo_list}')
            ###ajjc return
        else:
            log.info("rebalance:No open orders--proceed")
            
    #
    # Fix saved positions with actual positions (unfilled/cancelled orders from previous day or
    # some existing portfolio positions not generated from this algorithm)
    #

    if context.REBALANCE_DONE or (context.orders_done_today): # Only execute bulk of rebalance once
        if DEBUG:
            log.info('No rebalance more than once per day: context.orders_done_today={} time_till_trade={} curr_min={} portval={}'.format(context.orders_done_today, time_till_trade, time_now, int(portfolio_value)))
        return
    if DEBUG:
        log.info('rebalance_once: num_trading_minutes = {} num_portfolio_assets = {}'.format(minut(context), len(list(context.portfolio.positions))))
    expected_positions = {}
    for strategy in context.strategies:
        expected_positions = add_positions(expected_positions, strategy.expected_positions())

    actual_positions = { sec:position.amount for sec, position in list(context.portfolio.positions.items()) } #old:iteritems

    missing_positions = diff_positions(expected_positions, actual_positions)

    for strategy in context.strategies:
        missing_positions = strategy.fix_positions(missing_positions)

    #
    # Calculate how much money we have for rebalancing today
    #
    context.exposure.update(context, data)
    context.leverage = context.exposure.get_current_leverage(context, consider_open_orders = False)
    long_available_cash, short_available_cash = context.exposure.get_available_cash_long_short(context)
    long_exposure, short_exposure = context.exposure.get_long_short_exposure(context)
    long_cash_per_strategy  = (long_available_cash + long_exposure) / len(context.strategies)
    short_cash_per_strategy = (short_available_cash + short_exposure) / len(context.strategies)

    if DEBUG:
        log.debug( 'long_available_cash {} short_available_cash {} long_exposure {} short_exposure {}'.format(long_available_cash, short_available_cash, long_exposure, short_exposure))
        log.debug( 'long_cash_per_strategy {} short_cash_per_strategy {}'.format(long_cash_per_strategy, short_cash_per_strategy))
    print( 'leverage {} pv {} long_avl_cash {} short_avl_cash {} long_exp {} short_exp {}'.format(context.leverage, portfolio_value, long_available_cash, short_available_cash, long_exposure, short_exposure))
    #print( 'long_cash_per_strategy {} short_cash_per_strategy {}'.format(long_cash_per_strategy, short_cash_per_strategy))

    if DEBUG:
        log.debug( 'expected:\n {}'.format(expected_positions))
        log.debug( 'actual:\n {}'.format(actual_positions))
        log.debug( 'missing:\n {}'.format(missing_positions))
        log.debug( 'CurrentIBPortfolio:\n {}'.format(context.portfolio))


    #
    # calculate new positions
    #
    new_positions = {}
    prev_positions = {}
    check_positions = {}
    for strategy in context.strategies:
        s_new_positions, s_prev_positions = strategy.rebal_strat(data, long_cash_per_strategy, short_cash_per_strategy)
        new_positions = add_positions(new_positions, s_new_positions)
        prev_positions = add_positions(prev_positions, s_prev_positions)
    #
    # get rid of leftovers (maybe the algo was started with a non empty portfolio)
    #
    prev_positions = diff_positions(prev_positions, missing_positions)
    ###bad prev_positions = diff_positions(prev_positions, actual_positions)
    check_positions = diff_positions(prev_positions, actual_positions)

    #
    # Clear previous positions for this day
    #
    all_orders = diff_positions(new_positions, prev_positions)
    context.all_orders = all_orders
    if DEBUG:
        log.debug( 'prev_positions:\n {}'.format(prev_positions))
        log.debug( 'new_positions:\n {}'.format(new_positions))
        log.debug( 'check_positions:\n {}'.format(check_positions))
        if len(check_positions) > 0:
            print("BUG")
        log.debug( 'all_orders:\n {}'.format(all_orders))
        log.debug("-------- all_orders={}".format(context.all_orders))
    # all_positions_assets = 

    prevp_s        = pd.Series({name.symbol: val for name, val in prev_positions.items()}, name='prev_pos')
    newp_s         = pd.Series({name.symbol: val for name, val in new_positions.items()}, name='new_pos')
    orders_s       = pd.Series({name.symbol: val for name, val in context.all_orders.items()}, name='all_orders')

    ll=[prevp_s.to_frame(),newp_s.to_frame(),orders_s.to_frame()]
    df=pd.concat(ll)
    idx=df.index.drop_duplicates(keep='first')
    
    order_verif_df = pd.DataFrame(0, index=idx, columns=['prev_pos','new_pos','all_orders'])
    order_verif_df.update(prevp_s)
    order_verif_df.update(newp_s)
    order_verif_df.update(orders_s)
    order_verif_df.index.name = 'asset'

    today, name_sim, p, pa = create_logs_path(context,AlgoParms['IS_LIVE'], LOCAL_ZL_LIVE_PATH)    
    
    summary_file = p / f"order_verif_{today:%Y-%m-%d-%H-%M-%S}.csv"
    order_verif_df.to_csv(summary_file,float_format='%.4f')
    summary_file_cur = pa /  f"order_verif.csv"    
    order_verif_df.to_csv(summary_file_cur,float_format='%.4f')
    
    context.order_verif_df=order_verif_df

    summary_file = p / f"prev_positions_{today:%Y-%m-%d-%H-%M-%S}.csv"
    (pd.DataFrame.from_dict(data=dict(prev_positions), orient='index')
       .to_csv(summary_file, header=False,float_format='%.4f'))
    summary_file_cur = pa /  f"prev_positions.csv"    
    (pd.DataFrame.from_dict(data=dict(prev_positions), orient='index')
       .to_csv(summary_file_cur, header=False,float_format='%.4f'))

    summary_file = p / f"new_positions_{today:%Y-%m-%d-%H-%M-%S}.csv"
    (pd.DataFrame.from_dict(data=dict(new_positions), orient='index')
       .to_csv(summary_file, header=False,float_format='%.4f'))
    summary_file_cur = pa / f"new_positions.csv"
    (pd.DataFrame.from_dict(data=dict(new_positions), orient='index')
       .to_csv(summary_file_cur, header=False,float_format='%.4f'))

    summary_file = p / f"all_orders_{today:%Y-%m-%d-%H-%M-%S}.csv"
    (pd.DataFrame.from_dict(data=all_orders, orient='index')
       .to_csv(summary_file, header=False,float_format='%.4f'))
    summary_file_cur = pa /  f"all_orders.csv"
    (pd.DataFrame.from_dict(data=all_orders, orient='index')
       .to_csv(summary_file, header=False,float_format='%.4f'))

    # To avoid excessive slippage enter new positions in the order queue instead of ordering now
    #
    #context.all_orders = context.all_orders.update(all_orders)
    context.REBALANCE_DONE = True  # Only rebalance once per day flag
    if not AlgoParms['IS_LIVE']:
        if (not context.ORDERS_DONE) and (len(context.all_orders) >0) and (time_till_trade > 0) :
            log.info('\nIS_LIVE={} execute_orders: time_till_trade={} curr_min={} portval={}'.format(AlgoParms['IS_LIVE'], time_till_trade, time_now, int(portfolio_value)))
            context.ORDERS_DONE = True
            if context.leverage >1.3:
                lvg_withOO    = context.exposure.get_current_leverage(context, consider_open_orders = True)
                lvg_withoutOO = context.exposure.get_current_leverage(context, consider_open_orders = False)
                print("BadLeverage: lvg_withOO{} lvg_withoutOO{} c.leverage{}".format(lvg_withOO,lvg_withoutOO,context.leverage))
            execute_orders(context, data)

    return all_orders

    #context.order_mngr.set_orders(all_orders)
    #log.debug("-------- all_orders={}".format(all_orders))
    #context.order_mngr.process_order_queue(context, data)

def execute_orders(context,data):
    log.info('execute_orders: num_trading_minutes = {} num_portfolio_assets = {}'.format(minut(context), len(list(context.portfolio.positions))))
    #log.debug( 'execute_orders: CurrentIBPortfolio {}'.format(context.portfolio))

    today, name_sim, p, pa = create_logs_path(context,AlgoParms['IS_LIVE'], LOCAL_ZL_LIVE_PATH)    

    summary_file = p / f"curr_ib_portfolio_{today:%Y-%m-%d-%H-%M-%S}.csv"
    (pd.DataFrame.from_dict(data=context.portfolio.__dict__, orient='index')
       .to_csv(summary_file, header=False, float_format='%.4f'))
    summary_file_cur = pa / f"curr_ib_portfolio.csv"
    (pd.DataFrame.from_dict(data=context.portfolio.__dict__, orient='index')
       .to_csv(summary_file, header=False, float_format='%.4f'))
    
    context.orders_signature = sum(list(context.all_orders.values()))
    
    #ajjc 
    context.order_mngr.cancel_open_orders(data)
    
    context.order_mngr.set_orders(context.all_orders)
    log.notice("ORDERS_SIGN = {}".format(context.orders_signature))
    #pp.pprint("ORDERS_SIGN = {}".format(sum(list(context.all_orders.values()))))
    
    
    context.order_mngr.process_order_queue(context, data)
    log.debug("DONE: Orders submitted at: {}".format(minut(context)))

    ##original:
    ###schedule_function(log_stats, date_rules.every_day(), time_rules.market_close())
    ### Blue Seahawk tracking additions
    #schedule_function(recording_statements,date_rules.every_day(),
    #                  time_rules.market_close())
    ###for i in range(1, 391):
    ###    schedule_function(track_orders, date_rules.every_day(), time_rules.market_open(minutes=i))
    # This is included for the indication of profit per dollar invested since
    #   different factors might not always invest the same amount. Apples-to-apples comparison.
    #do_pvr = 1
    #if do_pvr:
    #    for i in range(1, 391):
    #        schedule_function(pvr, date_rules.every_day(), time_rules.market_open(minutes=i))


def log_stats(context, data):
    context.exposure.update(context, data)
    long_exposure_pct, short_exposure_pct = context.exposure.get_long_short_exposure_pct(context)
    record(lever=context.account.leverage,
    exposure=context.account.net_leverage,
    num_pos=len(context.portfolio.positions),
    long_exposure_pct=long_exposure_pct,
    short_exposure_pct=short_exposure_pct)

def recording_statements(context, data):
    print("num_positions={}".format(len(context.portfolio.positions)))
    #record(
    #num_positions=len(context.portfolio.positions))
    #record(cash = context.portfolio.cash/1000000.0)
    record(cap_used = context.portfolio.capital_used)
    longs = shorts = 0
    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            longs += 1
        if position.amount < 0:
            shorts += 1
    #record(long_lever=longs, short_lever=shorts)
    return_per_risk(context, data)

def return_per_risk(context,data):
    if not '_init_rpr' in context:
        context._init_rpr = True
        context._leverage_limit = 1.0
        context._borrowed = 0
        context._min_cash = context.portfolio.starting_cash
        context._funding = context.portfolio.starting_cash * context.PCT_FUNDED
    context._cash_used = context.portfolio.starting_cash - context.portfolio.cash
    context._capused=context.portfolio.capital_used
    context._pnl=context.portfolio.pnl
    context._totalret= 100.0 + ((context._pnl/context._funding)-1.0)*100.0
    context._min_cash = min(context._min_cash, context.portfolio.cash)
    cl = context.account.leverage
    pv = context.portfolio.portfolio_value
    if cl > 1.0 and cl > context._leverage_limit:
        context._borrowed = max(context._borrowed, cl*pv - context._leverage_limit*pv)
    if context._borrowed > 0:
        risked = context._cash_used + context._borrowed
        #risked = context.portfolio.starting_cash + context._borrowed
    else:
        risked = context._cash_used
        #risked = context.portfolio.starting_cash - context._min_cash
    rpr = 0
    rpr_eff = 0
    if risked != 0:
        rpr = (context.portfolio.portfolio_value - context.portfolio.starting_cash)/(context._capused)*100
        #rpr = (context.portfolio.portfolio_value - context.portfolio.starting_cash)/(risked)*100
    rpr_eff = (context._cash_used)/(risked)*100
    context.exposure.update(context, data)
    long_exposure_pct, short_exposure_pct = context.exposure.get_long_short_exposure_pct(context)
    #record(PctRetPerRisk = rpr_eff)
    #record(MoneyRisked = risked)
    #record(Borrowed = context._borrowed)
    record(pnl = context._pnl)
    record(totret = context._totalret)
    record(long_exposure_pct=long_exposure_pct,
           short_exposure_pct=short_exposure_pct)
    #record(CashUsed = context._cash_used)
    #record(rpr_eff = rpr_eff)

def track_orders(context, data):
    '''  Show orders when made and filled.
           Info: https://www.quantopian.com/posts/track-orders
    '''
    c = context
    if not c.TRACK_ORDERS_ON:
        c.TRACK_ORDERS_ON = True
    #if 'trac' not in c:
        c.t_opts = {        # __________    O P T I O N S    __________
                            'symbols'     : [],   # List of symbols to filter for, like ['TSLA', 'SPY']
            'log_neg_cash': 0,    # Show cash only when negative.
            'log_cash'    : 1,    # Show cash values in logging window or not.
            'log_ids'     : 1,    # Include order id's in logging window or not.
            'log_unfilled': 1,    # When orders are unfilled. (stop & limit excluded).
            'log_cancels' : 1,    # When orders are canceled.
            }    # Move these to initialize() for better efficiency.
        c.trac = {}
        c.t_dates  = {  # To not overwhelm the log window, start/stop dates can be entered.
                        'active': 0,
            'start' : [],   # Start dates, option like ['2007-05-07', '2010-04-26']
            'stop'  : []    # Stop  dates, option like ['2008-02-13', '2010-11-15']
            }
        log.info('track_orders active. Headers ...')
        log.info('             Shares     Shares')
        log.info('Min   Action Order  Sym  Now   at Price   PnL   Stop or Limit   Cash  Id')
    # If 'start' or 'stop' lists have something in them, triggers ...
    if c.t_dates['start'] or c.t_dates['stop']:
        _date = str(c.get_datetime().date())
        if   _date in c.t_dates['start']:    # See if there's a match to start
            c.t_dates['active'] = 1
        elif _date in c.t_dates['stop']:     #   ... or to stop
            c.t_dates['active'] = 0
    else: c.t_dates['active'] = 1           # Set to active b/c no conditions.
    if c.t_dates['active'] == 0: return     # Skip if not active.
    def _minute():   # To preface each line with the minute of the day.
        bar_dt = c.get_datetime().astimezone(_tz('US/Eastern'))
        return (bar_dt.hour * 60) + bar_dt.minute - 570 # (-570 = 9:31a)
    def _trac(to_log):      # So all logging comes from the same line number,
        log.info(' {}   {}'.format(str(_minute()).rjust(3), to_log))  # for vertical alignment in the logging window.

    for oid in c.trac.copy():               # Existing known orders
        o = c.get_order(oid)
        if c.t_opts['symbols'] and (o.sid.symbol not in c.t_opts['symbols']): continue
        if o.dt == o.created: continue        # No chance of fill yet.
        cash = ''
        prc  = data.current(o.sid, 'price') if data.can_trade(o.sid) else c.portfolio.positions[o.sid].last_sale_price
        if (c.t_opts['log_neg_cash'] and c.portfolio.cash < 0) or c.t_opts['log_cash']:
            cash = str(int(c.portfolio.cash))
        if o.status == 2:                     # Canceled
            do = 'Buy' if o.amount > 0 else 'Sell' ; style = ''
            if o.stop:
                style = ' stop {}'.format(o.stop)
                if o.limit: style = ' stop {} limit {}'.format(o.stop, o.limit)
            elif o.limit: style = ' limit {}'.format(o.limit)
            if c.t_opts['log_cancels']:
                _trac('  Canceled {} {} {}{} at {}   {}  {}'.format(do, o.amount,
                                                              o.sid.symbol, style, prc, cash, o.id[-4:] if c.t_opts['log_ids'] else ''))
            del c.trac[o.id]
        elif o.filled:                        # Filled at least some.
            filled = '{}'.format(o.amount)
            filled_amt = 0
            if o.status == 1:                   # Complete
                if 0 < c.trac[o.id] < o.amount:
                    filled   = 'all {}/{}'.format(o.filled - c.trac[o.id], o.amount)
                filled_amt = o.filled
            else:                                    # c.trac[o.id] value is previously filled total
                filled_amt = o.filled - c.trac[o.id]   # filled this time, can be 0
                c.trac[o.id] = o.filled                # save fill value for increments math
                filled = '{}/{}'.format(filled_amt, o.amount)
            if filled_amt:
                now = ' ({})'.format(c.portfolio.positions[o.sid].amount) if c.portfolio.positions[o.sid].amount else ' _'
                pnl = ''  # for the trade only
                amt = c.portfolio.positions[o.sid].amount ; style = ''
                if (amt - o.filled) * o.filled < 0:    # Profit-taking scenario including short-buyback
                    cb = c.portfolio.positions[o.sid].cost_basis
                    if cb:
                        pnl  = -filled_amt * (prc - cb)
                        sign = '+' if pnl > 0 else '-'
                        pnl  = '  ({}{})'.format(sign, '%.0f' % abs(pnl))
                if o.stop:
                    style = ' stop {}'.format(o.stop)
                    if o.limit: style = ' stop () limit {}'.format(o.stop, o.limit)
                elif o.limit: style = ' limit {}'.format(o.limit)
                if o.filled == o.amount: del c.trac[o.id]
                _trac('   {} {} {}{} at {}{}{}'.format(
              'Bot' if o.amount > 0 else 'Sold', filled, o.sid.symbol, now,
            '%.2f' % prc, pnl, style).ljust(52) + '  {}  {}'.format(cash, o.id[-4:] if c.t_opts['log_ids'] else ''))
        elif c.t_opts['log_unfilled'] and not (o.stop or o.limit):
            _trac('      {} {}{} unfilled  {}'.format(o.sid.symbol, o.amount,
                                                  ' limit' if o.limit else '', o.id[-4:] if c.t_opts['log_ids'] else ''))

    oo = get_open_orders().values()
    if not oo: return                       # Handle new orders
    cash = ''
    if (c.t_opts['log_neg_cash'] and c.portfolio.cash < 0) or c.t_opts['log_cash']:
        cash = str(int(c.portfolio.cash))
    for oo_list in oo:
        for o in oo_list:
            if c.t_opts['symbols'] and (o.sid.symbol not in c.t_opts['symbols']): continue
            if o.id in c.trac: continue         # Only new orders beyond this point
            prc = data.current(o.sid, 'price') if data.can_trade(o.sid) else c.portfolio.positions[o.sid].last_sale_price
            c.trac[o.id] = 0 ; style = ''
            now  = ' ({})'.format(c.portfolio.positions[o.sid].amount) if c.portfolio.positions[o.sid].amount else ' _'
            if o.stop:
                style = ' stop {}'.format(o.stop)
                if o.limit: style = ' stop {} limit {}'.format(o.stop, o.limit)
            elif o.limit: style = ' limit {}'.format(o.limit)
            _trac('{} {} {}{} at {}{}'.format('Buy' if o.amount > 0 else 'Sell',
                                          o.amount, o.sid.symbol, now, '%.2f' % prc, style).ljust(52) + '  {}  {}'.format(cash, o.id[-4:] if c.t_opts['log_ids'] else ''))
def pvr(context, data):
    ''' Custom chart and/or logging of profit_vs_risk returns and related information
    '''
    import time
    from datetime import datetime
    from pytz import timezone      # Python will only do once, makes this portable.
                                   #   Move to top of algo for better efficiency.
    c = context  # Brevity is the soul of wit -- Shakespeare [for readability]
    t_zone   = 'US/Eastern' #NYSE centric time zone
    if not c.PVR_ON:
    #if 'pvr' not in c:
        c.PVR_ON = True
        cur_time=datetime.now(timezone(t_zone))
        c.df=pd.DataFrame([c.recorded_vars], index=[cur_time])
        c.df.index.name = 'Date'
        
        #cur_time=datetime.now(timezone(t_zone)).strftime("%Y-%m-%d %H:%M")
        #df=pd.DataFrame().from_dict(c.recorded_vars)
        
        #myIndex = pd.date_range(cur_time, cur_time, freq='T')
        #df.reindex(myIndex)        
        
        # For real money, you can modify this to total cash input minus any withdrawals
        manual_cash = c.portfolio.starting_cash
        time_zone   = 'US/Pacific'   # Optionally change to your own time zone for wall clock time
        if AlgoParms['IS_LIVE']:
            sync_portfolio_to_broker(context, data)
            
        c.pvr = {
            'options': {
                # # # # # # # # # #  Options  # # # # # # # # # #
                'logging'         : 1,    # Info to logging window with some new maximums
                'log_summary'     : 1,  # Summary every x days. 252/yr

                'record_pvr'      : 1,    # Profit vs Risk returns (percentage)
                'record_pvrp'     : 0,    # PvR (p)roportional neg cash vs portfolio value
                'record_cash'     : 0,    # Cash available
                'record_max_lvrg' : 1,    # Maximum leverage encountered
                'record_max_risk' : 0,    # Highest risk overall
                'record_shorting' : 1,    # Total value of any shorts
                'record_max_shrt' : 0,    # Max value of shorting total
                'record_cash_low' : 0,    # Any new lowest cash level
                'record_q_return' : 1,    # Quantopian returns (percentage)
                'record_pnl'      : 1,    # Profit-n-Loss
                'record_risk'     : 0,    # Risked, max cash spent or shorts beyond longs+cash
                'record_leverage' : 1,    # End of day leverage (context.account.leverage)
                'record_cagr'     : 0,
                # All records are end-of-day or the last data sent to chart during any day.
                # The way the chart operates, only the last value of the day will be seen.
                # # # # # # # # #  End options  # # # # # # # # #
            },
            'pvr'        : 0,      # Profit vs Risk returns based on maximum spent
            'cagr'       : 0,
            'max_lvrg'   : 0,
            'max_shrt'   : 0,
            'max_risk'   : 0,
            'days'       : 0.0,
            'date_prv'   : '',
            'date_end'   : c.get_environment('end').date(),
            'cash_low'   : manual_cash,
            'cash'       : manual_cash,
            'start'      : manual_cash,
            'tz'         : time_zone,
            'begin'      : time.time(),  # For run time
            'run_str'    : '{} to {}  ${}  {} {}'.format(c.get_environment('start').date(), c.get_environment('end').date(), int(manual_cash), datetime.now(timezone(time_zone)).strftime("%Y-%m-%d %H:%M"), time_zone)
        }
        if c.pvr['options']['record_pvrp']: c.pvr['options']['record_pvr'] = 0 # if pvrp is active, straight pvr is off
        if c.get_environment('arena') not in ['backtest', 'live']: c.pvr['options']['log_summary'] = 1 # Every day when real money
        log.info(c.pvr['run_str'])
    p = c.pvr ; o = c.pvr['options'] ; pf = c.portfolio ; pnl = pf.portfolio_value - p['start']
    def _pvr(c):
        p['cagr'] = ((pf.portfolio_value / p['start']) ** (1 / (p['days'] / 252.))) - 1
        ptype = 'PvR' if o['record_pvr'] else 'PvRp'
        log.info('{} {} %/day   cagr {}   Portfolio value {}   PnL {}'.format(ptype, '%.4f' % (p['pvr'] / p['days']), '%.3f' % p['cagr'], '%.0f' % pf.portfolio_value, '%.0f' % pnl))
        log.info('  Profited {} on {} activated/transacted for PvR of {}%'.format('%.0f' % pnl, '%.0f' % p['max_risk'], '%.1f' % p['pvr']))
        log.info('  QRet {} PvR {} CshLw {} MxLv {} MxRisk {} MxShrt {}'.format('%.2f' % (100 * pf.returns), '%.2f' % p['pvr'], '%.0f' % p['cash_low'], '%.2f' % p['max_lvrg'], '%.0f' % p['max_risk'], '%.0f' % p['max_shrt']))
    def _minut():
        dt = c.get_datetime().astimezone(timezone(p['tz']))
        return str((dt.hour * 60) + dt.minute - 570).rjust(3)  # (-570 = 9:31a)
    date = c.get_datetime().date()
    if p['date_prv'] != date:
        p['date_prv'] = date
        p['days'] += 1.0
    do_summary = 0
    if o['log_summary'] and p['days'] % o['log_summary'] == 0 and _minut() == '100':
        do_summary = 1              # Log summary every x days
    if do_summary or date == p['date_end']:
        p['cash'] = pf.cash
    elif p['cash'] == pf.cash and not o['logging']: return  # for speed

    shorts = sum([z.amount * z.last_sale_price for s, z in pf.positions.items() if z.amount < 0])
    #shorts = sum([z.amount * z.cost_basis for s, z in pf.positions.items() if z.amount < 0])
    new_key_hi = 0                  # To trigger logging if on.
    cash       = pf.cash
    cash_dip   = int(max(0, p['start'] - cash))
    risk       = int(max(cash_dip, -shorts))

    if o['record_pvrp'] and cash < 0:   # Let negative cash ding less when portfolio is up.
        cash_dip = int(max(0, cash_dip * p['start'] / pf.portfolio_value))
        # Imagine: Start with 10, grows to 1000, goes negative to -10, should not be 200% risk.

    if int(cash) < p['cash_low']:             # New cash low
        new_key_hi = 1
        p['cash_low'] = int(cash)             # Lowest cash level hit
        if o['record_cash_low']: record(CashLow = p['cash_low'])

    if c.account.leverage > p['max_lvrg']:
        new_key_hi = 1
        p['max_lvrg'] = c.account.leverage    # Maximum intraday leverage
        if o['record_max_lvrg']: record(MxLv    = p['max_lvrg'])

    if shorts < p['max_shrt']:
        new_key_hi = 1
        p['max_shrt'] = shorts                # Maximum shorts value
        if o['record_max_shrt']: record(MxShrt  = p['max_shrt'])

    if risk > p['max_risk']:
        new_key_hi = 1
        p['max_risk'] = risk                  # Highest risk overall
        if o['record_max_risk']:  record(MxRisk = p['max_risk'])

    # Profit_vs_Risk returns based on max amount actually invested, long or short
    if p['max_risk'] != 0: # Avoid zero-divide
        p['pvr'] = 100 * pnl / p['max_risk']
        ptype = 'PvRp' if o['record_pvrp'] else 'PvR'
        if o['record_pvr'] or o['record_pvrp']: record(**{ptype: p['pvr']})

    if o['record_shorting']: record(Shorts = shorts)             # Shorts value as a positve
    if o['record_leverage']: record(Lv     = c.account.leverage) # Leverage
    if o['record_cash']    : record(Cash   = cash)               # Cash
    if o['record_risk']    : record(Risk   = risk)  # Amount in play, maximum of shorts or cash used
    if o['record_q_return']: record(QRet   = 100 * pf.returns)
    if o['record_pnl']     : record(PnL    = pnl)                # Profit|Loss
    if o['record_cagr']     : record(CAGR   = p['cagr'])                # Profit|Loss

    if o['logging'] and new_key_hi:
        log.info('{}{}{}{}{}{}{}{}{}{}{}{}'.format(_minut(),
            ' Lv '     + '%.1f' % c.account.leverage,
            ' MxLv '   + '%.2f' % p['max_lvrg'],
            ' QRet '   + '%.1f' % (100 * pf.returns),
            ' PvR '    + '%.1f' % p['pvr'],
            ' PnL '    + '%.0f' % pnl,
            ' Cash '   + '%.0f' % cash,
            ' CshLw '  + '%.0f' % p['cash_low'],
            ' Shrt '   + '%.0f' % shorts,
            ' MxShrt ' + '%.0f' % p['max_shrt'],
            ' Risk '   + '%.0f' % risk,
            ' MxRisk ' + '%.0f' % p['max_risk']
        ))
        
    if do_summary: _pvr(c)

    if c.PSF_name: #Portfolio Summary File for Telegram Bot
        #with open(c.PSF_name, 'a') as f:
            #tmpstring = '{}-{}\n'.format(datetime.now(timezone(t_zone)).strftime("%Y-%m-%d %H:%M"), t_zone)
   
            #for key, value in context.recorded_vars.items():
                #tmpstring = tmpstring + "\t{0:<20} {1}".format(key, value) + "\n"
            #print(tmpstring)
            ##context.bot.send_message(chat_id=update.message.chat_id, text=tmpstring)    
            #f.write(tmpstring)
            cur_time=datetime.now(timezone(t_zone))
            #cur_time=datetime.now(timezone(t_zone)).strftime("%Y-%m-%d %H:%M")
            if (c.df is None) or (len(c.df)==0) or (c.df.empty):
                c.df=pd.DataFrame([c.recorded_vars], index=[cur_time])
                c.df.index.name='ts'             
            else:
                newrow_df=pd.DataFrame([c.recorded_vars], index=[cur_time])
                c.df = pd.concat([df, newrow_df], ignore_index=False)  
            c.df.to_csv(c.PSF_name, index_label='daily_factor_and_port_dist_file')
    if c.get_datetime() == c.get_environment('end'):   # Summary at end of run
        _pvr(c) ; elapsed = (time.time() - p['begin']) / 60  # minutes
        log.info( '{}\nRuntime {} hr {} min'.format(p['run_str'], int(elapsed / 60), '%.1f' % (elapsed % 60)))

## Put any initialization logic here. The context object will be passed to
## the other methods in your algorithm.


## Stop_loss = 0.10 # Stop loss of 10%

#def initialize(context):

    ## Check price at the end of each day for stop loss.
    #schedule_function(exit_on_StopLoss,
                      #date_rules.every_day(),
                      #time_rules.market_close())

