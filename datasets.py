import quandl
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from collections import defaultdict
from scipy.stats import spearmanr
from talib import RSI, BBANDS, MACD, ATR, NATR, PPO
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from catboost import Pool, CatBoostRegressor
from statsmodels.api import OLS, add_constant
from zipline.utils.calendars import get_calendar
from alphalens import performance as perf
from alphalens.utils import get_clean_factor_and_forward_returns, rate_of_return, std_conversion
from alphalens.tears import create_summary_tear_sheet, create_full_tear_sheet

# Combines data creation processing and ML training for Sharadar Equities-Fund Data:
df_sefp = (pd.read_csv('~/Desktop/Eonum/SHARADAR_SEFP.csv',
                       parse_dates=['date'],
                       index_col=['date', 'ticker'],
                       infer_datetime_format=True)
           .sort_index())
quandl.ApiConfig.api_key = 'fvy5hW5Eeg8U4ecXQyVz'
tickers = quandl.get_table('SHARADAR/TICKERS')
daily = quandl.get_table('SHARADAR/DAILY')
# prices = pd.concat([quandl.get_table('SHARADAR/SEP'), quandl.get_table('SHARADAR/SFP')]).set_index(['ticker', 'date']).sort_index()
DATA_STORE = 'sefpassets.h5'

with pd.HDFStore(DATA_STORE) as store:
    store.put('quandl/wiki/prices', df_sefp)
    store.put('us_equities/stocks', tickers.set_index(['lastupdated', 'ticker']))
    store.put('us/equities/stocks', daily.set_index(['lastupdated', 'ticker']))
percentiles = [.001, .01, .02, .03, .04, .05]
percentiles += [1-p for p in percentiles[::-1]]
T = [1, 5, 10, 21, 42, 63]
min_obs = 7 * 12 * 21   # 21 * MONTH
lookheads = [1, 5, 21]
test_params = list(product(lookheads, [int(4.5 * 252), 252], [63, 21]))
ohlcv = ['open', 'closeadj', 'low', 'high', 'volume']
idx = pd.IndexSlice
with pd.HDFStore(DATA_STORE) as store:
    prices = (store['quandl/wiki/prices']
              .loc[idx['1998-01-01':'2022-01-21', :], ohlcv] # select OHLCV from timeframe
              .swaplevel()
              .sort_index())
    metadata = pd.merge(store['us_equities/stocks'].loc[:, ['scalemarketcap', 'sector']],
                         store['us/equities/stocks'].loc[:, 'marketcap'], how='outer', on=['ticker', 'lastupdated'])
    metadata.index.names = ['symbol', 'date']

prices.volume /= 1e3
prices.index.names = ['symbol', 'date']
metadata.index.name = 'symbol'
nobs = prices.groupby(level='symbol').size()
prices = prices.loc[idx[nobs[nobs > 100].index, :], :]   # needs to change back to min
# Limit universe to 1k sotkcs with highest market cap
# universe = metadata.loc[metadata['scalemarketcap'] == '5 - Large'].index
universe = metadata.marketcap.nlargest(1000).index
prices = prices.loc[idx[universe, :], :].drop_duplicates()
metadata = metadata.loc[universe]

class MultipleTimeSeriesCV:
    """Generates tuples of train_idx, test_idx pairs
    Assumes the MultiIndex contains levels 'symbol' and 'date'
    purges overlapping outcomes"""

    def __init__(self,
                 n_splits=3,
                 train_period_length=126,
                 test_period_length=21,
                 lookahead=None,
                 date_idx='date',
                 shuffle=False):
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle
        self.date_idx = date_idx

    def split(self, X, y=None, groups=None):
        unique_dates = X.index.get_level_values(self.date_idx).unique()
        days = sorted(unique_dates, reverse=True)
        split_idx = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            split_idx.append([int(train_start_idx / 9), int(train_end_idx / 9),
                              int(test_start_idx / 9), int(test_end_idx / 9)])
        dates = X.reset_index()[[self.date_idx]]

        for train_start, train_end, test_start, test_end in split_idx:
            train_idx = dates[(dates[self.date_idx] > days[train_start])
                              & (dates[self.date_idx] <= days[train_end])].index
            test_idx = dates[(dates[self.date_idx] > days[test_start])
                             & (dates[self.date_idx] <= days[test_end])].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx.to_numpy(), test_idx.to_numpy()

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
def ic_lgbm(preds, train_data):
    """Custom IC eval metric for lightgbm"""
    is_higher_better = True
    return 'ic', spearmanr(preds, train_data.get_label())[0], is_higher_better
def compute_bb(close):
    high, mid, low = BBANDS(close, timeperiod=20)
    return pd.DataFrame({'bb_high': high, 'bb_low': low}, index=close.index)
def compute_atr(stock_data):
    df = ATR(stock_data.high, stock_data.low,
             stock_data.closeadj, timeperiod=14)
    return df.sub(df.mean()).div(df.std())
def compute_macd(close):
    macd = MACD(close)[0]
    return (macd - np.mean(macd))/np.std(macd)
def get_fi(model):
    """Return normalized feature importance as pd.Series"""
    fi = model.feature_importance(importance_type='gain')
    return (pd.Series(fi / fi.sum(), index=model.feature_name()))
prices['dollar_vol'] = prices[['closeadj', 'volume']].prod(1).div(1e3)
dollar_vol_ma = (prices.dollar_vol.unstack('symbol').rolling(window=21, min_periods=1).mean())
prices['dollar_vol_rank'] = (dollar_vol_ma.rank(axis=1, ascending=False).stack('symbol').swaplevel())
prices = (prices.join(prices.groupby(level='symbol').closeadj.apply(compute_bb)))
prices['bb_high'] = prices.bb_high.sub(prices.closeadj).div(prices.bb_high).apply(np.log1p)
prices['bb_low'] = prices.closeadj.sub(prices.bb_low).div(prices.closeadj).apply(np.log1p)
prices['NATR'] = prices.groupby(level='symbol', group_keys=False).apply(lambda x: NATR(x.high, x.low, x.closeadj))
prices['rsi'] = prices.groupby(level='symbol').closeadj.apply(RSI)
prices['PPO'] = prices.groupby(level='symbol').closeadj.apply(PPO)
prices['ATR'] = prices.groupby('symbol', group_keys=False).apply(compute_atr)
prices['MACD'] = prices.groupby('symbol', group_keys=False).closeadj.apply(compute_macd)
metadata.sector = pd.factorize(metadata.sector)[0].astype(int)
prices = prices.join(metadata['sector'])
by_sym = prices.groupby(level='symbol').closeadj
for t in T:
    prices[f'r{t:02}'] = by_sym.pct_change(t)
    prices[f'r{t:02}dec'] = (prices[f'r{t:02}'].groupby(level='date').apply(lambda x: pd.qcut(
        x, q=10, labels=False, duplicates='drop')))
    prices[f'r{t:02}q_sector'] = (prices.groupby(['date', 'sector'])[f'r{t:02}'].transform(lambda x: pd.qcut(
        x, q=5, labels=False, duplicates='drop')))
print('----------------Trying delete 2 dropnas, Lookheads: ', lookheads)
for t in lookheads:
    prices[f'r{t:02}_fwd'] = prices.groupby(level='symbol')[f'r{t:02}'].shift(-t)
# prices[[f'r{t:02}' for t in T]].describe()
outliers = prices[prices.r01 > 1].index.get_level_values('symbol').unique()
prices = prices.drop(outliers, level='symbol')
prices['year'] = prices.index.get_level_values('date').year
prices['month'] = prices.index.get_level_values('date').month
prices['weekday'] = prices.index.get_level_values('date').weekday
prices.drop(['open', 'closeadj', 'low', 'high', 'volume'], axis=1).to_hdf(DATA_STORE, 'model_data')

labels = sorted(prices.filter(like='_fwd').columns)
features = prices.columns.difference(labels).tolist()
tickers = prices.index.get_level_values('symbol').unique()
cat_cols = ['year', 'month', 'sector', 'weekday']   # 'age', 'msize', 'sector']
lr = LinearRegression()
daily_ic_names = ['daily_ic_mean', 'daily_ic_mean_n', 'daily_ic_median', 'daily_ic_median_n']
param_names = ['learning_rate', 'num_leaves', 'feature_fraction', 'min_data_in_leaf']
base_params = dict(boosting='gbdt', objective='regression', verbose=-1)
cv_params = list(product([.01, .1, .3], [4, 8, 32, 128], [.3, .6, .95], [250, 500, 1000]))
label_dict = dict(zip(lookheads, labels))
test_param_sample = np.random.choice(list(range(len(test_params))), size=int(len(test_params)), replace=False)
test_params = [test_params[i] for i in test_param_sample]
num_iterations = [10, 25, 50, 75] + list(range(100, 501, 50))
metric_cols = (param_names + ['t'] + daily_ic_names + [str(n) for n in num_iterations])
for feature in ['year', 'weekday', 'month']:
    prices[feature] = pd.factorize(prices[feature], sort=True)[0]
print('-----------Before loops - Train Configs:', len(test_params), prices.info())
lr_metrics, lgb_ic, lgb_metrics = ([] for _ in range(3))
for lookahead, train_len, test_len in tqdm(test_params):
    label = f'r{lookahead:02}_fwd'
    df = pd.get_dummies(prices.loc[:, features + [label]].dropna(), columns=cat_cols, drop_first=True)
    X, y = df.drop(label, axis=1), df[label]
    n_splits = int(4 * 252 / test_len)
    cvp = np.random.choice(list(range(len(cv_params))), size=int(len(cv_params) / 2), replace=False)
    cv_params_ = [cv_params[i] for i in cvp]
    print(f'Lookahead: {lookahead:2.0f} | '
          f'Train: {train_len:3.0f} | '
          f'Test: {test_len:2.0f} | '
          f'Params: {len(cv_params_):3.0f} | '
          f'Train configs: {len(test_params)}')
    cv = MultipleTimeSeriesCV(n_splits=n_splits,
                              test_period_length=test_len,
                              lookahead=lookahead,
                              train_period_length=train_len)
    ic, preds = [], []
    for i, (train_idx, test_idx) in enumerate(cv.split(X=X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        preds.append(y_test.to_frame('y_true').assign(y_pred=y_pred))
        ic.append(spearmanr(y_test, y_pred)[0])
    preds = pd.concat(preds)
    lr_metrics.append([lookahead,
                       train_len,
                       test_len,
                       np.mean(ic),
                       spearmanr(preds.y_true, preds.y_pred)[0]
                      ])

    outcome_data = prices.loc[:, features + [label_dict[lookahead]]]
    lgb_data = lgb.Dataset(data=outcome_data.drop(label_dict[lookahead], axis=1),
                           label=outcome_data[label_dict[lookahead]],
                           categorical_feature=cat_cols,
                           free_raw_data=False)
    predictions, metrics, feature_importance, daily_ic = [], [], [], []
    for p, param_vals in enumerate(cv_params_):
        key = f'{lookahead}/{train_len}/{test_len}/' + '/'.join([str(p) for p in param_vals])
        params = dict(zip(param_names, param_vals))
        params.update(base_params)
        cv_preds, nrounds = [], []
        ic_cv = defaultdict(list)
        for i, (train_idx, test_idx) in enumerate(cv.split(X=outcome_data)):
            lgb_train = lgb_data.subset(used_indices=train_idx.tolist(), params=params).construct()
            model = lgb.train(params=params, train_set=lgb_train, num_boost_round=num_iterations[-1], verbose_eval=False)
            if i == 0:
                fi = get_fi(model).to_frame()
            else:
                fi[i] = get_fi(model)
            test_set = outcome_data.iloc[test_idx, :]
            X_test = test_set.loc[:, model.feature_name()]
            y_test = test_set.loc[:, label_dict[lookahead]]
            y_pred = {str(n): model.predict(X_test, num_iteration=n) for n in num_iterations}
            cv_preds.append(y_test.to_frame('y_test').assign(**y_pred).assign(i=i))
        cv_preds = pd.concat(cv_preds).assign(**params)
        predictions.append(cv_preds)
        by_day = cv_preds.groupby(level='date')
        ic_by_day = pd.concat([by_day.apply(lambda x: spearmanr(x.y_test, x[str(n)])[0]).to_frame(n)
                               for n in num_iterations], axis=1)
        daily_ic_mean = ic_by_day.mean()
        daily_ic_mean_n = daily_ic_mean.idxmax()
        daily_ic_median = ic_by_day.median()
        daily_ic_median_n = daily_ic_median.idxmax()
        ic = [spearmanr(cv_preds.y_test, cv_preds[str(n)])[0] for n in num_iterations]

        metrics = pd.Series(list(param_vals) + [t, daily_ic_mean.max(), daily_ic_mean_n, daily_ic_median.max(),
                                                daily_ic_median_n] + ic, index=metric_cols)
        msg = f'\t{p:3.0f} | {t:3.0f} | {params["learning_rate"]:5.2f} | '
        msg += f'{params["num_leaves"]:3.0f} | {params["feature_fraction"]:3.0%} | {params["min_data_in_leaf"]:4.0f} | '
        msg += f' {max(ic):6.2%} | {ic_by_day.mean().max(): 6.2%} | {daily_ic_mean_n: 4.0f} | {ic_by_day.median().max(): 6.2%} | {daily_ic_median_n: 4.0f}'
        print(msg)
        metrics.to_hdf('tuning_lgb.h5', 'metrics/' + key)
        ic_by_day.assign(**params).to_hdf('tuning_lgb.h5', 'daily_ic/' + key)
        fi.T.describe().T.assign(**params).to_hdf('tuning_lgb.h5', 'fi/' + key)
        cv_preds.to_hdf('tuning_lgb.h5', 'predictions/' + key)
lr_metrics = pd.DataFrame(lr_metrics, columns=['lookahead', 'train_length', 'test_length', 'ic_by_day', 'ic'])
with pd.HDFStore('tuning_lgb.h5') as store:
    for i, key in enumerate(
        [k[1:] for k in store.keys() if k[1:].startswith('metrics')]):
        _, t, train_length, test_length = key.split('/')[:4]
        attrs = {
            'lookahead': t,
            'train_length': train_length,
            'test_length': test_length
        }
        s = store[key].to_dict()
        s.update(attrs)
        if i == 0:
            lgb_metrics = pd.Series(s).to_frame(i)
        else:
            lgb_metrics[i] = pd.Series(s)
    keys = [k[1:] for k in store.keys()]
    for key in keys:
        _, t, train_length, test_length = key.split('/')[:4]
        if key.startswith('daily_ic'):
            df = (store[key].drop(['boosting', 'objective', 'verbose'], axis=1)
                  .assign(lookahead=t, train_length=train_length, test_length=test_length))
            lgb_ic.append(df)
    lgb_ic = pd.concat(lgb_ic).reset_index()
lgb_metrics = pd.melt(lgb_metrics.T.drop('t', axis=1), value_name='ic',
                  id_vars=(['lookahead', 'train_length', 'test_length'] + param_names + daily_ic_names),
                  var_name='boost_rounds').dropna().apply(pd.to_numeric)
int_cols = ['lookahead', 'train_length', 'test_length', 'boost_rounds']
id_vars = ['date'] + int_cols[:-1] + param_names
lgb_ic = pd.melt(lgb_ic, id_vars=id_vars, value_name='ic', var_name='boost_rounds').dropna()
lgb_ic.loc[:, int_cols] = lgb_ic.loc[:, int_cols].astype(int)
lgb_daily_ic = lgb_ic.groupby(id_vars[1:] + ['boost_rounds']).ic.mean().to_frame('ic').reset_index()

def get_lgb_params(data, t=5, best=0):
    param_cols = int_cols[-3:-1] + param_names + ['boost_rounds']
    df = data[data.lookahead==t].sort_values('ic', ascending=False).iloc[best]
    return df.loc[param_cols]
def get_lgb_key(t, p):
    key = f'{t}/{int(p.train_length)}/{int(p.test_length)}/{p.learning_rate}/'
    return key + f'{int(p.num_leaves)}/{p.feature_fraction}/{int(p.min_data_in_leaf)}'
for best in range(10):
    best_params = get_lgb_params(lgb_daily_ic, t=1, best=best)
    key = get_lgb_key(1, best_params)
    rounds = str(int(best_params.boost_rounds))
    if best == 0:
        best_predictions = pd.read_hdf('tuning_lgb.h5', 'predictions/' + key)
        best_predictions = best_predictions[rounds].to_frame(best)
    else:
        best_predictions[best] = pd.read_hdf('tuning_lgb.h5', 'predictions/' + key)[rounds]

    param = best_params.to_dict()
    for p in ['min_data_in_leaf', 'num_leaves']:
        param[p] = int(param[p])
    train_length = int(param.pop('train_length'))
    test_length = int(param.pop('test_length'))
    num_boost_round = int(param.pop('boost_rounds'))
    param.update(base_params)
    cv = MultipleTimeSeriesCV(n_splits=int(252 / test_length), test_period_length=test_length, lookahead=1, train_period_length=train_length)

    predictions = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X=outcome_data), 1):
        lgb_train = lgb_data.subset(used_indices=train_idx.tolist(), params=param).construct()
        model = lgb.train(params=param, train_set=lgb_train, num_boost_round=num_boost_round, verbose_eval=False)
        test_set = outcome_data.iloc[test_idx, :]
        y_test = test_set.filter(like='fwd').rename(columns={'r05_fwd': 'y_test'})
        y_pred = model.predict(test_set.loc[:, model.feature_name()])
        predictions.append(y_test.assign(prediction=y_pred))
    if best == 0:
        test_predictions = (pd.concat(predictions).rename(columns={'prediction': best}))
    else:
        test_predictions[best] = pd.concat(predictions).prediction

best_predictions = best_predictions.sort_index()
best_predictions.to_hdf(DATA_STORE, 'lgb/train/01')
test_predictions.to_hdf(DATA_STORE, 'lgb/test/01')
test_tickers = best_predictions.index.get_level_values('symbol').unique()
factor = best_predictions.iloc[:, :5].mean(1).dropna().tz_localize('UTC', level='date').swaplevel()
trade_prices = prices.loc[idx[tickers], 'open'].unstack('symbol').tz_localize('UTC')
factor_data = get_clean_factor_and_forward_returns(factor=factor, prices=trade_prices, quantiles=5, periods=(1, 5, 10, 21), max_loss=.4)
# asset_list = results.index.levels[1].unique()     - run_pipeline(pipe, start, end)
# prices = get_pricing(asset_list, start_date='2016-06-15', end_date='2017-08-30', fields='open_price')   (1,3,5,10,21)
mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(factor_data, by_date=True, by_group=False, demeaned=True, group_adjust=False)
mean_quant_ret, std_quantile = perf.mean_return_by_quantile(factor_data, by_group=False, demeaned=True)
mean_quant_rateret = mean_quant_ret.apply(rate_of_return, axis=0, base_period=mean_quant_ret.columns[0])
mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
    factor_data,
    by_date=True,
    by_group=False,
    demeaned=True,
    group_adjust=False,
)
mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(rate_of_return, base_period=mean_quant_ret_bydate.columns[0],)
compstd_quant_daily = std_quant_daily.apply(std_conversion, base_period=std_quant_daily.columns[0])
alpha_beta = perf.factor_alpha_beta(factor_data, demeaned=True)
mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
    mean_quant_rateret_bydate,
    factor_data["factor_quantile"].max(),
    factor_data["factor_quantile"].min(),
    std_err=compstd_quant_daily,
)

cat_names = ['max_depth', 'min_child_samples']
cat_params = list(product([3, 5, 7, 9], [20, 250, 500]))
cattest_params = list(product(lookheads, [int(4.5 * 252), 252], [63]))
class CatBoostIC(object):
    ### Custom IC eval metric for CatBoost
    def is_max_optimal(self): return True
    def evaluate(self, approxes, target, weight):
        target = np.array(target)
        approxes = np.array(approxes).reshape(-1)
        rho = spearmanr(approxes, target)[0]
        return rho, 1
    def get_final_error(self, error, weight): return error
num_iterations = [10, 25, 50, 75] + list(range(100, 1001, 100))
metric_cols = (cat_names + ['t'] + daily_ic_names + [str(n) for n in num_iterations])
for lookahead, train_length, test_length in cattest_params:
    cvp = np.random.choice(list(range(len(cat_params))), size=int(len(cat_params) / 1), replace=False)
    cat_params_ = [cat_params[i] for i in cvp]
    n_splits = int(4 * 252 / test_length)
    print(f'Lookahead: {lookahead:2.0f} | Train: {train_length:3.0f} | '
          f'Test: {test_length:2.0f} | Params: {len(cat_params_):3.0f} | Train configs: {len(cat_params)}')
    cv = MultipleTimeSeriesCV(n_splits=n_splits,
                              lookahead=lookahead,
                              test_period_length=test_length,
                              train_period_length=train_length)
    outcome_data = prices.loc[:, features + [label_dict[lookahead]]].dropna()
    cat_cols_idx = [outcome_data.columns.get_loc(c) for c in cat_cols]
    catboost_data = Pool(label=outcome_data[label_dict[lookahead]],
                         data=outcome_data.drop(label_dict[lookahead], axis=1),
                         cat_features=cat_cols_idx)
    predictions, metrics, feature_importance, daily_ic = [], [], [], []
    key = f'{lookahead}/{train_length}/{test_length}'

    for p, param_vals in enumerate(cat_params_):
        params = dict(zip(cat_names, param_vals))
        cv_preds, nrounds = [], []
        ic_cv = defaultdict(list)
        for i, (train_idx, test_idx) in enumerate(cv.split(X=outcome_data)):
            train_set = catboost_data.slice(train_idx.tolist())
            model = CatBoostRegressor(**params)
            model.fit(X=train_set, verbose_eval=False)
            test_set = outcome_data.iloc[test_idx, :]
            X_test = test_set.loc[:, model.feature_names_]
            y_test = test_set.loc[:, label_dict[lookahead]]
            y_pred = {str(n): model.predict(X_test, ntree_end=n) for n in num_iterations}
            cv_preds.append(y_test.to_frame('y_test').assign(**y_pred).assign(i=i))
        cv_preds = pd.concat(cv_preds).assign(**params)
        predictions.append(cv_preds)
        by_day = cv_preds.groupby(level='date')
        ic_by_day = pd.concat([by_day.apply(lambda x: spearmanr(x.y_test, x[str(n)])[0]).to_frame(n)
                               for n in num_iterations], axis=1)
        daily_ic_mean = ic_by_day.mean()
        daily_ic_mean_n = daily_ic_mean.idxmax()
        daily_ic_median = ic_by_day.median()
        daily_ic_median_n = daily_ic_median.idxmax()
        ic = [spearmanr(cv_preds.y_test, cv_preds[str(n)])[0] for n in num_iterations]
        metrics = pd.Series(list(param_vals) +
                            [t, daily_ic_mean.max(), daily_ic_mean_n,
                             daily_ic_median.max(), daily_ic_median_n] + ic, index=metric_cols)
        metrics.to_hdf('tuning_catboost.h5', 'metrics/' + key)
        ic_by_day.assign(**params).to_hdf('tuning_catboost.h5', 'daily_ic/' + key)
        cv_preds.to_hdf('tuning_catboost.h5', 'predictions/' + key)
