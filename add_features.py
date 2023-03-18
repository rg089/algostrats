import pandas as pd

def check_numeric(df, col):
    return df[col].dtype in ['float64', 'int64']

def difference_cols(df, a, b):
    df[f'{a}-{b}'] = df[a] - df[b]
    return df, f'{a}-{b}'

def get_ma_base_string(s):
    idx = s.find('_ma_')
    if idx == -1:
        return None
    return s[:idx]

def moving_avg(df, col, window_size=3, center=False):
    col_name = f'{col}_ma_{window_size}'
    df[col_name] = df[col].rolling(window_size, min_periods=1, center=center).mean()
    return df, col_name

def slope(df, col, window):
    col_name = f'{col}_slope_{window}'
    df[col_name] = df[col].diff(periods=window).fillna(df[col])/window
    return df, col_name

def max_change_helper(seq):
    ans = []
    tracker = {i:0 for i in range(seq[-1]+1)}
    for i in seq:
        tracker[i] += 1
        ans.append(tracker[i])
    return ans

def max_change(df, col):
    inc_tracker = df[col].diff().lt(0).cumsum().values
    dec_tracker = df[col].diff().gt(0).cumsum().values
    
    inc_values = max_change_helper(inc_tracker)
    dec_values = max_change_helper(dec_tracker)
    
    combined = [inc_values[i]-1 if inc_values[i] >= dec_values[i] \
                else -dec_values[i]+1 for i in range(len(inc_values))]
    
    col_name = f'{col}_changelen'
    df[col_name] = combined
    return df, col_name

def discretize(df, col):
    stats = df[col].describe()
    low_thresh, high_thresh = stats['25%'], stats['75%']
    df[f'{col}_val'] = df[col].apply(lambda x: 0 if x<=low_thresh else 2 if x>=high_thresh else 1)
    df[f'{col}_polarity'] = df[col].apply(lambda x: 1 if x>0 else -1)
    # df[f'{col}_discrete'] = df[f'{col}_val'] + df[f'{col}_polarity']
    return df, [f'{col}_val', f'{col}_polarity'] #, f'{col}_discrete']

def add_features(feed):
    columns_to_use = ['Open', 'High', 'Low', 'Close', 'Volume', 'row_num', 'Open_n', 
                  'High_n', 'Low_n', 'Close_n', 'Volume_n', 'SMA_10',
       'SMA_20', 'VOL_SMA_20', 'RSI_14', 'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0',
       'BBB_5_2.0', 'BBP_5_2.0', 'MACD_12_26_9', 'MACDh_12_26_9',
       'MACDs_12_26_9', 'VWAP_D', 'MOM_30', 'CMO_14']
        
    subtract_col_names = [('High', 'Low'), ('Open', 'Close'), ('SMA_20', 'SMA_10'), ('Open_n', 'Close_n'), ('High_n', 'Low_n'), ('Open', 'High')]
    subtract_cols = []

    for cols in subtract_col_names:
        feed, added_col = difference_cols(feed, cols[0], cols[1])
        subtract_cols.append(added_col)
        
    window_sizes = [1,5,10,20,50]
    pre_avg_cols = columns_to_use + subtract_cols
    avg_cols = []

    for window in window_sizes:
        for col in pre_avg_cols:
            feed, added_col = moving_avg(feed, col, window_size=window)
            avg_cols.append(added_col)
                
    pre_slope_cols = columns_to_use + subtract_cols + avg_cols
    window_sizes = [1,3,5,10,15]
    slope_cols = []

    for window in window_sizes:
        for col in pre_slope_cols:
            feed, added_col = slope(feed, col, window=window)
            slope_cols.append(added_col)
            
    intra_ma_diff_cols = []

    for i in range(len(avg_cols)-1):
        for j in range(i+1, len(avg_cols)):
            colA, colB = avg_cols[i], avg_cols[j]
            baseA, baseB = get_ma_base_string(colA), get_ma_base_string(colB)
            if baseA != baseB: continue
            
            feed, added_col = difference_cols(feed, colA, colB)
            intra_ma_diff_cols.append(added_col)
            
    pre_change_cols = columns_to_use + subtract_cols + avg_cols + slope_cols + intra_ma_diff_cols
    change_cols = []

    for col in pre_change_cols:
        feed, added_col = max_change(feed, col)
        change_cols.append(added_col)
        
    pre_discrete_cols = pre_change_cols + change_cols
    discrete_cols = []

    for col in pre_discrete_cols:
        feed, added_cols = discretize(feed, col)
        for added_col in added_cols: discrete_cols.append(added_col)
        
    return feed, pre_discrete_cols, discrete_cols