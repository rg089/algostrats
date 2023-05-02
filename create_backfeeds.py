import pandas as pd
import numpy as np
import import_ipynb
from feeds import BackFeed
from featfuncs import feat_aug,add_addl_features_feed,add_ta_features_feed,add_sym_feature_feed
import pickle
import pandas as pd
import warnings
import os
warnings.simplefilter("ignore")


def create_backfeeds(synthetic=True, simple=True, n=1, nw=4):
    data=pd.read_csv('./capvolfiltered.csv')
    tickers=list(data.iloc[0:10]['ticker'].values)
    for i in range(n):
        print(f'[INFO] Currently on backfeed: {i+1}')
        feed=BackFeed(tickers=tickers, nd=5, nw=nw, interval='5m', synthetic=synthetic, simple=simple)
        print('Processing feed')
        add_addl_features_feed(feed,tickers=feed.tickers)
        add_sym_feature_feed(feed,tickers=feed.tickers)
        fname = os.path.join('..', 'algodata', 'backfeeds', f'backfeed_{nw}w_{synthetic}_{simple}_{i+1}.pkl')
        pickle.dump(feed, open(fname, 'wb'))
    print(f'[INFO] Task Complete!')
    

if __name__=='__main__':
    create_backfeeds(synthetic=True, simple=True, n=10, nw=4)
    create_backfeeds(synthetic=True, simple=False, n=10, nw=4)
    create_backfeeds(synthetic=False, simple=False, n=1, nw=4)