import pandas as pd
import numpy as np
import import_ipynb
from backtest import Backtest
from feeds import BackFeed,DataFeed
from validation import Validate
from featfuncs import feat_aug,add_addl_features_feed,add_ta_features_feed,add_sym_feature_feed
from decision_tree import DecisionTree
import pickle
import pandas as pd
import warnings
from india_calendar import IBDay
warnings.simplefilter("ignore")
# # upload rulestrats.ipynb 
# from google.colab import files
# uploaded=files.upload()
# change to True if on colab
colab=False
from rulestrats import MomStrat,GapBet,AdaMomCMOADF, RuleStrat
from add_features import add_features
import pickle
from rlagents import RLStratAgentDyn, COLS
from stable_baselines3 import PPO,A2C,DQN
from stable_baselines3.common.vec_env import StackedObservations
from stable_baselines3.common.monitor import Monitor as Mon
from feed_env import Episode
import aspectlib
import os


class DTStrat(RuleStrat):
    def __init__(self,th=.00025, model_path='saved_models/dt_hp_11.pkl'):
        super(DTStrat,self).__init__()
        self.logL=[]
        self.th=th
        self.model_type='rule_based'
        self.data_cols='all'
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
    def check_entry_batch(self,dfD):
        # print(dfD)
        decisionsD={t:0 for t in dfD}
        stopD={t:0 for t in dfD}
        targetD={t:0 for t in dfD}
        
        for t in dfD.keys():
            data=dfD[t]
            last_row = data.iloc[-1:]
            action = self.model.predict(last_row)[0]
            decisionsD[t]=action
            
        return decisionsD,stopD,targetD
    
    def exit_predicate(self,row,df):
        return False

        
models = [model for model in os.listdir('saved_models') if model.startswith('dt')]

with open('additional_utils/cols.pkl', 'rb') as f:
        d = pickle.load(f)
imp_cols = d['imp_cols']
cols_to_use = d['cols_to_use'] + ['row_num']

def get_dist(bt):
    rewards = [bt.results[t][d]['tot'] for t in bt.results for d in bt.results[t]]
    reward_avg = np.mean(rewards)
    reward_std = np.std(rewards)
    return (f'{reward_avg:.1f}\u00B1{reward_std:.1f}')


final_data = {}
# data=pd.read_csv('./capvolfiltered.csv')
# tickers=list(data.iloc[0:10]['ticker'].values)
        
paths = [
    # 'realdata/augdata_01-Jan-2022_5m.csv',
    #         'realdata/augdata_03-Feb-2022_5m.csv',
    #         'realdata/augdata_02-Mar-2022_5m.csv', 
    #         'realdata/augdata_04-Apr-2022_5m.csv', 
    #         'realdata/augdata_02-May-2022_5m.csv',  
    #         'realdata/augdata_04-Jul-2022_5m.csv',
    #         'realdata/augdata_16-Dec-2022_5m.csv',
            'realdata/alldata.csv']


for pickle_path in paths:
    pickle_name = pickle_path.split("/")[-1].rstrip('.csv')
    pickle_name1 = f'datafeed_{pickle_name}_False_True_False.pkl'
    pickle_name2 = f'datafeed_{pickle_name}_True_False_False.pkl'
    
    
    if os.path.exists(os.path.join('..', 'algodata', 'realdata', pickle_name1)):
        feed = pickle.load(open(os.path.join('..', 'algodata', 'realdata', pickle_name1), 'rb'))
        print(f'Reading data from: {pickle_name1}')
        # print(feed.data[list(feed.data.keys())[0]].columns)
        for ticker in feed.data:
            df = feed.data[ticker]
            feed.data[ticker], _, _ = add_features(df, columns_to_use=['row_num'])
            
        feed.ndata={}
        for t in feed.tickers:
            print(f'[INFO] On ticker={t}')
            dfa=feed.data[t]
            dfL=[]
            feed.ndata[t]={}
            for d in dfa['Date'].unique():
                pdt=pd.to_datetime(d)
                pdtp=pdt-IBDay(1)
                df=dfa.loc[(pd.to_datetime(dfa['Date'])<=pdt)&
                            (pd.to_datetime(dfa['Date'])>=pdtp)]
                df['row_num'] = np.arange(len(df))
                df=df[~df.index.duplicated(keep='first')]
                df=df.sort_index()
                dfc=df.loc[df['Date']==d]
                feed.offsets[t][d]=df.shape[0]-dfc.shape[0]
                feed.ndata[t][d]=df
        
    else:
        feed = pickle.load(open(os.path.join('..', 'algodata', 'realdata', pickle_name2), 'rb'))
        print('Processing....')
        add_addl_features_feed(feed,tickers=feed.tickers)
        add_sym_feature_feed(feed,tickers=feed.tickers)
        print('Adding new features....')
        for ticker in feed.data:
            df = feed.data[ticker]
            feed.data[ticker], _, _ = add_features(df, columns_to_use=cols_to_use)
        
                
    for model_name in models:
        print(model_name)
        if model_name not in final_data:
            final_data[model_name] = []
            
        bt=Backtest(feed,tickers=feed.tickers,add_features=False,target=.001,stop=.01,txcost=0.001,
                    loc_exit=True,scan=True,topk=5,deploy=True,save_dfs=False)
        dtStrat = DTStrat(model_path=f'saved_models/{model_name}')
        ans = []

        bt.run_all(tickers=feed.tickers,model=dtStrat,verbose=False)
        curr = get_dist(bt)
        final_data[model_name].append(curr)
        
print(final_data)
df = pd.DataFrame(final_data, index=paths)
df.to_csv('results/dt_on_real.csv')
