from stable_baselines3 import PPO,A2C,DQN
from stable_baselines3.common.vec_env import StackedObservations
from stable_baselines3.common.monitor import Monitor as Mon
import warnings
warnings.simplefilter("ignore")

import import_ipynb
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from datetime import datetime as dt
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pickle
from threading import Thread
import threading
from IPython import display
import time,getopt,sys,os
from feeds import BackFeed,DataFeed
from featfuncs import feat_aug,add_addl_features_feed,add_ta_features_feed,add_sym_feature_feed
from featfuncs import add_global_indices_feed
from feed_env import Episode
import aspectlib
from rlagents import RLStratAgentDyn, COLS, RLStratAgentDynFeatures
from backtest import Backtest
from feeds import BackFeed,DataFeed
from validation import Validate
import pickle
import plotly.express as px
from india_calendar import IBDay
import plotly
from add_features import add_features

from utils import read_yaml, extract_suffix
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", "-c", type=str, default="configs/config_base.yaml", help="Path to the config file")

args = parser.parse_args()
config_path = args.config_path

config = read_yaml(config_path)

DATAPATH='../algodata'
if not os.path.exists(DATAPATH):
    os.makedirs(DATAPATH)

algorithm=PPO # The algorithm to use
synthetic= config['synthetic']
simple= config['simple']
nd,nw=5,config['weeks']
training_steps=config['training_steps']

config_suffix = extract_suffix(config_path)
modelname = config.get('modelname', f'{config_suffix}.pth')

deploy = config['deploy']
loadfeed = config['loadfeed']
datafeed = config['datafeed']
use_alt_data = config['use_alt']
datafiles = config['datafile']
win = config['win']
loadfeed_path = config.get('loadfile', '')
epochs = config.get('epochs', 1)
use_raw_features = config.get('old_features', True)
use_new_features = config.get('new_features', False)
create_feed = config.get('create_feed', False)
dynamic_test = config.get('dynamic_test_features', False)
use_prediscrete = config.get('use_prediscrete', False)
use_normalized =  config.get('use_normalized', False)


if not loadfeed or not loadfeed_path:
    loadfeed_path = os.path.join('..', 'algodata', f'btfeed_{config_suffix}.pkl') # Path to save
    
modelname = f'{modelname.rstrip(".pth")}_{use_raw_features}_{use_new_features}.pth'

colab=False
n_steps=2048

with open('additional_utils/cols.pkl', 'rb') as f:
        d = pickle.load(f)
imp_cols = d['imp_cols']
cols_to_use = d['cols_to_use']
prediscrete_imp_cols = d['prediscrete_imp_cols']
imp_cols_n = d['imp_cols_n']


if use_prediscrete:
    imp_cols = prediscrete_imp_cols


def stringify(x):
    return pd.to_datetime(x['Datetime']).strftime('%d-%b-%Y')


if type(datafiles) == str: 
    datafiles = [datafiles]

   
if use_new_features and use_raw_features:
    using_cols = COLS + imp_cols
elif use_raw_features:
    using_cols = COLS
elif use_new_features:
    if use_normalized: using_cols = imp_cols_n
    else: using_cols = imp_cols

continuing_cols = using_cols + ['Date', 'datetime']
needed_cols = ['row_num', 'Close_n', 'Open_n', 'Open', 'Close', 'High', 'Low', 'Open_n', 'High_n', 'Low_n', 'Volume_n']
for col in needed_cols:
    if col not in continuing_cols: continuing_cols.append(col)

for epoch in range(epochs):
    for idx, datafile in enumerate(datafiles):
        print(f'[INFO] EPOCH: {epoch} FILE: {datafile}!')
        
        datafile_suffix = os.path.basename(datafile).rstrip('.csv')
        datafeed_path = os.path.join('..', 'algodata', 'realdata', 
                f'datafeed_{datafile_suffix}_{use_raw_features}_{use_new_features}_{use_prediscrete}.pkl')
        if use_normalized:
            datafeed_path = datafeed_path.replace('.pkl', '_normalized.pkl')

        if os.path.exists(datafeed_path) and not create_feed:
            feed = pickle.load(open(datafeed_path, 'rb'))
            print('[INFO] Loading pickle file for datafeed!')
            
            for t in feed.ndata:
                for d in feed.ndata[t]:
                    if feed.ndata[t][d].isnull().values.any(): 
                        feed.ndata[t][d]=feed.ndata[t][d].fillna(1)
                        # print(t,d)
                    if feed.ndata[t][d].isin([-np.inf,np.inf]).values.any():
                        feed.ndata[t][d]=feed.ndata[t][d].replace([np.inf, -np.inf],1)
                        
        else:
            DATAFILE=os.path.join(DATAPATH, datafile)
            print(f'Reading datafile: {DATAFILE}')
            df=pd.read_csv(DATAFILE)
            
            if 'Date' not in df.columns: 
                print('Adding Date')
                df['Date']=df.apply(stringify,axis=1)

            print('Creating feed')
            
            data=pd.read_csv('./capvolfiltered.csv')
            tickers=[t for t in list(df['ticker'].unique()) if t in list(data['ticker'].values)]
            feed=DataFeed(tickers=tickers,dfgiven=True,df=df)
            
            print('Processing feed')
            add_addl_features_feed(feed,tickers=feed.tickers)
            add_sym_feature_feed(feed,tickers=feed.tickers)

            if use_new_features:
                for ticker in feed.data:
                    df = feed.data[ticker]
                    df, pre_discrete_cols, discrete_cols = add_features(df, columns_to_use=cols_to_use)
                    feed.data[ticker] = df.loc[:, continuing_cols]
                    
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
                        
            for t in feed.ndata:
                for d in feed.ndata[t]:
                    if feed.ndata[t][d].isnull().values.any(): 
                        feed.ndata[t][d]=feed.ndata[t][d].fillna(1)
                        # print(t,d)
                    if feed.ndata[t][d].isin([-np.inf,np.inf]).values.any():
                        feed.ndata[t][d]=feed.ndata[t][d].replace([np.inf, -np.inf],1)
                
            if not colab: 
                with open(datafeed_path,'wb') as f: pickle.dump(feed,f)
            elif colab: 
                with open('/tmp/btdatafeed.pickle','wb') as f: pickle.dump(feed,f)

                
        print(f'[INFO] Use raw features: {use_raw_features}, Use new features: {use_new_features}') 
        print(f'Using: {len(using_cols)} COLS! Ex: {using_cols[:7]+using_cols[-7:]}')              

        def get_alt_data_live():
            aD={'gdata':feed.gdata}
            return aD

        agent=RLStratAgentDynFeatures(algorithm,monclass=Mon,soclass=StackedObservations,verbose=1,
                        metarl=True,myargs=(n_steps,use_alt_data,win), using_cols=using_cols)
        agent.use_memory=True #depends on whether RL algorithm uses memory for state computation
        agent.debug=False
        
        agent.data_cols = continuing_cols

        if modelname and os.path.exists('./saved_models/'+modelname): 
            agent.load_model(filepath='./saved_models/'+modelname)
            print(f'Loading model from {modelname}.')
        else:
            print(f'[INFO] Will save model as {modelname}!')

        @aspectlib.Aspect
        def my_decorator(*args, **kwargs):
            state,rew,done,exit_type = yield
            args[0].policy.reward((rew,done,{'exit_type':exit_type}))
            return state,rew,done,exit_type

        aspectlib.weave(Episode, my_decorator, methods='env_step')

        bt=Backtest(feed,tickers=feed.tickers,add_features=False,target=5,stop=5,txcost=0.001,
                    loc_exit=True,scan=True,topk=10,deploy=deploy,save_dfs=False,
                    save_func=None)

        def run_btworld():
            global bt,feed,agent
            while agent.training:
                bt.run_all(tickers=feed.tickers,model=agent,verbose=False)

        agent.start(training_steps=training_steps)

        btworldthread=Thread(target=run_btworld,name='btworld')
        btworldthread.start()

        def check_bt_training_status():
            threadL=[thread.name for thread in threading.enumerate()]
            if 'monitor' not in threadL and 'btworld' not in threadL:
                print(f'Training Over after {agent.model.num_timesteps} steps')
                return False
            else:
                print(f'Model Training for {agent.model.num_timesteps} steps')
                return True

        while check_bt_training_status():
            time.sleep(2)

        if modelname: 
            torch.save(agent.model.policy.state_dict(),'./saved_models/'+modelname)

        train_results_folder = os.path.join('results', f'{config_suffix}',
                                            'training')
        os.makedirs(train_results_folder, exist_ok=True)

        df=pd.read_csv('/tmp/aiagents.monitor.csv',comment='#')
        df.to_csv(os.path.join(train_results_folder, 'training_monitor.csv'), index=False)

        fig = px.line(df['r'].rolling(window=500).mean().values)
        plotly.offline.plot(fig, filename=os.path.join(train_results_folder, 'train_curve.html'))