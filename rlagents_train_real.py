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
from rlagents import RLStratAgentDyn
from backtest import Backtest
from feeds import BackFeed,DataFeed
from validation import Validate
import pickle
import plotly.express as px
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

if not loadfeed or not loadfeed_path:
    loadfeed_path = os.path.join('..', 'algodata', f'btfeed_{config_suffix}.pkl') # Path to save

# print(f'[INFO] The columns in RLAgents are: {COLS}!')

colab=False
script=True

n_steps=2048 # reduce for debugging only else 2048

OHLCV_COLS=['Open_n','High_n','Low_n','Close_n','Volume_n']
TA_COLS=['SMA_10', 'SMA_20','VOL_SMA_20','RSI_14','BBL_5_2.0','BBM_5_2.0','BBU_5_2.0',
       'BBB_5_2.0', 'BBP_5_2.0','MACD_12_26_9','MACDh_12_26_9','MACDs_12_26_9','VWAP_D',
        'MOM_30','CMO_14']
TA_COLS_MIN=['SMA_10', 'SMA_20','CMO_14']
# COLS=['row_num']+OHLCV_COLS+TA_COLS
COLS=['row_num']+OHLCV_COLS+TA_COLS

cols_to_use = []
if use_raw_features:
    cols_to_use += COLS

def stringify(x):
    return pd.to_datetime(x['Datetime']).strftime('%d-%b-%Y')

def add_features_to_feed(feed):
    global cols_to_use
    for ticker in feed.data:
        df = feed.data[ticker]
        df, pre_discrete_cols, discrete_cols = add_features(df)
        feed.data[ticker] = df
        if use_new_features:
            cols_to_use += [col for col in discrete_cols if col not in cols_to_use]
    return feed

if type(datafiles) == str: 
    datafiles = [datafiles]
    
printed_cols = False
    
for epoch in range(epochs):
    for idx, datafile in enumerate(datafiles):
        print(f'[INFO] EPOCH: {epoch} FILE: {datafile}!')
        
        datafile_suffix = os.path.basename(datafile).rstrip('.csv')
        datafeed_path = os.path.join('..', 'algodata', 'realdata', 
                f'datafeed_{datafile_suffix}_{use_raw_features}_{use_new_features}.pkl')
        
        if not loadfeed and not datafeed:
            data=pd.read_csv('./capvol100.csv')
            tickers=list(data.iloc[0:50]['ticker'].values)
            print('Creating feed')
            feed=BackFeed(tickers=tickers,nd=nd,nw=nw,interval='5m',synthetic=synthetic,simple=simple)
            print('Processing feed')
            add_addl_features_feed(feed,tickers=feed.tickers)
            add_sym_feature_feed(feed,tickers=feed.tickers)
            if use_new_features:
                feed = add_features_to_feed(feed)
            if not synthetic: add_global_indices_feed(feed)
            if not colab: 
                with open(loadfeed_path,'wb') as f: pickle.dump(feed,f)
                
        elif loadfeed and not datafeed:
            if not colab: 
                with open('../../temp_data/btfeed.pickle','rb') as f: feed=pickle.load(f)
            elif colab: 
                with open(loadfeed_path,'rb') as f: feed=pickle.load(f)
            print(f'[INFO] Loaded feed from the pickle file: {loadfeed_path}')

        if not loadfeed and datafeed:
            if os.path.exists(datafeed_path):
                feed = pickle.load(open(datafeed_path, 'rb'))
                print('[INFO] Loading pickle file for datafeed!')
            else:
                DATAFILE=os.path.join(DATAPATH, datafile)
                print(f'Reading datafile: {DATAFILE}')
                df=pd.read_csv(DATAFILE)
                if 'Date' not in df.columns: 
                    print('Adding Date')
                    df['Date']=df.apply(stringify,axis=1)
                print('Creating feed')
                feed=DataFeed(tickers=list(df.ticker.unique()[0:10]),dfgiven=True,df=df)
                print('Processing feed')
                add_addl_features_feed(feed,tickers=feed.tickers)
                add_sym_feature_feed(feed,tickers=feed.tickers)
                if use_new_features:
                    feed = add_features_to_feed(feed)
                # add_global_indices_feed(feed)
                if not colab: 
                    with open(datafeed_path,'wb') as f: pickle.dump(feed,f)
                elif colab: 
                    with open('/tmp/btdatafeed.pickle','wb') as f: pickle.dump(feed,f)
                
        elif loadfeed and datafeed:
            if not colab: 
                with open('../../temp_data/btdatafeed.pickle','rb') as f: feed=pickle.load(f)
            elif colab:
                with open('/tmp/btdatafeed.pickle','rb') as f: feed=pickle.load(f)
                
        if not printed_cols:
            print(f'[INFO] Use raw features: {use_raw_features}, Use new features: {use_new_features}')
            print(f'Finally using: total {len(cols_to_use)}) columns (last 100: {cols_to_use[-100:]})!')                

        def get_alt_data_live():
            aD={'gdata':feed.gdata}
            return aD

        agent=RLStratAgentDyn(algorithm,monclass=Mon,soclass=StackedObservations,verbose=1,win=win,
                        metarl=True,myargs=(n_steps,use_alt_data), use_cols=cols_to_use)
        agent.use_memory=True #depends on whether RL algorithm uses memory for state computation
        agent.debug=False
        agent.data_cols = ['datetime', 'Date'] + cols_to_use
        if use_alt_data: agent.set_alt_data(alt_data_func=get_alt_data_live)


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
                    loc_exit=True,scan=True,topk=5,deploy=deploy,save_dfs=False,
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
            modelname_current = modelname.rstrip('.pth') + f'_{epoch+1}_file_{idx+1}.pth'
            torch.save(agent.model.policy.state_dict(),'./saved_models/'+modelname_current)

        train_results_folder = os.path.join('results', f'{config_suffix}',
                                            'training', f'epoch_{epoch+1}_{idx+1}')
        os.makedirs(train_results_folder, exist_ok=True)

        df=pd.read_csv('/tmp/aiagents.monitor.csv',comment='#')
        df.to_csv(os.path.join(train_results_folder, 'training_monitor.csv'), index=False)

        fig = px.line(df['r'].rolling(window=10).mean().values)
        plotly.offline.plot(fig, filename=os.path.join(train_results_folder, 'train_curve.html'))