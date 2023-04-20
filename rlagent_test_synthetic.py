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
from india_calendar import IBDay
from threading import Thread
import threading
from IPython import display
from backtest import Backtest
from rlagents import RLStratAgentDyn, COLS, RLStratAgentDynFeatures
import time,getopt,sys,os

from feeds import BackFeed,DataFeed
from featfuncs import feat_aug,add_addl_features_feed,add_ta_features_feed,add_sym_feature_feed
from featfuncs import add_global_indices_feed

from feed_env import Episode
import aspectlib
import yaml
import pickle
import plotly.express as px
import plotly
import plotly.graph_objects as go

import plotly.express as px
from add_features import add_features

from utils import read_yaml, extract_suffix, remove_extension
import argparse


# Plotting

def annotate_action(rew,act,df):
    if rew[1]>=0:color='Green'
    else: color='Red'
    if act[0]==1:text='Buy'
    elif act[0]==-1:text='Sell'
    ann=dict(font=dict(color=color,size=15),x=df.index[rew[0]],y=df.iloc[rew[0]]['Close'],
             showarrow=True,text=text)
    return ann

def annotate_exit(rew,act,anns,df):
    if rew[1]>=0:color='Green'
    else: color='Red'
    X=[a['x'] for a in anns if a is not None]
    if df.index[rew[2]] in X: 
        idx=X.index(df.index[rew[2]])
        anns[idx]['text']='Ex&'+anns[idx]['text']
    else:
        anns+=[dict(font=dict(color=color,size=15),x=df.index[rew[2]],y=df.iloc[rew[2]]['Close'],
                    showarrow=True,text='Exit')]

def plot_ticker_date(bt,ticker,date):
    global fig
    df=feed.ndata[ticker][date]
    df=df.loc[df['Date']==date]
    fig = go.Figure(data=
        [go.Candlestick(x = df.index,
                        open  = df["Open"],
                        high  = df["High"],
                        low   = df["Low"],
                        close = df["Close"])]
    )
    reward=np.round(bt.results[ticker][date]["tot"],2)
    fig.update_layout(
        title=f'{ticker} on {date} return {reward}',
        yaxis_title="Price"
    )
    anns=[]
    for r,a in zip(bt.results[ticker][date]['rew'],bt.results[ticker][date]['acts']):
        anns+=[annotate_action(r,a,df)]
    for r,a in zip(bt.results[ticker][date]['rew'],bt.results[ticker][date]['acts']):
        anns+=[annotate_exit(r,a,anns,df)]
    for a in anns: 
        if a is not None: fig.add_annotation(a)
    # fig.show()
    return fig

def combine_plotly_figs_to_html(plotly_figs, html_fname, include_plotlyjs='cdn', 
                                separator=None, auto_open=False):
    with open(html_fname, 'w') as f:
        f.write(plotly_figs[0].to_html(include_plotlyjs=include_plotlyjs))
        for fig in plotly_figs[1:]:
            if separator:
                f.write(separator)
            f.write(fig.to_html(full_html=False, include_plotlyjs=False))

    if auto_open:
        import pathlib, webbrowser
        uri = pathlib.Path(html_fname).absolute().as_uri()
        webbrowser.open(uri)
        

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", "-c", type=str, default="configs/config_base.yaml", help="Path to the config file")

args = parser.parse_args()
config_path = args.config_path

config = read_yaml(config_path)

# print(f'[INFO] The columns in RLAgents are: {COLS}!')

DATAPATH='algodata'
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
datafiles = config.get('datafile_test', [])
win = config['win']
top_k = config['top_k']
use_raw_features = config.get('old_features', True)
use_new_features = config.get('new_features', False)
create_feed = config.get('create_feed', False)
dynamic_test = config.get('dynamic_test_features', False)
use_prediscrete = config.get('use_prediscrete', False)
# # datafeed_path = os.path.join('..', 'algodata', 'realdata', f'datafeed_{config_suffix}.pkl')
# loadfeed_path = config.get('loadfile', '')
# if not loadfeed or not loadfeed_path:
#     loadfeed_path = os.path.join('..', 'algodata', f'btfeed_{config_suffix}_test.pkl') # Path to save
#     print(f'[INFO] Will save the generated feed at {loadfeed_path}')


modelname = f'{modelname.rstrip(".pth")}_{use_raw_features}_{use_new_features}.pth'

colab=False
DATAPATH='../algodata'

n_steps=2048 # reduce for debugging only else 2048

with open('additional_utils/cols.pkl', 'rb') as f:
        d = pickle.load(f)
imp_cols = d['imp_cols']
cols_to_use = d['cols_to_use']
prediscrete_imp_cols = d['prediscrete_imp_cols']

if use_prediscrete:
    imp_cols = prediscrete_imp_cols

def stringify(x):
    return pd.to_datetime(x['Datetime']).strftime('%d-%b-%Y')

if type(datafiles) ==  str:
    datafiles = [datafiles]
    
if use_new_features and use_raw_features:
    using_cols = COLS + imp_cols
elif use_raw_features:
    using_cols = COLS
elif use_new_features:
    using_cols = imp_cols

print(f'Will use: {using_cols}')
if dynamic_test and use_new_features: # Done to avoid error in get_state (as the new features haven't been added)
    continuing_cols = cols_to_use + ['Date', 'datetime']
else:
    continuing_cols = using_cols + ['Date', 'datetime']
    
needed_cols = ['row_num', 'Close_n', 'Open_n', 'Open', 'Close', 'High', 'Low']
for col in needed_cols:
    if col not in continuing_cols: continuing_cols.append(col)
    
rew_dist = {'file': [], 'ticker': [], 'reward': []}


for datafile in datafiles:
    datafile_suffix = os.path.basename(datafile).rstrip('.csv')
    datafeed_path = os.path.join('..', 'algodata', 'realdata', 
                f'datafeed_{datafile_suffix}_{use_raw_features}_{use_new_features}.pkl')
    if use_prediscrete:
        datafeed_path = os.path.join('..', 'algodata', 'realdata', 
                f'datafeed_{datafile_suffix}_{use_raw_features}_{use_new_features}_{use_prediscrete}.pkl')   

    if (os.path.exists(datafeed_path)) and not create_feed:
        # if os.path.exists(datafeed_path_new):
        #     datafeed_path = datafeed_path_new
        feed = pickle.load(open(datafeed_path, 'rb'))
        print(f'[INFO] Loading pickle file for datafeed: {datafeed_path}!')
        
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
        feed=DataFeed(tickers=list(df.ticker.unique()[0:10]),dfgiven=True,df=df)
        
        print('Processing feed')
        add_addl_features_feed(feed,tickers=feed.tickers)
        add_sym_feature_feed(feed,tickers=feed.tickers)

        if use_new_features and not dynamic_test:
            print('Adding features to feed!')
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

    def get_alt_data_live():
        aD={'gdata':feed.gdata}
        return aD
    
    use_alt_data=False
    agent=RLStratAgentDynFeatures(algorithm,monclass=Mon,soclass=StackedObservations,verbose=1,win=win,
                    metarl=True,myargs=(n_steps,use_alt_data), using_cols=using_cols, dynamic_test=dynamic_test,
                    cols_to_use=cols_to_use)
    agent.use_memory=True #depends on whether RL algorithm uses memory for state computation
    agent.debug=False

    agent.training=False
    agent.data_cols = continuing_cols
    
    if modelname and os.path.exists('./saved_models/'+modelname): 
        agent.load_model(filepath='./saved_models/'+modelname)
        print(f'Loading model from {modelname}.')
    else:
        print(f'[INFO] Model {modelname} not found! Continuing...')
        continue

    @aspectlib.Aspect
    def my_decorator(*args, **kwargs):
        state,rew,done,exit_type = yield
        args[0].policy.reward((rew,done,{'exit_type':exit_type}))
        return state,rew,done,exit_type

    aspectlib.weave(Episode, my_decorator, methods='env_step')

    bt=Backtest(feed,tickers=feed.tickers,add_features=False,target=5,stop=5,txcost=0.001,
                loc_exit=True,scan=False,topk=5,deploy=deploy,save_dfs=False,
                save_func=None)

    bt.run_all(tickers=feed.tickers,model=agent,verbose=False)

    test_results_folder = os.path.join('results', f'{config_suffix}', 'test')
    if datafile:
        test_results_folder = os.path.join(test_results_folder, datafile.rstrip('.csv'))
    os.makedirs(test_results_folder, exist_ok=True)
    

    with open(os.path.join(test_results_folder, 'reward_sum.txt'), 'w') as f:
        reward_sum = sum([bt.results[t][d]['tot'] for t in bt.results for d in bt.results[t]])
        f.write(str(reward_sum))
    
    with open(os.path.join(test_results_folder, 'reward_distribution.txt'), 'w') as f:
        reward_dict = {t: [bt.results[t][d]['tot'] for d in bt.results[t]] for t in bt.results}
        reward_avg = {t: np.mean(reward_dict[t]) for t in reward_dict}
        reward_std = {t: np.std(reward_dict[t]) for t in reward_dict}
        # dist = f'{reward_avg:.1f}\u00B1{reward_std:.1f}'
        dist = '\n'.join([f'{t}: {reward_avg[t]:.2f}\u00B1{reward_std[t]:.2f}' for t in reward_avg])
        f.write(dist)
    
    overall_rewards = [bt.results[t][d]['tot'] for t in bt.results for d in bt.results[t]]
    
    for t in reward_avg:
        rew_dist['ticker'].append(t)
        rew_dist['file'].append(datafile)
        rew_dist['reward'].append(f'{reward_avg[t]:.2f}\u00B1{reward_std[t]:.2f}')
        
    overall_reward = f'{np.mean(overall_rewards):.2f}\u00B1{np.std(overall_rewards):.2f}'
    rew_dist['ticker'].append('overall')
    rew_dist['file'].append(datafile)
    rew_dist['reward'].append(overall_reward)  

    bt_dump_path = os.path.join(test_results_folder, 'bt.pkl')
    with open(bt_dump_path, 'wb') as f:
        pickle.dump(bt.results, f)

    figs=[]
    for t in bt.results:
        for d in bt.results[t]:
            figs+=[plot_ticker_date(bt,t,d)]


    plotting_save_path = os.path.join(test_results_folder, 'plots.html')
    combine_plotly_figs_to_html(figs, plotting_save_path)
    
    
    df = pd.DataFrame(rew_dist)
    df_path = os.path.join(test_results_folder, 'distribution.csv')
    df.to_csv(df_path, index=False)