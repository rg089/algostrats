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
from backtest import Backtest
from rlagents import RLStratAgentDyn, COLS
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

print(f'[INFO] The columns in RLAgents are: {COLS}!')

DATAPATH='algodata'
if not os.path.exists(DATAPATH):
    os.makedirs(DATAPATH)

algorithm=PPO # The algorithm to use
synthetic= config['synthetic']
simple= config['simple']
nd,nw=5,config['weeks']
training_steps=config['training_steps']

config_suffix = extract_suffix(config_path)
if 'modelname' in config:
    base_modelname = config['modelname']
else:
    base_modelname = f'{config_suffix}.pth'
    if config_suffix.endswith('_trainfile_test'):
        base_modelname = f'{config_suffix.rstrip("_trainfile_test")}.pth'
        
model_suffixes = config.get('model_suffixes', [''])
if '' not in model_suffixes:
    model_suffixes.append('')
    
deploy = config['deploy']
loadfeed = config['loadfeed']
datafeed = config['datafeed']
use_alt_data = config['use_alt']
datafiles = config.get('datafile_test', [])
win = config['win']
top_k = config['top_k']

datafeed_path = os.path.join('..', 'algodata', 'realdata', f'datafeed_{config_suffix}.pkl')
loadfeed_path = config.get('loadfile', '')
if not loadfeed or not loadfeed_path:
    loadfeed_path = os.path.join('..', 'algodata', f'btfeed_{config_suffix}_test.pkl') # Path to save
    print(f'[INFO] Will save the generated feed at {loadfeed_path}')

colab=False
script=True
DATAPATH='../algodata'

n_steps=2048 # reduce for debugging only else 2048

def stringify(x):
    return pd.to_datetime(x['Datetime']).strftime('%d-%b-%Y')

if type(datafiles) ==  str:
    datafiles = [datafiles]
    
for datafile in datafiles:        
    if not loadfeed and not datafeed:
        data=pd.read_csv('./capvol100.csv')
        tickers=list(data.iloc[0:top_k]['ticker'].values)
        print('Creating feed')
        feed=BackFeed(tickers=tickers,nd=nd,nw=nw,interval='5m',synthetic=synthetic)
        print('Processing feed')
        add_addl_features_feed(feed,tickers=feed.tickers)
        add_sym_feature_feed(feed,tickers=feed.tickers)
        add_global_indices_feed(feed)
        if colab: 
            with open('/tmp/btfeed.pickle','wb') as f: pickle.dump(feed,f)
        else: 
            with open(loadfeed_path,'wb') as f: pickle.dump(feed,f)
            
    elif loadfeed and not datafeed:
        if colab: 
            with open('/tmp/btfeed.pickle','rb') as f: feed=pickle.load(f)
        else: 
            with open(loadfeed_path, 'rb') as f: feed=pickle.load(f)
        print(f'[INFO] Loaded feed from the pickle file: {loadfeed_path}')

    if not loadfeed and datafeed:
        #DATAFILE=DATAPATH+'augdata_'+date+'_5m.csv'
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

    def get_alt_data_live():
        aD={'gdata':feed.gdata}
        return aD
    
    for model_suffix in model_suffixes:
        
        modelname_wtho_ext, ext = remove_extension(base_modelname)
        modelname = f'{modelname_wtho_ext}{model_suffix}{ext}'
        print(f'\n[INFO] Will use model: {modelname}!')
        
        use_alt_data=False
        agent=RLStratAgentDyn(algorithm,monclass=Mon,soclass=StackedObservations,verbose=1,win=win,
                        metarl=True,myargs=(n_steps,use_alt_data))
        agent.use_memory=True #depends on whether RL algorithm uses memory for state computation
        agent.debug=False
        if use_alt_data: agent.set_alt_data(alt_data_func=get_alt_data_live)

        agent.training=False

        if modelname and os.path.exists('./saved_models/'+modelname): 
            agent.load_model(filepath='./saved_models/'+modelname)
            print(f'Loading model from {modelname}.')
        else:
            print(f'[INFO] Model {modelname} not found! Continuing...')

        @aspectlib.Aspect
        def my_decorator(*args, **kwargs):
            state,rew,done,exit_type = yield
            args[0].policy.reward((rew,done,{'exit_type':exit_type}))
            return state,rew,done,exit_type

        aspectlib.weave(Episode, my_decorator, methods='env_step')

        bt=Backtest(feed,tickers=feed.tickers,add_features=False,target=5,stop=5,txcost=0.001,
                    loc_exit=True,scan=False,topk=5,deploy=deploy,save_dfs=False,
                    save_func=None)

        agent.data_cols=agent.data_cols+['Date']

        bt.run_all(tickers=feed.tickers,model=agent,verbose=False)

        test_results_folder = os.path.join('results', f'{config_suffix}', 'test', modelname)
        if datafile:
            test_results_folder = os.path.join(test_results_folder, datafile.rstrip('.csv'))
        os.makedirs(test_results_folder, exist_ok=True)

        with open(os.path.join(test_results_folder, 'reward_sum.txt'), 'w') as f:
            reward_sum = sum([bt.results[t][d]['tot'] for t in bt.results for d in bt.results[t]])
            f.write(str(reward_sum))
            
        with open(os.path.join(test_results_folder, 'reward_distribution.txt'), 'w') as f:
            rewards = [bt.results[t][d]['tot'] for t in bt.results for d in bt.results[t]]
            reward_avg = np.mean(rewards)
            reward_std = np.std(rewards)
            f.write(f'{reward_avg:.1f}\u00B1{reward_std:.1f}')

        bt_dump_path = os.path.join(test_results_folder, 'bt.pkl')
        with open(bt_dump_path, 'wb') as f:
            pickle.dump(bt.results, f)

        figs=[]
        for t in bt.results:
            for d in bt.results[t]:
                figs+=[plot_ticker_date(bt,t,d)]

        plotting_save_path = os.path.join(test_results_folder, 'plots.html')
        combine_plotly_figs_to_html(figs, plotting_save_path)