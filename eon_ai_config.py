# HCA Strategy configuration
# Setup server constants here
# Add paths to PYTHON_PATH
import os, inspect, sys
from pprint import pprint

#MultiDim Dict
# https://en.wikipedia.org/wiki/Autovivification
from collections import defaultdict
Tree = lambda: defaultdict(Tree)
py   = Tree()

# HCA Install location: Change, based on machine/os install
# aws_ec2
from os import getenv
def get_script_dir(follow_symlinks=True):
         if getattr(sys, 'frozen', False): # py2exe, PyInstaller, cx_Freeze
                  path = os.path.abspath(sys.executable)
         else:
                  path = inspect.getabsfile(get_script_dir)
         if follow_symlinks:
                  path = os.path.realpath(path)
         return os.path.dirname(path)


path = get_script_dir()

HCA_RELEASE_STRAT_DIR = getenv("HCA_RELEASE_STRAT_DIR", path)
sys.path.append(os.path.abspath(HCA_RELEASE_STRAT_DIR))

py['install']=os.path.expanduser(HCA_RELEASE_STRAT_DIR) # Mode settings
#py['install']=os.path.expanduser() # Mode settings
py['sitename'] = "hca"
py['secret'] = '\x18\xcax\xcb\x0c\xeb\x91|\xd8d\xba\x0f\x0f\x14\xf2C}SC\x97\xc7|U\x16'

# Set DEBUG flag
py['DEBUG'] = False #True #False
print ('DEBUG flag =', py['DEBUG'])

# Eval Settings
py['gpu']= False #True #False #False #True

# Paths
# Normalized prefix for install, based on other computing devices
py['norm_install'] = os.path.realpath(os.path.abspath(os.path.join(os.path.split( inspect.getfile(inspect.currentframe() ))[0],py['install'])))
print( "norm_install={}".format(py['norm_install']))
if py['norm_install']  not in sys.path:
         sys.path.insert(0,py['norm_install'] )

print ('norm_install path =', py['norm_install'])

py['strat_names']={}
#Micro Sectors

###Start:EonumTech
EonumTech_list=None # Read in from .h5 db in eon_ai.py
py['strats']['segm_sect']['EonumTech']['algo']              = py['install'] + '/eon_ai.py'
py['strats']['segm_sect']['EonumTech']['LEVERAGE_FAC']      = 1.0
py['strats']['segm_sect']['EonumTech']['LONG_PCT']	    = 0.7 #1.0 #0.5 #0.5 #0.7 #1.0 #0.6 #1.0 #0.7 #1.0
py['strats']['segm_sect']['EonumTech']['PRATE_PCT']	    = 0.1
py['strats']['segm_sect']['EonumTech']['LIMIT_ORDER_PCT']   = 0.02
py['strats']['segm_sect']['EonumTech']['LMT_PCT']	    = 0.03
py['strats']['segm_sect']['EonumTech']['MINUTES_TO_REBAL']  = 1 #30 #1 #15 ### 1
py['strats']['segm_sect']['EonumTech']['IS_LIVE']	    = True #False #True #True #False #True #False #True # #False #True
py['strats']['segm_sect']['EonumTech']['UNIVERSE']	    = EonumTech_list
###End:EonumTech

###Start:SaasTech
SaasTech_list=(
    # SaaS List 05/05/2020
    #changes made 4/1/2020
    'ADBE', #(ADOBE SYSTEMS INC)154B
    'ADSK', # (AUTODESK INC) – New in 2019 index 34B
    'APPF', # (APPFOLIO INC – A)3.8B
    'AVLR', # (AVALARA INC) – New in 2019 index 5.99B
    'BL', # (BLACKLINE INC)2.94B
    'BOX', # (BOX INC – CLASS A)2.1B
    'CHGG', #4.4B  28%TTM added 4/1/2020
    'CRM', # (SALESFORCE.COM INC)130B
    'CSOD', # (CORNERSTONE ONDEMAND INC)1.93B
    ### 'DATA', #Aquired by SalesForce 08-01-2019 'DATA', # (TABLEAU SOFTWARE INC-CL A)
    #'DBX', # (DROPBOX INC-CLASS A)
    'DOCU', # (DOCUSIGN INC)16.66B
    #'DOMO', # (DOMO INC – CLASS B) – New in 2019 index 261M
    'EVBG', # (EVERBRIDGE INC)3.66B
    'FIVN', # (FIVE9 INC)4.837B
    'HUBS', # (HUBSPOT INC)5.82B
    #'LOGM', # (LOGMEIN INC)4B Change in revenue not met.
    #'MDSO', # (MEDIDATA SOLUTIONS INC) not found
    'MIME', # (MIMECAST LTD)2.2B
    ### 'MOBL', # (MOBILE IRON INC)436M Bubble. Mobile security TTM 6.2% #2020-12-06: Aquired by Ivanti-private
    'NEWR', # (NEW RELIC INC)2.7B Cloud Analyzer
    'NOW', # (SERVICENOW INC)54B Management cloud software 38%ttm
    'OKTA', # (OKTA INC) – New in 2019 index 15B  Identity Security
     'PD',  #PagerDuty 1.3B  Analyzer of data for action
    #'PLAN', # (ANAPLAN INC) – New in 2019 index 4B  44.6%
    'PS',   #Pluralsight 1.5B Could be good for COVID-19
    'QTWO', # (Q2 HOLDINGS INC) 2.8B  Payment soulutions for banks 40% rev growth
    'RNG', # (RINGCENTRAL INC-CLASS A)18.4B 34%
    'RP', # (REALPAGE INC)5B  13.6% ttm
    ###Aquired by TWILIO 2019-02-01 'SEND', # (SENDGRID INC)
    'SHOP', # (SHOPIFY INC – CLASS A)49B  49%ttm
    'SMAR', # (SMARTSHEET INC-CLASS A)5B 52%TTM
    'SPLK', # (SPLUNK INC)19B 39%ttm
    'SPSC', # (SPS COMMERCE INC 1.6B 12.4TTM
    'TEAM', # (ATLASSIAN CORP PLC-CLASS A)33B  36%TTM
    'TLND', # (TALEND SA – ADR)630m 29.3%ttm
    'TWLO', # (TWILIO INC – A)12.5B 74.55TTM
    'TWOU', # (2U INC)1.28B 40%TTM
    ###ULTI', # (ULTIMATE SOFTWARE GROUP INC) #Delisted 2019-05-03
    'VEEV', # (VEEVA SYSTEMS INC-CLASS A)23.2B  28%TTM
    'WDAY', # (WORKDAY INC-CLASS A)28B 28.5%TTM
    'WORK', #(Slack) 15B 57%TTM
    #'WIX', # (WIX.COM LTD)web development subscription growth suspect. 5B
    'ZEN', # (ZENDESK INC)7B 36.4TTM
    'ZM', #ZOOM 40B
    #'ZUO', # (ZUORA INC – CLASS A) subscription based not growing enough
     )
py['strats']['segm_sect']['SaasTech']['algo']               = py['install'] + '/eon_ai.py'
py['strats']['segm_sect']['SaasTech']['LEVERAGE_FAC']       = 1.0
py['strats']['segm_sect']['SaasTech']['LONG_PCT']	    = 1.0 #0.7 #1.0
py['strats']['segm_sect']['SaasTech']['PRATE_PCT']	    = 0.1
py['strats']['segm_sect']['SaasTech']['LIMIT_ORDER_PCT']    = 0.02
py['strats']['segm_sect']['SaasTech']['LMT_PCT']	    = 0.03
py['strats']['segm_sect']['SaasTech']['MINUTES_TO_REBAL']   = 1 #30 #1 #15 ### 1
py['strats']['segm_sect']['SaasTech']['IS_LIVE']	    = True #True #True #False #True #False #True # #False #True
py['strats']['segm_sect']['SaasTech']['UNIVERSE']	    = SaasTech_list
###End:SaasTech


###DEBUG pprint(py)

#Usage
#import hca_config as scfg #Strategy Config
#cfg_list =  [(x,y,mcfg.py['models'][x][y]) for x in scfg.py['models'] for y in scfg.py['models'][x] ]

