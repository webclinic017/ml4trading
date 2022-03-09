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
###Start:SaasTech + RHTech
#ajjc 07/28/2020 Modify Universe: Remove some Corona factors #-#
#ajjc 08/07/2020 Modify Universe: Remove all Corona factors #-#
SaasTech_list=(
    #CoronaTech_list=(    #Added 04-26-2020
    # Corona List 05/05/2020
    #-#''JNJ',
    #-#'GILD',
    #-#''SNY',
    #-#'MRNA',
    #-#''INO',
    #-#'NVAX',
    #-#'ABBV',
    #-#'VIR',
    #dup in Saas 'ZM',

    #-#'PTON',
    #-#'TDOC',
    #'MMM',
    #-#'CLX',
    # duplicated in SaasTech 'ADBE',
    # 'CSCO',
    #-#'CTXS',
    #tbtf 'MSFT',
    #tbtf 'ORCL','SAP',
    #-#''LRN',
     # duplicated in SaasTech 'CHGG',
     # duplicated in SaasTech 'NOW',
    #tbtf 'AMZN',
    #tbtf 'COST',
    #tbtf 'IBM',
    #jjc added 04-27-2020
    #-#'PG',
    #-#'DGX',
    #-#'LH',
    #-#'BYND',
    #)
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
    'PLAN', # (ANAPLAN INC) – New in 2019 index 4B  44.6%
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
py['strats']['segm_sect']['SaasTech']['algo']               = py['install'] + '/hca_segmsec.py'
py['strats']['segm_sect']['SaasTech']['LEVERAGE_FAC']       = 1.0
py['strats']['segm_sect']['SaasTech']['LONG_PCT']	    = 0.7 #1.0
py['strats']['segm_sect']['SaasTech']['PRATE_PCT']	    = 0.1
py['strats']['segm_sect']['SaasTech']['LIMIT_ORDER_PCT']    = 0.02
py['strats']['segm_sect']['SaasTech']['LMT_PCT']	    = 0.03
py['strats']['segm_sect']['SaasTech']['MINUTES_TO_REBAL']   = 1 #30 #1 #15 ### 1
py['strats']['segm_sect']['SaasTech']['IS_LIVE']	    = False #True #True #False #True #False #True # #False #True
py['strats']['segm_sect']['SaasTech']['UNIVERSE']	    = SaasTech_list




###End:SaasTech


#Genomics and Biotechnolgy  38 stocks

###Start:BioTech
BioTech_list = (
'ILMN',
'PTCT',
'ONCE',
'SRPT',
'QGEN',
'A',
'ALNY',
'RARE',
'BMRN',
'BPMC',
'RGNX',
'BLUE',
'VRTX',
'NVTA',
'GHDX',
'BOLD',
'ARWR',
'QURE',
'EDIT',
'SGMO',
'CELG',
'PACB',
'GILD',
'CRSP',
'NTRA',
'FLDM',
'RCKT',
'VCYT',
'WVE',
'DRNA',
'SRNE',
'NTLA',
'FIXX',
'VYGR',
'PRQR',
'ADVM',
'SLDB',
) #'TOCA'
py['strats']['segm_sect']['BioTech']['algo'] 			= py['install'] + '/hca_segmsec.py'
py['strats']['segm_sect']['BioTech']['LEVERAGE_FAC']       = 1.0
py['strats']['segm_sect']['BioTech']['LONG_PCT']			= 1.0
py['strats']['segm_sect']['BioTech']['PRATE_PCT']			= 0.1
py['strats']['segm_sect']['BioTech']['LIMIT_ORDER_PCT']		= 0.02
py['strats']['segm_sect']['BioTech']['LMT_PCT']			= 0.03
py['strats']['segm_sect']['BioTech']['MINUTES_TO_REBAL']	= 1
py['strats']['segm_sect']['BioTech']['IS_LIVE']			= True
py['strats']['segm_sect']['BioTech']['UNIVERSE']			= BioTech_list

###End:BioTech
###Start:IoTTech

# Internet of Things 35 Stocks
IoTTech_list = (
'GRMN',
'DSCM',
'ST',
'SWKS',
'STM',
'CY',
'ADT',
'SLAB',
'QCOM',
'CSCO',
'ALRM',
'HON',
'ADI',
'ROK',
'INTC',
'JCI',
'EMR',
'IBM',
'ABB',
'IDCC',
'BDC',
'ITRI',
'BMI',
'AMBA',
'RMBS',
'FIT',
'NTGR',
'ORBC',
'SWIR',
'PI',
'SENS',
'ARLO',
'VUZI',
'REZI',
'GTX'
)

py['strats']['segm_sect']['IoTTech']['algo'] 			= py['install'] + '/hca_segmsec.py'
py['strats']['segm_sect']['IoTTech']['LEVERAGE_FAC']       = 1.0
py['strats']['segm_sect']['IoTTech']['LONG_PCT']			= 1.0
py['strats']['segm_sect']['IoTTech']['PRATE_PCT']			= 0.1
py['strats']['segm_sect']['IoTTech']['LIMIT_ORDER_PCT']		= 0.02
py['strats']['segm_sect']['IoTTech']['LMT_PCT']			= 0.03
py['strats']['segm_sect']['IoTTech']['MINUTES_TO_REBAL']	= 1
py['strats']['segm_sect']['IoTTech']['IS_LIVE']			= True
py['strats']['segm_sect']['IoTTech']['UNIVERSE']			= IoTTech_list


###End:IoTTech

###Start:FinTech
# Fintech 20 stocks

FinTech_list = (
'PYPL',
'INTU',
'SSNC',
'FISV',
'SQ',
'IS',
'FDC',
'GWRE',
'BKI',
'TREE',
'HQY',
'ENV',
'PAGS',
'VIRT',
'EPAY',
'BCOR',
'LC',
'YRD',
'MITK',
'ONDK',
)
py['strats']['segm_sect']['FinTech']['algo'] 			= py['install'] + '/hca_segmsec.py'
py['strats']['segm_sect']['FinTech']['LEVERAGE_FAC']       = 1.0
py['strats']['segm_sect']['FinTech']['LONG_PCT']			= 1.0
py['strats']['segm_sect']['FinTech']['PRATE_PCT']			= 0.1
py['strats']['segm_sect']['FinTech']['LIMIT_ORDER_PCT']		= 0.02
py['strats']['segm_sect']['FinTech']['LMT_PCT']			= 0.03
py['strats']['segm_sect']['FinTech']['MINUTES_TO_REBAL']	= 1
py['strats']['segm_sect']['FinTech']['IS_LIVE']			= True
py['strats']['segm_sect']['FinTech']['UNIVERSE']			= FinTech_list

###End:CannabisTech

###Start:CannabisTech

# Marijuana 20 Stocks
CannabisTech_list = (
'ACB',
'GWPH',
'CGC',
'CRON',
'TLRY',
'HEXO',
'CRBP',
'APHA',
'INSY',
'TPB',
'CARA',
'SWM',
#'BTI',
'SMG',
#'MO',
#'PM',
#'VGR',
#'ARNA',
'UVV',
'XXII',
)
py['strats']['segm_sect']['CannabisTech']['algo'] 			= py['install'] + '/hca_segmsec.py'
py['strats']['segm_sect']['CannabisTech']['LEVERAGE_FAC']       = 1.0
py['strats']['segm_sect']['CannabisTech']['LONG_PCT']			= 1.0
py['strats']['segm_sect']['CannabisTech']['PRATE_PCT']			= 0.1
py['strats']['segm_sect']['CannabisTech']['LIMIT_ORDER_PCT']		= 0.02
py['strats']['segm_sect']['CannabisTech']['LMT_PCT']			= 0.03
py['strats']['segm_sect']['CannabisTech']['MINUTES_TO_REBAL']	= 1
py['strats']['segm_sect']['CannabisTech']['IS_LIVE']			= True
py['strats']['segm_sect']['CannabisTech']['UNIVERSE']			= CannabisTech_list

##End:WaterTech

###Start:WaterTech



#Water portfolio 36 stocks
WaterTech_list=(
'XYL',
'AOS',
'ECL',
'ROP',
'IEX',
'IDXX',
'AWK',
'DHR',
'FLS',
'A',
'ACM',
'TTEK',
'WTR',
'PNR',
'VMI',
'WTS',
'RXN',
'AWR',
'FELE',
'CWT',
'WMS',
'MWA',
'SBS',
'AQUA',
'SJW',
'MLI',
'BMI',
'ITRI',
'LNN',
'MSEX',
'ERII',
'AEGN',
'PRMW',
'GRC',
'YORW',
'WAAS',
)
py['strats']['segm_sect']['WaterTech']['algo'] 			= py['install'] + '/hca_segmsec.py'
py['strats']['segm_sect']['WaterTech']['LEVERAGE_FAC']       = 1.0
py['strats']['segm_sect']['WaterTech']['LONG_PCT']			= 1.0
py['strats']['segm_sect']['WaterTech']['PRATE_PCT']			= 0.1
py['strats']['segm_sect']['WaterTech']['LIMIT_ORDER_PCT']		= 0.02
py['strats']['segm_sect']['WaterTech']['LMT_PCT']			= 0.03
py['strats']['segm_sect']['WaterTech']['MINUTES_TO_REBAL']	= 1
py['strats']['segm_sect']['WaterTech']['IS_LIVE']			= True
py['strats']['segm_sect']['WaterTech']['UNIVERSE']			= WaterTech_list


##End:CyberTech

###Start:CyberTech
    #Cyber Security based on HACK ETF
CyberTech_list=(
         'CSCO','SPLK','PANW','CYBR','CACI','FEYE','PFPT','AKAM','FTNT','TUFN',
         'SYMC','CHKP','SAIC','CVLT','JNPR','NET','SWI',
         'QLYS','TENB','CARB','SAIL','PSN','OKTA','ZS','CRWD','BAH','FSCT',
         'VRSN','LDOS','EVBG','MIME','VRNS','FFIV','MANT','RPD','VRNT','BA',
         'RDWR','NTCT','ULE','SCWX','MOBL','ZIXI','OSPN','ATEN','NCC',
    )

py['strats']['segm_sect']['CyberTech']['algo'] 			= py['install'] + '/hca_segmsec.py'
py['strats']['segm_sect']['CyberTech']['LEVERAGE_FAC']       = 1.0
py['strats']['segm_sect']['CyberTech']['LONG_PCT']			= 1.0
py['strats']['segm_sect']['CyberTech']['PRATE_PCT']			= 0.1
py['strats']['segm_sect']['CyberTech']['LIMIT_ORDER_PCT']		= 0.02
py['strats']['segm_sect']['CyberTech']['LMT_PCT']			= 0.03
py['strats']['segm_sect']['CyberTech']['MINUTES_TO_REBAL']	= 1
py['strats']['segm_sect']['CyberTech']['IS_LIVE']			= True
py['strats']['segm_sect']['CyberTech']['UNIVERSE']			= CyberTech_list


##End:CyberTech

###Start:CoronaTech

CoronaTech_list=(
    'JNJ','GILD','SNY','MRNA','INO',
    'NVAX','ABBV',
    #'VIR',
    'ZM','PTON',
    'TDOC','MMM',
    #'CLX','ADBE','CSCO',
    'CTXS','MSFT','ORCL','SAP','LRN','CHGG','NOW','AMZN',
    #'COST',
    #'IBM',
)

py['strats']['segm_sect']['CoronaTech']['algo'] 			= py['install'] + '/hca_segmsec.py'
py['strats']['segm_sect']['CoronaTech']['LEVERAGE_FAC']                 = 1.0
py['strats']['segm_sect']['CoronaTech']['LONG_PCT']			= 1.0
py['strats']['segm_sect']['CoronaTech']['PRATE_PCT']			= 0.1
py['strats']['segm_sect']['CoronaTech']['LIMIT_ORDER_PCT']		= 0.02
py['strats']['segm_sect']['CoronaTech']['LMT_PCT']			= 0.03
py['strats']['segm_sect']['CoronaTech']['MINUTES_TO_REBAL']	        = 1
py['strats']['segm_sect']['CoronaTech']['IS_LIVE']			= False #True
py['strats']['segm_sect']['CoronaTech']['IS_PERSIST']			= False
py['strats']['segm_sect']['CoronaTech']['UNIVERSE']			= CoronaTech_list

##End:CoronaTech
###Start:RHTech

RHTech_list=(
         # Robinhood Restrictions as of 01/30/2021         
         'AAL', #Aurora Cannabis (NYSE:ACB#NYSE">ACB</a>): 1 share, standard limits
         'AG', #First Majestic Silver Corp (NYSE:AG#NYSE">AG</a>): 1 share, standard limits
         'AMC', #AMC Entertainment (NYSE:AMC#NYSE">AMC</a>): 1 share, 10 options
         'AMD', #Advanced Micro Devices Inc (NASDAQ:AMD#NASDAQ">AMD</a>): 1 share, standard limits
         'BB', #BlackBerry Ltd (NYSE:BB#NYSE">BB</a>): 1 share, 10 options
         'BBBY', #Bed Bath &amp; Beyond Inc (NASDAQ:BBBY#NASDAQ">BBBY</a>): 1 share, 10 options
         'BYDDY', #BYD Co (OTC:BYDDY#OTC">BYDDY</a>): 1 share
         'BYND', #Beyond Meat&nbsp;(NASDAQ:BYND#NASDAQ">BYND</a>): 1 share, standard limits
         'CCIV', #Churchill Capital Corp IV (NYSE:CCIV#NYSE">CCIV</a>): 1 share, standard limits
         'CLOV', #Clover Health (NASDAQ:CLOV#NASDAQ">CLOV</a>) 1 share, standard limits
         'CRIS', #Curis (NASDAQ:CRIS#NASDAQ">CRIS</a>): 1 share, standard limits
         'CTRM', #Castor Maritime Inc (NASDAQ:CTRM#NASDAQ">CTRM</a>): 5 shares
         'EXPR', #Express Inc (NYSE:EXPR#NYSE">EXPR</a>): 5 shares', 10 options
         'EZGO', #EZGO Technologies (NASDAQ:EZGO#NASDAQ">EZGO</a>): 5 share
         'GM', #General Motors Corporation (NYSE:GM#NYSE">GM</a>): 1 share, standard limits
         'GME', #<br>	<li>GameStop Corp: 1 share, 5 options
         #'GTE', #Gran Tierra Energy (NYSE:GTE#NYSE">GTE</a>): 5 share, standard limits
         'HIMS', #Hims &amp; Hers Health (NYSE:HIMS#NYSE">HIMS</a>): 1 share, standard limits
         'INO', #Inovio Pharmaceuticals Inc (NASDAQ:INO#NASDAQ">INO</a>): 1 share, standard limits
         'IPOE', #Social Capital Hedosophia Holdings Corp V (NYSE:IPOE#NYSE">IPOE</a>): 1 share, standard limits
         'IPOF', #Social Capital Hedosophia Holdings Corp VI (NYSE:IPOF#NYSE">IPOF</a>): 1 share, standard limits
         #'JAGX', #Jaguar Health Inc (NASDAQ:JAGX#NASDAQ">JAGX</a>): 5 share, standard limits
         'KOSS', #Koss Corp (NASDAQ:KOSS#NASDAQ">KOSS</a>): 1 share
         'LLIT', #Lianluo Smart (NASDAQ:LLIT#NASDAQ">LLIT</a>): 5 share
         'MRNA', #Moderna Inc (NASDAQ:MRNA#NASDAQ">MRNA</a>): 1 share, standard limits
         'NAKD', #Naked Brands Group (NASDAQ:NAKD#NASDAQ">NAKD</a>): 5 shares
         'NCTY', #Nokia Oyj (NYSE:NOK#NYSE">NOK</a>): 5 shares, 10 options
         'NVAX', #Novavax Inc (NASDAQ:NVAX#NASDAQ">NVAX</a>): 1 share, standard limits
         'OPEN', #Opendoor Technologies Inc (NASDAQ:OPEN#NASDAQ">OPEN</a>): 1 share', standard limits
         'RKT', #Rocket Companies Inc (NYSE:RKT#NYSE">RKT</a>): 1 share', standard limits
         'RLX', #RLX Technology (NYSE:RLX#NYSE">RLX</a>): 1 share, standard limits
         #SymNotFound: 'RYCEY', #Rolls-Royce Holdings (OTC:RYCEY#OTC">RYCEY</a>): 5 shares, standard limits
         'SBUX', #Starbucks Corp (NASDAQ:SBUX#NASDAQ">SBUX</a>): 1 share, standard limits
         'SHLS', #Shoals Technologies Group (NASDAQ:SHLS#NASDAQ">SHLS</a>): 1 share
         'SIEB', #Siebert Financial Corp (NASDAQ:SIEB#NASDAQ">SIEB</a>): 1 share, standard limits
         #SymNotFound: 'SLV', #iShares Silver Trust (NYSE:SLV#NYSE">SLV</a>): 1 share, standard limits
         #??bad Losses 'SNDL', #Sundial Growers Inc (NASDAQ:SNDL#NASDAQ">SNDL</a>): 5 shares, 10 options
         #SymNotFound: 'SOXL', #Direxion Daily Semiconductor Bull 3x Shares (NYSE:SOXL#NYSE">SOXL</a>): 1 share, standard limits
         'SRNE', #Sorrento Therapeutics Inc (NASDAQ:SRNE#NASDAQ">SRNE</a>): 1 share, standard limits
         'STPK', #Tengasco (NYSE:TGC#NYSE">TGC</a>): 5 shares
         'TIRX', #Tian Ruixiang Holdings (NASDAQ:TIRX#NASDAQ">TIRX</a>): 1 share
         'TR', #Tootsie Roll Industries (NYSE:TR#NYSE">TR</a>): 1 share, 10 options
         'TRVG', #Trivago (NASDAQ:TRVG#NASDAQ">TRVG</a>): 55 shares, 10 options
         'WKHS', #Workhorse Group (NASDAQ:WKHS#NASDAQ">WKHS</a>): 1 share, standard limits
         #SymNotFound: 'XM', #Qualtrics International (NASDAQ:XM#NASDAQ">XM</a>): 1 share, standard limits
         #'ZOM', #Zomedica Corp (NYSE:ZOM#NYSE">ZOM</a>): 5 shares
)

py['strats']['segm_sect']['RHTech']['algo'] 			= py['install'] + '/hca_segmsec.py'
py['strats']['segm_sect']['RHTech']['LEVERAGE_FAC']             = 1.0
py['strats']['segm_sect']['RHTech']['LONG_PCT']			= 1.0
py['strats']['segm_sect']['RHTech']['PRATE_PCT']	        = 0.1
py['strats']['segm_sect']['RHTech']['LIMIT_ORDER_PCT']		= 0.02
py['strats']['segm_sect']['RHTech']['LMT_PCT']			= 0.03
py['strats']['segm_sect']['RHTech']['MINUTES_TO_REBAL']	        = 1
py['strats']['segm_sect']['RHTech']['IS_LIVE']			= False #True
py['strats']['segm_sect']['RHTech']['IS_PERSIST']		= False
py['strats']['segm_sect']['RHTech']['UNIVERSE']			= RHTech_list

##End:RHTech


###DEBUG pprint(py)

#Usage
#import hca_config as scfg #Strategy Config
#cfg_list =  [(x,y,mcfg.py['models'][x][y]) for x in scfg.py['models'] for y in scfg.py['models'][x] ]

