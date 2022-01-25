#!/bin/bash
#set -v

. ~/hca/hca.env

DT=`date +%F`
echo "Date=${DT}"

echo "zipline run -s ${DT} -f /home/hca-ws2005/hca/hca_eon_live-main/strats-eonum/eon_ai.py --bundle sharadar-eqfd --broker ib --broker-uri 127.0.0.1:7499:1301 --broker-acct DU1359971 --data-frequency daily --state-file /home/hca-ws2005/hca/hca_eon_live-main/strats-eonum/Data/strategy.state --realtime-bar-target /home/hca-ws2005/hca/hca_eon_live-main/strats-eonum/Data/realtime-bars/"


zipline run -s ${DT} -f /home/hca-ws2005/hca/hca_eon_live-main/strats-eonum/eon_ai.py --bundle sharadar-eqfd --broker ib --broker-uri 127.0.0.1:7499:1301 --broker-acct DU1359971 --data-frequency daily --state-file /home/hca-ws2005/hca/hca_eon_live-main/strats-eonum/Data/strategy.state --realtime-bar-target /home/hca-ws2005/hca/hca_eon_live-main/strats-eonum/Data/realtime-bars/
