#!/bin/bash

#initialise variables
rates_array=(500 1000 5000 9000)
file_array=("traf_500mbps.csv" "traf_1000mbps.csv" "traf_5000mbps.csv" "traf_9000mbs.csv") 
for i in 0 1 2 3
do
	echo $i
	echo ${file_array[i]}
done
#sleep 30
echo "starting VPP and sleep 15"
vpp_start-default.sh &
sleep 15
echo "Setup xconnect"
vpp_setup-xconnect.sh
sleep 2
#sudo $BINS/vppctl -s /tmp/cli.sock &
#sleep 5

for i in 0 1 2 3
do
	echo "starting MoonGen with rate: ${rates_array[$i]} mpbs"
	cd /usr/local/src/MoonGen/; sudo ./moongen-simple start load-latency:0:1:rate=${rates_array[i]},timeLimit=10m &
	echo "sleep 5 secs"
	sleep 5

	echo "starting perf and sleeping 60s"
	sudo perf stat -e page-faults,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-stores,L1-icache-load-misses,LLC-load-misses,LLC-loads,LLC-store-misses,LLC-stores,branch-load-misses,branch-loads,dTLB-load-misses,dTLB-loads,dTLB-store-misses,dTLB-stores,iTLB-load-misses,iTLB-loads,node-load-misses,node-loads,node-store-misses,node-stores -x, -o "$HOME/test$i.csv" -r 1 -p `pidof vpp` -I 100 &
	sleep 60

	pid=$(pidof perf)
	pidArray=($pid)
	sudo kill -9 $pid
	sleep 5
	echo "program: ${pidArray[0]} (perf), killed"

	pid_moon=$(pidof MoonGen)
	sudo kill -9 $pid_moon
	sleep 5
	echo "program: $pid_moon (MoonGen), killed"

done

sudo killall vpp_main
