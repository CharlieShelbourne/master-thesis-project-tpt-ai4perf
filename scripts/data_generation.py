import time
import threading
import os
import paramiko
import subprocess


def start_vpp():
    time.sleep(0.2)
    print('starting vpp')
    print(sshcmd('vpp_start-default.sh &'))
    #time.sleep(5)
    #sshcmd('vpp_setup-xconnect.sh')
    #time.sleep(5)
    #sshcmd('sudo $BINS/vppctl -s /tmp/cli.sock')
    #time.sleep(5)
    #sshcmd('^C')

def run_perf(sample_period):

    #sshcmd('sudo perf stat -e branches,branch-misses,bus-cycles,cache-misses,cache-references,cycles,instructions,ref-cycles,alignment-faults,bpf-output,context-switches,cpu-clock,cpu-migrations,dummy,emulation-faults,major-faults,minor-faults,page-faults,task-clock,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-stores,L1-icache-load-misses,LLC-load-misses,LLC-loads,LLC-store-misses,LLC-stores,branch-load-misses,branch-loads,dTLB-load-misses,dTLB-loads,dTLB-store-misses,dTLB-stores,iTLB-load-misses,iTLB-loads,node-load-misses,node-loads,node-store-misses,node-stores -x, -o out_test.csv -r 1 -p `pidof vpp` -I 100')
    sshcmd('sudo perf stat -e branches,branch-misses,bus-cycles,cache-misses,cache-references,cycles,instructions,ref-cycles,alignment-faults,bpf-output,context-switches,cpu-clock,cpu-migrations,dummy,emulation-faults,major-faults,minor-faults,page-faults,task-clock,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-stores,L1-icache-load-misses,LLC-load-misses,LLC-loads,LLC-store-misses,LLC-stores,branch-load-misses,branch-loads,dTLB-load-misses,dTLB-loads,dTLB-store-misses,dTLB-stores,iTLB-load-misses,iTLB-loads,node-load-misses,node-loads,node-store-misses,node-stores -x, -r 1 -p `pidof vpp` -I '+sample_period+'')

def run_moongen(rate,loop):
    print('run moongen'+rate)
    sshcmd('cd /usr/local/src/MoonGen/; sudo ./moongen-simple start load-latency:0:1:rate='+rate+',timeLimit=10m')
    if loop == 0:
        print('sleep 40')
        time.sleep(40)
    else:
        print('sleep 20')
        time.sleep(20)


def sshcmd(command,prt= False):

    ssh = subprocess.Popen(["ssh", 'charlie@vpp.r2.enst.fr', command],
                           shell=False,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,)

    if prt == True:
        for line in ssh.stderr:
            print(line)

    return ssh


rates = ['500','1000','5000','9000']
run_perf('100')
for i, rate in enumerate(rates):
    print(rate)
    run_moongen(rate,i)

os.system('scp charlie@vpp.r2.enst.fr:out_test.csv /home/francesco/charlie_thesis_dir')

