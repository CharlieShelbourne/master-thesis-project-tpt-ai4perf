import pyshark
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import csv
import math

from scipy import stats
from scipy.stats import poisson

from matplotlib.backends.backend_pdf import PdfPages

#cap = pyshark.FileCapture('/home/francesco/Documents/Thesis_project/Traces/malware_1.pcap')
#cap = pyshark.FileCapture('/home/francesco/Documents/Thesis_project/Traces/packet_injection.pcap')
#cap = pyshark.FileCapture('/home/francesco/Documents/Thesis_project/Traces/packet_injection_6.pcap')
#cap = pyshark.FileCapture('/home/francesco/Documents/Thesis_project/Traces/maccdc2010_00025_20100313235851_172M.pcap')
#cap = pyshark.FileCapture('/home/francesco/Documents/Thesis_project/Traces/Datacentre/univ1_pt1.pcap')


def print_pkt_length(pkt):
    #print(pkt.captured_length)
    #print(pkt.sniff_timestamp)
    print(pkt)

#cap.apply_on_packets(print_pkt_length)

def extractAndPlot(capture,file,label, plot = True):
    packet_lengths =[]
    time_stamp = []

    for packet in tqdm(capture):
        #packet_lengths.append(packet.captured_length)
        #packet_lengths.append(packet.length)
        packet_lengths.append(packet.frame_info.get_field_value('len'))
        time_stamp.append(packet.frame_info.get_field_value('time_delta'))

    packet_lengths = np.asarray(packet_lengths)
    packet_lengths = packet_lengths.astype(int)
    time_stamp = np.asarray(time_stamp)
    time_stamp = time_stamp.astype(float)
    time_stamp_cum = np.cumsum(time_stamp)

    if plot == True:
        multi_plot_data(packet_lengths, time_stamp, time_stamp_cum,file,label)

    return packet_lengths, time_stamp, time_stamp_cum


def multi_plot_data(packet_lengths,time_stamp,time_stamp_cum,file,label): #plot hitograms CCDFs and pkt length/time stamp
    #plt.figure(1)
    #plt.subplot(231)
    #plt.hist(packet_lengths, bins='auto')  # arguments are passed to np.histogram
    #plt.title("PDF packet lengths")
    #plt.xlabel("packet no.")

    plt.subplot(221)
    x = np.sort(packet_lengths)
    y = (np.arange(len(packet_lengths)) / len(packet_lengths))
    plt.plot(x, y, '.', color='red')
    plt.title("CDF packet lengths "+label)
    plt.xlabel("packet size")

    plt.subplot(222)
    x= np.sort(packet_lengths)
    y = 1 - (np.arange(len(packet_lengths)) / (len(packet_lengths)+1))
    plt.loglog(x,y, '.', color='red')
    plt.title("CCDF packet lengths "+label)
    plt.xlabel("packet size")

    #plt.subplot(234)
    #plt.hist(time_stamp, bins='auto')  # arguments are passed to np.histogram
    #plt.title("PDF time stamps")
    #plt.xlabel("packet no.")

    plt.subplot(223)
    x = np.sort(time_stamp)
    y = (np.arange(len(time_stamp)) / len(time_stamp))
    plt.plot(x, y, '.', color='red')
    plt.title("CDF time stamps "+label)
    plt.xlabel("time")

    plt.subplot(224)
    x = np.sort(time_stamp)
    y = 1 - (np.arange(len(time_stamp)) / (len(time_stamp)+1))
    plt.loglog(x, y, '.', color='red')
    plt.title("CCDF time stamps "+label)
    plt.xlabel("time")

    plt.tight_layout()

    plt.savefig('/home/francesco/Documents/Thesis_project/Traces/'+file+'.png')
    plt.close()
    #plt.show()

    #plt.figure(1)
    #plt.subplot(211)
    #plt.plot(time_stamp_cum, packet_lengths / time_stamp)
    #plt.xlabel('time')
    #plt.ylabel('pkt length/processing time')

    #plt.subplot(212)
    #plt.plot(time_stamp_cum, packet_lengths)
    #plt.xlabel('time')
    #plt.xlabel('packet length')
    #plt.tight_layout()
    #plt.show()

    #plt.savefig('/home/francesco/Documents/Thesis_project/Results/pkt_rate_and_length')



def split_trace(trace):
    seg_len = int(len(trace) / 3)

    seg_1 = trace[0:seg_len]
    seg_2 = trace[seg_len:2 * seg_len]
    seg_3 = trace[2 * seg_len:3 * seg_len]

    return [seg_1,seg_2,seg_3]


def CCDF_compare(A,B):

    A = split_trace(A)
    B = split_trace(B)

    for i in range(len(A)):

        plt.subplot(3,2,i+1)
        packet_rank = 1 - (np.sort(A[i]) / np.max(A[i]))
        plt.plot(packet_rank, '.', color='red')
        plt.title("CCDF of packet lengths")
        plt.xlabel("packet no.")


        plt.subplot(3,2,i+4)
        packet_rank = 1 - (np.sort(B[i]) / np.max(B[i]))
        plt.plot(packet_rank, '.', color='red')
        plt.title("CCDF of time stamps")
        plt.tight_layout()
        plt.xlabel("packet no.")

    plt.show()


def KS_test(seg):
    D,p= stats.kstest(seg, stats.uniform(0, np.max(seg)).cdf)
    print('KS statistic:',D)
    print('p value:',p)
    return D,p



def poissons(x,mean):
    for val in x:
        fact = np.math.factorial(int(np.round(val)))

    return (mean**(x)*np.exp(-mean))/fact

######################################################################################################################
###################################################    MAIN      ####################################################
######################################################################################################################
if __name__ == '__main__':
    pl = []
    ts = []
    labels = ['pcpDC1','pcpDC2','pcpPINJ1','pcpPINJ2']
    for l, file in enumerate(['Datacentre/univ1_pt1','Datacentre/univ1_pt2','Packet_injection/maccdc2010_00025_20100313235851_172M','Packet_injection/maccdc2012_00008_197M']):

        cap = pyshark.FileCapture('/home/francesco/Documents/Thesis_project/Traces/'+ file+'.pcap')
        #cap = pyshark.FileCapture('/home/francesco/Documents/Thesis_project/Traces/Datacentre/univ1_pt2.pcap')
        print("pcap loaded")

    #CCDF_compare(packet_lengths,time_stamp)
        packet_lengths, time_stamp, cum_sum = extractAndPlot(cap,file,labels[l],plot = True)
    #     pl.append(packet_lengths)
    #     ts.append(time_stamp)
        df_quartiles = pd.DataFrame(columns = ['lower','median','upper'],index=['packet lengths','time stamps'])

        df_quartiles['median'].loc['packet lengths'] = np.quantile(packet_lengths,0.5)
        df_quartiles['lower'].loc['packet lengths'] = np.quantile(packet_lengths, 0.25)
        df_quartiles['upper'].loc['packet lengths'] = np.quantile(packet_lengths, 0.75)

        df_quartiles['median'].loc['time stamps'] = np.quantile(packet_lengths, 0.5)
        df_quartiles['lower'].loc['time stamps'] = np.quantile(packet_lengths, 0.25)
        df_quartiles['upper'].loc['time stamps'] = np.quantile(packet_lengths, 0.75)

        df_quartiles.to_csv('/home/francesco/Documents/Thesis_project/Traces/'+file+'.csv')

        df_moments = pd.DataFrame(columns = ['mean','skewness','kurtosis'],index=['packet lengths','time stamps'])

        df_moments['mean'].loc['packet lengths'] = np.mean(packet_lengths)
        df_moments['skewness'].loc['packet lengths'] = stats.skew(packet_lengths)
        df_moments['kurtosis'].loc['packet lengths'] = stats.kurtosis(packet_lengths)

        df_moments['mean'].loc['time stamps'] = np.mean(packet_lengths)
        df_moments['skewness'].loc['time stamps'] = stats.skew(packet_lengths)
        df_moments['kurtosis'].loc['time stamps'] = stats.kurtosis(packet_lengths)

        df_moments.to_csv('/home/francesco/Documents/Thesis_project/Traces/' + file + '.csv')
    #
    # ks_test_pl =[]
    # ks_test_ts =[]
    # for p in range(len(pl)):
    #     for i in range(p,len(pl)):
    #         stats.ks_2samp(pl[p], pl[i])
    #         stats.ks_2samp(ts[p], ts[i])
        # test_distributions = ['weibull_min','weibull_max','beta','lognorm']
        #
        # kstests_pl = pd.DataFrame(columns = test_distributions,index=['ks_stat','p_value'])
        # kstests_ts = pd.DataFrame(columns=test_distributions, index=['ks_stat', 'p_value'])
        # for distribution in test_distributions:
        #     kstests_pl[distribution].loc['ks_stat'],kstests_pl[distribution].loc['p_value'] = stats.kstest(packet_lengths,'norm')
        #     kstests_ts[distribution].loc['ks_stat'],kstests_ts[distribution].loc['p_value'] = stats.kstest(time_stamp,'norm')
        # kstests_pl.to_csv('/home/francesco/Documents/Thesis_project/Results/Traces/'+labels[l]+'packet_length_ks.csv')
        # kstests_ts.to_csv('/home/francesco/Documents/Thesis_project/Results/Traces/' + labels[l] + 'time_stamp_ks.csv')
    #print(cap[2].frame_info.get_field_value('time_delta'))
    #print(cap[2].frame_info)
    #     for i, val in enumerate(time_stamp):
    #         time_stamp[i]= int(np.round(val*100))
    #
    # #split trace into 3 parts
    #     #pkt_segs = split_trace(packet_lengths)
    #     #time_segs = split_trace(time_stamp)
    #     #cst_segs = split_trace(cum_sum)
    #
    #     data = [packet_lengths, time_stamp]

    #print(poisson.pmf(test, mean))
    #print((mean**test)*np.exp(-mean))#/np.math.factorial(int(np.round(test))))


    #calculate mean of samples - max likelyhood estimator for poisson distribution
        # pp = PdfPages('/home/francesco/Documents/Thesis_project/Results/Trace_packet_dist.pdf')
        # ks_res = np.zeros((2,2))
        # for i, metric in enumerate(data):
        #
        #     #for j, seg in enumerate(metric):
        #     fig, ax = plt.subplots(1, 1)
        #     psn_param = np.mean(metric)
        #
        #     mean, var, skew, kurt = poisson.stats(psn_param, moments='mvsk')
        #     #fig, ax = plt.subplots(1, 1)
        #
        #     ax.plot(np.sort(metric), poisson.pmf(np.sort(metric), psn_param), ms=8,
        #             label='Poisson\'s '+r'$\lambda$: ' + str(round(psn_param, 3)))
        #     n, x, _ = ax.hist(metric,histtype='step',density= True,label='Emperical Distribution')
        #     density = stats.gaussian_kde(np.sort(metric))
        #     ax.plot(x, density(x), 'r--', label='Gaussian Kernel Smoothing ')
        #
        #     if i == 0:
        #         ax.set_title('Trace Packet Length Distributions')
        #         ax.set_xlabel('packet length (bytes)')
        #     elif i == 1:
        #         ax.set_title('Trace Packet Time Duration Distributions ')
        #         ax.set_xlabel('time from previous packet (ms)')
        #     ax.legend()
        #
        #     pp.savefig()
        #     plt.show()
        #     ks_res[i,0],ks_res[i,1] = KS_test(metric)
        #
        #writer = csv.writer(open("/home/francesco/Documents/Thesis_project/Results/trace_poisson_kstest.csv", 'a'))
        #writer.writerow(ks_res)



    # Compair trace segment distributions with KS tests

        # pp.close()
        #
        # print("\nIf the KS statistic is small and p value is high we cannot reject that the distributions of the samples are the same")
        #for n,d in enumerate(data):
        #    res = np.zeros((3,2))

        #    if n == 0:
        #        print('Packet length')
        #    else:
        #        print('Time stamp')

        #    for i in range(len(d)-1):
        #        res[i,0],res[i,1] = stats.ks_2samp(d[i],d[i+1])
        #        res[len(d)-1,0],res[len(d)-1,1] = stats.ks_2samp(d[0],d[len(d)-1])

    #Save KS test to CSV
            #writer = csv.writer(open("/home/francesco/Documents/Thesis_project/Results/trace_samples_compare_kstest.csv", 'a'))
            #writer.writerow(res)



    #multi_plot_data(pkt_segs[0],time_segs[0],cst_segs[0])
    #multi_plot_data(pkt_segs[1],time_segs[1],cst_segs[1])
    #multi_plot_data(pkt_segs[2],time_segs[2],cst_segs[2])


