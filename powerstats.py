from __future__ import print_function
from __future__ import division
import numpy as np
import argparse
import matplotlib.pyplot as plt
import datetime


class Channel(object):
    labels = {}
    args = None
    
    def __init__(self, chan_num=None):
        if chan_num:
            self.chan_num = chan_num
            self.label = Channel.labels[chan_num] # TODO add error handling if no label
            self._load()
        
    def _load(self):
        filename = Channel.args.data_dir + "channel_{:d}.dat".format(self.chan_num) 
        try:
            with open(filename) as data_file:
                lines = data_file.readlines()
        except IOError:
            self.data = None
            return
        
        self.data = np.zeros(len(lines), 
                             dtype=[('timestamp', np.uint32), ('watts', float)])
        i = 0
        for line in lines:
            line = line.split()
            timestamp = int(line[0])
            watts = float(line[1])
            if (not self.args.no_high_vals
            or self.label == "mains" 
            or self.label == "aggregate"
            or self.label == "agg" 
            or watts < 4000):    
                self.data[i] = (timestamp, watts) 
                i += 1
                
        if i != len(lines):
            self.data = np.resize(self.data, i)
        
    def __str__(self):
        str_format = "{:>2d}  {:<11s}  {:1d}  {:>7d}" + "  {:>7.1f}"*8
        
        if self.data is None:
            return str_format.format(self.chan_num, self.label[:11],
                                     1,0,0,0,0,0,0,0,0,0)
        
        is_sorted = self._sort()
        pwr = self.data['watts']
        dt  = self.data['timestamp'][1:-1] - self.data['timestamp'][0:-2]
        
        return str_format.format(
                       self.chan_num, self.label[:11], is_sorted, self.data.size,
                       pwr.min(), pwr.mean(), pwr.max(), pwr.std(),
                        dt.min(),  dt.mean(),  dt.max(),  dt.std())
    
    @staticmethod
    def print_header():
        print("                                 |---------POWER (W)----------|      |-------SAMPLE PERIOD (s)----|")
        print(" #       NAME    S    COUNT      MIN     MEAN      MAX    STDEV      MIN     MEAN      MAX    STDEV")
        
    def _sort(self):
        """If self.data is sorted by timecode then return true,
        else sort rows in self.data by timecode and return false."""
       
        sorted_data = self.data.__copy__()
        sorted_data.sort(order='timestamp')
        if (sorted_data == self.data).all():
            return True
        else:
            self.data = sorted_data
            return False
        
    def plot(self):
        if self.data is None:
            return
        
        x = np.empty(self.data.size, dtype="object")
        for i in range(self.data.size):
            x[i] = datetime.datetime.fromtimestamp(self.data["timestamp"][i])
        plt.plot(x, self.data['watts'], label=self.label)


def load_labels(args):
    with open(args.data_dir + args.labels_file) as labels_file:
        lines = labels_file.readlines()
    
    labels = {}
    for line in lines:
        line = line.split()
        labels[int(line[0])] = line[1] # TODO add error handling if line[0] not an int
        
    return labels


def setup_argparser():
    # Process command line _args
    parser = argparse.ArgumentParser(description="Generate simple stats for "
                                                 "electricity power data logs.")
       
    parser.add_argument('--data-dir', dest='data_dir', type=str
                        ,default=""
                        ,help='directory for storing data')
    
    parser.add_argument('--labels-file', dest='labels_file', type=str
                        ,default="labels.dat"
                        ,help="filename for labels data (default:'labels.dat')")
    
    parser.add_argument('--no-high-values', dest='no_high_vals', action='store_const',
                        const=True, default=False, 
                        help='Remove values >4000W for IAMs (default=False)')

    args = parser.parse_args()

    # append trailing slash to data_directory if necessary
    if args.data_dir and args.data_dir[-1] != "/":
        args.data_dir += "/"
       
    return args


def main():
    args = setup_argparser()
    
    labels = load_labels(args)

    Channel.labels = labels
    Channel.args = args
    Channel.print_header()
    
    plt.hold(True)
    for chan_num in labels.keys():
        chan = Channel(chan_num)
        print(chan)
        chan.plot()
        
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()