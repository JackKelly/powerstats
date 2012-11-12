from __future__ import print_function
from __future__ import division
import numpy as np
import argparse

def print_stats(duration, lst):
    if lst:
        arr = np.array(lst) / 1000
        forward_dif = arr[1:-1]-arr[0:-2]
        print("Mean num secs per sample: {:>5.2f}s".format(forward_dif.mean()))
        print("Max secs between samples: {:>5.2f}s".format(forward_dif.max()))
        print("Min secs between samples: {:>5.2f}s".format(forward_dif.min()))
        print("stdev between samples   : {:>5.2f}s".format(forward_dif.std()))


class Channel(object):
    labels = {}
    args = None
    
    def __init__(self, chan_num=None):
        if chan_num:
            self.chan_num = chan_num
            self.name = Channel.labels[chan_num] # TODO add error handling if no label
            self._load()
        
    def _load(self):
        filename = Channel.args.data_dir + "channel_{:d}.dat".format(self.chan_num) 
        with open(filename) as data_file:
            lines = data_file.readlines()
        
        self.data = np.empty(len(lines), 
                             dtype=[('timecode', np.uint32), ('watts', float)])
        i = 0
        for line in lines:
            line = line.split()
            self.data[i] = (line[0], line[1]) 
            i += 1
        
    def __str__(self):
        is_sorted = self._sort()
        pwr = self.data['watts']
        dt  = self.data['timecode'][1:-1] - self.data['timecode'][0:-2]
        
        return ("{:>2d}  {:<11s}  {:1d}  {:>6d}" + "  {:>7.1f}"*8).format(
                       self.chan_num, self.name[:11], is_sorted, self.data.size,
                       pwr.min(), pwr.mean(), pwr.max(), pwr.std(),
                        dt.min(),  dt.mean(),  dt.max(),  dt.std())
    
    @staticmethod
    def print_header():
        print("                                |---------POWER (W)----------|      |-------SAMPLE PERIOD (s)----|")
        print(" #       NAME    S   COUNT      MIN     MEAN      MAX    STDEV      MIN     MEAN      MAX    STDEV")
        
    def _sort(self):
        """If self.data is sorted by timecode then return true,
        else sort rows in self.data by timecode and return false."""
        
        sorted_data = self.data.__copy__()
        sorted_data.sort(order='timecode')
        if (sorted_data == self.data).all():
            return True
        else:
            self.data = sorted_data
            return False
                
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

    args = parser.parse_args()

    # append trailing slash to data_directory if necessary
    if args.data_dir and args.data_dir[-1] != "/":
        args.data_dir += "/"
       
    return args


def main():
    args = setup_argparser()
    
    args.data_dir = "/home/jack/workspace/python/rfm_ecomanager_logger/data/" # TODO remove
    
    labels = load_labels(args)

    Channel.labels = labels
    Channel.args = args
    Channel.print_header()
    
    for chan_num in labels.keys():
        chan = Channel(chan_num)
        print(chan)


if __name__ == "__main__":
    main()