from __future__ import print_function
from __future__ import division
import numpy as np
import argparse
import matplotlib.pyplot as plt
import datetime
import sys


class Channel(object):
    labels = {}
    args = None
    first_timestamp = None
    last_timestamp = None
    
    def __init__(self, chan_num=None):
        if chan_num:
            self.chan_num = chan_num
            self.label = Channel.labels[chan_num] # TODO add error handling if no label
            self._load()
            self.is_aggregate_chan = True if self.label in ["mains", "aggregate", "agg"] \
                                          else False
        
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
            if Channel.args.start and timestamp < Channel.args.start:
                continue
            if Channel.args.end and timestamp > Channel.args.end:
                break
            watts = float(line[1])
            if (self.args.allow_high_vals
            or self.is_aggregate_chan
            or watts < 4000):    
                self.data[i] = (timestamp, watts) 
                i += 1
                
        # Resize self.data if we didn't take every line
        if not i:
            self.data = None
            return
        elif i != len(lines):
            self.data = np.resize(self.data, i)
            
        # Update Channel.first_timestamp and .last_timestamp
        if (not Channel.first_timestamp 
        or self.data["timestamp"][0] < Channel.first_timestamp):
            Channel.first_timestamp = self.data["timestamp"][0]
                
        if (not Channel.last_timestamp
        or self.data["timestamp"][-1] > Channel.last_timestamp):
            Channel.last_timestamp = self.data["timestamp"][-1]
        
    def __str__(self):
        str_format = "{:>2d}  {:<11s}  {:1d}  {:>7d}" + "  {:>7.1f}"*9
        
        if self.data is None:
            return str_format.format(self.chan_num, self.label[:11],
                                     1,0,0,0,0,0,0,0,0,0,0)
        
        is_sorted = self._sort()
        pwr = self.data['watts']
        dt  = self.data['timestamp'][1:-1] - self.data['timestamp'][0:-2]
        
        return str_format.format(
                       self.chan_num, self.label[:11], is_sorted, self.data.size,
                       pwr.min(), pwr.mean(), pwr.max(), pwr.std(),
                        dt.min(),  dt.mean(),  dt.max(),  dt.std(),
                        self._percent_missed())
        
    def _percent_missed(self):
        total_time = Channel.last_timestamp - Channel.first_timestamp
        num_expected_samples = total_time / 6
        return (1 - (self.data.size / num_expected_samples)) * 100
    
    @staticmethod
    def print_header():
        if not Channel.first_timestamp:
            print("NO DATA! Command line options --start =",
                   Channel.args.start, "--end =", Channel.args.end)
            return
        
        last = datetime.datetime.fromtimestamp(Channel.last_timestamp)
        first = datetime.datetime.fromtimestamp(Channel.first_timestamp)
        
        print("Start time        =", Channel.first_timestamp, first)
        print("End time          =", Channel.last_timestamp, last)
        print("Total time period =", last - first)        
        print("")
        print("                                 |---------POWER (W)----------|      |-------SAMPLE PERIOD (s)----|")
        print(" #       NAME    S    COUNT      MIN     MEAN      MAX    STDEV      MIN     MEAN      MAX    STDEV  %missed")    
    
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


def convert_to_int(string, name):
    if string:
        try:
            return int(string)
        except ValueError:
            print("ERROR:", name, "time must be an integer", file=sys.stderr)
            sys.exit(2)


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
    
    parser.add_argument('--start', dest='start', type=str
                        ,default=""
                        ,help="Unix timestamp to start time period.")    

    parser.add_argument('--end', dest='end', type=str
                        ,default=""
                        ,help="Unix timestamp to end time period.")    

    args = parser.parse_args()

    args.allow_high_vals = not args.no_high_vals

    # append trailing slash to data_directory if necessary
    if args.data_dir and args.data_dir[-1] != "/":
        args.data_dir += "/"
    
    args.start = convert_to_int(args.start, "start")
    args.end   = convert_to_int(args.end,   "end")
    
    if args.start and args.end and args.start > args.end:
        print("ERROR: start time", args.start, "is after end time", args.end,
              file=sys.stderr)
        sys.exit(2)
       
    return args


def main():
    args = setup_argparser()
    
    print("data-dir = ", args.data_dir)
    
    labels = load_labels(args)

    Channel.labels = labels
    Channel.args = args
    
    channels = {}
    
    for chan_num in labels.keys():
        channels[chan_num] = Channel(chan_num)

    Channel.print_header()
    plt.hold(True)    
    for dummy, chan in channels.iteritems():
        print(chan)
        chan.plot()
        
    if Channel.first_timestamp:
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()