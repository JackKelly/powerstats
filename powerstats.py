#! /usr/bin/python

from __future__ import print_function
from __future__ import division
import numpy as np
import argparse
import matplotlib.pyplot as plt
import datetime
import sys
import os
from table import Table

class Channel(object):
    labels = {}
    args = None
    first_timestamp = None
    last_timestamp = None
    
    table = Table(col_width=[5,11,6,3] + [6,6] + [6,6,6,6] + [10, 6],
                  data_format=["{:d}","{:s}","{:d}","{}"] + ["{:.1f}"]*7 + ["{:.1f}"],
                  col_sep=1)
    
    # Create two-row header
    table.header_row([(4, ""), (2, "POWER (W)", "-"), (4, "SAMPLE PERIOD (s)", "-"), (2, "")])
    table.header_row(["#", 
                      "name", 
                      "count", 
                      "s",
                      "min", "max",
                      "min", "mean", "max", "stdev",
                      "% missed",
                      "kwh"
                      ])
    
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
            
    def _kwh(self):
        if self.data is None:
            return 0
        
        kwh = 0
        for i in range(0, self.data.size-1):
            if self.data[i]['watts']:
                dt = self.data[i+1]['timestamp']-self.data[i]['timestamp']
                if dt > 300: # assume it's off if we haven't heard from it
                    dt = 300
                    
                dt = dt / 3600 # convert from seconds to hours
                kwh += (dt * self.data[i]['watts']) / 1000
            
        return kwh
        
    def add_to_table(self):        
        if self.data is None:
            Channel.table.data_row(self.chan_num, self.label,
                                     1,0,0,0,0,0,0,0,0,0)
            return
        
        if Channel.args.sort:
            is_sorted = self._sort()
        else:
            is_sorted = "-"
        
        pwr = self.data['watts']
        dt  = self.data['timestamp'][1:-1] - self.data['timestamp'][0:-2]
        
        Channel.table.data_row([
                       self.chan_num, self.label, self.data.size, is_sorted,
                       pwr.min(), pwr.max(),
                        dt.min(),  dt.mean(),  dt.max(),  dt.std(),
                       self._percent_missed(),
                       self._kwh()])
        
    def _percent_missed(self):
        total_time = Channel.last_timestamp - Channel.first_timestamp
        num_expected_samples = total_time / 6
        return (1 - (self.data.size / num_expected_samples)) * 100
    
    @staticmethod
    def timeperiod_table():
        if not Channel.first_timestamp:
            print("NO DATA! Command line options --start={}, --end={}"
                           .format(Channel.args.start, Channel.args.end))
            return
        
        last = datetime.datetime.fromtimestamp(Channel.last_timestamp)
        first = datetime.datetime.fromtimestamp(Channel.first_timestamp)
           
        htable = Table(col_width=[17,20,20])
        
        htable.data_row(["Start time",
                         Channel.first_timestamp.__str__(),
                         first.__str__()])
        
        htable.data_row(["End time",
                         Channel.last_timestamp.__str__(),
                         last.__str__()])
        
        htable.data_row(["Total time period",
                         (last - first).__str__()])        

        return htable
    
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
    
    parser.add_argument('--sort', dest='sort', action='store_const',
                        const=True, default=False, 
                        help='Pre-sort by date. Vital for MIT data (default=False)')
    
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

    if os.environ.get('DISPLAY'):
        plt.hold(True)
            
    for dummy, chan in channels.iteritems():
        chan.add_to_table()
        if os.environ.get('DISPLAY'):
            chan.plot()
        
    print(Channel.timeperiod_table())
    print(Channel.table)
        
    if os.environ.get('DISPLAY') and Channel.first_timestamp:
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()