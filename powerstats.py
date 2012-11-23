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

MAX_PERIOD = 300 # seconds
SAMPLE_PERIOD = 6 # seconds        

class Channel(object):
    labels = {}
    args = None
    first_timestamp = None
    last_timestamp = None
    axes = None
    
    table = Table(col_width=[5,11,6,3] + [6,6] + [6,6,6,6] + [10, 6],
                  data_format=["{:d}","{:s}","{:d}","{}"] + ["{:.1f}"]*6 + ["{:.1%}", "{:.1f}"],
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
        self.dt = None
        if chan_num:
            self.chan_num = chan_num
            self.label = Channel.labels[chan_num] # TODO add error handling if no label
            self._load()
            self.is_aggregate_chan = True if self.label in ["mains", "aggregate", "agg"] \
                                          else False
        
    def _load(self):
        filename = Channel.args.data_dir + "/channel_{:d}.dat".format(self.chan_num) 
        print("Loading", filename, "...", end="")
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
            
        self.dt = self.data['timestamp'][1:-1] - self.data['timestamp'][0:-2]
            
        print("done.")
            
    def _kwh(self):
        if self.data is None:
            return 0

        dt_limited = np.where(self.dt>MAX_PERIOD, SAMPLE_PERIOD, self.dt)
        watt_seconds = (dt_limited * self.data['watts'][:-2]).sum()           
        return watt_seconds / 3600000
        
    def add_to_table(self):
        if self.data is None:
            Channel.table.data_row([self.chan_num, self.label,
                                     0,'-',0,0,0,0,0,0,1,0])
            return
        
        if Channel.args.sort:
            is_sorted = self._sort()
        else:
            is_sorted = "-"        

        Channel.table.data_row([
                       self.chan_num, self.label, self.data.size, is_sorted,
                       self.data['watts'].min(), self.data['watts'].max(),
                       self.dt.min(), self.dt.mean(), self.dt.max(), self.dt.std(),
                       self._proportion_missed(),
                       self._kwh()])
        
    def _proportion_missed(self):
        total_time = Channel.last_timestamp - Channel.first_timestamp
        num_expected_samples = total_time / SAMPLE_PERIOD
        prop_missed = (1 - ((self.data.size+1) / num_expected_samples))
        if prop_missed < 0:
            prop_missed = 0 
        return prop_missed
    
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
        Channel.axes.plot(x, self.data['watts'], label=self.label)
        
    @staticmethod
    def output_text_tables():
        print("Output text tables")
        print(Channel.timeperiod_table())
        print(Channel.table)

    @staticmethod
    def output_html_tables():
        html = "<p>"
        html += Channel.timeperiod_table().html()
        html += "</p><p>"
        html += Channel.table.html()
        html += "</p>"
        return html


def load_labels(args):
    with open(args.data_dir + "/" + args.labels_file) as labels_file:
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
    
    parser.add_argument('--no-plot', dest='plot', action='store_const',
                        const=False, default=True, 
                        help='Do not plot graph (default=False if X is available)')
    
    parser.add_argument('--html-dir', dest='html_dir', type=str,
                        default="", 
                        help='Send stats and graphs directory as HTML')    
    
    parser.add_argument('--start', dest='start', type=str
                        ,default=""
                        ,help="Unix timestamp to start time period.")    

    parser.add_argument('--end', dest='end', type=str
                        ,default=""
                        ,help="Unix timestamp to end time period.")    

    args = parser.parse_args()

    args.allow_high_vals = not args.no_high_vals

    # process paths
    args.data_dir = os.path.realpath(args.data_dir)
    if args.html_dir:
        args.html_dir = os.path.realpath(args.html_dir)
        # if directory doesn't exist then create it
        if not os.path.isdir(args.html_dir):
            os.makedirs(args.html_dir)
    
    # process start and end times
    args.start = convert_to_int(args.start, "start")
    args.end   = convert_to_int(args.end,   "end")
    if args.start and args.end and args.start > args.end:
        print("ERROR: start time", args.start, "is after end time", args.end,
              file=sys.stderr)
        sys.exit(2)
       
    # turn off plotting if X is not attached
    if not os.environ.get('DISPLAY'):
        args.plot = False
       
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
        
    print("")

    if args.plot:
        fig = plt.figure(figsize=(14,6))
        Channel.axes = fig.add_subplot(111) 
        Channel.axes.set_xlabel("time")
        Channel.axes.set_ylabel("watts")
            
    for dummy, chan in channels.iteritems():
        chan.add_to_table()
        if args.plot:
            chan.plot()
    
    if args.html_dir:
        html_file = open(args.html_dir + "/index.html", "w")
        html_file.write("<!DOCTYPE html>\n<html>\n<body>")
        html_file.write(Channel.output_html_tables())
        html_file.write("<img src=\"fig.png\"/>")
        html_file.write("</body>\n</html>")
        html_file.close()
    else:
        Channel.output_text_tables()
        
    if args.plot and Channel.first_timestamp:
        plt.tight_layout()
        leg = plt.legend()
        for t in leg.get_texts():
            t.set_fontsize('small')
                
        if args.html_dir:
            plt.savefig(args.html_dir + "/fig.png", bbox_inches=0)
        else:
            plt.show()
            


if __name__ == "__main__":
    main()