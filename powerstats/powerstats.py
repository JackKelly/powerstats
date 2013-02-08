#! /usr/bin/python

from __future__ import print_function, division
import numpy as np
import argparse
import os
import matplotlib
if not os.environ.get('DISPLAY'):
    matplotlib.use('Agg') # use Agg backend if X isn't available
import matplotlib.pyplot as plt
import datetime
import sys
import pickle
import copy
from table import Table

MAX_PERIOD = 300 # seconds
SAMPLE_PERIOD = 6 # seconds        

class Channel(object):
    labels = {}
    args = None
    axes = None
    max_chan_num = 0
    
    table = Table(col_width=[5,11,6,3] + [6,6] + [6,6,6,6] + [10, 6],
                  data_format=["{:d}","{:s}","{:d}","{}","{:.1f}","{:.1f}","{}","{:.1f}","{}","{:.1f}","{:.1%}", "{:.1f}"],
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

    cache_table = Table(col_width=[5,11,6] + [6,6] + [6,6,6] + [10, 6],
                        data_format=["{:d}","{:s}","{:d}","{:.1f}","{:.1f}","{}","{:.1f}","{}","{:.1%}", "{:.1f}"])
    cache_table.header_row([(3, ""), (2, "POWER (W)", "-"), (3, "SAMPLE PERIOD (s)", "-"), (2, "")])
    cache_table.header_row(["#", 
                      "name", 
                      "count", 
                      "min", "max",
                      "min", "mean", "max",
                      "% missed",
                      "kwh"
                      ])
    
    totals_table = copy.deepcopy(cache_table)

    data_to_plot = False
    
    def __init__(self, chan_num=None):
        self._dt = None # delta time.
        self._proportion_missed_cache = None
        self._kwh_cache = None
        self._cache = {}
        
        if chan_num:
            self.chan_num = chan_num
            if self.chan_num > Channel.max_chan_num:
                Channel.max_chan_num = self.chan_num
            self.label = Channel.labels[chan_num] # TODO add error handling if no label
            self.is_aggregate_chan = True if self.label in ["mains", "aggregate", "agg"] \
                                          else False
            self._load()
        
    def _load(self):
        filename = "channel_{:d}.dat".format(self.chan_num)
        print("Loading ", filename, "... ", end="", sep="")        
        self.data_filename = Channel.args.data_dir + "/" + filename

        # Load cache if necessary
        if Channel.args.cache:
            self._load_cache()        
        
        # Load data file
        try:
            with open(self.data_filename) as data_file:
                if self._cache:
                    data_file.seek(self._cache['filesize'])
                lines = data_file.readlines()
        except IOError:
            self.data = None
            print("doesn't exist. Skipping.")
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
                
        print(i, "lines loaded... ", end="")
                
        # Resize self.data if we didn't take every line
        if not i:
            self.data = None
            print("No new data loaded!")
            return
        elif i != len(lines):
            self.data = np.resize(self.data, i)

        # Calculate delta time vector
        self._dt = self.data['timestamp'][1:] - self.data['timestamp'][:-1]
                    
        print("Done.")

    def _load_cache(self):
        # Try loading the pickled cache file
        self.pkl_filename = Channel.args.data_dir + \
                       "/channel_{:d}_cache.pkl".format(self.chan_num)

        try:
            pkl_file = open(self.pkl_filename, "rb")
        except:
            self._cache = {}
        else:
            self._cache = pickle.load(pkl_file)
            pkl_file.close()

    def add_cache_to_table(self, table):
        if not self._cache:
            return
        
        table.update_first_timestamp(self._cache['first_timestamp'])
        table.update_last_timestamp(self._cache['last_timestamp'])
        
        table.data_row([
          self.chan_num, self.label,
          self._cache['count'],
          self._cache['watts_min'], self._cache['watts_max'],
          self._cache['dt_min'], self._cache['dt_mean'], self._cache['dt_max'],
          self._cache['missed'],
          self._cache['kwh']
          ])

    def update_and_save_cache(self):
        if not Channel.args.cache or self.data is None:
            return
        
        if self._cache:
            self._cache['watts_min'] = min(self._cache['watts_min'], self.data['watts'].min())
            self._cache['watts_max'] = max(self._cache['watts_max'], self.data['watts'].max())
            self._cache['dt_min'] = min(self._cache['dt_min'], self._dt.min())
            self._cache['dt_max'] = max(self._cache['dt_max'], self._dt.max())
            total_size = self._cache['count'] + self.data.size 
            self._cache['dt_mean'] = ((self._cache['dt_mean']*(self._cache['count']-1)) + self._dt.sum()) / (total_size-2)
            self._cache['missed'] = ((self._cache['missed']*self._cache['count']) + (self._proportion_missed()*self.data.size)) / total_size 
            self._cache['count'] = total_size
            self._cache['kwh'] += self._kwh()
        else:
            self._cache['first_timestamp'] = self.data['timestamp'][0]
            self._cache['watts_min'] = self.data['watts'].min()
            self._cache['watts_max'] = self.data['watts'].max()
            self._cache['dt_min'] = self._dt.min()
            self._cache['dt_max'] = self._dt.max()
            self._cache['dt_mean'] = self._dt.mean()
            self._cache['missed'] = self._proportion_missed()
            self._cache['count'] = self.data.size
            self._cache['kwh'] = self._kwh()
            
        self._cache['last_timestamp'] = self.data['timestamp'][-1]
        self._cache['filesize'] = os.path.getsize(self.data_filename)
            
        with open(self.pkl_filename, "wb") as output:
            # "with" ensures we close the file, even if an exception occurs.
            pickle.dump(self._cache, output)            

    def _kwh(self):
        if self.data is None:
            return 0

        if not self._kwh_cache:
            dt_limited = np.where(self._dt>MAX_PERIOD, SAMPLE_PERIOD, self._dt)
            watt_seconds = (dt_limited * self.data['watts'][:-1]).sum()           
            self._kwh_cache = watt_seconds / 3600000
            
        return self._kwh_cache
        
    def add_to_table(self):
        if self._dt is None or self._dt.size < 2:
            Channel.table.data_row([self.chan_num, self.label,
                                     0,'-',0,0,0,0,0,0,1,0])
            return
        
        if Channel.args.sort:
            is_sorted = self._sort()
        else:
            is_sorted = "-"        

        Channel.table.update_first_timestamp(self.data['timestamp'][0])
        Channel.table.update_last_timestamp(self.data['timestamp'][-1])

        Channel.table.data_row([
                       self.chan_num, self.label, self.data.size, is_sorted,
                       self.data['watts'].min(), self.data['watts'].max(),
                       self._dt.min(), self._dt.mean(), self._dt.max(), self._dt.std(),
                       self._proportion_missed(),
                       self._kwh()])

        
    def _proportion_missed(self):
        if not self._proportion_missed_cache:
            n_missed = ((self._dt // SAMPLE_PERIOD) * (self._dt > 10)).sum()
            n_expected = n_missed + (self._dt <= 10).sum()
            self._proportion_missed_cache = n_missed / n_expected
        return self._proportion_missed_cache
        
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
        if self._dt is None or self.data is None:
            return
        else:
            Channel.data_to_plot = True
        
        # Power consumption
        x = np.empty(self.data.size, dtype="object")
        for i in range(self.data.size):
            x[i] = datetime.datetime.fromtimestamp(self.data["timestamp"][i])
            
        pwr_line, = Channel.pwr_axes.plot(x, self.data['watts'], label=
                                          str(self.chan_num)+" "+self.label)
                
        # Plot missed samples
        for i in (self._dt > 11).nonzero()[0]:
            start = x[i]
            end   = x[i+1]
            rect = plt.Rectangle((start, -self.chan_num), # bottom left corner
                                 (end-start).total_seconds()/86400, # length
                                 1, # width
                                 color=pwr_line.get_c())
            Channel.hit_axes.add_patch(rect)
        
    @staticmethod
    def output_text_tables():
        print("NEW DATA:\n", Channel.table)
        if Channel.args.cache and Channel.cache_table.data:
            print("OLD DATA:\n", Channel.cache_table)
            print("TOTALS:\n", Channel.totals_table)

    @staticmethod
    def output_html_tables():
        html = "<h3>NEW DATA:</h3>\n"
        html += Channel.table.html()
        if Channel.args.cache and Channel.cache_table.data:
            html += "<h3>OLD DATA:</h3>\n"
            html += Channel.cache_table.html()
            html += "<h3>TOTALS:</h3>\n"
            html += Channel.totals_table.html()

        return html


def load_labels(args):
    """
    Loads data from labels.dat file.
    
    Returns:
        A dict mapping channel numbers (ints) to appliance names (str)
    """
    with open(args.data_dir + "/" + args.labels_file) as labels_file:
        lines = labels_file.readlines()
    
    labels = {}
    for line in lines:
        line = line.split()
        labels[int(line[0])] = line[1] # TODO add error handling if line[0] not an int
        
    print("Loaded {} lines from labels.dat".format(len(labels)))
        
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
                                                 "electricity power data logs."
                                    ,epilog="example: ./powerstats.py  --data-dir ~/data")
       
    parser.add_argument('--data-dir'
                        ,dest="data_dir"
                        ,default=os.environ.get("DATA_DIR")
                        ,help='directory from which to retrieve data.')
    
    parser.add_argument('--numeric-subdirs'
                        ,dest='numeric_subdirs'
                        ,action='store_true'
                        ,help='Data is stored within numerically named subdirs in base data dir.')
    
    parser.add_argument('--labels-file'
                        ,default="labels.dat"
                        ,help="filename (without path) for labels data (default:'labels.dat').")
    
    parser.add_argument('--no-high-values', dest='allow_high_vals', action='store_false'
                        ,help='Remove values >4000W for IAMs.')
    
    parser.add_argument('--sort', action='store_true'
                        ,help='Pre-sort by date. Vital for MIT data.')
    
    parser.add_argument('--no-plot', dest='plot', action='store_false'
                        ,help='Do not plot graph.')

    parser.add_argument('--html', action='store_true'
                        ,help='Output HTML to data-dir/html/')
    
    parser.add_argument('--html-dir', dest="html_dir"
                        ,help='Output stats and graphs as HTML to this directory.')    
    
    parser.add_argument('--start'
                        ,help="Unix timestamp to start time period.")    

    parser.add_argument('--end'
                        ,help="Unix timestamp to end time period.")    

    parser.add_argument('--cache', action='store_true'
                        ,help='Cache data for this timeperiod, starting from end of last cached period.')

    args = parser.parse_args()

    # process data dir
    if args.data_dir:
        args.data_dir = os.path.realpath(args.data_dir)
    else:
        sys.exit("\nERROR: Please specify a data directory either using the --data-dir \n"
                 "       command line option or using the $DATA_DIR environment variable.\n")

    # process numeric_subdirs
    if args.numeric_subdirs:
        # find the highest number data_dir
        existing_subdirs = os.walk(args.data_dir).next()[1]
        if existing_subdirs:
            existing_subdirs.sort()
            args.data_dir += "/" + existing_subdirs[-1]

    # process html
    if args.html:
        args.html_dir = args.data_dir + "/html"

    # process html_dir
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

    # Feedback to the user
    print("\nSELECTED OPTIONS:")
    print("*  input data directory = ", args.data_dir)
    feedback_arg(args.numeric_subdirs, "using numeric subdirectories.")
    feedback_arg(args.allow_high_vals, "allowing high IAM values.")
    feedback_arg(args.sort, "pre-sorting data")
    feedback_arg(args.plot, "plotting")
    feedback_arg(args.html_dir, "outputting HTML", "to file://{}/index.html".format(args.html_dir))
    feedback_arg(args.cache, "caching data")
    print("*  window starting at {}".format(args.start) if args.start else "*  window starting at beginning of data")
    print("*  window ending at {}".format(args.end) if args.end else "*  window finishing at end of data")
    print("")

    return args


def scale_width(ax, scale):
    """
    Scales the width of a matplotlib.axes object.
    
    Args:
        - ax (matplotlib.axes)
        - scale (float)
    """
    # Taken from http://stackoverflow.com/a/4701285/732596    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*scale, box.height])


def feedback_arg(arg, text, optional_text=""):
    """Provide feedback to the user about selected options."""
    print("*", "" if arg else " not", text, optional_text if arg else "")


def main():
    # Load command-line arguments
    args = setup_argparser()
    Channel.args = args    
    
    # Load labels.dat
    try:
        labels = load_labels(args)
    except IOError, e:
        sys.exit(e)
    Channel.labels = labels
    
    # Load channel data
    channels = {}
    for chan_num in labels.keys():
        channels[chan_num] = Channel(chan_num)
        
    print("")

    # Setup matplotlib figure (but don't plot anything yet)
    if args.plot:
        fig = plt.figure(figsize=(14,6))
        Channel.pwr_axes = fig.add_subplot(2,1,1)
        Channel.pwr_axes.set_title("Power consumption")
        Channel.pwr_axes.set_xlabel("time")
        Channel.pwr_axes.set_ylabel("watts")
        
        Channel.hit_axes = fig.add_subplot(2,1,2) # for plotting missed samples
        Channel.hit_axes.set_title("Drop-outs")  
        Channel.hit_axes.xaxis.axis_date()  
    
    # Produce data for stats tables, update cache and plot channel data
    for dummy, chan in channels.iteritems():
        chan.add_to_table()
        if args.cache:
            chan.add_cache_to_table(Channel.cache_table)
            chan.update_and_save_cache()
            chan.add_cache_to_table(Channel.totals_table)
        if args.plot:
            chan.plot()
    
    # Output stats tables as HTML or to stdout
    if args.html_dir:
        html_file = open(args.html_dir + "/index.html", "w")
        html_file.write("<!DOCTYPE html>\n<html>\n<body>")
        html_file.write(Channel.output_html_tables())
        if Channel.data_to_plot:
            html_file.write("<img src=\"fig.png\"/>")
        else:
            html_file.write("<p>No new data to plot!</p>")
        html_file.write("</body>\n</html>")
        html_file.close()
    else:
        Channel.output_text_tables()
        
    # Finish formatting plots and output to screen or file
    if args.plot and Channel.data_to_plot:
        # Format axes
        Channel.hit_axes.autoscale_view()      
        Channel.hit_axes.set_xlim( Channel.pwr_axes.get_xlim() )
        Channel.hit_axes.set_ylim([-Channel.max_chan_num, 0])
        date_formatter = matplotlib.dates.DateFormatter("%d/%m\n%H:%M")
        Channel.hit_axes.xaxis.set_major_formatter( date_formatter )     
        Channel.pwr_axes.xaxis.set_major_formatter( date_formatter )     
        plt.tight_layout()
        
        # Shrink axes by 20% to make space for legend   
        scale_width(Channel.hit_axes, 0.8)
        scale_width(Channel.pwr_axes, 0.8)
        
        # Create and format legend
        leg = Channel.pwr_axes.legend(bbox_to_anchor=(1, 1), loc="upper left")
        for t in leg.get_texts():
            t.set_fontsize('small')
                
        # Output plot to chosen destination
        if args.html_dir:
            plt.savefig(args.html_dir + "/fig.png", bbox_inches=0)
        else:
            plt.show()

    if not Channel.data_to_plot:
        print("No new data to plot.")


if __name__ == "__main__":
    main()
