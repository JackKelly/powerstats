#! /usr/bin/python

from __future__ import print_function, division
import numpy as np
import argparse
import os
import matplotlib
if not os.environ.get('DISPLAY'):
    matplotlib.use('Agg') # use Agg backend if X isn't available
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
import sys
import pickle
import copy
import time
import pytz
import ConfigParser
from table import Table

MAX_PERIOD = 300 # seconds
SAMPLE_PERIOD = 6 # seconds        

class Channel(object):
    axes = None
    
    def __init__(self, chan_num=None, label=None):
        self._dt = None # delta time.
        self._proportion_missed_cache = None
        self._kwh_cache = None
        self._cache = {}
        self._is_sorted = None
        self.chan_num = chan_num
        self.label = label

    def load(self, filename, input_tz, data_type=None, start=None, end=None,
             use_cache=False, sort=False):
        """
        Args:
            filename (str): (optional) filename with full path
            input_tz (pytz.timezone)  
            
            data_type (str): (optional) one of the following:
                - "apparent_power"
                - "real_power"
            start (int or float): UNIX timestamp. Ignore all data recorded
                before this time (no timezone conversion done)
            end (int of float): UNIX timestamp. Ignore all data recorded
                after this time (no timezone conversion done)
        """
        self.data_filename = filename
        print("Loading ", self.data_filename, "... ", end="", sep="")
        
        # Load data file
        try:
            with open(self.data_filename) as data_file:
                if self._cache:
                    print(self._cache['filesize'], end=", ")
                    data_file.seek(self._cache['filesize'])
                lines = data_file.readlines()
        except IOError:
            self.data = None
            print("doesn't exist. Skipping.")
            return

        print(len(lines), end=", ")
        self.data = np.zeros(len(lines), 
                             dtype=[('datetime', datetime.datetime),
                                    ('watts', float)])
        
        if data_type is None or data_type == "real_power":
            power_column = 1
        elif data_type == "apparent_power":
            power_column = 2
        
        i = 0
        for line in lines:
            line = line.split()
            timestamp = float(line[0])
            
            if start and timestamp < start:
                continue
            if end and timestamp > end:
                print("timestamp", timestamp, "is after end =", end)
                break
            
            watts = float(line[power_column])
            date_time = datetime.datetime.fromtimestamp(timestamp, input_tz)
                    
            self.data[i] = (date_time, watts)
            i += 1
                
        print(i, end=", ")
                
        # Resize self.data if we didn't take every line
        if not i:
            self.data = None
            print("No new data loaded!")
            return
        elif i != len(lines):
            self.data = np.resize(self.data, i)

        # Calculate delta time vector
        timedeltas = self.data['datetime'][1:] - self.data['datetime'][:-1]
        self._dt = [timedelta.seconds for timedelta in timedeltas]
        self._dt = np.array(self._dt)
        
        if sort:
            self._is_sorted = self._sort()
                    
        print("Done.")

    def load_cache(self, cache_filename, input_tz):
        # Try loading the pickled cache file
        self.cache_filename = cache_filename
        try:
            pkl_file = open(self.cache_filename, "rb")
        except:
            self._cache = {}
        else:
            self._cache = pickle.load(pkl_file)
            pkl_file.close()
            
            # Backwards compatibility for when _cache stored float timestamp
            # not a datetime object:                    
            for s in ['first', 'last']:
                if self._cache.get(s + '_datetime') is None:
                    self._cache[s + '_datetime'] = datetime.datetime.fromtimestamp(
                                                     self._cache[s + '_timestamp'])
                    self._cache[s + '_datetime'] = input_tz.localize(self._cache[s + '_datetime'])

    def add_cache_to_table(self, table):
        if not self._cache:
            return table

        table.update_first_datetime(self._cache['first_datetime'])
        table.update_last_datetime(self._cache['last_datetime'])
        
        table.data_row([
          self.chan_num, self.label,
          self._cache['count'],
          self._cache['watts_min'], self._cache['watts_max'],
          self._cache['dt_min'], self._cache['dt_mean'], self._cache['dt_max'],
          self._cache['missed'],
          self._cache['kwh']
          ])
        
        return table

    def update_and_save_cache(self):
        if self.data is None:
            return
        
        if self._cache:
            self._cache['watts_min'] = min(self._cache['watts_min'], 
                                           self.data['watts'].min())
            self._cache['watts_max'] = max(self._cache['watts_max'], 
                                           self.data['watts'].max())
            self._cache['dt_min'] = min(self._cache['dt_min'], self._dt.min())
            self._cache['dt_max'] = max(self._cache['dt_max'], self._dt.max())
            total_size = self._cache['count'] + self.data.size 
            self._cache['dt_mean'] = ((self._cache['dt_mean'] *
                                      (self._cache['count']-1)) + 
                                      self._dt.sum()) / (total_size-2)
            self._cache['missed'] = ((self._cache['missed'] *
                                      self._cache['count']) + 
                                     (self._proportion_missed() * 
                                      self.data.size)) / total_size 
            self._cache['count'] = total_size
            self._cache['kwh'] += self._kwh()
        else:
            self._cache['first_datetime'] = self.data['datetime'][0]
            self._cache['watts_min'] = self.data['watts'].min()
            self._cache['watts_max'] = self.data['watts'].max()
            self._cache['dt_min'] = self._dt.min()
            self._cache['dt_max'] = self._dt.max()
            self._cache['dt_mean'] = self._dt.mean()
            self._cache['missed'] = self._proportion_missed()
            self._cache['count'] = self.data.size
            self._cache['kwh'] = self._kwh()
            
        self._cache['last_datetime'] = self.data['datetime'][-1]
        self._cache['filesize'] = os.path.getsize(self.data_filename)
            
        with open(self.cache_filename, "wb") as output:
            pickle.dump(self._cache, output) 

    def _kwh(self):
        if self.data is None:
            return 0

        if self._kwh_cache is None:
            dt_limited = np.where(self._dt>MAX_PERIOD, SAMPLE_PERIOD, self._dt)
            watt_seconds = (dt_limited * self.data['watts'][:-1]).sum()
            self._kwh_cache = watt_seconds / 3600000
            
        return self._kwh_cache
        
    def add_new_data_to_table(self, table):
        if self._dt is None or self._dt.size < 2:
            table.data_row([self.chan_num, self.label,
                                     0,'-',0,0,0,0,0,0,1,0])
            return table

        table.update_first_datetime(self.data['datetime'][0])
        table.update_last_datetime(self.data['datetime'][-1])

        is_sorted = "-" if self._is_sorted is None else self._is_sorted
        
        table.data_row([self.chan_num, self.label, self.data.size, is_sorted,
                        self.data['watts'].min(), self.data['watts'].max(),
                        self._dt.min(), self._dt.mean(), self._dt.max(),
                        self._dt.std(), self._proportion_missed(),
                        self._kwh()])

        return table
        
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
        sorted_data.sort(order='datetime')
        if (sorted_data == self.data).all():
            return True
        else:
            self.data = sorted_data
            return False
        
    def data_is_available(self):
        if self._dt is None or self.data is None:
            return False
        else:
            return True
        
    def plot_new_data(self, axes):
        if self.data is None:
            return
        
        # Power consumption
        return axes.plot(self.data['datetime'], self.data['watts'],
                              label=str(self.chan_num) + ' ' + self.label)
    
    def plot_missed_samples(self, axes, color):
        if self._dt is None:
            return
        
        for i in (self._dt > 11).nonzero()[0]:
            start = self.data['datetime'][i]
            end   = self.data['datetime'][i+1]
            rect = plt.Rectangle((start, -self.chan_num), # bottom left corner
                                 (end-start).total_seconds()/86400, # length
                                 1, # width
                                 color=color)
            axes.add_patch(rect)

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
    parser = argparse.ArgumentParser(description='Generate simple stats for '
                                                 'electricity power data logs.',
                                     epilog='example: ./powerstats.py '
                                            ' --data-dir ~/data')
       
    parser.add_argument('--data-dir', dest='base_data_dir', default=None,
                        help='directory from which to retrieve data.')
    
    parser.add_argument('--high-freq-data-dir', dest='high_freq_data_dir',
                        default=None, help='Directory for the .dat files'
                        ' recording real and apparent power, for example'
                        ' recorded using snd_card_power_meter.')
    
    parser.add_argument('--numeric-subdirs', dest='use_numeric_subdirs',
                        action='store_true',
                        help='Data is stored within numerically named subdirs '
                             'in base data dir.\n'
                             'Defaults to TRUE if data-dir is taken from'
                             ' $DATA_DIR env variable.')
    
    parser.add_argument('--labels-file',
                        default='labels.dat',
                        help='filename (without path) for labels data'
                             ' (default:\'labels.dat\').')
    
    parser.add_argument('--sort', action='store_true',
                        help='Pre-sort by date. Vital for MIT data.')
    
    parser.add_argument('--no-plot', dest='plot', action='store_false',
                        help='Do not plot  graph.')

    parser.add_argument('--html', action='store_true',
                        help='Output HTML to data-dir/html/')
    
    parser.add_argument('--html-dir', dest='html_dir',
                        help='Output stats and graphs as HTML to this directory.')    
    
    parser.add_argument('--start',
                        help='Unix timestamp to start time period,'
                             ' no TZ conversion done.') 

    parser.add_argument('--end',
                        help='Unix timestamp to end time period,'
                             ' no TZ conversion done.')

    parser.add_argument('--cache', action='store_true', dest='use_cache',
                        help='Cache data for this timeperiod, starting from '
                        'end of last cached period.')

    parser.add_argument('--input-timezone', dest='input_timezone', 
                        default=None,
                        help='Timezone of input data.'
                             ' e.g. \'UTC\' or \'Europe/London\'. This option'
                             ' overrides the timezone specified in the metadata.dat'
                             ' file, if such a file exists.  If no'
                             ' value is provided then the input timezone will'
                             ' be taken from the data-dir/metadata.dat file.'
                             ' If no such file exists then it defaults to UTC.')

    args = parser.parse_args()

    # process data dir
    if args.base_data_dir is None:
        args.base_data_dir = os.environ.get("DATA_DIR")
        args.use_numeric_subdirs = True

    if args.base_data_dir:
        args.base_data_dir = os.path.realpath(args.base_data_dir)
        if not os.path.exists(args.base_data_dir):
            sys.exit("\nERROR: " + args.base_data_dir + " does not exist.")
    else:
        sys.exit("\nERROR: Please specify a data directory either using the --data-dir \n"
                 "       command line option or using the $DATA_DIR environment variable.\n")

    # process use_numeric_subdirs
    # find the highest number subdirectory
    if args.use_numeric_subdirs:
        existing_subdirs = os.walk(args.base_data_dir).next()[1]
        
        # Remove any subdirs which contain alphabetic characters
        numeric_subdirs = [subdir for subdir in existing_subdirs 
                           if not any(char.isalpha() for char in subdir)]
    else:
        numeric_subdirs = None

    if numeric_subdirs:
        numeric_subdirs.sort()
        args.data_dir = args.base_data_dir + "/" + numeric_subdirs[-1]
    else:
        args.data_dir = args.base_data_dir        

    # process timezone
    if args.input_timezone is None:
        # look for metadata.dat file
        metadata_filename = args.data_dir + "/metadata.dat" 
        if os.path.exists(metadata_filename):
            metadata_parser = ConfigParser.RawConfigParser()
            metadata_parser.read(metadata_filename)
            args.input_timezone = metadata_parser.get("datetime", "timezone")
        else:
            args.input_timezone = 'Europe/London' # default

    # process html
    if args.html:
        args.html_dir = args.data_dir + "/html"
        
    # high freq data dir
    if args.high_freq_data_dir is None:
        args.high_freq_data_dir = args.base_data_dir + "/high-freq-mains"

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
    print("*  base data directory  = ", args.base_data_dir)
    print("*  input data directory = ", args.data_dir)
    print("*  high freq data dir   = ", args.high_freq_data_dir)
    print("*  input data timezone  = ", args.input_timezone)
    feedback_arg(args.use_numeric_subdirs, "using numeric subdirectories.")
    feedback_arg(args.sort, "pre-sorting data")
    feedback_arg(args.plot, "plotting")
    feedback_arg(args.html_dir, "outputting HTML", "to file://{}/index.html".format(args.html_dir))
    feedback_arg(args.use_cache, "caching data")
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


def load_high_freq_mains(high_freq_mains_dir, start_datetime, end_datetime, 
                         input_tz):
    
    if start_datetime is None or end_datetime is None:
        return None, None
    
    start_timestamp = time.mktime(start_datetime.timetuple())
    end_timestamp = time.mktime(end_datetime.timetuple())
    
    print("Loading high frequency mains data...")
    dir_listing = os.listdir(high_freq_mains_dir)
    
    # Find all .dat files
    # 'f' is short for 'file'
    dat_files = [f for f in dir_listing 
                 if f.startswith('mains-') and f.endswith('.dat')]

    # find set of dat files which start before end_timestamp
    dat_files_filtered = [f for f in dat_files if
                          int(f.lstrip('mains-').rstrip('.dat'))
                          < end_timestamp]
    
    if not dat_files_filtered:
        print("No high frequency dat files found with start times before",
              end_datetime)
        return None, None
    
    # open last dat file (a limitation of this code: there may be multiple dat 
    # files covering the time period between start_timestamp and end_timestamp
    # but we only open one.  In the vast majority of cases, this limitation
    # won't be an issue)
    dat_files_filtered.sort()
    data_filename = high_freq_mains_dir + "/" + dat_files_filtered[-1]

    real_power = Channel()
    real_power.label = "real power"
    real_power.load(filename=data_filename, input_tz=input_tz, 
                    data_type="real_power", 
                    start=start_timestamp)
    
    apparent_power = Channel()
    apparent_power.label = "apparent power"
    apparent_power.load(filename=data_filename, input_tz=input_tz,
                        data_type="apparent_power",
                        start=start_timestamp)
        
    return real_power, apparent_power
    

def main():
    # Load command-line arguments
    args = setup_argparser()

    # Setup Tables for storing data
    NAME_COL_WIDTH = 20
    
    new_data_table = Table(col_width=[5,NAME_COL_WIDTH,6,3] + [6,6] + [6,6,6,6] + [10, 6],
                  data_format=["{:d}","{:s}","{:d}","{}","{:.1f}","{:.1f}","{}","{:.1f}","{}","{:.1f}","{:.1%}", "{:.1f}"],
                  col_sep=1)
    
    # Create two-row header
    new_data_table.header_row([(4, ""), (2, "POWER (W)", "-"), (4, "SAMPLE PERIOD (s)", "-"), (2, "")])
    new_data_table.header_row(["#", 
                      "name", 
                      "count", 
                      "s",
                      "min", "max",
                      "min", "mean", "max", "stdev",
                      "% missed",
                      "kwh"
                      ])

    if args.use_cache:
        cache_table = Table(col_width=[5,NAME_COL_WIDTH,6] + [6,6] + [6,6,6] + [10, 6],
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

    data_available = False
    
    input_tz = pytz.timezone(args.input_timezone)
    
    # Load labels.dat
    try:
        labels = load_labels(args)
    except IOError, e:
        sys.exit(e)
    
    # Load channel data
    print("File name, seek position (if using cache), lines read, lines processed\n")
    channels = []
    for chan_num, label in labels.iteritems():
        channels.append(Channel(chan_num, label))
        
        if args.use_cache:
            cache_filename = (args.data_dir +
                              "/channel_{:d}_cache.pkl".format(chan_num))
            channels[-1].load_cache(cache_filename, input_tz)
        
        filename = args.data_dir + "/channel_{:d}.dat".format(chan_num)        
        channels[-1].load(filename, input_tz, start=args.start, end=args.end,
                          use_cache=args.use_cache, sort=args.sort)
                
    print("")

    # Setup matplotlib figure (but don't plot_new_data anything yet)
    if args.plot:
        fig = plt.figure(figsize=(20,10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        pwr_axes = plt.subplot(gs[0])
        pwr_axes.set_title("Power consumption")
        pwr_axes.set_xlabel("time")
        pwr_axes.set_ylabel("watts")
        pwr_axes.xaxis.axis_date(input_tz)          
        
        # Axes for plotting missed samples
        hit_axes = plt.subplot(gs[1])
        hit_axes.set_title("Drop-outs")  
        hit_axes.xaxis.axis_date(input_tz)  
    
    # Produce data for stats tables, update cache and plot_new_data channel data
    for chan in channels:
        
        new_data_table = chan.add_new_data_to_table(new_data_table)
        
        if args.use_cache:
            cache_table = chan.add_cache_to_table(cache_table)
            chan.update_and_save_cache()
            totals_table = chan.add_cache_to_table(totals_table)
            
        data_available |= chan.data_is_available()
        
        if args.plot and chan.data_is_available():
            pwr_line, = chan.plot_new_data(pwr_axes)
            chan.plot_missed_samples(hit_axes, color=pwr_line.get_c())
            
    
    # Load sound card power meter data (if available)
    
    if os.path.isdir(args.high_freq_data_dir):
        real_power, apparent_power = load_high_freq_mains(args.high_freq_data_dir,
                                                          new_data_table.first_datetime,
                                                          new_data_table.last_datetime,
                                                          input_tz)
        
        if real_power is not None:
            real_power.chan_num = channels[-1].chan_num + 1
            apparent_power.chan_num = channels[-1].chan_num + 2
            
            new_data_table = real_power.add_new_data_to_table(new_data_table)
            new_data_table = apparent_power.add_new_data_to_table(new_data_table)
            if args.plot:
                real_power.plot_new_data(pwr_axes)
                apparent_power.plot_new_data(pwr_axes)
            
            channels.append(real_power)
            channels.append(apparent_power) 
    
    # Output stats tables as HTML or to stdout
    if args.html_dir:
        html = "<!DOCTYPE html>\n<html>\n<body>"
        html += "<h3>NEW DATA:</h3>\n" + new_data_table.html()
        if args.use_cache and cache_table.data is not None:
            html += "<h3>OLD DATA:</h3>\n" + cache_table.html()
            html += "<h3>TOTALS:</h3>\n" + totals_table.html()
        if data_available:
            html += "<img src=\"fig.png\"/>"
        else:
            html += "<p>No new data to plot_new_data!</p>"
        html += "</body>\n</html>"
        
        with open(args.html_dir + "/index.html", "w") as html_file:
            html_file.write(html)

    else:
        # output text tables
        print("NEW DATA:", new_data_table, sep="\n")
        if args.use_cache and cache_table.data is not None:
            print("OLD DATA:", cache_table, sep="\n")
            print("TOTALS:", totals_table, sep="\n")        
        
    # Finish formatting plots and output to screen or file
    if args.plot and data_available:
        # Format axes
        hit_axes.autoscale_view()      
        hit_axes.set_xlim( pwr_axes.get_xlim() )
        hit_axes.set_ylim([-channels[-1].chan_num, 0])
        date_formatter = matplotlib.dates.DateFormatter("%d/%m\n%H:%M", 
                                                        tz=input_tz)
        hit_axes.xaxis.set_major_formatter( date_formatter )     
        pwr_axes.xaxis.set_major_formatter( date_formatter )     
        plt.tight_layout()
        
        # Shrink axes by 20% to make space for legend   
        scale_width(hit_axes, 0.8)
        scale_width(pwr_axes, 0.8)
        
        # Create and format legend
        leg = pwr_axes.legend(bbox_to_anchor=(1, 1), loc="upper left")
        for t in leg.get_texts():
            t.set_fontsize('small')
                
        # Output plot_new_data to chosen destination
        if args.html_dir:
            plt.savefig(args.html_dir + "/fig.png", bbox_inches=0)
        else:
            plt.show()

    if not data_available:
        print("No new data to plot_new_data.")


if __name__ == "__main__":
    main()
