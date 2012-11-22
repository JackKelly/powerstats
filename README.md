Produce simple statistics from [REDD](http://redd.csail.mit.edu)-formatted
power data.  Statistics include (for each channel):

* Min, mean, max and stdev for time between consecutive samples.
  (This is useful for checking on the health of sensors.)
* Min, mean, max and stdev for power consumption
* Is the channel sorted by timecode?

## Related projects

* [rfm_ecomanager_logger](/JackKelly/rfm_ecomanager_logger) Python script for
logging power data from the [rfm_edf_ecomanager RF base unit](/JackKelly/rfm_edf_ecomanager/)
code running on a Nanode.