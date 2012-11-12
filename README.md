Produce simple statistics from [REDD](http://redd.csail.mit.edu)-formatted
power data.  Statistics include:

* Mean, min, max and stdev for time between consecutive samples for each channel.
  (This is useful for checking on the health of sensors.)
  
* Is the channel sorted by timecode?

                      |-----POWER (W)-----|  |---SAMPLE PERIOD-(s)-|
  #  NAME         S?  MEAN  MIN  MAX  STDEV    MEAN  MIN  MAX  STDEV
 10  aggregate    1   2032 2030 2999   1000  1234.0 1023 1000  123.5