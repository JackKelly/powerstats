from __future__ import print_function
import sys
import datetime

class Table:
    def __init__(self, col_width=7, data_format=None, col_sep=1):
        """
        params:
        
          - col_width (int or list of ints): the number of characters to
            use for the width of each
            
          - data_format (list of strings): a list of formatting strings, e.g.
            ["{:d}", "{:.1f}"]
          
          - col_sep (int): number of characters to separate each column."""
          
        self.header = []
        self.data = []
        self.col_width = col_width
        self.sep_col_widths = isinstance(self.col_width, list)            
        self.data_format = data_format
        self.col_sep = col_sep
        self.first_timestamp = None
        self.last_timestamp = None
        
        if (self.sep_col_widths and self.data_format and
            len(self.data_format) != len(self.col_width)):
            print("ERROR: data_format and col_width contain a different number of elements!\n"
                  "Ignoring data_format.", file=sys.stderr)
            self.data_format = None
        
    def header_row(self, h):
        self.header.append(h)
        
    def data_row(self, d):
        self.data.append(d)
        
    def update_first_timestamp(self, t):
        if self.first_timestamp:
            if t < self.first_timestamp:
                self.first_timestamp = t
        else:
            self.first_timestamp = t

    def update_last_timestamp(self, t):
        if self.last_timestamp:
            if t > self.last_timestamp:
                self.last_timestamp = t
        else:
            self.last_timestamp = t

    def html(self):
        if not self.data:
            return ""
        
        s = "" 
        
        time_details = self._time_details()
        if time_details:
            s += time_details.html()  + "\n"
            
        s += "<table border=\"1\">\n"
        for row in self.header:
            s += "  <tr>\n"
            s += self._list_to_html_row(row, header=True)
            s += "  </tr>\n"
            
        for row in self.data:
            s += "  <tr align=\"right\">\n"
            s += self._list_to_html_row(row)
            s += "  </tr>\n"
            
        s+= "</table>"
        
        return s        
        
    def __str__(self):
        if not self.data:
            return ""        
        
        s = "" 
        
        time_details = self._time_details()
        if time_details:
            s += time_details.__str__()  + "\n"
        
        for row in self.header:
            s += self._list_to_plain_text_row(row, header=True).upper()
            s += "\n"
            
        for row in self.data:
            s += self._list_to_plain_text_row(row)
            s += "\n"
            
        return s

    def _time_details(self):
        if not self.first_timestamp:
            return
        
        time_details = Table(col_width = [9,20])
        start_dt = datetime.datetime.fromtimestamp(self.first_timestamp)
        end_dt   = datetime.datetime.fromtimestamp(self.last_timestamp)
        time_details.data_row(["Start", start_dt])
        time_details.data_row(["End", end_dt])
        time_details.data_row(["Duration", end_dt - start_dt])
        return time_details
    
    def _list_to_plain_text_row(self, lst, header=False):
        text = ""
        col_i = 0
        for col in lst:
            if isinstance(col, tuple):
                cell_span = col[0]
                cell_text = col[1]
                if len(col) == 2: 
                    align = "^"
                elif len(col) == 3:
                    align = col[2] + "^"                     
            else:
                cell_text = col
                cell_span = 1
                if header:
                    align = "^"
                else:
                    align = ">"
                               
            if self.sep_col_widths:
                cell_width = sum( self.col_width[col_i:col_i+cell_span] )
            else:
                cell_width = self.col_width * cell_span

            if self.data_format and "." in self.data_format[col_i]:
                precision = self.data_format[col_i][self.data_format[col_i].find(".")+1]
                precision = int(precision)
                max_val = 10**(cell_width-precision-1)
            else:
                max_val = 10**cell_width
                    
            if (isinstance(cell_text, (int, long, float)) and 
                cell_text > max_val) :
                cell_text = "{:.0e}".format(cell_text)
            elif not header and self.data_format:   
                try:             
                    cell_text = self.data_format[col_i].format(cell_text)
                except ValueError:
                    cell_text = str(cell_text)
            else:
                cell_text = str(cell_text)
                
            cell_width += (cell_span-1)*self.col_sep
                
            format_str = "{:" + align + str(cell_width) + "}"
            col_text = format_str.format(cell_text[:cell_width])
            
            col_text += ("{:^" + str(self.col_sep) + "}").format("|")          
            
            text += col_text
            col_i += cell_span
                            
        return text

    def _list_to_html_row(self, lst, header=False):
        html = ""
        col_i = 0
        for col in lst:
            if isinstance(col, tuple):
                cell_span = col[0]
                cell_text = col[1]
            else:
                cell_text = col
                cell_span = 1
                                                  
            if not header and self.data_format:
                cell_text = self.data_format[col_i].format(cell_text)                
            else:
                cell_text = str(cell_text)
            
            if header:
                row_type = "h"
            else:
                row_type = "d"
                
            html += "    <t" + row_type                
                
            if cell_span > 1:
                html += " colspan=\"{:d}\"".format(cell_span)
            html += ">" + cell_text + "</t" + row_type + ">\n"
            
            col_i += cell_span
                                        
        return html


def main():
    data_format = ["{:d}","{:s}","{:d}","{:d}"] + ["{:.1f}"]*9
    
    table = Table(col_width=[5,11,6,3 ] + [5,5,5,6]*2 + [10],
                  data_format=data_format,
                  col_sep=1)
    
    # Create two-row header
    table.header_row([(4, ""), (4, "POWER (W)", "-"), (4, "SAMPLE PERIOD (s)", "-"), (1, "")])
    table.header_row(["#", 
                      "name", 
                      "count", 
                      "s", 
                      "min", "mean", "max", "stdev",
                      "min", "mean", "max", "stdev",
                      "% missed"
                      ])
    
    # Add a row of data
    table.data_row([5,"aggregate",1234567,2,
                    1234,92342342.234324,5.5,7.55,
                    5,5,5,7.01,
                    7])

    print(table)
    print(table.html())

if __name__ == "__main__":
    main()
        