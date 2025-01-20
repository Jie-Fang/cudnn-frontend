import collections # For namedtuple
import re          # For regular expressions

from collections import OrderedDict
from helpers.utility     import split_comma, get_shell_list
from helpers.utility_py3 import *
from sys         import exc_info
from copy        import deepcopy
       
def wrap_val(val, none_val, bound):
    
    if val == None:
        return none_val
    
    if val < 0:
        return bound + val
        
    return val
    
def get_range_from_slice(slice_val, bound):
    start = wrap_val(slice_val.start, 0, bound)
    
    stop = wrap_val(slice_val.stop, bound, bound)
        
    step = slice_val.step
    
    if step == None:
        step = 1
    elif step < 0:
        start, stop = stop, start
        
        step *= -1
        
    return range(start, stop, step)
    
def get_range(slice_or_idx, bound):
    if isinstance(slice_or_idx, slice):
        return get_range_from_slice(slice_or_idx, bound)
        
    if slice_or_idx == None:
        raise Exception("Index value is None?")
        
    return range(wrap_val(slice_or_idx, None, bound), wrap_val(slice_or_idx, None, bound) + 1, 1)
    
def get_empty_2d_array(size):
    if len(size) != 2:
        raise Exception("Cannot create empty matrix with size %s" % str(size))
        
    result = []
    
    for col_i in range(size[0]):
        result.append( [""] * size[1] )
        
    return result
            
class Table:
    def __init__(self, size):
        if len(size) != 2:
            raise Exception("Cannot construct table with size: %s" % str(size))
            
        if size[0] == 0 or size[1] == 0:
            raise Exception("Table dimensions can't have any 0: %s" % str(size))
            
        self.col_count = size[0]
        self.row_count = size[1]
        
        self.cells = get_empty_2d_array(size)
        
    def resize(self, new_size):
        new_cells = get_empty_2d_array(new_size)
        
        for col_idx in range(new_size[0]):
            for row_idx in range(new_size[1]):
                if col_idx < self.col_count and row_idx < self.row_count:
                    new_cells[col_idx][row_idx] = self.cells[col_idx][row_idx]
                    
        self.col_count = new_size[0]
        self.row_count = new_size[1]
        
        self.cells = new_cells
        
    def __setitem__(self, pos, values):
        if len(pos) != 2:
            raise Exception("Invalid position in table: %s" % str(pos))
            
        (col, row) = pos
        
        max_col = col
        
        if isinstance(max_col, slice):
            max_col = max_col.stop
            
        max_row = row
        
        if isinstance(max_row, slice):
            max_row = max_row.stop
            
        if max_col >= self.col_count or max_row >= self.row_count:
            self.resize((max(max_col+1, self.col_count), max(max_row+1, self.row_count)))
            
        try:
            if isinstance(col, slice) and isinstance(row, slice):
                for (val_col_idx, self_col_idx) in enumerate(get_range_from_slice(col, self.col_count)):
                    for (val_row_idx, self_row_idx) in enumerate(get_range_from_slice(row, self.row_count)):
                        self.cells[self_col_idx][self_row_idx] = values[val_col_idx][val_row_idx]
                        
            elif isinstance(col, slice):
                for (val_col_idx, self_col_idx) in enumerate(get_range_from_slice(col, self.col_count)):
                    self_row_idx = wrap_val(row, None, self.row_count)
                    
                    self.cells[self_col_idx][self_row_idx] = values[val_col_idx]
                    
            elif isinstance(row, slice):
                for (val_row_idx, self_row_idx) in enumerate(get_range_from_slice(row, self.row_count)):
                    self_col_idx = wrap_val(col, None, self.col_count)
                    
                    self.cells[self_col_idx][self_row_idx] = values[val_row_idx]
                    
            else:
                self.cells[col][row] = values

                
        except Exception as e:
            # Store traceback info (to find where real error spawned)
            t, v, tb = exc_info()
            
            table_size = (self.col_count, self.row_count)
            
            # Re-raise exception with line info
            exception = Exception("[SPREADSHEET] Error setting table position %s with size %s" % (str(pos), str(table_size)))
            raise t(exception).with_traceback(tb)
        
    def __getitem__(self, pos):
        if len(pos) != 2:
            raise Exception("Invalid position in table: %s" % str(pos))
            
        (col, row) = pos
        
        try:
            if isinstance(col, slice) and isinstance(row, slice):
                
                col_range = get_range_from_slice(col, self.col_count)
                row_range = get_range_from_slice(row, self.row_count)
                
                result = get_empty_2d_array((len([idx for idx in col_range]), len([idx for idx in row_range])))
                
                for (val_col_idx, self_col_idx) in enumerate(col_range):
                    for (val_row_idx, self_row_idx) in enumerate(row_range):
                        result[val_col_idx][val_row_idx] = self.cells[self_col_idx][self_row_idx]
                        
            elif isinstance(col, slice):
                result = []
                
                for (val_col_idx, self_col_idx) in enumerate(get_range_from_slice(col, self.col_count)):
                    self_row_idx = wrap_val(row, None, self.row_count)
                    
                    result.append(self.cells[self_col_idx][self_row_idx])
                    
                return result
                    
            elif isinstance(row, slice):
                result = []
                
                for (val_row_idx, self_row_idx) in enumerate(get_range_from_slice(row, self.row_count)):
                    self_col_idx = wrap_val(col, None, self.col_count)
                    
                    result.append(self.cells[self_col_idx][self_row_idx])
                    
                return result
            else:
                return self.cells[col][row]
            
        except Exception as e:
            # Store traceback info (to find where real error spawned)
            t, v, tb = exc_info()
            
            table_size = (self.col_count, self.row_count)
            
            # Re-raise exception with line info
            raise t, Exception("[SPREADSHEET] Error getting table position %s with size %s" % (str(pos), str(table_size))), tb

        return result
        
    def __str__(self):
        return "\n".join([",".join(self[:, row_idx]) for row_idx in range(self.row_count)])
            
def extract_result(results, extract_value):
    if(results == None):
        return ""

    if(results.parsed == None):
        return ""

    extracted = getattr(results.parsed, extract_value)

    if(extracted == None):
        return ""

    return getattr(extracted, extracted._fields[0])

class Spreadsheet:
    def __init__(self, layers, extract_value):
        # Initialize member variables
        self.split_names = OrderedDict()
        self.disjoint_map = OrderedDict()
        
        for layer in layers:
            disjoint_str = str(layer.test_diff_flags)
            
            if disjoint_str not in self.disjoint_map:
                self.disjoint_map[disjoint_str] = len(self.disjoint_map)
                
            if layer.split_name not in self.split_names:
                self.split_names[layer.split_name] = len(self.split_names)
            
        self.table = Table((len(self.disjoint_map) + 2, len(self.split_names) + 1))
        
        self.table[0,  0] = "Layer Name (Split)"
        self.table[-1, 0] = "Layer Flags (In Common)"

        self.table[1:-1, 0] = self.disjoint_map.keys()
        
        self.table[0,  1:] = self.split_names.keys()
        
        self.extract_value = extract_value
        
    def add_run(self, layer, results):
        split_idx = self.split_names[layer.split_name]
        
        diff_flags_str = str(layer.test_diff_flags)     
        
        disjoint_idx = self.disjoint_map[diff_flags_str]
            
        if self.table[disjoint_idx+1, split_idx+1] != "":
            raise Exception("Conflict found for %s with column %s" % (layer.split_name, layer))
            
        self.table[disjoint_idx+1, split_idx+1] = str(extract_result(results, self.extract_value))
        
    def generate(self, file_name):
        if(file_name == None):
            return

        # Output perf data in csv format
        with open(file_name, 'w') as sheet_file:
            sheet_file.write(str(self.table))

if __name__ == "__main__":
    table = Table((4, 4))
    
    table[:, 1] = ["apple", "tomato", "strawberry", "orange"]
    
    table[1:-1, 2] = ["john", "jacob", "jingle", "heimer"]
    
    print(str(table))
