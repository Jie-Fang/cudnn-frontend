# This module defines the class "ETATimer"

import time
import sys

class ETATimer:
    # Initialize all variables (and assign user input to class variables)
    def __init__(self, message, total_updates):
        self.start = time.time()
        self.time_per = 0
        self.updates = 0
        self.total_updates = total_updates
        self.message = message

    # Update current updates; which also means update time per element
    def add_update(self):
        self.updates += 1
        self.time_per = (time.time() - self.start) / (self.updates)

    # Calculate total timer (total updates * total time per update)
    def total_time(self):
        return self.total_updates * self.time_per

    # Calculate elapsed time (current_time - start_time)
    def elapsed(self):
        return (time.time() - self.start)

    # Return float of seconds remaining ex: 10.5
    def get_eta(self):
        return self.total_time() - self.elapsed()

    # Returns ETA formatted ex: "3 days 10 hours 4 minutes 20 seconds"
    def format_time(self, time):
        sec_left = time

        day_left = int(int(sec_left) / (24*60*60))
        sec_left -= day_left*(24*60*60)
        
        hour_left = int(int(sec_left) / (60 * 60))
        sec_left -= hour_left*(60*60)

        min_left = int(int(sec_left) / (60))
        sec_left -= min_left*(60)

        result = ""
        if(day_left > 0):
            result += "%d days " % day_left
        
        if(hour_left > 0):
            result += "%d hours " % hour_left

        if(min_left > 0):
            result += "%d minutes " % min_left
       
        if(sec_left > 0):
            result += "%d seconds " % sec_left

        return result[:-1]

    # Prints current status ex: [Timing] Completed 2 of 4 (10 seconds remaining)
    def print_status(self):
        if(self.updates < self.total_updates):
            out_str = "[%s] Completed %d of %d (%s remaining)\n" % (self.message, self.updates, self.total_updates, self.format_time(self.get_eta()))

        else:
            out_str = "[%s] Completed All %d (%s elapsed time)\n" % (self.message, self.total_updates, self.format_time(self.elapsed()))

        sys.stdout.write(out_str)
        sys.stdout.flush()


    # Add an update and print the status
    def update_and_print(self):
        self.add_update()
        self.print_status()

if __name__ == "__main__":
    timer = ETATimer("Test Message of 10", 10)
    
    for i in range(10):
        timer.update_and_print()
    
    print("Done")
