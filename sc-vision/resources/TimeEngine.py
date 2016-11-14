import datetime
import time

class TimeEngine:
    def __init__(self,
                program_start_time = None,
                time_limit_enabled = False,
                hours_to_run = 0,
                minutes_to_run = 0,
                seconds_to_run = 0,
                capture_interval = None
                ):
        if program_start_time is None:
            self.program_start_time = self._now()
        else:
            self.program_start_time = program_start_time
        self.time_limit_enabled = time_limit_enabled
        self.hours_to_run = hours_to_run
        self.minutes_to_run = minutes_to_run
        self.seconds_to_run = seconds_to_run
        self.time_limit = float(hours_to_run) * 3600.0
        self.time_limit += float(minutes_to_run) * 60.0
        self.time_limit += float(seconds_to_run)
        if capture_interval is None:
            self.capture_interval = 5.0
        else:
            self.capture_interval = capture_interval

        self.timer = None

    def _now(self):
        return datetime.datetime.fromtimestamp(time.time())

    def _difference_between_times(self, start_time, stop_time = None):
        if stop_time is None:
            stop_time = self._now()
        time_difference = stop_time - start_time
        seconds_elapsed = time_difference.total_seconds()
        return seconds_elapsed

    def time_limit_reached(self):
        if self.time_limit_enabled:
            seconds_elapsed = self._difference_between_times(self.program_start_time)
            return seconds_elapsed >= self.time_limit
        else:
            return False

    def start_interval_timer(self):
        self.timer = self._now()

    def wait_for_interval_expiration(self):
        while self._difference_between_times(self.timer) < self.capture_interval:
            pass

    def wait(self, duration):
        time.sleep(duration)

# EOF
