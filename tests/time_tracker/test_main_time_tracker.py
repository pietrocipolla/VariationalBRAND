from unittest import TestCase

from root.controller.time_tracker.clones.main_time_tracker import main_time_tracker


class Test(TestCase):
    def test_main_time_tracker(self):
        main_time_tracker()
