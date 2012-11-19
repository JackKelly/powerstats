import powerstats
import unittest
import numpy as np

class TestChannel(unittest.TestCase):
    
    def setUp(self):
        self.chan = powerstats.Channel()
        self.chan.data = np.empty(5, 
                             dtype=[('timestamp', np.uint32), ('watts', float)])
        self.chan.data[0] = (1, 10)
        self.chan.data[1] = (3, 3)
        self.chan.data[2] = (5, 0.5)
        self.chan.data[3] = (2, 200)
        self.chan.data[4] = (4, 4)
        
        print(self.chan.data)
        
    def test_sort(self):
        is_sorted = self.chan._sort()
        
        correct_matrix = np.empty(5, 
                             dtype=[('timestamp', np.uint32), ('watts', float)])
        correct_matrix[0] = (1, 10)
        correct_matrix[1] = (2, 200)
        correct_matrix[2] = (3, 3)
        correct_matrix[3] = (4, 4)
        correct_matrix[4] = (5, 0.5)
        
        self.assertFalse(is_sorted)
        self.assertTrue((correct_matrix == self.chan.data).all())
        
        print(self.chan.data)


if __name__ == "__main__":
    unittest.main()
