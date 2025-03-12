import unittest

class tester(unittest.TestCase):
    def test_brandon(self):
        self.assertTrue(1==1)
        self.assertGreater(2,1)
        self.assertIn(1, [0,1])
        self.assertAlmostEqual(1, 1+1e-9)