import unittest
from solve import solve_LU, solve_MP, SingularMatrixException, is_correct_LU, is_correct_MP
from make_matrixes import open_sle

ACCURACY = 8
correct_cases = open_sle(sle_type='correct')
singular_cases = open_sle(sle_type='singular')
result_equality_LU = is_correct_LU(cases=correct_cases, accuracy=ACCURACY)
result_equality_MP = is_correct_MP(cases=correct_cases, accuracy=ACCURACY)

class TestSLE(unittest.TestCase): #SLE -  System of linear equations (СЛАУ)

    def test_exception_singular_LU(self): # достаточно протестировать по одному случаю с нулевым определителем
        matrix, vector = singular_cases[-1]
        self.assertRaises(SingularMatrixException, solve_LU, matrix, vector)

    def test_exception_singular_MP(self): # достаточно протестировать по одному случаю с нулевым определителем
        matrix, vector = singular_cases[-1]
        self.assertRaises(SingularMatrixException, solve_MP, matrix, vector)

    def test_correct_LU(self):
        for is_equal in result_equality_LU:         # проверяется, что округленные до восьмого
            with self.subTest():                 # знака вектора решений numpy и solve_LU равны
                self.assertIs(is_equal, True)

    def test_correct_MP(self):
        for is_equal in result_equality_MP:
            with self.subTest():
                self.assertIs(is_equal, True)

if __name__ == '__main__':
    unittest.main()
