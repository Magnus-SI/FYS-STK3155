import numpy as np
import pytest
from Ridge import inv_svd, Ridge
from project1 import OLS
from terrain import OLS3

class TestClass():
    def test_inv_svd(self):
        """
        Test that inv_svd actually gives the inverse
        """
        A =  np.random.rand(10,10)
        tol = 1e-10
        AinvA = A@inv_svd(A)
        cond = True
        for i in range(10):
            for j in range(10):
                if j != i and abs(AinvA[i,j])>tol:
                    cond = False
                if i==j and abs(AinvA[i,j]-1.)>tol:
                    cond = False
        assert cond

    def test_OLS(self):
        """
        Test that OLS gives correct answers within a tolerance
        Tests a first order polynomial case  with exact answer calulated
        """
        x = np.array([1,2,4,5,7])
        y = np.array([2,3,7,5,11])
        betas_precalculated = np.array([1/3., 79./57])
        X = np.polynomial.polynomial.polyvander(x,1)
        betas_calculated = OLS(X,y)
        tol = 1e-14
        assert np.all(np.abs(betas_calculated-betas_precalculated)<tol)

    def test_OLS3(self):
        """
        Test that OLS3 gives correct answers wihhin a tolerance
        Tests a first order polynomial case  with exact answer calulated
        """
        x = np.array([1,2,4,5,7])
        y = np.array([2,3,7,5,11])
        betas_precalculated = np.array([1/3., 79./57])
        X = np.polynomial.polynomial.polyvander(x,1)
        betas_calculated = OLS3(X,y)
        tol = 1e-14
        assert np.all(np.abs(betas_calculated-betas_precalculated)<tol)

    def test_Ridge(self):
        """
        Test that ridge gives correct answers within a tolerance when lambda=0
        Tests a first order polynomial case  with exact answer calulated
        """
        R = Ridge(0)
        x = np.array([1,2,4,5,7])
        y = np.array([2,3,7,5,11])
        betas_precalculated = np.array([1/3., 79./57])
        X = np.polynomial.polynomial.polyvander(x,1)
        betas_calculated = R(X,y)
        tol = 1e-14
        assert np.all(np.abs(betas_calculated-betas_precalculated)<tol)