"""Unit tests for the Calculator class."""

import pytest

from calculator import Calculator


class TestCalculator:
    """Tests for Calculator.add, subtract, multiply, divide."""

    def setup_method(self):
        self.calc = Calculator()

    def test_add(self):
        assert self.calc.add(2, 3) == 5
        assert self.calc.add(-1, 1) == 0
        assert self.calc.add(0, 0) == 0

    def test_subtract(self):
        assert self.calc.subtract(5, 3) == 2
        assert self.calc.subtract(3, 5) == -2
        assert self.calc.subtract(0, 0) == 0

    def test_multiply(self):
        assert self.calc.multiply(2, 3) == 6
        assert self.calc.multiply(-1, 1) == -1
        assert self.calc.multiply(0, 5) == 0

    def test_divide(self):
        assert self.calc.divide(6, 3) == 2
        assert self.calc.divide(5, 2) == 2.5
        assert self.calc.divide(-10, 2) == -5

    def test_divide_by_zero(self):
        with pytest.raises(ValueError, match="Cannot divide by zero."):
            self.calc.divide(1, 0)

    def test_operator_overloading(self):
        """Test operator overloading interfaces (__add__, __sub__, __mul__, __truediv__)."""

        # Test __add__
        result = self.calc.__add__(5)
        assert result == 5  # add(5, 0) = 5

        # Test __sub__
        result = self.calc.__sub__(3)
        assert result == -3  # subtract(0, 3) = -3

        # Test __mul__
        result = self.calc.__mul__(4)
        assert result == 4  # multiply(1, 4) = 4

        # Test __truediv__
        result = self.calc.__truediv__(2)
        assert result == 0.5  # divide(1, 2) = 0.5

    def test_operator_overloading_with_method_syntax(self):
        """Verify operator overloading produces same results as original methods."""

        # __add__ should match add(a, b) where b=0
        assert self.calc.__add__(5) == self.calc.add(5, 0)

        # __sub__ should match subtract where first arg is 0
        assert self.calc.__sub__(3) == self.calc.subtract(0, 3)

        # __mul__ should match multiply where first arg is 1
        assert self.calc.__mul__(4) == self.calc.multiply(1, 4)

        # __truediv__ should match divide where first arg is 1
        assert self.calc.__truediv__(2) == self.calc.divide(1, 2)