"""Simple calculator class with basic arithmetic operations."""


class Calculator:
    """A simple calculator that supports addition, subtraction, multiplication, and division."""

    def add(self, a: float, b: float) -> float:
        """Return the sum of a and b."""
        return a + b

    def subtract(self, a: float, b: float) -> float:
        """Return the difference of a minus b."""
        return a - b

    def multiply(self, a: float, b: float) -> float:
        """Return the product of a and b."""
        return a * b

    def divide(self, a: float, b: float) -> float:
        """Return the result of dividing a by b.

        Raises:
            ValueError: if b is zero.
        """
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b

    def __add__(self, other: float) -> float:
        """Add another number to the result of add(self, other).

        This enables operator overloading syntax: calc + value
        """
        if not isinstance(other, (int, float)):
            return NotImplemented
        return self.add(other, 0)

    def __sub__(self, other: float) -> float:
        """Subtract another number from the result of add(self, other).

        This enables operator overloading syntax: calc - value
        """
        if not isinstance(other, (int, float)):
            return NotImplemented
        return self.subtract(0, other)

    def __mul__(self, other: float) -> float:
        """Multiply by another number using the result of add(self, other).

        This enables operator overloading syntax: calc * value
        """
        if not isinstance(other, (int, float)):
            return NotImplemented
        return self.multiply(1, other)

    def __truediv__(self, other: float) -> float:
        """Divide by another number using the result of divide(self, other).

        This enables operator overloading syntax: calc / value
        """
        if not isinstance(other, (int, float)):
            return NotImplemented
        return self.divide(1, other)