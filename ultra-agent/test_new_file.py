def add_numbers(a, b):
    """Adds two numbers."""
    return a + b

def multiply_numbers(a, b):
    """Multiplies two numbers."""
    return a * b

# Simple test case for addition
if __name__ == "__main__":
    # Test addition
    add_result = add_numbers(5, 3)
    print(f"5 + 3 = {add_result}")
    assert add_result == 8
    print("Addition test passed!")

    # Test multiplication
    multiply_result = multiply_numbers(5, 3)
    print(f"5 * 3 = {multiply_result}")
    assert multiply_result == 15
    print("Multiplication test passed!")