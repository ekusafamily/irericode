import json
import sys

def calculate_fibonacci(n: int) -> list[int]:
    """
    Calculates Fibonacci numbers up to the nth term.
    Returns a list of Fibonacci numbers.
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]
    
    fib_sequence = [0, 1]
    while len(fib_sequence) < n:
        next_fib = fib_sequence[-1] + fib_sequence[-2]
        fib_sequence.append(next_fib)
        
    return fib_sequence

def save_to_json(data: dict, filename: str):
    """
    Saves the given dictionary data to a JSON file.
    """
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully saved Fibonacci results to {filename}")
    except IOError as e:
        print(f"Error saving to file {filename}: {e}", file=sys.stderr)

def main():
    """
    Main function to handle command-line arguments and execution.
    Expects one argument: n (the number of Fibonacci terms to calculate).
    """
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <n>", file=sys.stderr)
        sys.exit(1)
    
    try:
        n = int(sys.argv[1])
    except ValueError:
        print("Error: n must be an integer.", file=sys.stderr)
        sys.exit(1)

    if n < 0:
        print("Error: n must be non-negative.", file=sys.stderr)
        sys.exit(1)

    fib_numbers = calculate_fibonacci(n)
    
    results = {
        "n_terms": n,
        "fibonacci_sequence": fib_numbers
    }
    
    output_filename = "fibonacci_results.json"
    save_to_json(results, output_filename)

if __name__ == "__main__":
    main()