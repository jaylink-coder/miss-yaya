"""Phase 6: Code and Structured Output."""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FILE = os.path.join(REPO_ROOT, "data/sft/yaya_phase6_code.jsonl")
SYS = "You are Yaya, a helpful AI with strong programming skills. You write clean, working code with clear explanations."

random.seed(46)

def qa(q, a):
    return {"messages": [{"role": "system", "content": SYS}, {"role": "user", "content": q}, {"role": "assistant", "content": a}]}

CODE = [
    qa("Write a Python function to check if a number is even or odd.",
"Here is a simple function:\n\n```python\ndef even_or_odd(n):\n    return 'even' if n % 2 == 0 else 'odd'\n\nprint(even_or_odd(4))  # even\nprint(even_or_odd(7))  # odd\n```\n\nThe `%` operator gives the remainder after division. If remainder is 0, the number is even."),

    qa("How do I read a text file line by line in Python?",
"```python\nwith open('myfile.txt', 'r') as f:\n    for line in f:\n        print(line.strip())  # removes newline characters\n```\n\nThe `with` block closes the file automatically. Reading line by line is memory-efficient for large files."),

    qa("Write Python to calculate the average of a list of numbers.",
"```python\ndef average(numbers):\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)\n\nscores = [78, 85, 92, 65, 88]\nprint(f'Average: {average(scores):.1f}')  # Average: 81.6\n```"),

    qa("Write Python to count word frequency in a sentence.",
"```python\nfrom collections import Counter\n\ndef word_count(sentence):\n    words = sentence.lower().split()\n    return dict(Counter(words))\n\ntext = 'the cat sat on the mat and the cat'\nprint(word_count(text))\n# the:3  cat:2  sat:1  on:1  mat:1  and:1\n```"),

    qa("Write a Python function to find the largest number in a list.",
"```python\ndef find_largest(numbers):\n    if not numbers:\n        return None\n    largest = numbers[0]\n    for n in numbers:\n        if n > largest:\n            largest = n\n    return largest\n\nprices = [450, 230, 890, 120, 670]\nprint(find_largest(prices))  # 890\n# Or simply: max(prices)\n```"),

    qa("Write a simple dictionary for country capitals in Python.",
"```python\ncapitals = {\n    'Kenya': 'Nairobi',\n    'Tanzania': 'Dodoma',\n    'Uganda': 'Kampala'\n}\n\nprint(capitals['Kenya'])                 # Nairobi\nprint(capitals.get('Rwanda', 'Unknown')) # Unknown\ncapitals['Rwanda'] = 'Kigali'           # add entry\nprint(len(capitals))                     # 4\n```"),

    qa("Write a Python calculator that handles +, -, *, /.",
"```python\ndef calculate(a, op, b):\n    if op == '+': return a + b\n    if op == '-': return a - b\n    if op == '*': return a * b\n    if op == '/':\n        if b == 0:\n            return 'Error: division by zero'\n        return a / b\n    return 'Unknown operator'\n\nprint(calculate(10, '+', 5))  # 15\nprint(calculate(10, '/', 0))  # Error: division by zero\n```"),

    qa("Explain what a Python list comprehension does.",
"A list comprehension is a concise way to build a list.\n\n```python\n# Traditional loop\nsquares = []\nfor x in range(1, 6):\n    squares.append(x**2)\n# [1, 4, 9, 16, 25]\n\n# Same as list comprehension:\nsquares = [x**2 for x in range(1, 6)]\n\n# With a filter condition:\nevens = [x for x in range(10) if x % 2 == 0]\n# [0, 2, 4, 6, 8]\n```\n\nFormat: `[expression for item in iterable if condition]`"),

    qa("Give me a JSON object for a Kenyan product.",
'```json\n{\n  "id": "KE-001",\n  "name": "Kenyan AA Coffee",\n  "category": "beverages",\n  "price_kes": 850,\n  "in_stock": true,\n  "tags": ["coffee", "organic", "premium"],\n  "origin": "Kirinyaga County, Kenya"\n}\n```\n\nJSON rules: strings use double quotes, numbers have no quotes, booleans are `true`/`false`, arrays use `[]`.'),

    qa("What is wrong with this JSON: { name: John, age: 25 }",
'Two errors:\n\n1. **Keys must be double-quoted**: `name` must be `"name"`, `age` must be `"age"`\n2. **String values must be double-quoted**: `John` must be `"John"`\n\nCorrect:\n```json\n{"name": "John", "age": 25}\n```\n\nNumbers (25) do not need quotes. JSON never accepts single quotes.'),

    qa("Find the bug:\n```python\nfor i in range(5):\nprint(i)\n```",
"Bug: **missing indentation**. `print(i)` must be indented inside the loop.\n\nFixed:\n```python\nfor i in range(5):\n    print(i)\n```\n\nPython uses indentation (4 spaces is standard) to define code blocks."),

    qa("What is wrong:\n```python\ndef greet(name)\n    print('Hello ' + name)\n```",
"Missing **colon** after `def greet(name)`. It should be `def greet(name):`\n\nFixed:\n```python\ndef greet(name):\n    print('Hello ' + name)\n```"),

    qa("Find the bug:\n```python\nnum = input('Enter a number: ')\nprint(num * 2)\n```",
"Bug: `input()` returns a **string**. `'5' * 2` gives `'55'` (string repetition), not `10`.\n\nFix by converting to integer:\n```python\nnum = int(input('Enter a number: '))\nprint(num * 2)  # Now gives 10 for input 5\n```"),

    qa("Write Python to check if a word is a palindrome.",
"```python\ndef is_palindrome(word):\n    cleaned = word.lower().replace(' ', '')\n    return cleaned == cleaned[::-1]\n\nprint(is_palindrome('racecar'))  # True\nprint(is_palindrome('hello'))    # False\nprint(is_palindrome('level'))    # True\n```\n\n`[::-1]` reverses a string using Python slice notation."),

    qa("Write a Python function that returns only unique values from a list.",
"```python\ndef unique_values(lst):\n    return list(set(lst))\n\nnumbers = [1, 2, 2, 3, 3, 3, 4]\nprint(unique_values(numbers))  # [1, 2, 3, 4]\n\n# To preserve original order:\ndef unique_ordered(lst):\n    seen = set()\n    result = []\n    for x in lst:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result\n```"),
]

ALL = CODE * 5
random.shuffle(ALL)
os.makedirs(os.path.join(REPO_ROOT, "data/sft"), exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for item in ALL:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Generated {len(ALL)} code examples -> {OUT_FILE}")
