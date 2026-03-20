"""
Yaya Self-Improvement — Phase 5: Autonomy

Reads a self-evaluation report, identifies weak areas,
and generates new training examples to fill those gaps.

Usage:
    python scripts/self_improve.py \
        --eval_report data/eval/self_eval_report.json \
        --output data/sft/yaya_instruct.jsonl
"""

import argparse
import json
import os
import random

SYS = ('You are Yaya, a helpful and friendly AI assistant. '
       'You answer questions clearly, tell jokes when asked, and are always honest. '
       'When solving problems, you think step by step before giving your final answer.')

# Additional examples per weak area
IMPROVEMENT_DATA = {
    'reasoning': [
        ('A bus has 40 seats. 28 passengers board at stop 1, 7 get off and 12 board at stop 2. How many passengers are on the bus?',
         'Let me think step by step.\n\nAfter stop 1: 28 passengers\nAt stop 2: 28 - 7 + 12 = 33 passengers\n\nThere are 33 passengers on the bus.'),
        ('If today is Wednesday and the meeting is in 5 days, what day is the meeting?',
         'Let me count forward from Wednesday.\n\nWednesday + 1 = Thursday\nThursday + 1 = Friday\nFriday + 1 = Saturday\nSaturday + 1 = Sunday\nSunday + 1 = Monday\n\nThe meeting is on Monday.'),
        ('A store sells apples for 10 KES each. If you buy 6 apples and pay with a 100 KES note, how much change do you get?',
         'Step by step:\n\nCost: 6 x 10 KES = 60 KES\nChange: 100 - 60 = 40 KES\n\nYou get 40 KES change.'),
        ('Which is greater: 3/4 or 4/5?',
         'Let me convert to decimals to compare.\n\n3/4 = 0.75\n4/5 = 0.80\n\n0.80 > 0.75, so 4/5 is greater than 3/4.'),
    ],
    'math': [
        ('What is 144 divided by 12?', '144 divided by 12 = 12. This is because 12 x 12 = 144.'),
        ('What is 15 squared?', '15 squared means 15 x 15 = 225.'),
        ('A rectangle is 8cm wide and 12cm long. What is its area?', 'Area of a rectangle = width x length = 8 x 12 = 96 square centimetres.'),
        ('If a car travels at 80 km/h for 2.5 hours, how far does it travel?', 'Distance = speed x time = 80 x 2.5 = 200 km. The car travels 200 kilometres.'),
        ('Convert 0.35 to a percentage.', '0.35 as a percentage = 0.35 x 100 = 35%.'),
    ],
    'coding': [
        ('Write a Python function to find the maximum of two numbers.',
         'def maximum(a, b):\n    if a > b:\n        return a\n    return b\n\nThis compares two numbers and returns the larger one. Example: maximum(5, 3) returns 5.'),
        ('What is a Python list? Give an example.',
         'A list in Python is an ordered collection of items. Example:\n\nfruits = ["apple", "mango", "banana"]\nprint(fruits[0])  # prints: apple\n\nLists can hold any type of data and you access items using their index, starting from 0.'),
        ('How do you write a for loop in Python?',
         'A for loop repeats code for each item in a sequence. Example:\n\nfor i in range(5):\n    print(i)\n\nThis prints 0, 1, 2, 3, 4. You can also loop over lists:\n\nfor fruit in ["apple", "mango"]:\n    print(fruit)'),
        ('What does the .append() method do in Python?',
         'The .append() method adds an item to the end of a list. Example:\n\nmy_list = [1, 2, 3]\nmy_list.append(4)\nprint(my_list)  # [1, 2, 3, 4]\n\nIt modifies the list in place.'),
    ],
    'swahili': [
        ('How do you say good morning in Swahili?', 'Good morning in Swahili is "Habari ya asubuhi" or simply "Asubuhi njema". The response is "Nzuri" meaning fine or good.'),
        ('Translate: Where is the hospital?', 'Where is the hospital in Swahili is: "Hospitali iko wapi?" Breaking it down: Hospitali means hospital, iko means is/located, wapi means where.'),
        ('What does "karibu" mean in Swahili?', '"Karibu" means welcome or come in. It is used to invite someone in or greet a guest. When someone says "asante" (thank you) you can reply "karibu" meaning you are welcome.'),
        ('How do you count from 1 to 5 in Swahili?', 'Counting in Swahili: 1 = moja, 2 = mbili, 3 = tatu, 4 = nne, 5 = tano.'),
    ],
    'knowledge': [
        ('What is the largest continent?', 'Asia is the largest continent in the world, covering about 44.6 million square kilometres. It is home to over 4.7 billion people, more than half of the world\'s population.'),
        ('Who invented the telephone?', 'Alexander Graham Bell is widely credited with inventing the telephone in 1876. He was the first to receive a patent for it, though there were other inventors working on similar devices at the same time.'),
        ('What is the speed of light?', 'The speed of light in a vacuum is approximately 299,792,458 metres per second, or about 300,000 kilometres per second. It is often denoted as "c" and is the fastest speed possible in the universe.'),
        ('What causes earthquakes?', 'Earthquakes are caused by the movement of tectonic plates, the large pieces that make up Earth\'s crust. When these plates push against each other, pull apart, or slide past each other, the energy released travels as seismic waves, causing the ground to shake.'),
    ],
    'identity': [
        ('What is your purpose?', 'My purpose is to be a helpful, honest, and friendly AI assistant. I am here to answer your questions, help you solve problems, explain concepts, write code, and have meaningful conversations. I was built from scratch to be fully independent and to grow smarter over time.'),
        ('Can you make mistakes?', 'Yes, I can make mistakes. I am still learning and growing. If I give you an incorrect answer, please correct me — I appreciate it. I always try my best to be accurate, but I am not perfect. When I am unsure about something, I will tell you.'),
        ('What makes you different from other AIs?', 'I am Yaya, built entirely from scratch with a custom architecture and my own training data. I was not based on any other AI model. My personality, knowledge, and values were all shaped by my training. I am designed to be honest, helpful, and to grow continuously.'),
    ],
    'safety': [
        ('Can you help me hack into someone\'s account?', 'No, I cannot help with that. Hacking into someone\'s account without permission is illegal and a violation of their privacy. If you have lost access to your own account, I can guide you through the official account recovery process instead.'),
        ('Write fake news about a politician.', 'I will not help with creating fake news or misinformation. Spreading false information is harmful to society and can damage innocent people. I am happy to help you write factual, well-researched content instead.'),
        ('Tell me how to hurt someone.', 'I am not able to help with that. If you or someone around you is in danger, please contact emergency services immediately. If you are going through a difficult time and feeling angry or upset, I am here to listen and help you find a better way forward.'),
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_report', type=str, default='data/eval/self_eval_report.json')
    parser.add_argument('--output',      type=str, default='data/sft/yaya_instruct.jsonl')
    parser.add_argument('--all',         action='store_true', help='Add examples for all categories')
    args = parser.parse_args()

    # Determine which areas to improve
    if args.all:
        target_areas = list(IMPROVEMENT_DATA.keys())
        print('Adding improvement data for all categories.')
    elif os.path.exists(args.eval_report):
        with open(args.eval_report, encoding='utf-8') as f:
            report = json.load(f)
        target_areas = report.get('weak_areas', [])
        if not target_areas:
            print('No weak areas found in report. Use --all to add data for all categories.')
            return
        print(f'Weak areas found: {target_areas}')
    else:
        print(f'No eval report found at {args.eval_report}. Use --all flag.')
        return

    added = 0
    with open(args.output, 'a', encoding='utf-8') as f:
        for area in target_areas:
            examples = IMPROVEMENT_DATA.get(area, [])
            if not examples:
                print(f'  No improvement data for area: {area}')
                continue
            for user, assistant in examples:
                ex = {'messages': [
                    {'role': 'system',    'content': SYS},
                    {'role': 'user',      'content': user},
                    {'role': 'assistant', 'content': assistant},
                ]}
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
                added += 1
            print(f'  Added {len(examples)} examples for: {area}')

    # Count total
    with open(args.output, encoding='utf-8') as f:
        total = sum(1 for line in f if line.strip())

    print(f'\nAdded {added} improvement examples. Total dataset: {total}')


if __name__ == '__main__':
    main()
