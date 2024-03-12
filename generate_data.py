import random
import json

VARIANT_START = 10
VARIANT_END = 20

NUMBERS_START = 50
NUMBERS_END = 80

variant_numbers = random.sample(range(-100, 100),
                                random.randint(VARIANT_START, VARIANT_END))

numbers = [random.choice(variant_numbers)
           for _ in range(random.randint(NUMBERS_START, NUMBERS_END))]

with open('../../../../GitHub/VSTU/Технологии экспериментальных исследований/Lab5/test_statistical_data.json', 'w') as f:
    array_data = {"type": "ARRAY", "data": numbers}
    json.dump(array_data, f, indent=2)
    f.write('\n')
