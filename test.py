import json

test_file = json.load(open("arc-prize-2025/arc-agi_training_challenges.json"))

print(json.dumps(test_file["ff805c23"]["train"], indent=2))

# print(json.dumps(test_file["ff805c23"], indent=2))