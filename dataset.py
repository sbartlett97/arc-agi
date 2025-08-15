import json
import os

class ARCDataset:
    def __init__(self, data_dir, split, mode):
        self.data_dir = data_dir
        self.split = split
        self.mode = mode
        self.challenge_data, self.suport_data, self.solutions = self.load_data(split)

    def load_data(self, split):
        with open(os.path.join("./", self.data_dir, f'arc-agi_{self.split}_challenges.json'), 'r') as f:
            challenge_data = json.load(f)
        
        if self.mode == "train":
            with open(os.path.join("./", self.data_dir, f'arc-agi_{self.split}_solutions.json'), 'r') as f:
                solutions = json.load(f)
        else:
            solutions = None

        challenges = [] # List of challenges in format [[[..], [..], .., [..]], .., [[..], [..], .., [..]]]
        challenge_supports = [] # List of support data for challenge at index - in format [[{"input": [[..],[..],...,[..]], "output": [[..],[..],...,[..]]}], ..., [{"input": [[..],[..],...,[..]], "output": [[..],[..],...,[..]]}]]
        
        for key in challenge_data.keys():
            challenges.append(challenge_data[key]["test"])
            challenge_supports.append(challenge_data[key]["train"])

        if solutions is not None:
            solutions = [solutions[key] for key in challenge_data.keys()]

        return challenges, challenge_supports, solutions

    def __len__(self):
        return len(self.challenge_data)
    
    def __getitem__(self, idx):
        if self.mode == "train":
            return self.challenge_data[idx], self.suport_data[idx], self.solutions[idx]
        else:
            return self.challenge_data[idx], self.suport_data[idx]
    
    def __iter__(self):
        if self.mode == "train":
            return iter(zip(self.challenge_data, self.challenge_supports, self.solutions))
        else:
            return iter(zip(self.challenge_data, self.challenge_supports))
    

if __name__ == "__main__":
    dataset = ARCDataset(data_dir="arc-prize-2025", split="training", mode="train")
    print(len(dataset))
    print("Challenge: ", json.dumps(dataset[0][0], indent=2))
    print("Support: ", json.dumps(dataset[0][1], indent=2))
    print("Solution: ", json.dumps(dataset[0][2], indent=2))