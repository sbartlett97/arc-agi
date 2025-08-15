import json
import os

import numpy as np

class ARCDataset:
    def __init__(self, data_dir, split, mode, augment_support=False):
        self.data_dir = data_dir
        self.split = split
        self.mode = mode
        self.challenge_data, self.suport_data, self.solutions = self.load_data(split)
        if augment_support:
            self.suport_data = self.augment_support()

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
    
    def augment_support(self):
        """
        Augment the support data by adding 100 samples to the support data.
        """
        augmented_support_data = self.suport_data.copy()
         
        support_0_augmented = augmented_support_data[0].copy()
        support_1_augmented = augmented_support_data[1].copy()

        for _ in range(100):
            support_0_augmented["input"] = support_0_augmented["input"] + np.ones(support_0_augmented["input"].shape)
            support_0_augmented["output"] = support_0_augmented["output"] + np.ones(support_0_augmented["output"].shape)
            support_1_augmented["input"] = support_1_augmented["input"] + np.ones(support_1_augmented["input"].shape)
            support_1_augmented["output"] = support_1_augmented["output"] + np.ones(support_1_augmented["output"].shape)
            augmented_support_data.append(support_0_augmented)
            augmented_support_data.append(support_1_augmented)
            support_0_augmented = augmented_support_data[-2].copy()
            support_1_augmented = augmented_support_data[-1].copy()
        return augmented_support_data

if __name__ == "__main__":
    dataset = ARCDataset(data_dir="arc-prize-2025", split="training", mode="train")
    print(len(dataset))
    print("Challenge: ", json.dumps(dataset[0][0], indent=2))
    print("Support: ", json.dumps(dataset[0][1], indent=2))
    print("Solution: ", json.dumps(dataset[0][2], indent=2))