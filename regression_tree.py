class TreeNode:
    def __init__(self, examples: list[dict]) -> None:
        self.examples = examples
        self.left = None
        self.right = None
        self.split_point = None
        self.features = self.examples[0].keys().remove("bpd")

    def split(self):
        best_split = {
            "feature": None,
            "split_value": None,
            "mse": float("inf"),
            "index": None}

        for feature in self.features:
            self.examples.sort(key=lamba x: x[feature])



class RegressionTree:
    def __init__(self, examples):
        # Don't change the following two lines of code.
        self.root = TreeNode(examples)
        self.train()

    def train(self):
        # Don't edit this line.
        self.root.split()

    def predict(self, example):
        # Write your code here.
        return 0