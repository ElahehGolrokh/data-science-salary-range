from metaflow import FlowSpec, step, Parameter
from omegaconf import OmegaConf


class Pipeline(FlowSpec):

    @step
    def start(self):
        print("Starting pipeline...")
        self.next(self.check_conditions)

    @step
    def check_conditions(self):
        print("Checking conditions...")
        self.next(self.prepare)

    @step
    def prepare(self):
        print("Preparing data...")
        self.next(self.train)

    @step
    def train(self):
        print("Training model...")
        self.next(self.evaluate)

    @step
    def evaluate(self):
        print("Evaluating model...")
        self.next(self.export)

    @step
    def export(self):
        print("Exporting model...")
        self.next(self.end)

    @step
    def end(self):
        print('done.')


if __name__ == '__main__':
    Pipeline()