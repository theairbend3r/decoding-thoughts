class StimulusClassifierConfig:
    def __init__(self):
        self.class_ignore_list = ["person", "fungus", "plant"]
        self.label_level = 0
        self.validation_dataset_size = 0.2
        self.model_input_num_channel = 3
        self.epochs = 20
        self.learning_rate = 0.001
        self.batch_size = 64
