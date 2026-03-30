from abc import ABC, abstractmethod

class ModelConfig:
    """Configuration object for models - demonstrates composition."""
    def __init__(self, model_name: str, learning_rate: float = 0.01, epochs: int = 10):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.epochs = epochs

    def __repr__(self):
        return f"[Config] {self.model_name} | lr={self.learning_rate} | epochs={self.epochs}"


class BaseModel(ABC):
    """Abstract Base Class - demonstrates abstraction and class attribute."""
    model_count = 0

    def __init__(self, config: ModelConfig):
        self.config = config
        BaseModel.model_count += 1

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def evaluate(self, data):
        pass


class LinearRegressionModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def train(self, data):
        print(f"LinearRegression: Training on {len(data)} samples for {self.config.epochs} epochs (lr={self.config.learning_rate})")

    def evaluate(self, data):
        print("LinearRegression: Evaluation MSE = 0.042")


class NeuralNetworkModel(BaseModel):
    def __init__(self, config: ModelConfig, layers: list):
        super().__init__(config)
        self.layers = layers

    def train(self, data):
        print(f"NeuralNetwork {self.layers}: Training on {len(data)} samples for {self.config.epochs} epochs (lr={self.config.learning_rate})")

    def evaluate(self, data):
        print("NeuralNetwork: Evaluation Accuracy = 91.5%")


class DataLoader:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)


class Trainer:
    def __init__(self, model: BaseModel, dataloader: DataLoader):
        self.model = model
        self.dataloader = dataloader

    def run(self):
        print(f"--- Training {self.model.config.model_name} ---")
        self.model.train(self.dataloader)
        self.model.evaluate(self.dataloader)


# ==================== RUN THE FRAMEWORK ====================
if __name__ == "__main__":
    linear_config = ModelConfig("LinearRegression", learning_rate=0.01, epochs=10)
    neural_config = ModelConfig("NeuralNetwork", learning_rate=0.001, epochs=20)
    
    print(linear_config)
    print(neural_config)
    
    linear_model = LinearRegressionModel(linear_config)
    neural_model = NeuralNetworkModel(neural_config, layers=[64, 32, 1])
    
    print(f"Models created: {BaseModel.model_count}")
    
    dataset = [0.1, 0.2, 0.3, 0.4, 0.5]
    dataloader = DataLoader(dataset)
    
    trainer_linear = Trainer(linear_model, dataloader)
    trainer_linear.run()
    
    trainer_nn = Trainer(neural_model, dataloader)
    trainer_nn.run()