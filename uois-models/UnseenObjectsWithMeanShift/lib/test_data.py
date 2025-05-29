from datasets.pushing_dataset import PushingDataset
from datasets.humanplay_dataset import HumanPlayDataset

# data = PushingDataset()
data = HumanPlayDataset()

print(len(data))
print(data[0]['depth'].shape)