import torchvision as tv
import torch as T

_transformer = tv.transforms.Compose([
    tv.transforms.Resize([256, 256]),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

_dataset = tv.datasets.ImageFolder('../../Datasets/horse2zebra/', _transformer)


def _sampler(pos, neg):  # 2: horse, 3: zebra
    _labels_pos = [i for i, (_, l) in enumerate(_dataset.imgs) if l == pos]
    _labels_neg = [i for i, (_, l) in enumerate(_dataset.imgs) if l == neg]
    _sampler_pos = T.utils.data.sampler.SubsetRandomSampler(_labels_pos)
    _sampler_neg = T.utils.data.sampler.SubsetRandomSampler(_labels_neg)
    return _sampler_pos, _sampler_neg


_train_sampler = _sampler(3, 2)
data_A = T.utils.data.DataLoader(_dataset, sampler=_train_sampler[0], batch_size=1)
data_B = T.utils.data.DataLoader(_dataset, sampler=_train_sampler[1], batch_size=1)

_valid_sampler = _sampler(1, 0)
valid_A = T.utils.data.DataLoader(_dataset, sampler=_train_sampler[0], batch_size=3)
valid_B = T.utils.data.DataLoader(_dataset, sampler=_train_sampler[1], batch_size=3)
