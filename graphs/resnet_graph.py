from graphs.base_graph import BIGGraph, NodeType
from torchvision import transforms
import torchvision
from torch.utils.data import Subset
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.models.resnet as resnet
import models.resnets as resnet2

from model_merger import ModelMerge
from matching_functions import match_tensors_identity, match_tensors_zipit
from copy import deepcopy

def set_seed(seed):
    """Sets the seed for numpy and pytorch"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
class ResNetGraph(BIGGraph):

    def __init__(self, model,
                 shortcut_name='shortcut',
                 layer_name='layer',
                 head_name='linear',
                 num_layers=3):
        super().__init__(model)

        self.shortcut_name = shortcut_name
        self.layer_name = layer_name
        self.num_layers = num_layers
        self.head_name = head_name

    def add_basic_block_nodes(self, name_prefix, input_node):
        shortcut_prefix = name_prefix + f'.{self.shortcut_name}'
        shortcut_output_node = input_node
        if shortcut_prefix in self.named_modules and len(self.get_module(shortcut_prefix)) > 0:
            # There's a break in the skip connection here, so add a new prefix
            input_node = self.add_nodes_from_sequence('', [NodeType.PREFIX], input_node)

            shortcut_output_node = self.add_nodes_from_sequence(
                name_prefix=shortcut_prefix,
                list_of_names=['0', '1'],
                input_node=input_node
            )


        skip_node = self.add_nodes_from_sequence(
            name_prefix=name_prefix,
            list_of_names=[
                'conv1', 'bn1', NodeType.PREFIX, 'conv2', 'bn2', NodeType.SUM],
            input_node=input_node
        )

        self.add_directed_edge(shortcut_output_node, skip_node)

        return skip_node

    def add_bottleneck_block_nodes(self, name_prefix, input_node):
        shortcut_prefix = name_prefix + f'.{self.shortcut_name}'
        shortcut_output_node = input_node
        if shortcut_prefix in self.named_modules and len(self.get_module(shortcut_prefix)) > 0:
            # There's a break in the skip connection here, so add a new prefix
            input_node = self.add_nodes_from_sequence('', [NodeType.PREFIX], input_node)

            shortcut_output_node = self.add_nodes_from_sequence(
                name_prefix=shortcut_prefix,
                list_of_names=['0', '1'],
                input_node=input_node
            )

        skip_node = self.add_nodes_from_sequence(
            name_prefix=name_prefix,
            list_of_names=[
                'conv1', 'bn1', NodeType.PREFIX,
                'conv2', 'bn2', NodeType.PREFIX,
                'conv3', 'bn3', NodeType.SUM],
            input_node=input_node
        )

        self.add_directed_edge(shortcut_output_node, skip_node)

        return skip_node

    def add_layer_nodes(self, name_prefix, input_node):
        source_node = input_node

        for layer_index, block in enumerate(self.get_module(name_prefix)):
            block_class = block.__class__.__name__

            if block_class == 'BasicBlock':
                source_node = self.add_basic_block_nodes(name_prefix+f'.{layer_index}', source_node)
            elif block_class == 'Bottleneck':
                source_node = self.add_bottleneck_block_nodes(name_prefix+f'.{layer_index}', source_node)
            else:
                raise NotImplementedError(block_class)

        return source_node

    def graphify(self):
        input_node = self.create_node(node_type=NodeType.INPUT)
        input_node = self.add_nodes_from_sequence('', ['conv1', 'bn1'], input_node, sep='')

        for i in range(1, self.num_layers+1):
            input_node = self.add_layer_nodes(f'{self.layer_name}{i}', input_node)

        input_node = self.add_nodes_from_sequence('',
            [NodeType.PREFIX, 'avgpool', self.head_name, NodeType.OUTPUT], input_node, sep='')

        return self


def resnet20(model):
    return ResNetGraph(model)

def resnet50(model):
    return ResNetGraph(model,
                       shortcut_name='downsample',
                       head_name='fc',
                       num_layers=4)

def resnet18(model):
    return resnet50(model)


def get_cifar100_dataset(args, validation_split):
    """Creates augmented train, validation, and test data loaders."""
    cifar_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    cifar_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    # Data

    transforms_1 = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ]
    )

    transforms_2 = transforms.Compose(
        [
            transforms.AugMix(severity=5, mixture_width=6),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ]
    )

    transform_trains = [
        transforms_1,
        transforms_2,
    ]  # TODO maybe stronger different augmentations(?)

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ]
    )

    trainsets = [
        torchvision.datasets.CIFAR100(
            root=args.data_dir, train=True, download=True, transform=transform
        )
        for transform in transform_trains
    ]

    valset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=False, download=True, transform=transform_test
    )

    return trainsets, valset


def get_dataloaders(args, validation_split=0.0):
    set_seed(args.seed)

    trainsets, testset = get_cifar100_dataset(args, validation_split)

    # subsetting with train_data_fraction from args
    # uniform between classes
    if args.train_data_fraction < 1.0:
        # make map of class and indices of the samples
        target_to_sample_idxs = [[] for _ in range(len(trainsets[0].classes))]
        for i, (_, target) in enumerate(trainsets[0]):
            target_to_sample_idxs[target].append(i)

        # shuffle each class
        for sample_idxs in target_to_sample_idxs:
            np.random.shuffle(sample_idxs)
            # only keep fraction
            sample_idxs[:] = sample_idxs[
                : int(len(sample_idxs) * args.train_data_fraction)
            ]

        # concat
        subsampled_idxs = np.concatenate(target_to_sample_idxs)
        # shuffle again
        np.random.shuffle(subsampled_idxs)
        # make subset
        trainsets = [Subset(trainset, subsampled_idxs) for trainset in trainsets]

    train_loaders = [
        DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            num_workers=2,
        )
        for trainset in trainsets
    ]

    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    return train_loaders, test_loader, test_loader  # TODO validation_split






if __name__ == '__main__':

    import argparse
    import os
    import torch.nn.functional as F

    # set cuda save math
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    batch_size = 256
    args = argparse.Namespace()
    args.batch_size = batch_size
    args.test_batch_size = batch_size
    args.baseline = False
    args.seed = 1
    args.data = "cifar100"
    args.train_data_fraction = .5
    args.device = torch.device("cuda:0")
    args.data_dir = "/datasets01/cifar100/022818/data"
    train_loaders, valid_loader, _ = get_dataloaders(args)


    model = resnet.resnet18(num_classes=100).eval()
    models = [deepcopy(model).to(args.device) for _ in range(2)]
    del model

    optimizers = [
        torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        for model in models
    ]

    if os.path.exists("model2.pt"):
        models[0].load_state_dict(torch.load("model1.pt"))
        models[1].load_state_dict(torch.load("model2.pt"))
    else:
        for i in range(100):
            n_corrects = [0 for _ in range(len(models))]
            for batch_idx, data_and_targets in enumerate(zip(*train_loaders)):
                data_for_models = [data.to(args.device) for data, _ in data_and_targets]
                target = [target for _, target in data_and_targets][0].to(
                    args.device
                )  # targets don't change
                [optimizer.zero_grad() for optimizer in optimizers]
                outputs = [model(data) for model, data in zip(models, data_for_models)]

                loss = sum(
                    F.cross_entropy(postact, target) for postact in outputs
                ) / len(outputs)
                loss.backward()
                [optimizer.step() for optimizer in optimizers]

                preds = [postact.argmax(dim=1, keepdim=True) for postact in outputs]
                for i, pred in enumerate(preds):
                    n_corrects[i] += pred.eq(target.view_as(pred)).sum().item()

            print(
                ",".join(
                    [
                        str(n_correct / len(train_loaders[0].dataset))
                        for n_correct in n_corrects
                    ]
                )
            )

        # save both models:
        torch.save(models[0].state_dict(), "model1.pt")
        torch.save(models[1].state_dict(), "model2.pt")



    with torch.no_grad():
        graph1 = resnet18(deepcopy(models[0])).graphify()
        graph2 = resnet18(deepcopy(models[1])).graphify()

        merge = ModelMerge(graph1, graph2, device=args.device)
        merge.transform(deepcopy(models[0]).eval(), train_loaders[0], transform_fn=match_tensors_zipit)
        models.append(merge.merged_model)

        for i in torch.linspace(0.25, 0.75, 3):
            model4 = deepcopy(models[0])
            for param1, param2, param3 in zip(
                models[0].parameters(), models[1].parameters(), model4.parameters()
            ):
                param3.data = (1 - i) * param1.data + i * param2.data
            models.append(model4)

        # eval models:
        for loader in [train_loaders[0], valid_loader]:
            accs = []
            for model in models:
                model.eval()
                correct = 0
                with torch.no_grad():
                    for data, target in loader:
                        data, target = data.to(args.device), target.to(args.device)
                        output = model(data)
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                accs.append(correct / len(loader.dataset))
            print(accs)
