from sklearn import metrics
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from utils.pytorch_utils import forward
from sklearn.metrics import precision_recall_fscore_support


class Evaluator(object):
    def __init__(self, model,num_pretrain_class=None, device='cuda:0'):
        """Evaluator.

        Args:
          model: object
        """
        self.model = model
        self.num_pretrain_class = num_pretrain_class
        self.device = device
        
    def evaluate(self, data_loader):
        """Forward evaluation data and calculate statistics.

        Args:
          data_loader: object

        Returns:
          statistics: dict, 
              {'average_precision': (classes_num,), 'auc': (classes_num,)}
        """

        # Forward
        output_dict = forward(
            model=self.model, 
            generator=data_loader, 
            return_target=True)
        # (audios_num, classes_num)
        clipwise_output = output_dict['clipwise_output'][:, :-1]

        # for c in clipwise_output:
        #     print(c)

        target = output_dict['target']    # (audios_num, classes_num)
        target = torch.tensor(target)
        if self.num_pretrain_class is not None:
            batch_size, _ = target.shape
            target = torch.cat([torch.zeros(batch_size, self.num_pretrain_class).to(self.device), target.to(self.device)], dim=-1)
                # Loss
        if target.is_cuda:
            target = target.cpu()
            target = target.numpy()
        # average_precision = metrics.average_precision_score(
        #     target, clipwise_output, average=None)

        # auc = metrics.roc_auc_score(target, clipwise_output, average=None)
        
        target_acc = np.argmax(target, axis=1)
        clipwise_output_acc = np.argmax(clipwise_output, axis=1)
        print('Returned target_acc and clipwise_output_acc')

        return target_acc,clipwise_output_acc
        # acc = accuracy_score(target_acc, clipwise_output_acc)

        # f1_score, precision, recall, _ = precision_recall_fscore_support(target_acc, clipwise_output_acc,
        #                                                                  average='weighted')
        # # print(target_acc, clipwise_output_acc, acc,f1_score, precision, recall)
        # statistics = {'average_precision': average_precision, 'accuracy': acc, 'f1_score': f1_score, 'precision':precision,'recall':recall}

        # return statistics