import torch.nn.functional as F


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class StudentTeacherOptim:

    def __init__(self, student_params, teacher_params, momentum):

        self.studen_params = student_params
        self.teacher_params = teacher_params
        self.m = momentum

    def step(self):
        # teacher parameters are updated through the momentum rule
        for t_param, s_param in zip(self.teacher_params, self.studen_params):
            t_param.data = t_param.data * self.m + s_param.data * (1. - self.m)
            


# losses
def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    -> REPLACED Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    batch_size = input_logits.shape[0] # to normalize across batch
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / (num_classes * batch_size)

