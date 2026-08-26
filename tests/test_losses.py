import unittest
import torch
from nue.losses import TrainingLosses

class TestLosses(unittest.TestCase):
    def test_ntp_loss(self):
        losses = TrainingLosses()
        logits = torch.randn(2, 4, 10, requires_grad=True)
        targets = torch.randint(0, 10, (2, 4))
        loss = losses.ntp_loss(logits, targets)
        self.assertIsNotNone(loss.grad_fn)

    def test_ad_loss(self):
        losses = TrainingLosses()
        student_logits = torch.randn(2, 4, 10, requires_grad=True)
        teacher_logits = torch.randn(2, 4, 10)
        targets = torch.randint(0, 10, (2, 4))
        loss = losses.ad_loss(student_logits, teacher_logits, targets)
        self.assertIsNotNone(loss.grad_fn)

if __name__ == '__main__':
    unittest.main()
