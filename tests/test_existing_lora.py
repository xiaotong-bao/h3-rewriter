import sys
from pathlib import Path
import unittest
import torch
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grpo'))
from existing_lora import freeze_reference


class ExistingLoraTests(unittest.TestCase):
    def test_reference_preserves_fp32_values_and_is_independent(self):
        base = torch.nn.Sequential(torch.nn.Linear(3, 3)).to(torch.bfloat16)
        config = LoraConfig(r=2, lora_alpha=4, target_modules=['0'])
        model = get_peft_model(base, config)
        for name, p in model.named_parameters():
            if '.default.' in name:
                p.data = torch.full(p.shape, .1234567, dtype=torch.float32)
        model.add_adapter('ref', config)
        freeze_reference(model)
        for name, p in model.named_parameters():
            if '.default.' in name:
                ref = model.get_parameter(name.replace('.default.', '.ref.'))
                self.assertEqual(ref.dtype, p.dtype)
                self.assertTrue(torch.equal(ref, p))
                self.assertFalse(ref.requires_grad)
                self.assertNotEqual(ref.data_ptr(), p.data_ptr())
                before = ref.detach().clone()
                p.data.add_(1.)
                self.assertTrue(torch.equal(ref, before))


if __name__ == '__main__':
    unittest.main()
