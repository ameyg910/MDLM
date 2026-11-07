from maskforge.diffusion.loss import mdlm_loss

import torch 

def test_mdlm_loss():
    input_ids = torch.zeros((2, 4), dtype=torch.long)
    mask_positions = torch.tensor([
        [False, True, True, False], 
        [False, True, True, True],
    ])
    t = torch.tensor([0.2, 0.8])
    other_logit = torch.log(torch.exp(torch.tensor(2.0))-1)

    logits = torch.zeros((2, 4, 2))
    logits[:, :, 1] = other_logit

    loss = mdlm_loss(
        logits=logits,
        input_ids=input_ids,
        mask_positions=mask_positions,
        t=t,
    )

    expected_loss = 13.75 

    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-5)


def test_no_true_values(): 
        input_ids = torch.zeros((2, 4), dtype=torch.long)
        mask_positions = torch.tensor([
            [False, False, False, False], 
            [False, False, False, False],
        ])
        t = torch.tensor([0.2, 0.8])
        other_logit = torch.log(torch.exp(torch.tensor(2.0))-1)
    
        logits = torch.zeros((2, 4, 2))
        logits[:, :, 1] = other_logit
    
        loss = mdlm_loss(
            logits=logits,
            input_ids=input_ids,
            mask_positions=mask_positions,
            t=t,
        )
    
        expected_loss = 0.0
    
        assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-5)