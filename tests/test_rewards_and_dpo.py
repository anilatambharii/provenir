from provenir.rewards.primitives import ExactMatchReward, FormatReward
from provenir.train.algorithms import DPOConfig, DPOTrainer


def test_reward_primitives_and_dpo_config() -> None:
    exact = ExactMatchReward()
    assert exact.score({"prediction": "Paris", "reference": "Paris"}) == 1.0

    fmt = FormatReward(min_length=3)
    assert fmt.score({"prediction": "abc"}) == 1.0
    assert fmt.score({"prediction": "ab"}) == 0.0

    cfg = DPOConfig(beta=0.1)
    trainer = DPOTrainer(config=cfg)
    assert trainer.config.beta == 0.1
