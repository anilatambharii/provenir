from provenir.rewards.primitives import ExactMatchReward, FormatReward
from provenir.rewards.registry import RewardRegistry


def test_reward_registry_registers_and_scores() -> None:
    registry = RewardRegistry()
    registry.register("exact", ExactMatchReward())
    registry.register("format", FormatReward(min_length=4))

    exact = registry.get("exact")
    fmt = registry.get("format")

    assert exact.score({"prediction": "abc", "reference": "abc"}) == 1.0
    assert fmt.score({"prediction": "abcd"}) == 1.0
    assert fmt.score({"prediction": "abc"}) == 0.0
