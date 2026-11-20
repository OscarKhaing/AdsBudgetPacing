"""Base class shared by every pacing agent."""


class Agent:
    """A pacing agent chooses one bid multiplier alpha per hour of the day.

    The environment bids alpha * ecpc * pctr on every auction in that hour.
    Subclasses override `act`; agents that adapt or learn also override `update`.
    """

    name = "agent"

    def reset(self):
        """Called at the start of every episode (one simulated day)."""

    def act(self, obs: dict) -> float:
        """Return the bid multiplier alpha for the hour described by `obs`."""
        raise NotImplementedError

    def update(self, obs, alpha, reward, next_obs, done):
        """Observe one transition (s, a, r, s'). Default: do nothing."""
