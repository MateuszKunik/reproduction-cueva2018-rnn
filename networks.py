import torch
import torch.nn as nn


class ContinuousTimeRNN(nn.Module):
    def __init__(
            self,
            num_inputs: int,
            num_outputs: int,
            num_units: int,
            noise_std: float,
            time_step: float,
            time_constant: float,
            rec_init: str = "orthogonal",
    ):
        super().__init__()
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_units = num_units

        self.noise_std = noise_std
        self.time_step = time_step
        self.time_constant = time_constant
        self.relaxation_rate = time_step / time_constant

        self.input_weights = nn.Linear(
            num_inputs, num_units, bias=False
        )
        self.recurrent_weights = nn.Linear(
            num_units, num_units, bias=False
        )
        self.output_weights = nn.Linear(
            num_units, num_outputs, bias=False
        )
        self.state_bias = nn.Parameter(
            torch.zeros(num_units)
        )
        self._init_weights(rec_init)


    def _init_weights(self, rec_init):
        nn.init.normal_(
            self.input_weights.weight,
            mean=0.0, std=1.0 / (self.num_inputs ** 0.5)
        )

        if rec_init == "orthogonal":
            nn.init.orthogonal_(self.recurrent_weights.weight)
        elif rec_init == "gaussian":
            nn.init.normal_(
                self.recurrent_weights.weight,
                mean=0.0, std=1.5 / (self.num_units ** 0.5)
            )

        nn.init.zeros_(self.output_weights.weight)
        nn.init.zeros_(self.state_bias)


    def forward(self, inputs, initial_state=None):
        batch_size, num_steps, _ = inputs.shape

        if initial_state is not None:
            state = initial_state
        else:
            state = torch.zeros(
                size=(batch_size, self.num_units),
                device=inputs.device,
                dtype=inputs.dtype
            )

        activities = []
        outputs = []

        for t in range(num_steps):
            input_t = inputs[:, t, :]

            drive = (
                self.recurrent_weights(torch.tanh(state)) +
                self.input_weights(input_t) +
                self.state_bias
            )
            state_noise = self.noise_std * torch.randn_like(state)
            next_state = (
                state + self.relaxation_rate * (-state + drive + state_noise)
            )
            activity_t = torch.tanh(next_state)
            activities.append(activity_t)

            output_t = self.output_weights(activity_t)
            outputs.append(output_t)

            state = next_state

        activities = torch.stack(activities, dim=1)
        outputs = torch.stack(outputs, dim=1)
        
        return activities, outputs