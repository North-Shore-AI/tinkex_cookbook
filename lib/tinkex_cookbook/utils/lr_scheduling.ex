defmodule TinkexCookbook.Utils.LRScheduling do
  @moduledoc """
  Learning rate schedule helpers.
  """

  @type lr_schedule :: :linear | :cosine | :constant

  @spec compute_schedule_lr_multiplier(lr_schedule(), non_neg_integer(), pos_integer()) :: float()
  def compute_schedule_lr_multiplier(lr_schedule, step, total_steps) do
    case lr_schedule do
      :linear ->
        1.0 - step / total_steps

      :cosine ->
        0.5 * (1.0 + :math.cos(:math.pi() * step / total_steps))

      :constant ->
        1.0

      other ->
        raise ArgumentError, "Unknown learning rate schedule: #{inspect(other)}"
    end
  end
end
