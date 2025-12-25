defmodule TinkexCookbook.Utils.LRSchedulingTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Utils.LRScheduling

  test "linear schedule" do
    assert_in_delta LRScheduling.compute_schedule_lr_multiplier(:linear, 5, 10), 0.5, 1.0e-6
  end

  test "cosine schedule" do
    assert_in_delta LRScheduling.compute_schedule_lr_multiplier(:cosine, 0, 10), 1.0, 1.0e-6
    assert_in_delta LRScheduling.compute_schedule_lr_multiplier(:cosine, 10, 10), 0.0, 1.0e-6
  end

  test "constant schedule" do
    assert LRScheduling.compute_schedule_lr_multiplier(:constant, 3, 10) == 1.0
  end
end
