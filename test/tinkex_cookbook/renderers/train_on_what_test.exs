defmodule TinkexCookbook.Renderers.TrainOnWhatTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Renderers.TrainOnWhat

  describe "TrainOnWhat enum values" do
    test "last_assistant_message/0 returns the correct string value" do
      assert TrainOnWhat.last_assistant_message() == "last_assistant_message"
    end

    test "all_assistant_messages/0 returns the correct string value" do
      assert TrainOnWhat.all_assistant_messages() == "all_assistant_messages"
    end

    test "all_messages/0 returns the correct string value" do
      assert TrainOnWhat.all_messages() == "all_messages"
    end

    test "all_tokens/0 returns the correct string value" do
      assert TrainOnWhat.all_tokens() == "all_tokens"
    end

    test "all_user_and_system_messages/0 returns the correct string value" do
      assert TrainOnWhat.all_user_and_system_messages() == "all_user_and_system_messages"
    end

    test "customized/0 returns the correct string value" do
      assert TrainOnWhat.customized() == "customized"
    end
  end

  describe "values/0" do
    test "returns all valid enum values" do
      values = TrainOnWhat.values()

      assert length(values) == 6
      assert "last_assistant_message" in values
      assert "all_assistant_messages" in values
      assert "all_messages" in values
      assert "all_tokens" in values
      assert "all_user_and_system_messages" in values
      assert "customized" in values
    end
  end

  describe "valid?/1" do
    test "returns true for valid string values" do
      assert TrainOnWhat.valid?("last_assistant_message")
      assert TrainOnWhat.valid?("all_assistant_messages")
      assert TrainOnWhat.valid?("all_messages")
      assert TrainOnWhat.valid?("all_tokens")
      assert TrainOnWhat.valid?("all_user_and_system_messages")
      assert TrainOnWhat.valid?("customized")
    end

    test "returns false for invalid values" do
      refute TrainOnWhat.valid?("invalid")
      refute TrainOnWhat.valid?("")
      refute TrainOnWhat.valid?(nil)
      refute TrainOnWhat.valid?(:last_assistant_message)
    end
  end

  describe "from_string/1" do
    test "returns {:ok, value} for valid strings" do
      assert {:ok, "last_assistant_message"} = TrainOnWhat.from_string("last_assistant_message")
      assert {:ok, "all_assistant_messages"} = TrainOnWhat.from_string("all_assistant_messages")
    end

    test "returns {:error, reason} for invalid strings" do
      assert {:error, _} = TrainOnWhat.from_string("invalid")
      assert {:error, _} = TrainOnWhat.from_string("")
    end
  end
end
