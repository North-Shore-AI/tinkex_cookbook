defmodule TinkexCookbook.TinkexApi.RequestStreamingTest do
  use ExUnit.Case, async: true

  alias Tinkex.API.Request

  test "prepare_body/4 streams large JSON payloads" do
    body = %{"data" => String.duplicate("a", 100_000)}

    assert {:ok, _headers, {:stream, stream}} =
             Request.prepare_body(body, [{"content-type", "application/json"}], nil, [])

    assert IO.iodata_to_binary(Enum.to_list(stream)) == Jason.encode!(body)
  end

  test "prepare_body/4 keeps small JSON payloads as binary" do
    body = %{"data" => "small"}

    assert {:ok, _headers, encoded} =
             Request.prepare_body(body, [{"content-type", "application/json"}], nil, [])

    assert is_binary(encoded)
    assert encoded == Jason.encode!(body)
  end
end
