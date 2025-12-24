defmodule TinkexCookbook.PortsTest do
  use ExUnit.Case, async: false

  import Mox

  alias TinkexCookbook.Ports
  alias TinkexCookbook.Ports.VectorStore

  setup :verify_on_exit!

  describe "new/1" do
    test "overrides a port adapter with module-only reference" do
      ports = Ports.new(ports: [vector_store: TinkexCookbook.Ports.VectorStoreMock])

      assert {TinkexCookbook.Ports.VectorStoreMock, []} =
               Ports.resolve(ports, :vector_store)
    end

    test "overrides a port adapter with module and opts" do
      ports =
        Ports.new(ports: [vector_store: {TinkexCookbook.Ports.VectorStoreMock, [foo: :bar]}])

      assert {TinkexCookbook.Ports.VectorStoreMock, [foo: :bar]} =
               Ports.resolve(ports, :vector_store)
    end
  end

  describe "delegation" do
    test "vector store calls are routed to the adapter" do
      ports =
        Ports.new(ports: [vector_store: {TinkexCookbook.Ports.VectorStoreMock, [region: :test]}])

      TinkexCookbook.Ports.VectorStoreMock
      |> expect(:get_or_create_collection, fn [region: :test], "docs", %{} ->
        {:ok, :collection}
      end)

      assert {:ok, :collection} = VectorStore.get_or_create_collection(ports, "docs", %{})
    end
  end
end
