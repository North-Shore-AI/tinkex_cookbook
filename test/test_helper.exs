# Exclude integration tests by default
# Run with: mix test --include integration
ExUnit.start(exclude: [:integration])

# Compile support modules
Code.require_file("support/mock_tokenizer.ex", __DIR__)
Code.require_file("support/mock_tokenizer_special.ex", __DIR__)
Code.require_file("support/mock_tinkex.ex", __DIR__)

Mox.defmock(TinkexCookbook.Ports.VectorStoreMock, for: TinkexCookbook.Ports.VectorStore)
Mox.defmock(TinkexCookbook.Ports.DatasetStoreMock, for: TinkexCookbook.Ports.DatasetStore)
Mox.defmock(TinkexCookbook.Ports.EmbeddingClientMock, for: TinkexCookbook.Ports.EmbeddingClient)
Mox.defmock(TinkexCookbook.Ports.LLMClientMock, for: TinkexCookbook.Ports.LLMClient)
Mox.defmock(TinkexCookbook.Ports.HubClientMock, for: TinkexCookbook.Ports.HubClient)
Mox.defmock(TinkexCookbook.Ports.BlobStoreMock, for: TinkexCookbook.Ports.BlobStore)
