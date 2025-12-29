# Exclude integration tests by default
# Run with: mix test --include integration
ExUnit.start(exclude: [:integration])

# Compile support modules
Code.require_file("support/mock_tokenizer.ex", __DIR__)
Code.require_file("support/mock_tokenizer_special.ex", __DIR__)
Code.require_file("support/mock_tinkex.ex", __DIR__)

# Define mocks using CrucibleTrain port behaviours
Mox.defmock(TinkexCookbook.Mocks.VectorStoreMock, for: CrucibleTrain.Ports.VectorStore)
Mox.defmock(TinkexCookbook.Mocks.DatasetStoreMock, for: CrucibleTrain.Ports.DatasetStore)
Mox.defmock(TinkexCookbook.Mocks.EmbeddingClientMock, for: CrucibleTrain.Ports.EmbeddingClient)
Mox.defmock(TinkexCookbook.Mocks.LLMClientMock, for: CrucibleTrain.Ports.LLMClient)
Mox.defmock(TinkexCookbook.Mocks.HubClientMock, for: CrucibleTrain.Ports.HubClient)
Mox.defmock(TinkexCookbook.Mocks.BlobStoreMock, for: CrucibleTrain.Ports.BlobStore)
Mox.defmock(TinkexCookbook.Mocks.TrainingClientMock, for: CrucibleTrain.Ports.TrainingClient)
